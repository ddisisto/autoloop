"""Sliding compressibility, stationarity assessment, and autocorrelation."""

import numpy as np
import pandas as pd

from utils import compressibility

AUTOCORR_MAX_LAG = 2000


def sliding_compressibility(decoded_texts: pd.Series, window_size: int) -> np.ndarray:
    """Compute compressibility over a sliding window of tokens.

    Returns array of length len(decoded_texts). Positions 0..window_size-2 are
    NaN (insufficient tokens for a full window). Use comp_stats() from
    analyze.summary to get NaN-safe scalar summaries; use the raw array only
    for time-series plotting or slicing where positional alignment matters.
    """
    texts = decoded_texts.to_numpy()
    n = len(texts)
    results = np.full(n, np.nan)

    for i in range(window_size - 1, n):
        window_text = "".join(texts[i - window_size + 1 : i + 1])
        results[i] = compressibility(window_text.encode("utf-8"))

    return results


def stationarity_blocks(
    series: np.ndarray, n_blocks: int = 5
) -> dict:
    """Split a series into n_blocks and compute per-block statistics.

    Returns dict with block means, stds, linear trend slope, and classification.
    """
    block_size = len(series) // n_blocks
    trimmed = series[: block_size * n_blocks]
    blocks = trimmed.reshape(n_blocks, block_size)

    means = blocks.mean(axis=1)
    stds = blocks.std(axis=1)
    slope = np.polyfit(range(n_blocks), means, 1)[0]

    overall_std = trimmed.std()
    total_drift = abs(slope * n_blocks)
    if overall_std > 0 and total_drift > 0.1 * overall_std:
        classification = "transient"
    else:
        block_mean_std = means.std()
        if overall_std > 0 and block_mean_std > 0.05 * overall_std:
            classification = "structured"
        else:
            classification = "stationary"

    return {
        "block_means": means.tolist(),
        "block_stds": stds.tolist(),
        "slope": float(slope),
        "total_drift": float(total_drift),
        "overall_std": float(overall_std),
        "classification": classification,
    }


def entropy_autocorrelation(entropy: np.ndarray, max_lag: int = AUTOCORR_MAX_LAG) -> np.ndarray:
    """Normalized autocorrelation of entropy series at lags 0..max_lag.

    Returns array of length max_lag+1 where index i is the autocorrelation at lag i.
    """
    x = entropy - entropy.mean()
    var = np.dot(x, x)
    if var == 0:
        return np.zeros(max_lag + 1)
    n = len(x)
    max_lag = min(max_lag, n - 1)
    acf = np.correlate(x, x, mode="full")
    mid = len(acf) // 2
    acf = acf[mid:mid + max_lag + 1] / var
    return acf

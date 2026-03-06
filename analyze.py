"""Post-hoc analysis for autoloop runs.

Computes derived metrics from generation Parquet files:
- Output compressibility over sliding windows
- Stationarity assessment (5-block comparison)
- Per-run summary statistics
"""

import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compressibility(text: bytes) -> float:
    """Compression ratio: compressed_size / original_size. Lower = more compressible."""
    compressed = gzip.compress(text, compresslevel=6)
    return len(compressed) / len(text)


def sliding_compressibility(decoded_texts: pd.Series, window_size: int) -> np.ndarray:
    """Compute compressibility over a sliding window of tokens.

    Returns one value per step starting from index window_size-1.
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

    # Classification heuristic: trend is "significant" if slope * n_blocks
    # exceeds 10% of the overall std
    overall_std = trimmed.std()
    total_drift = abs(slope * n_blocks)
    if overall_std > 0 and total_drift > 0.1 * overall_std:
        classification = "transient"
    else:
        # Check for structured non-stationarity: block means vary but no trend
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


def summarize_run(df: pd.DataFrame) -> dict:
    """Compute summary statistics for a single run."""
    exp = df[df.phase == "experiment"]

    return {
        "n_prefill": int((df.phase == "prefill").sum()),
        "n_experiment": len(exp),
        "entropy_mean": float(exp.entropy.mean()),
        "entropy_std": float(exp.entropy.std()),
        "entropy_min": float(exp.entropy.min()),
        "entropy_max": float(exp.entropy.max()),
        "log_prob_mean": float(exp.log_prob.mean()),
        "log_prob_std": float(exp.log_prob.std()),
        "eos_count": int(exp.eos.sum()),
        "eos_rate": float(exp.eos.mean()),
    }


def analyze_run(parquet_path: Path, context_length: int) -> dict:
    """Full analysis pipeline for a single run.

    Returns dict with summary stats, stationarity, and compressibility data.
    """
    df = pd.read_parquet(parquet_path)
    exp = df[df.phase == "experiment"].reset_index(drop=True)

    summary = summarize_run(df)

    # Stationarity on entropy
    entropy_stationarity = stationarity_blocks(exp.entropy.to_numpy())

    # Compressibility: primary W=L, secondary W=L//4
    log.info("Computing compressibility W=%d for %s", context_length, parquet_path.name)
    comp_primary = sliding_compressibility(exp.decoded_text, context_length)

    w_secondary = max(context_length // 4, 16)
    log.info("Computing compressibility W=%d for %s", w_secondary, parquet_path.name)
    comp_secondary = sliding_compressibility(exp.decoded_text, w_secondary)

    # Stationarity on primary compressibility (skip NaN prefix)
    comp_valid = comp_primary[~np.isnan(comp_primary)]
    comp_stationarity = stationarity_blocks(comp_valid)

    return {
        "summary": summary,
        "entropy_stationarity": entropy_stationarity,
        "compressibility_stationarity": comp_stationarity,
        "compressibility_primary": comp_primary,
        "compressibility_secondary": comp_secondary,
    }

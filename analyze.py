"""Post-hoc analysis for autoloop runs.

Computes derived metrics from generation Parquet files:
- Output compressibility over sliding windows (arbitrary W values)
- Stationarity assessment (5-block comparison)
- Per-run summary statistics

Results are cached as .analysis.pkl sidecars, keyed by window sizes.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from utils import compressibility

log = logging.getLogger(__name__)


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


def _cache_path(parquet_path: Path, window_sizes: list[int]) -> Path:
    """Return the path for a cached analysis result."""
    w_tag = "_".join(f"W{w}" for w in sorted(window_sizes))
    return parquet_path.with_suffix(f".{w_tag}.analysis.pkl")


def _is_cache_valid(parquet_path: Path, cache_path: Path) -> bool:
    """Check if cached analysis exists and is newer than the parquet."""
    if not cache_path.exists():
        return False
    return cache_path.stat().st_mtime > parquet_path.stat().st_mtime


def analyze_run(parquet_path: Path, window_sizes: list[int]) -> dict:
    """Full analysis pipeline for a single run.

    Args:
        parquet_path: Path to the run's parquet file.
        window_sizes: List of window sizes for compressibility computation.

    Returns dict with summary stats, stationarity, and compressibility data.
    Compressibility stored as {W: array} dict keyed by window size.
    Uses disk cache to avoid recomputing.
    """
    cache = _cache_path(parquet_path, window_sizes)
    if _is_cache_valid(parquet_path, cache):
        log.info("Loading cached analysis for %s", parquet_path.name)
        with open(cache, "rb") as f:
            return pickle.load(f)

    df = pd.read_parquet(parquet_path)
    exp = df[df.phase == "experiment"].reset_index(drop=True)

    summary = summarize_run(df)
    entropy_stationarity = stationarity_blocks(exp.entropy.to_numpy())

    comp = {}
    for w in window_sizes:
        log.info("Computing compressibility W=%d for %s", w, parquet_path.name)
        comp[w] = sliding_compressibility(exp.decoded_text, w)

    # Stationarity on largest window compressibility
    w_max = max(window_sizes)
    comp_valid = comp[w_max][~np.isnan(comp[w_max])]
    comp_stationarity = stationarity_blocks(comp_valid)

    result = {
        "summary": summary,
        "entropy_stationarity": entropy_stationarity,
        "compressibility_stationarity": comp_stationarity,
        "compressibility": comp,
        "window_sizes": sorted(window_sizes),
    }

    with open(cache, "wb") as f:
        pickle.dump(result, f)
    log.info("Cached analysis to %s", cache.name)

    return result

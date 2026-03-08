"""Post-hoc analysis for autoloop runs.

Computes derived metrics from generation Parquet files:
- Output compressibility over sliding windows (arbitrary W values)
- Stationarity assessment (5-block comparison)
- Per-run summary statistics

Results are cached in a single .analysis.pkl per parquet file.
Cache is incrementally updatable: requesting new window sizes only
computes the missing ones. Cache is invalidated when parquet changes.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from utils import compressibility

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------

def default_window_sizes(L: int) -> list[int]:
    """Default window sizes for a given context length."""
    return [L, max(L // 4, 16)]


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


def summarize_run(exp: pd.DataFrame) -> dict:
    """Compute summary statistics from experiment-phase DataFrame."""
    return {
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


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _cache_path(parquet_path: Path) -> Path:
    """Single cache file per parquet."""
    return parquet_path.with_suffix(".analysis.pkl")


def _load_cache(parquet_path: Path) -> dict | None:
    """Load cache if valid (parquet hasn't changed since cache was written)."""
    cp = _cache_path(parquet_path)
    if not cp.exists():
        return None
    pq_mtime = parquet_path.stat().st_mtime
    with open(cp, "rb") as f:
        cache = pickle.load(f)
    if cache.get("parquet_mtime") != pq_mtime:
        log.info("Cache stale for %s (parquet changed)", parquet_path.name)
        return None
    return cache


def _save_cache(parquet_path: Path, cache: dict) -> None:
    """Save cache with current parquet mtime."""
    cache["parquet_mtime"] = parquet_path.stat().st_mtime
    cp = _cache_path(parquet_path)
    with open(cp, "wb") as f:
        pickle.dump(cache, f)
    log.info("Cached analysis to %s", cp.name)


def _load_experiment_df(parquet_path: Path) -> pd.DataFrame:
    """Load experiment-phase rows from a parquet file."""
    df = pd.read_parquet(parquet_path)
    return df[df.phase == "experiment"].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_run(
    parquet_path: Path,
    window_sizes: list[int],
    exp: pd.DataFrame | None = None,
) -> dict:
    """Analyze a single run, computing only what's missing from cache.

    Args:
        parquet_path: Path to the run's parquet file.
        window_sizes: List of window sizes for compressibility computation.
        exp: Pre-loaded experiment-phase DataFrame. If None, loaded from parquet.

    Returns dict with summary stats, stationarity, and compressibility data.
    Compressibility stored as {W: array} dict keyed by window size.

    Cache is incremental: previously computed window sizes are reused,
    only missing ones are computed. Cache invalidates when parquet changes.
    """
    requested = set(window_sizes)
    cache = _load_cache(parquet_path)

    if cache is not None:
        cached_windows = set(cache.get("compressibility", {}).keys())
        missing = requested - cached_windows
        if not missing:
            log.info("Full cache hit for %s", parquet_path.name)
            return cache
        log.info("Partial cache hit for %s (have W=%s, need W=%s)",
                 parquet_path.name, sorted(cached_windows), sorted(missing))
    else:
        missing = requested
        cache = {}

    # Load data if needed
    if exp is None:
        exp = _load_experiment_df(parquet_path)

    # Summary and entropy stationarity (recompute on cache miss)
    if "summary" not in cache:
        cache["summary"] = summarize_run(exp)
        cache["entropy_stationarity"] = stationarity_blocks(exp.entropy.to_numpy())

    # Compute missing compressibility windows
    comp = cache.get("compressibility", {})
    for w in sorted(missing):
        log.info("Computing compressibility W=%d for %s", w, parquet_path.name)
        comp[w] = sliding_compressibility(exp.decoded_text, w)
    cache["compressibility"] = comp

    # Stationarity on largest window compressibility
    w_max = max(comp.keys())
    comp_valid = comp[w_max][~np.isnan(comp[w_max])]
    cache["compressibility_stationarity"] = stationarity_blocks(comp_valid)

    # Track which windows are available
    cache["window_sizes"] = sorted(comp.keys())

    _save_cache(parquet_path, cache)
    return cache

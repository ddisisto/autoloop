"""Analysis package for autoloop runs.

Public API:
    analyze_run(parquet_path, window_sizes, exp=None) -> dict
    default_window_sizes(L) -> list[int]
    comp_stats(cache, W) -> dict
    summarize_run(exp) -> dict
    sliding_compressibility(decoded_texts, window_size) -> np.ndarray
    stationarity_blocks(series, n_blocks=5) -> dict
    entropy_autocorrelation(entropy, max_lag=2000) -> np.ndarray
    load_experiment_df(parquet_path) -> pd.DataFrame

Submodules:
    analyze.metrics    — scalar metric extraction (surprisal, EOS, decorrelation)
    analyze.semantic   — vocabulary, Heaps' law, coherence, repetition onset
    analyze.cache      — .analysis.pkl cache management
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analyze.cache import load_cache, save_cache, load_experiment_df
from analyze.compressibility import (
    sliding_compressibility,
    stationarity_blocks,
    entropy_autocorrelation,
)
from analyze.summary import default_window_sizes, comp_stats, summarize_run

log = logging.getLogger(__name__)

# Re-export for external use
__all__ = [
    "analyze_run",
    "default_window_sizes",
    "comp_stats",
    "summarize_run",
    "sliding_compressibility",
    "stationarity_blocks",
    "entropy_autocorrelation",
    "load_experiment_df",
]


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
    cache = load_cache(parquet_path)

    if cache is not None:
        cached_windows = set(cache.get("compressibility", {}).keys())
        missing = requested - cached_windows
        needs_autocorr = "entropy_autocorrelation" not in cache
        if not missing and not needs_autocorr:
            log.info("Full cache hit for %s", parquet_path.name)
            return cache
        log.info("Partial cache hit for %s (have W=%s, need W=%s)",
                 parquet_path.name, sorted(cached_windows), sorted(missing))
    else:
        missing = requested
        cache = {}

    if exp is None:
        exp = load_experiment_df(parquet_path)

    if "summary" not in cache:
        cache["summary"] = summarize_run(exp)
        cache["entropy_stationarity"] = stationarity_blocks(exp.entropy.to_numpy())
    if "entropy_autocorrelation" not in cache:
        cache["entropy_autocorrelation"] = entropy_autocorrelation(exp.entropy.to_numpy())

    comp = cache.get("compressibility", {})
    for w in sorted(missing):
        log.info("Computing compressibility W=%d for %s", w, parquet_path.name)
        comp[w] = sliding_compressibility(exp.decoded_text, w)
    cache["compressibility"] = comp

    w_max = max(comp.keys())
    comp_valid = comp[w_max][~np.isnan(comp[w_max])]
    cache["compressibility_stationarity"] = stationarity_blocks(comp_valid)

    cache["window_sizes"] = sorted(comp.keys())

    save_cache(parquet_path, cache)
    return cache

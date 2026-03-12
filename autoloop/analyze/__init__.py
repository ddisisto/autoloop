"""Analysis package for autoloop runs.

Public API:
    analyze_run(parquet_path, window_sizes, exp=None) -> dict
    default_window_sizes(L) -> list[int]
    comp_stats(cache, W) -> dict  — NaN-safe scalar stats (use this, not raw arrays)
    summarize_run(exp) -> dict
    sliding_compressibility(decoded_texts, window_size) -> np.ndarray  — has leading NaN
    stationarity_blocks(series, n_blocks=5) -> dict
    entropy_autocorrelation(entropy, max_lag=2000) -> np.ndarray
    load_experiment_df(parquet_path) -> pd.DataFrame

Note: compressibility arrays have NaN at positions 0..W-2 (no full window).
For scalar summaries, always use comp_stats(). Access raw arrays only for
time-series work where positional alignment matters.

Submodules:
    analyze.metrics    — scalar metric extraction (surprisal, EOS, decorrelation)
    analyze.semantic   — vocabulary, Heaps' law, coherence, repetition onset
    analyze.cache      — .analysis.pkl cache management
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .cache import load_cache, save_cache, load_experiment_df
from .compressibility import (
    sliding_compressibility,
    stationarity_blocks,
    entropy_autocorrelation,
)
from .summary import default_window_sizes, comp_stats, summarize_run

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


def _window_metrics() -> list:
    """Discover registered window metrics from the metrics registry."""
    from ..metrics import by_scale
    return by_scale("window")


def analyze_run(
    parquet_path: Path,
    window_sizes: list[int],
    exp: pd.DataFrame | None = None,
) -> dict:
    """Analyze a single run, computing only what's missing from cache.

    Args:
        parquet_path: Path to the run's parquet file.
        window_sizes: List of window sizes for window-level metrics.
        exp: Pre-loaded experiment-phase DataFrame. If None, loaded from parquet.

    Returns dict with summary stats, stationarity, and per-metric windowed
    arrays. Each window metric is stored as {metric_id: {W: array}}.

    Cache is incremental: previously computed window sizes are reused,
    only missing ones are computed. Cache invalidates when parquet changes.
    """
    requested = set(window_sizes)
    cache = load_cache(parquet_path)

    # Determine what needs computing
    window_mets = _window_metrics()
    needs_work = False

    if cache is not None:
        needs_autocorr = "entropy_autocorrelation" not in cache
        for m in window_mets:
            cached_windows = set(cache.get(m.id, {}).keys())
            if requested - cached_windows:
                needs_work = True
                break
        if not needs_work and not needs_autocorr:
            log.info("Full cache hit for %s", parquet_path.name)
            return cache
    else:
        cache = {}
        needs_work = True

    if exp is None:
        exp = load_experiment_df(parquet_path)

    if "summary" not in cache:
        cache["summary"] = summarize_run(exp)
        cache["entropy_stationarity"] = stationarity_blocks(exp.entropy.to_numpy())
    if "entropy_autocorrelation" not in cache:
        cache["entropy_autocorrelation"] = entropy_autocorrelation(exp.entropy.to_numpy())

    # Compute each registered window metric at requested window sizes
    all_window_sizes: set[int] = set()
    for m in window_mets:
        existing = cache.get(m.id, {})
        missing = requested - set(existing.keys())
        if missing and m.window_fn is not None:
            cached_ws = sorted(existing.keys()) if existing else []
            log.info("Computing %s for %s (have W=%s, need W=%s)",
                     m.id, parquet_path.name, cached_ws, sorted(missing))
            for w in sorted(missing):
                log.info("  %s W=%d", m.id, w)
                existing[w] = m.window_fn(exp.decoded_text, w)
        cache[m.id] = existing
        all_window_sizes.update(existing.keys())

    # Stationarity of the largest window metric (compressibility if present)
    comp = cache.get("compressibility", {})
    if comp:
        w_max = max(comp.keys())
        comp_valid = comp[w_max][~np.isnan(comp[w_max])]
        cache["compressibility_stationarity"] = stationarity_blocks(comp_valid)

    cache["window_sizes"] = sorted(all_window_sizes)

    save_cache(parquet_path, cache)
    return cache

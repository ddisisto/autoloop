"""Analysis cache management for autoloop runs.

Single .analysis.pkl per parquet file, incrementally updatable,
mtime-validated against source parquet.
"""

import logging
import pickle
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CACHE_VERSION = 2  # bump to invalidate all caches


def cache_path(parquet_path: Path) -> Path:
    """Single cache file per parquet."""
    return parquet_path.with_suffix(".analysis.pkl")


def load_cache(parquet_path: Path) -> dict | None:
    """Load cache if valid (parquet hasn't changed since cache was written)."""
    cp = cache_path(parquet_path)
    if not cp.exists():
        return None
    pq_mtime = parquet_path.stat().st_mtime
    with open(cp, "rb") as f:
        cache = pickle.load(f)
    if cache.get("parquet_mtime") != pq_mtime:
        log.info("Cache stale for %s (parquet changed)", parquet_path.name)
        return None
    if cache.get("cache_version") != CACHE_VERSION:
        log.info("Cache version mismatch for %s (have %s, want %d)",
                 parquet_path.name, cache.get("cache_version"), CACHE_VERSION)
        return None
    return cache


def save_cache(parquet_path: Path, cache: dict) -> None:
    """Save cache with current parquet mtime."""
    cache["cache_version"] = CACHE_VERSION
    cache["parquet_mtime"] = parquet_path.stat().st_mtime
    cp = cache_path(parquet_path)
    with open(cp, "wb") as f:
        pickle.dump(cache, f)
    log.info("Cached analysis to %s", cp.name)


def load_experiment_df(parquet_path: Path) -> pd.DataFrame:
    """Load experiment-phase rows from a parquet file."""
    df = pd.read_parquet(parquet_path)
    return df[df.phase == "experiment"].reset_index(drop=True)

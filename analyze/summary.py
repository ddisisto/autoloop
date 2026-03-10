"""Run-level summary statistics and helper functions."""

import numpy as np
import pandas as pd


def default_window_sizes(L: int) -> list[int]:
    """Standard window grid [32,64,128,256].

    Floor at 32 (gzip overhead dominates below this).
    Always includes windows above L — longer-range patterns may be
    visible even beyond the context window.
    """
    return [32, 64, 128, 256]


def comp_stats(cache: dict, W: int) -> dict[str, float]:
    """Extract compressibility summary stats for window size W from analysis cache.

    This is the correct interface for scalar compressibility values. Raw
    compressibility arrays have leading NaN (first W-1 positions lack a full
    window) — this function filters those before computing stats.

    Returns dict with mean, std, min, max. Returns all-NaN dict if W missing.
    """
    nan_result = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    comp = cache.get("compressibility", {}).get(W)
    if comp is None:
        return nan_result
    valid = comp[~np.isnan(comp)] if isinstance(comp, np.ndarray) else np.array([x for x in comp if not np.isnan(x)])
    if len(valid) == 0:
        return nan_result
    return {
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "min": float(valid.min()),
        "max": float(valid.max()),
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

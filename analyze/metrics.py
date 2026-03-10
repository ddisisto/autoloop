"""Scalar metric extraction functions for autoloop runs.

Pure computation on experiment-phase DataFrames and autocorrelation arrays.
No caching, no I/O, no file discovery.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from analyze.summary import summarize_run


def surprisal_stats(exp: pd.DataFrame) -> dict:
    """Compute mean, variance, skewness, and excess kurtosis of surprisal (-log_prob)."""
    surprisal = -exp["log_prob"].values
    return {
        "surprisal_mean": float(np.mean(surprisal)),
        "surprisal_var": float(np.var(surprisal)),
        "surprisal_skew": float(skew(surprisal)),
        "surprisal_kurtosis": float(kurtosis(surprisal)),
    }


def eos_interarrival(exp: pd.DataFrame) -> dict:
    """Compute inter-arrival time statistics between EOS tokens."""
    eos_steps = exp.index[exp["eos"]].values
    n_gaps = len(eos_steps) - 1 if len(eos_steps) >= 2 else 0

    if n_gaps == 0:
        return {
            "eos_gap_mean": float("nan"),
            "eos_gap_std": float("nan"),
            "eos_gap_median": float("nan"),
            "eos_gap_min": float("nan"),
            "eos_gap_max": float("nan"),
            "eos_gap_count": 0,
        }

    gaps = np.diff(eos_steps)
    return {
        "eos_gap_mean": float(np.mean(gaps)),
        "eos_gap_std": float(np.std(gaps)),
        "eos_gap_median": float(np.median(gaps)),
        "eos_gap_min": int(np.min(gaps)),
        "eos_gap_max": int(np.max(gaps)),
        "eos_gap_count": int(len(gaps)),
    }


def decorrelation_lag(acf: np.ndarray, threshold: float = np.exp(-1)) -> int:
    """Find the first lag where autocorrelation drops below threshold."""
    below = np.where(acf < threshold)[0]
    if len(below) == 0:
        return len(acf) - 1
    return int(below[0])


def run_scalars(exp: pd.DataFrame, acf: np.ndarray) -> dict:
    """Compute all scalar metrics for one run as a flat dict."""
    result = {}
    result.update(summarize_run(exp))
    result.update(surprisal_stats(exp))
    result.update(eos_interarrival(exp))
    result["decorrelation_lag"] = decorrelation_lag(acf)
    return result

"""Scalar metric extraction functions for autoloop runs.

Pure computation on experiment-phase DataFrames and autocorrelation arrays.
No caching, no I/O, no file discovery.

Individual functions here are the canonical implementations. The central
metrics registry (autoloop.metrics) wraps them with (exp, cache) -> float
signatures for automatic discovery.
"""

import pandas as pd

from .summary import summarize_run

# Canonical implementation lives in the central registry module.
# Re-exported here for backward compatibility (precollapse.py imports from here).
from ..metrics import decorrelation_lag  # noqa: F401


def run_scalars(exp: pd.DataFrame, cache: dict) -> dict:
    """Compute all run-level scalar metrics via the central registry.

    Iterates registered run-level metrics and calls each run_fn.
    Also includes basic summary stats from summarize_run().
    """
    from ..metrics import by_scale

    result = {}
    result.update(summarize_run(exp))

    for m in by_scale("run"):
        if m.run_fn is not None:
            result[m.id] = m.run_fn(exp, cache)

    return result

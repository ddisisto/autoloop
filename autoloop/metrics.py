"""Metric registry for autoloop.

Central definitions for every metric in the system. Each metric is
declared once here and automatically available for analysis, plotting,
the explorer, and control.

Scales:
    step   — per-token values stored as parquet columns
    window — computed over sliding windows of decoded text
    run    — scalar summaries computed once per run

Adding a new metric:
    1. Write the compute function (if not step-level)
    2. Call register(MetricDef(...))
    3. Done — analysis, explorer, and plotting discover it automatically
"""

import logging
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

Scale = Literal["step", "window", "run"]


@dataclass(frozen=True)
class MetricDef:
    """Declaration of a single metric.

    Attributes:
        id: Unique key, e.g. "entropy", "compressibility", "heaps_beta".
        name: Human-readable display name.
        scale: Temporal scale — "step", "window", or "run".
        description: One-line description.
        unit: Unit string, e.g. "nats", "ratio".
        column: For step metrics, the parquet column name.
        window_fn: For window metrics, fn(decoded_texts, window_size) -> ndarray.
        run_fn: For run metrics, fn(exp_df, cache_dict) -> float.
        sensor_fn: For real-time reads, fn(records, window) -> float.
    """
    id: str
    name: str
    scale: Scale
    description: str
    unit: str = ""
    column: str | None = None
    window_fn: Callable | None = None
    run_fn: Callable | None = None
    sensor_fn: Callable | None = None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, MetricDef] = {}


def register(metric: MetricDef) -> MetricDef:
    """Register a metric definition. Returns it for chaining."""
    if metric.id in _REGISTRY:
        raise ValueError(f"Metric {metric.id!r} already registered")
    _REGISTRY[metric.id] = metric
    return metric


def get(metric_id: str) -> MetricDef:
    """Look up a registered metric by ID. Raises KeyError if not found."""
    return _REGISTRY[metric_id]


def all_metrics() -> list[MetricDef]:
    """All registered metrics, sorted by id."""
    return sorted(_REGISTRY.values(), key=lambda m: m.id)


def by_scale(scale: Scale) -> list[MetricDef]:
    """All metrics at a given temporal scale."""
    return [m for m in all_metrics() if m.scale == scale]


# ---------------------------------------------------------------------------
# Shared compute helpers
# ---------------------------------------------------------------------------

def heaps_beta_ols(words: list[str], n_checkpoints: int = 20) -> tuple[float, int, int]:
    """Core OLS computation for Heaps' law exponent.

    Samples vocabulary growth at n_checkpoints points, fits log-log OLS.

    Args:
        words: Lowercased word list (pre-filtered for len > 1).
        n_checkpoints: Number of sample points for OLS fit.

    Returns:
        (beta, n_words, n_unique) tuple.
    """
    n_words = len(words)
    if n_words < 50:
        return 0.0, n_words, 0

    seen: set[str] = set()
    step_size = max(1, n_words // n_checkpoints)
    ns: list[int] = []
    vs: list[int] = []
    for i, w in enumerate(words):
        seen.add(w)
        if (i + 1) % step_size == 0:
            ns.append(i + 1)
            vs.append(len(seen))
    n_unique = len(seen)

    if len(ns) < 3:
        return 0.0, n_words, n_unique

    log_n = np.log(np.array(ns, dtype=float))
    log_v = np.log(np.array(vs, dtype=float))
    n_pts = len(log_n)
    sum_x = log_n.sum()
    sum_y = log_v.sum()
    sum_xy = (log_n * log_v).sum()
    sum_x2 = (log_n ** 2).sum()
    denom = n_pts * sum_x2 - sum_x ** 2
    if abs(denom) > 1e-12:
        beta = float((n_pts * sum_xy - sum_x * sum_y) / denom)
    else:
        beta = 0.0

    return beta, n_words, n_unique


# ---------------------------------------------------------------------------
# Sensor functions (real-time, from engine records)
# ---------------------------------------------------------------------------

def _sensor_heaps_beta(records: list[dict], window: int) -> float:
    """Heaps' beta from trailing engine records."""
    tail = records[-window:] if len(records) > window else records
    exp_tail = [r for r in tail if r.get("phase") == "experiment"]
    if not exp_tail:
        exp_tail = tail
    text = "".join(r["decoded_text"] for r in exp_tail)
    words = [w.lower() for w in text.split() if len(w) > 1]
    beta, _, _ = heaps_beta_ols(words)
    return beta


def _sensor_entropy_mean(records: list[dict], window: int) -> float:
    """Mean entropy from trailing engine records."""
    tail = records[-window:] if len(records) > window else records
    exp_tail = [r for r in tail if r.get("phase") == "experiment"]
    if not exp_tail:
        exp_tail = tail
    ent = [r["entropy"] for r in exp_tail]
    return sum(ent) / len(ent) if ent else 0.0


def _sensor_entropy_std(records: list[dict], window: int) -> float:
    """Entropy std from trailing engine records."""
    tail = records[-window:] if len(records) > window else records
    exp_tail = [r for r in tail if r.get("phase") == "experiment"]
    if not exp_tail:
        exp_tail = tail
    ent = [r["entropy"] for r in exp_tail]
    if not ent:
        return 0.0
    mean = sum(ent) / len(ent)
    return (sum((e - mean) ** 2 for e in ent) / len(ent)) ** 0.5


# ---------------------------------------------------------------------------
# Run-level compute functions
# ---------------------------------------------------------------------------

def _run_entropy_mean(exp: pd.DataFrame, cache: dict) -> float:
    """Mean entropy from experiment DataFrame."""
    return float(exp.entropy.mean())


def _run_entropy_std(exp: pd.DataFrame, cache: dict) -> float:
    """Entropy std from experiment DataFrame."""
    return float(exp.entropy.std())


def _run_decorrelation_lag(exp: pd.DataFrame, cache: dict) -> float:
    """Decorrelation lag from analysis cache."""
    acf = cache.get("entropy_autocorrelation")
    if acf is None:
        return float("nan")
    threshold = np.exp(-1)
    below = np.where(acf < threshold)[0]
    if len(below) == 0:
        return float(len(acf) - 1)
    return float(below[0])


def _run_heaps_beta(exp: pd.DataFrame, cache: dict) -> float:
    """Heaps' beta from full experiment DataFrame."""
    text = "".join(exp.decoded_text)
    words = [w.lower() for w in text.split() if len(w) > 1]
    beta, _, _ = heaps_beta_ols(words)
    return beta


# ---------------------------------------------------------------------------
# Built-in metric registrations
# ---------------------------------------------------------------------------

# Step-level (parquet columns, computed in engine.step())
register(MetricDef(
    "entropy", "Softmax Entropy", "step",
    "Shannon entropy of the softmax distribution",
    unit="nats", column="entropy",
))
register(MetricDef(
    "log_prob", "Log Probability", "step",
    "Log probability of the chosen token",
    column="log_prob",
))
register(MetricDef(
    "eos", "EOS Flag", "step",
    "Whether the generated token was EOS",
    column="eos",
))
register(MetricDef(
    "temperature", "Temperature", "step",
    "Sampling temperature at each step",
    column="temperature",
))
register(MetricDef(
    "context_length", "Context Length", "step",
    "Context window size at each step",
    column="context_length",
))

# Window-level (sliding window computations)
register(MetricDef(
    "compressibility", "Compressibility", "window",
    "Gzip compression ratio over sliding window",
    unit="ratio",
    window_fn=None,  # set below after import
))

# Run-level (scalar summaries)
register(MetricDef(
    "heaps_beta", "Heaps' Beta", "run",
    "Vocabulary growth exponent (Heaps' law)",
    run_fn=_run_heaps_beta,
    sensor_fn=_sensor_heaps_beta,
))
register(MetricDef(
    "decorrelation_lag", "Decorrelation Lag", "run",
    "First lag where entropy ACF drops below 1/e",
    unit="steps",
    run_fn=_run_decorrelation_lag,
))
register(MetricDef(
    "entropy_mean", "Entropy Mean", "run",
    "Mean softmax entropy over run",
    unit="nats",
    run_fn=_run_entropy_mean,
    sensor_fn=_sensor_entropy_mean,
))
register(MetricDef(
    "entropy_std", "Entropy Std", "run",
    "Standard deviation of softmax entropy",
    unit="nats",
    run_fn=_run_entropy_std,
    sensor_fn=_sensor_entropy_std,
))

# Derived step-level (computed from other columns)
register(MetricDef(
    "eos_ema", "EOS Rate (EMA)", "step",
    "Exponential moving average of EOS flag (span=1000)",
    column="eos",  # source column; actual computation uses eos_ema()
))


def _bind_window_fns() -> None:
    """Bind window functions that require sibling imports.

    Called at module load time. Uses object.__setattr__ because
    MetricDef is frozen.
    """
    from .analyze.compressibility import sliding_compressibility
    m = _REGISTRY["compressibility"]
    object.__setattr__(m, "window_fn", sliding_compressibility)


_bind_window_fns()

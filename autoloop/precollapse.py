"""Pre-collapse trajectory analysis: detection and analysis.

Detects collapse events, extracts pre-collapse features, maps attractor
content, and characterizes W/L dynamics during the descent into attractors.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import analyze_run, default_window_sizes, sliding_compressibility
from .analyze.metrics import decorrelation_lag

log = logging.getLogger(__name__)

RUNS_DIR = Path("data/runs")
STANDARD_W = default_window_sizes(0)


# ---------------------------------------------------------------------------
# Collapse detection
# ---------------------------------------------------------------------------

@dataclass
class CollapseEvent:
    """A detected collapse event within a run."""
    onset_step: int               # first step of sustained low entropy
    end_step: int | None          # step where entropy recovers (None if permanent)
    duration: int                 # steps in collapsed state
    attractor_text: str           # repeated text pattern
    attractor_period: int         # tokens per cycle
    entropy_floor: float          # mean entropy during collapse
    pre_onset_entropy_mean: float # mean entropy in pre-collapse window
    pre_onset_entropy_slope: float  # linear entropy slope approaching onset


@dataclass
class BasinTransition:
    """A transition between attractor basins."""
    collapse_start: int
    escape_step: int
    duration: int       # steps in basin before escape
    floor: float        # mean entropy during collapse
    spike: float        # max entropy at escape
    landing: float      # mean entropy in 2000 steps after escape
    direction: str      # "deeper" or "shallower"


@dataclass
class RunAnalysis:
    """Pre-collapse analysis for a single run."""
    run_id: str
    L: int
    T: float
    seed: int
    n_steps: int
    events: list[CollapseEvent] = field(default_factory=list)
    # Run-level summaries
    total_collapsed_steps: int = 0
    collapse_fraction: float = 0.0
    first_collapse_step: int | None = None
    dominant_attractor: str = ""
    dominant_period: int = 0
    # Collapse intensity (works for oscillating/partial collapse too)
    collapse_intensity: float = 0.0  # fraction of steps below threshold (no sustain requirement)
    regime: str = ""  # "escaped", "oscillating", "collapsed", "deep_collapsed"
    # Pre-collapse trajectory features (before first collapse)
    pre_entropy_mean: float = float("nan")
    pre_entropy_std: float = float("nan")
    pre_entropy_slope: float = float("nan")
    pre_entropy_var_decay: float = float("nan")  # slope of rolling variance
    pre_eos_rate: float = float("nan")
    pre_decorr_lag: int = 0
    # Multi-scale dynamics
    pre_comp_spread: float = float("nan")  # |comp_Wmax - comp_Wmin| pre-collapse
    descent_comp_slope: dict = field(default_factory=dict)  # W -> slope during descent
    wl_convergence: dict = field(default_factory=dict)  # W/L analysis
    basin_transitions: list[BasinTransition] = field(default_factory=list)  # escape events


def detect_collapses(
    entropy: np.ndarray,
    threshold: float = 0.1,
    min_sustain: int = 500,
    recovery_threshold: float = 0.3,
    min_recovery: int = 200,
) -> list[tuple[int, int | None]]:
    """Find sustained low-entropy regions (collapse events).

    Args:
        entropy: per-step entropy array
        threshold: entropy below this = collapsed
        min_sustain: minimum consecutive steps to count as collapse
        recovery_threshold: entropy above this = recovered
        min_recovery: minimum consecutive steps above recovery_threshold

    Returns list of (onset_step, end_step) tuples. end_step is None if
    collapse persists to end of run.
    """
    n = len(entropy)
    events = []
    i = 0

    while i < n:
        # Find start of low-entropy region
        if entropy[i] < threshold:
            start = i
            # Scan forward for sustained low entropy
            j = i + 1
            while j < n and entropy[j] < threshold:
                j += 1
            # Check if sustained enough
            if j - start >= min_sustain:
                # Find recovery point (if any)
                end = None
                k = j
                while k < n:
                    # Check for sustained recovery
                    if entropy[k] >= recovery_threshold:
                        run_start = k
                        while k < n and entropy[k] >= recovery_threshold:
                            k += 1
                        if k - run_start >= min_recovery:
                            end = run_start
                            break
                    k += 1
                events.append((start, end))
                i = end if end is not None else n
            else:
                i = j
        else:
            i += 1

    return events


def extract_attractor(decoded_texts: np.ndarray, start: int, length: int = 2000) -> tuple[str, int]:
    """Extract the repeating pattern from a collapsed region.

    Returns (attractor_text, period) where period is the token count per cycle.
    """
    end = min(start + length, len(decoded_texts))
    tokens = decoded_texts[start:end]
    text = "".join(tokens)

    # Try periods from 1 to 200 tokens
    best_period = len(tokens)
    best_score = 0.0

    for period in range(1, min(201, len(tokens) // 3)):
        # Check how well this period explains the sequence
        matches = 0
        total = 0
        for i in range(period, min(len(tokens), period * 20)):
            if tokens[i] == tokens[i % period]:
                matches += 1
            total += 1
        if total > 0:
            score = matches / total
            if score > best_score:
                best_score = score
                best_period = period

    # Extract the attractor text
    attractor = "".join(tokens[:best_period])
    # If score is low, it's not a clean cycle — return longer sample
    if best_score < 0.8:
        attractor = "".join(tokens[:min(100, len(tokens))])
        best_period = 0  # mark as non-periodic

    return attractor, best_period


def pre_collapse_features(
    entropy: np.ndarray,
    eos: np.ndarray,
    onset: int,
    window: int = 2000,
) -> dict:
    """Extract trajectory features from the window before collapse onset."""
    start = max(0, onset - window)
    pre_e = entropy[start:onset]
    pre_eos = eos[start:onset]

    if len(pre_e) < 10:
        return {
            "pre_entropy_mean": float("nan"),
            "pre_entropy_std": float("nan"),
            "pre_entropy_slope": float("nan"),
            "pre_entropy_var_decay": float("nan"),
            "pre_eos_rate": float("nan"),
        }

    # Linear slope of entropy approaching collapse
    x = np.arange(len(pre_e))
    slope = np.polyfit(x, pre_e, 1)[0]

    # Rolling variance decay: is variance shrinking as we approach?
    block_size = max(len(pre_e) // 10, 10)
    n_blocks = len(pre_e) // block_size
    if n_blocks >= 3:
        block_vars = [pre_e[i*block_size:(i+1)*block_size].var() for i in range(n_blocks)]
        var_slope = np.polyfit(range(n_blocks), block_vars, 1)[0]
    else:
        var_slope = float("nan")

    return {
        "pre_entropy_mean": float(pre_e.mean()),
        "pre_entropy_std": float(pre_e.std()),
        "pre_entropy_slope": float(slope),
        "pre_entropy_var_decay": float(var_slope),
        "pre_eos_rate": float(pre_eos.mean()),
    }


def multiscale_descent(
    cache: dict,
    onset: int,
    pre_window: int = 5000,
) -> dict:
    """Characterize compressibility at multiple W sizes during the descent."""
    comp_data = cache.get("compressibility", {})
    if not comp_data:
        return {"pre_comp_spread": float("nan"), "descent_slopes": {}, "w_ratios": {}}

    start = max(0, onset - pre_window)
    w_sizes = sorted(comp_data.keys())

    # Pre-collapse means at each W
    pre_means = {}
    descent_slopes = {}
    for W in w_sizes:
        c = comp_data[W][start:onset]
        valid = c[~np.isnan(c)]
        if len(valid) > 10:
            pre_means[W] = float(valid.mean())
            x = np.arange(len(valid))
            descent_slopes[W] = float(np.polyfit(x, valid, 1)[0])
        else:
            pre_means[W] = float("nan")

    # Spread between largest and smallest W
    valid_means = {w: m for w, m in pre_means.items() if not np.isnan(m)}
    if len(valid_means) >= 2:
        vals = list(valid_means.values())
        spread = max(vals) - min(vals)
    else:
        spread = float("nan")

    return {
        "pre_comp_spread": spread,
        "descent_slopes": descent_slopes,
        "pre_comp_by_w": pre_means,
    }


def detect_basin_transitions(
    entropy: np.ndarray,
    threshold: float = 0.1,
    min_collapse: int = 200,
    landing_window: int = 2000,
) -> list[BasinTransition]:
    """Find all transitions out of collapsed basins."""
    transitions = []
    in_low = False
    low_start = 0

    for i in range(len(entropy)):
        if entropy[i] < threshold and not in_low:
            in_low = True
            low_start = i
        elif entropy[i] > 0.5 and in_low:
            duration = i - low_start
            if duration >= min_collapse:
                spike_window = entropy[max(0, i-10):min(len(entropy), i+50)]
                spike = float(spike_window.max())
                floor = float(entropy[low_start:i].mean())
                post = entropy[i:min(len(entropy), i + landing_window)]
                landing = float(post.mean()) if len(post) > 0 else float("nan")
                direction = "deeper" if landing < floor * 2 else "shallower"
                transitions.append(BasinTransition(
                    collapse_start=low_start,
                    escape_step=i,
                    duration=duration,
                    floor=floor,
                    spike=spike,
                    landing=landing,
                    direction=direction,
                ))
            in_low = False

    return transitions


def wl_convergence_profile(cache: dict, L: int) -> dict:
    """Analyze how compressibility behaves as W approaches L."""
    comp_data = cache.get("compressibility", {})
    if not comp_data:
        return {}

    w_sizes = sorted(comp_data.keys())
    profile = {}
    slopes = {}

    for W in w_sizes:
        c = comp_data[W]
        valid = c[~np.isnan(c)]
        if len(valid) < 100:
            continue
        ratio = W / L if L > 0 else 0
        profile[ratio] = float(valid.mean())
        # Temporal slope: is compressibility trending over the run?
        x = np.arange(len(valid))
        slopes[ratio] = float(np.polyfit(x, valid, 1)[0])

    # Slope divergence: how much do small-W and large-W trends differ?
    if len(slopes) >= 2:
        sorted_ratios = sorted(slopes.keys())
        slope_div = slopes[sorted_ratios[-1]] - slopes[sorted_ratios[0]]
    else:
        slope_div = float("nan")

    # Block-level W/L analysis: how does the spread evolve over time?
    n_blocks = 10
    block_spreads = []
    if len(w_sizes) >= 2:
        w_min, w_max = w_sizes[0], w_sizes[-1]
        c_min, c_max = comp_data[w_min], comp_data[w_max]
        n = min(len(c_min), len(c_max))
        bs = n // n_blocks
        for b in range(n_blocks):
            start, end = b * bs, (b + 1) * bs
            s_min = c_min[start:end]
            s_max = c_max[start:end]
            mask = ~np.isnan(s_min) & ~np.isnan(s_max)
            if mask.any():
                block_spreads.append(float(np.mean(s_max[mask] - s_min[mask])))

    return {
        "comp_by_wl_ratio": profile,
        "slope_by_wl_ratio": slopes,
        "slope_divergence": slope_div,
        "block_spreads": block_spreads,
    }


# ---------------------------------------------------------------------------
# Per-run analysis
# ---------------------------------------------------------------------------

def analyze_precollapse(
    parquet_path: Path,
    threshold: float = 0.1,
    pre_window: int = 2000,
) -> RunAnalysis:
    """Full pre-collapse analysis for a single run."""
    # Parse run ID
    stem = parquet_path.stem
    run_id = stem

    # Extract L, T, seed from filename
    try:
        parts = stem.split("_")
        L = int(parts[0][1:])
        T = float(parts[1][1:])
        seed = int(parts[2][1:])
    except (IndexError, ValueError):
        L, T, seed = 0, 0.0, 0

    # Load data
    df = pd.read_parquet(parquet_path)
    exp = df[df.phase == "experiment"].reset_index(drop=True)
    entropy = exp["entropy"].values
    eos = exp["eos"].values
    decoded = exp["decoded_text"].values

    result = RunAnalysis(
        run_id=run_id, L=L, T=T, seed=seed, n_steps=len(exp),
    )

    # Load or compute analysis cache (for compressibility)
    cache_path = parquet_path.with_suffix(".analysis.pkl")
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = analyze_run(parquet_path, STANDARD_W, exp=exp)

    # Detect collapse events
    raw_events = detect_collapses(entropy, threshold=threshold)

    for onset, end in raw_events:
        duration = (end if end is not None else len(entropy)) - onset

        # Attractor content
        attractor_text, period = extract_attractor(decoded, onset)

        # Entropy floor during collapse
        collapse_end = end if end is not None else len(entropy)
        floor = entropy[onset:collapse_end].mean()

        # Pre-onset features
        pre = pre_collapse_features(entropy, eos, onset, window=pre_window)

        event = CollapseEvent(
            onset_step=onset,
            end_step=end,
            duration=duration,
            attractor_text=attractor_text,
            attractor_period=period,
            entropy_floor=floor,
            pre_onset_entropy_mean=pre["pre_entropy_mean"],
            pre_onset_entropy_slope=pre["pre_entropy_slope"],
        )
        result.events.append(event)

    # Run-level summaries
    if result.events:
        result.total_collapsed_steps = sum(e.duration for e in result.events)
        result.collapse_fraction = result.total_collapsed_steps / len(entropy)
        result.first_collapse_step = result.events[0].onset_step

        # Dominant attractor = longest collapse event
        longest = max(result.events, key=lambda e: e.duration)
        result.dominant_attractor = longest.attractor_text[:80]
        result.dominant_period = longest.attractor_period

        # Pre-first-collapse trajectory
        first_onset = result.events[0].onset_step
        pre = pre_collapse_features(entropy, eos, first_onset, window=pre_window)
        result.pre_entropy_mean = pre["pre_entropy_mean"]
        result.pre_entropy_std = pre["pre_entropy_std"]
        result.pre_entropy_slope = pre["pre_entropy_slope"]
        result.pre_entropy_var_decay = pre["pre_entropy_var_decay"]
        result.pre_eos_rate = pre["pre_eos_rate"]

        # Decorrelation lag in pre-collapse window
        pre_start = max(0, first_onset - pre_window)
        pre_entropy = entropy[pre_start:first_onset]
        if len(pre_entropy) > 100:
            from .analyze.compressibility import entropy_autocorrelation
            acf = entropy_autocorrelation(pre_entropy, max_lag=min(500, len(pre_entropy) - 1))
            result.pre_decorr_lag = decorrelation_lag(acf)

        # Multi-scale dynamics
        ms = multiscale_descent(cache, first_onset, pre_window=min(5000, first_onset))
        result.pre_comp_spread = ms["pre_comp_spread"]
        result.descent_comp_slope = ms["descent_slopes"]

    # Collapse intensity: fraction of steps below threshold (no sustain requirement)
    result.collapse_intensity = float((entropy < threshold).mean())

    # Regime classification
    if result.collapse_intensity < 0.05:
        result.regime = "escaped"
    elif result.collapse_intensity < 0.30:
        result.regime = "oscillating"
    elif result.collapse_fraction > 0.50:
        result.regime = "deep_collapsed"
    elif result.collapse_fraction > 0:
        result.regime = "collapsed"
    else:
        # High intensity but no sustained events — micro-collapse oscillation
        result.regime = "oscillating"

    # W/L convergence: how compressibility changes as W approaches L
    result.wl_convergence = wl_convergence_profile(cache, L)

    # Basin transitions: escape events between attractors
    result.basin_transitions = detect_basin_transitions(entropy, threshold=threshold)

    return result

"""Pre-collapse trajectory analysis for autoloop runs.

Detects collapse events, extracts pre-collapse features, maps attractor
content, and characterizes W/L dynamics during the descent into attractors.

Key outputs:
- Per-run collapse event detection with onset step, attractor period, and content
- Pre-collapse trajectory features: entropy slope, variance decay, EOS rate
- Multi-scale compressibility dynamics (W/L ratios) through the descent
- Cross-run summary CSV for systematic comparison

Usage:
    python precollapse.py                          # analyze all T<=1.0 runs
    python precollapse.py --runs data/runs/L*.parquet  # specific runs
    python precollapse.py --threshold 0.1          # custom entropy threshold
    python precollapse.py --csv data/precollapse.csv   # write CSV
    python precollapse.py --detail L0256_T0.80_S42     # detailed single-run report
"""

import argparse
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from autoloop.analyze import analyze_run, default_window_sizes, sliding_compressibility
from autoloop.analyze.metrics import decorrelation_lag

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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
    """Extract trajectory features from the window before collapse onset.

    Args:
        entropy: full entropy array
        eos: full EOS boolean array
        onset: step index of collapse onset
        window: number of steps before onset to analyze
    """
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
    """Characterize compressibility at multiple W sizes during the descent.

    Returns dict with:
        - pre_comp_spread: |comp_Wmax - comp_Wmin| in pre-collapse window
        - descent_slopes: {W: slope} of compressibility approaching onset
        - w_l_convergence: at what W/L ratio does compressibility diverge most?
    """
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


def detect_basin_transitions(
    entropy: np.ndarray,
    threshold: float = 0.1,
    min_collapse: int = 200,
    landing_window: int = 2000,
) -> list[BasinTransition]:
    """Find all transitions out of collapsed basins.

    Detects: sustained low entropy → escape spike → landing.
    """
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
    """Analyze how compressibility behaves as W approaches L.

    The W/L boundary is where the measurement window equals the model's
    context — this is where "memory saturation" dynamics become visible.

    Returns:
        - comp_by_wl_ratio: {W/L_ratio: mean_compressibility} across the run
        - slope_divergence: difference in comp slope between W<<L and W≈L
        - saturation_w: smallest W where comp stabilizes (slope change < 5%)
    """
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
            from analyze.compressibility import entropy_autocorrelation
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


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def summary_row(ra: RunAnalysis) -> dict:
    """Flatten RunAnalysis to a dict suitable for CSV."""
    row = {
        "run_id": ra.run_id,
        "L": ra.L,
        "T": ra.T,
        "seed": ra.seed,
        "n_steps": ra.n_steps,
        "n_collapse_events": len(ra.events),
        "total_collapsed_steps": ra.total_collapsed_steps,
        "collapse_fraction": round(ra.collapse_fraction, 4),
        "first_collapse_step": ra.first_collapse_step,
        "dominant_attractor": ra.dominant_attractor,
        "dominant_period": ra.dominant_period,
        "pre_entropy_mean": round(ra.pre_entropy_mean, 4) if not np.isnan(ra.pre_entropy_mean) else None,
        "pre_entropy_std": round(ra.pre_entropy_std, 4) if not np.isnan(ra.pre_entropy_std) else None,
        "pre_entropy_slope": f"{ra.pre_entropy_slope:.2e}" if not np.isnan(ra.pre_entropy_slope) else None,
        "pre_entropy_var_decay": f"{ra.pre_entropy_var_decay:.2e}" if not np.isnan(ra.pre_entropy_var_decay) else None,
        "pre_eos_rate": round(ra.pre_eos_rate, 4) if not np.isnan(ra.pre_eos_rate) else None,
        "pre_decorr_lag": ra.pre_decorr_lag,
        "pre_comp_spread": round(ra.pre_comp_spread, 4) if not np.isnan(ra.pre_comp_spread) else None,
        "collapse_intensity": round(ra.collapse_intensity, 4),
        "regime": ra.regime,
        "n_basin_transitions": len(ra.basin_transitions),
        "n_deeper_transitions": sum(1 for t in ra.basin_transitions if t.direction == "deeper"),
        "mean_spike": round(np.mean([t.spike for t in ra.basin_transitions]), 3) if ra.basin_transitions else None,
        "mean_floor": round(np.mean([t.floor for t in ra.basin_transitions]), 4) if ra.basin_transitions else None,
    }
    # Add per-W descent slopes
    for W in STANDARD_W:
        val = ra.descent_comp_slope.get(W)
        row[f"descent_slope_W{W}"] = f"{val:.2e}" if val is not None else None
    # W/L convergence
    wl = ra.wl_convergence
    if wl:
        row["wl_slope_divergence"] = round(wl.get("slope_divergence", float("nan")), 6) if not np.isnan(wl.get("slope_divergence", float("nan"))) else None
        for ratio, comp in sorted(wl.get("comp_by_wl_ratio", {}).items()):
            row[f"comp_wl_{ratio:.2f}"] = round(comp, 4)
    return row


def detail_report(ra: RunAnalysis) -> str:
    """Detailed text report for a single run."""
    lines = [
        f"{'='*70}",
        f"Pre-Collapse Analysis: {ra.run_id}",
        f"L={ra.L}, T={ra.T}, seed={ra.seed}, steps={ra.n_steps}",
        f"{'='*70}",
        "",
        f"Regime: {ra.regime} (intensity={ra.collapse_intensity:.3f})",
        f"Collapse events: {len(ra.events)}",
        f"Total collapsed steps: {ra.total_collapsed_steps} ({100*ra.collapse_fraction:.1f}%)",
        f"First collapse at step: {ra.first_collapse_step}",
        "",
    ]

    for i, ev in enumerate(ra.events):
        end_str = str(ev.end_step) if ev.end_step is not None else "END"
        lines.extend([
            f"--- Event {i+1}: steps {ev.onset_step} → {end_str} ({ev.duration} steps) ---",
            f"  Entropy floor: {ev.entropy_floor:.4f}",
            f"  Pre-onset entropy: mean={ev.pre_onset_entropy_mean:.3f}, slope={ev.pre_onset_entropy_slope:.2e}",
            f"  Attractor period: {ev.attractor_period} tokens",
            f"  Attractor text: {repr(ev.attractor_text[:120])}",
            "",
        ])

    if ra.events:
        lines.extend([
            "--- Pre-first-collapse trajectory ---",
            f"  Entropy: mean={ra.pre_entropy_mean:.3f}, std={ra.pre_entropy_std:.3f}",
            f"  Entropy slope: {ra.pre_entropy_slope:.2e} (negative = descending)",
            f"  Variance decay: {ra.pre_entropy_var_decay:.2e} (negative = narrowing)",
            f"  EOS rate: {ra.pre_eos_rate:.4f}",
            f"  Decorrelation lag: {ra.pre_decorr_lag}",
            f"  Multi-scale spread: {ra.pre_comp_spread:.4f}",
            "",
            "  Compressibility descent slopes by W:",
        ])
        for W in sorted(ra.descent_comp_slope.keys()):
            slope = ra.descent_comp_slope[W]
            lines.append(f"    W={W:3d}: {slope:.2e}")

    # W/L convergence profile
    wl = ra.wl_convergence
    if wl and wl.get("comp_by_wl_ratio"):
        lines.extend(["", "--- W/L Convergence Profile ---"])
        for ratio in sorted(wl["comp_by_wl_ratio"].keys()):
            comp = wl["comp_by_wl_ratio"][ratio]
            slope = wl.get("slope_by_wl_ratio", {}).get(ratio, float("nan"))
            lines.append(f"  W/L={ratio:.2f}: comp={comp:.3f}, trend={slope:.2e}")
        sd = wl.get("slope_divergence", float("nan"))
        if not np.isnan(sd):
            lines.append(f"  Slope divergence (large_W - small_W): {sd:.2e}")
        spreads = wl.get("block_spreads", [])
        if spreads:
            lines.append(f"  Block spreads (Wmax-Wmin over time): {' '.join(f'{s:.3f}' for s in spreads)}")

    # Basin transitions
    if ra.basin_transitions:
        n_deeper = sum(1 for t in ra.basin_transitions if t.direction == "deeper")
        lines.extend([
            "",
            f"--- Basin Transitions ({len(ra.basin_transitions)} escapes, {n_deeper} to deeper basins) ---",
        ])
        for t in ra.basin_transitions:
            lines.append(
                f"  step {t.collapse_start:>6d}→{t.escape_step:>6d} ({t.duration:>5d} steps): "
                f"floor={t.floor:.3f}, spike={t.spike:.1f}, landing={t.landing:.3f} ({t.direction})"
            )

    return "\n".join(lines)


def print_summary(df: pd.DataFrame, results: list[RunAnalysis] | None = None) -> None:
    """Print regime-grouped summary table to stdout.

    Args:
        df: DataFrame of summary_row dicts (columns: run_id, regime, etc.).
        results: Optional list of RunAnalysis (unused, kept for API compat).
    """
    print(f"\n{'='*80}")
    print(f"Pre-Collapse Summary: {len(df)} runs analyzed")
    print(f"{'='*80}")

    for regime in ["deep_collapsed", "collapsed", "oscillating", "escaped"]:
        regime_df = df[df["regime"] == regime]
        if len(regime_df) == 0:
            continue
        print(f"\n--- {regime.upper()} ({len(regime_df)} runs) ---")

        if regime in ("deep_collapsed", "collapsed"):
            print(f"{'Run':<22s} {'Intens':>6s} {'Onset':>7s} {'Frac':>6s} {'Pre-H':>6s} "
                  f"{'Slope':>10s} {'Spread':>6s} {'Per':>4s} Attractor")
            for _, row in regime_df.iterrows():
                onset = row["first_collapse_step"]
                onset_str = f"{int(onset):>7d}" if onset is not None else "    N/A"
                pre_h = f"{row['pre_entropy_mean']:.3f}" if row["pre_entropy_mean"] is not None else "  N/A"
                slope = row["pre_entropy_slope"] if row["pre_entropy_slope"] is not None else "       N/A"
                spread = f"{row['pre_comp_spread']:.3f}" if row["pre_comp_spread"] is not None else " N/A"
                attractor = (row["dominant_attractor"] or "")[:30]
                print(f"{row['run_id']:<22s} {row['collapse_intensity']:>6.3f} {onset_str} "
                      f"{row['collapse_fraction']:>6.2f} {pre_h:>6s} {slope:>10s} "
                      f"{spread:>6s} {row['dominant_period']:>4d} {attractor}")
        else:
            print(f"{'Run':<22s} {'Intens':>6s} {'Entropy':>8s}")
            for _, row in regime_df.iterrows():
                e_mean = f"{row.get('pre_entropy_mean', 'N/A')}" if row.get("pre_entropy_mean") is not None else "N/A"
                print(f"{row['run_id']:<22s} {row['collapse_intensity']:>6.3f} {e_mean:>8s}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_runs(patterns: list[str] | None, runs_dir: Path = RUNS_DIR) -> list[Path]:
    """Find parquet files matching patterns, or all T<=1.0 runs."""
    if patterns:
        from glob import glob
        paths = []
        for p in patterns:
            paths.extend(Path(x) for x in sorted(glob(p)))
        return [p for p in paths if p.suffix == ".parquet"]

    # Default: all fixed-param runs with T <= 1.0
    all_runs = sorted(runs_dir.glob("L*_T*_S*.parquet"))
    filtered = []
    for p in all_runs:
        try:
            T = float(p.stem.split("_")[1][1:])
            if T <= 1.20:
                filtered.append(p)
        except (IndexError, ValueError):
            continue
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-collapse trajectory analysis")
    parser.add_argument("--runs", nargs="*", help="Parquet file patterns (default: all T<=1.2)")
    parser.add_argument("--threshold", type=float, default=0.1, help="Entropy collapse threshold")
    parser.add_argument("--pre-window", type=int, default=2000, help="Steps before onset to analyze")
    parser.add_argument("--csv", type=str, help="Output CSV path")
    parser.add_argument("--detail", type=str, help="Run ID for detailed report")
    args = parser.parse_args()

    paths = find_runs(args.runs)
    log.info("Analyzing %d runs", len(paths))

    results = []
    for p in paths:
        log.info("Processing %s", p.stem)
        ra = analyze_precollapse(p, threshold=args.threshold, pre_window=args.pre_window)
        results.append(ra)

        if args.detail and args.detail in ra.run_id:
            print(detail_report(ra))

    # Summary table
    rows = [summary_row(ra) for ra in results]
    df = pd.DataFrame(rows)

    # Sort by L, T, seed
    df = df.sort_values(["L", "T", "seed"]).reset_index(drop=True)

    if args.csv:
        df.to_csv(args.csv, index=False)
        log.info("Wrote %s", args.csv)
    else:
        print_summary(df, results)

    # If no --detail given but --csv also not given, print detailed reports for interesting cases
    if not args.detail and not args.csv:
        # Show detail for runs with recovery (temporary collapse)
        for ra in results:
            recoveries = [e for e in ra.events if e.end_step is not None]
            if recoveries:
                print(f"\n{detail_report(ra)}")


if __name__ == "__main__":
    main()

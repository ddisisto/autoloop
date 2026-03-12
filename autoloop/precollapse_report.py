"""Pre-collapse report formatting and CLI entry point.

Contains summary_row, detail_report, print_summary, and standalone CLI.
"""

import argparse
import logging
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from .precollapse import (
    RUNS_DIR,
    STANDARD_W,
    RunAnalysis,
    analyze_precollapse,
)

log = logging.getLogger(__name__)


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
    """Print regime-grouped summary table to stdout."""
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


def find_runs(patterns: list[str] | None, runs_dir: Path = RUNS_DIR) -> list[Path]:
    """Find parquet files matching patterns, or all T<=1.0 runs."""
    if patterns:
        paths = []
        for p in patterns:
            paths.extend(Path(x) for x in sorted(glob(p)))
        return [p for p in paths if p.suffix == ".parquet"]

    # Default: all fixed-param runs with T <= 1.2
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

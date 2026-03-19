#!/usr/bin/env python3
"""Regime separability analysis: which metrics best distinguish regimes?

Computes all registered metrics across sweep runs, classifies regimes,
and analyzes which metrics (and metric pairs) best separate them.

Usage:
    python scripts/regime_analysis.py              # full analysis
    python scripts/regime_analysis.py --compute    # just compute + save CSV
    python scripts/regime_analysis.py --report     # report from saved CSV
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoloop.analyze import analyze_run, default_window_sizes, load_experiment_df
from autoloop.analyze.scalars import run_scalars
from autoloop.plot import parse_run_name
from autoloop.runlib import discover_runs

log = logging.getLogger(__name__)
RUNS_DIR = Path("data/runs")
OUT_CSV = Path("data/regime_analysis.csv")


# ---------------------------------------------------------------------------
# Regime classification (inline, from precollapse logic)
# ---------------------------------------------------------------------------

def classify_regime(
    entropy: np.ndarray,
    heaps_beta: float,
    comp_w256_mean: float = float("nan"),
) -> str:
    """Classify a run into one of four regimes.

    Primary discriminators from data analysis:
    - β < 0.40 → collapse (clean wall, Cohen's d=3.4 vs suppressed)
    - entropy_mean separates low (collapse+suppressed) from high (rich+noise)
    - comp_W256 > 0.65 at high entropy → noise (near-uniform sampling)
    """
    ent_mean = float(np.mean(entropy))

    # Collapse: vocabulary dies, β drops below 0.40
    if heaps_beta < 0.40:
        return "collapse"

    # High entropy zone: distinguish rich from noise
    if ent_mean > 3.5:
        # Noise: near-incompressible (model sampling near-uniformly)
        if not np.isnan(comp_w256_mean) and comp_w256_mean > 0.65:
            return "noise"
        return "rich"

    # Low entropy + moderate/high β: suppressed dynamics
    return "suppressed"


# ---------------------------------------------------------------------------
# Compute all metrics for all sweep runs
# ---------------------------------------------------------------------------

def compute_all(runs_dir: Path) -> pd.DataFrame:
    """Compute all registered metrics for all sweep runs."""
    parquets = discover_runs(runs_dir, run_type="sweep")
    log.info("Found %d sweep runs", len(parquets))

    rows: list[dict] = []
    for pq in sorted(parquets):
        stem = pq.stem
        params = parse_run_name(pq)
        log.info("Processing %s", stem)

        exp = load_experiment_df(pq)
        window_sizes = default_window_sizes(params["L"])
        cache = analyze_run(pq, window_sizes=window_sizes, exp=exp)

        # All run-level scalars via registry
        scalars = run_scalars(exp, cache)

        # Regime classification
        entropy = exp.entropy.to_numpy()
        comp_data = cache.get("compressibility", {})
        comp_w256 = float("nan")
        if 256 in comp_data:
            arr = comp_data[256]
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                comp_w256 = float(np.mean(valid))
        regime = classify_regime(entropy, scalars.get("heaps_beta", 0.0), comp_w256)

        # Window metrics at multiple window sizes
        for metric_key, prefix in [("compressibility", "comp"), ("lz_complexity", "lz")]:
            wdata = cache.get(metric_key, {})
            for w in sorted(wdata.keys()):
                arr = wdata[w]
                valid = arr[~np.isnan(arr)]
                if len(valid) > 0:
                    scalars[f"{prefix}_W{w}_mean"] = float(np.mean(valid))
                    scalars[f"{prefix}_W{w}_std"] = float(np.std(valid))

        row = {
            "run_id": stem,
            "L": params["L"],
            "T": params["T"],
            "seed": params["S"],
            "regime": regime,
            **scalars,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(["L", "T", "seed"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Analysis: separability
# ---------------------------------------------------------------------------

def regime_separability(df: pd.DataFrame) -> None:
    """Analyze which metrics best separate regimes."""
    regimes = df["regime"].unique()
    n_regimes = len(regimes)
    regime_counts = df["regime"].value_counts()

    print("\n" + "=" * 70)
    print("REGIME DISTRIBUTION")
    print("=" * 70)
    for reg in sorted(regimes):
        runs = df[df.regime == reg]
        L_range = f"L={sorted(runs.L.unique())}"
        T_range = f"T=[{runs["T"].min():.2f}..{runs["T"].max():.2f}]"
        print(f"  {reg:15s}  n={regime_counts[reg]:3d}  {L_range}  {T_range}")

    # Find numeric columns that could be metrics
    metric_cols = [c for c in df.columns
                   if df[c].dtype in (np.float64, np.int64, float, int)
                   and c not in ("L", "T", "seed")]

    print(f"\nMetric columns available: {len(metric_cols)}")

    # Per-metric separability: F-statistic (between-group / within-group variance)
    print("\n" + "=" * 70)
    print("METRIC SEPARABILITY (F-statistic, higher = better separator)")
    print("=" * 70)

    f_scores: dict[str, float] = {}
    for col in metric_cols:
        vals = df[col].values
        if np.all(np.isnan(vals)) or np.std(vals) == 0:
            continue

        groups = [df.loc[df.regime == r, col].dropna().values for r in regimes]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) < 2:
            continue

        # Manual F-statistic to avoid scipy dependency at script level
        grand_mean = np.nanmean(vals)
        n_total = sum(len(g) for g in groups)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

        df_between = len(groups) - 1
        df_within = n_total - len(groups)

        if df_within > 0 and ss_within > 0:
            f_stat = (ss_between / df_between) / (ss_within / df_within)
            f_scores[col] = f_stat

    # Sort by F-score
    ranked = sorted(f_scores.items(), key=lambda x: x[1], reverse=True)
    for col, f_stat in ranked:
        print(f"  {col:30s}  F={f_stat:10.2f}")

    # Per-regime metric profiles
    print("\n" + "=" * 70)
    print("REGIME PROFILES (mean ± std)")
    print("=" * 70)

    top_metrics = [col for col, _ in ranked[:12]]
    for col in top_metrics:
        print(f"\n  {col}:")
        for reg in sorted(regimes):
            vals = df.loc[df.regime == reg, col].dropna()
            if len(vals) > 0:
                print(f"    {reg:15s}  {vals.mean():10.4f} ± {vals.std():8.4f}  "
                      f"[{vals.min():.4f} .. {vals.max():.4f}]")

    # Pairwise regime confusion: which regimes are hardest to tell apart?
    print("\n" + "=" * 70)
    print("PAIRWISE REGIME DISCRIMINATION (best single metric)")
    print("=" * 70)

    from itertools import combinations
    for r1, r2 in combinations(sorted(regimes), 2):
        g1 = df[df.regime == r1]
        g2 = df[df.regime == r2]
        best_col = None
        best_sep = 0.0
        for col in top_metrics:
            v1 = g1[col].dropna()
            v2 = g2[col].dropna()
            if len(v1) == 0 or len(v2) == 0:
                continue
            # Cohen's d: separation in std units
            pooled_std = np.sqrt((v1.var() + v2.var()) / 2)
            if pooled_std > 0:
                d = abs(v1.mean() - v2.mean()) / pooled_std
                if d > best_sep:
                    best_sep = d
                    best_col = col
        if best_col:
            print(f"  {r1:15s} vs {r2:15s}  "
                  f"best={best_col:25s}  Cohen's d={best_sep:.2f}")

    # Transition indicators: for runs near regime boundaries
    print("\n" + "=" * 70)
    print("BOUNDARY RUNS (conditions with multiple regimes at same T or L)")
    print("=" * 70)

    for T_val in sorted(df["T"].unique()):
        at_T = df[df["T"] == T_val]
        if at_T.regime.nunique() > 1:
            print(f"\n  T={T_val:.2f}: {dict(at_T.regime.value_counts())}")
            for _, row in at_T.sort_values("L").iterrows():
                print(f"    L={row.L:4d}  regime={row.regime:15s}  "
                      f"ent={row.entropy_mean:.3f}  β={row.heaps_beta:.3f}  "
                      f"declag={row.decorrelation_lag:.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute", action="store_true",
                        help="Compute metrics and save CSV only")
    parser.add_argument("--report", action="store_true",
                        help="Report from saved CSV only")
    args = parser.parse_args()

    if args.report:
        if not OUT_CSV.exists():
            print(f"No saved data at {OUT_CSV}. Run without --report first.")
            return
        df = pd.read_csv(OUT_CSV)
        regime_separability(df)
        return

    # Compute
    df = compute_all(RUNS_DIR)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} runs to {OUT_CSV}")
    print(f"Columns: {list(df.columns)}")

    if not args.compute:
        regime_separability(df)


if __name__ == "__main__":
    main()

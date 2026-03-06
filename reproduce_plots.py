"""Regenerate all standard plots from available run data, with caching.

Produces plots for every useful slice of the pilot sweep:
  - Per context length (all temperatures, seed 42)
  - Per temperature (all context lengths, seed 42)
  - All seed-42 runs together

Analysis results are cached per-run (see analyze.py), so only new or
modified runs trigger expensive recomputation. Plots are skipped if all
output figures are newer than all input parquets (unless --force).

Usage:
    python reproduce_plots.py
    python reproduce_plots.py --force
    python reproduce_plots.py --plots entropy phase
    python reproduce_plots.py --downsample 50
"""

import argparse
import glob
import logging
from pathlib import Path

import pandas as pd

from analyze import analyze_run
from plot import (
    PLOT_TYPES,
    RunBundle,
    ensure_figures_dir,
    make_label,
    make_output_prefix,
    parse_run_name,
    plot_compressibility_timeseries,
    plot_entropy_timeseries,
    plot_phase_portrait,
    plot_temporal_phase,
    plot_violin,
)

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "data" / "runs"
FIGURES_DIR = REPO_ROOT / "data" / "figures"

NEEDS_ANALYSIS = {"compressibility", "phase", "temporal", "violin"}

# Each slice: (label, glob pattern relative to RUNS_DIR, output prefix)
SLICES: list[tuple[str, str, str]] = [
    # Per context length, all temperatures (seed 42)
    ("L=64, all T, seed 42",   "L0064_T*_S42.parquet",  "L0064_Tmulti_S42"),
    ("L=256, all T, seed 42",  "L0256_T*_S42.parquet",  "L0256_Tmulti_S42"),
    ("L=1024, all T, seed 42", "L1024_T*_S42.parquet",  "L1024_Tmulti_S42"),
    # Per temperature, all context lengths (seed 42)
    ("T=0.50, all L, seed 42", "L*_T0.50_S42.parquet",  "Lmulti_T0.50_S42"),
    ("T=1.00, all L, seed 42", "L*_T1.00_S42.parquet",  "Lmulti_T1.00_S42"),
    ("T=1.50, all L, seed 42", "L*_T1.50_S42.parquet",  "Lmulti_T1.50_S42"),
    # All seed-42 runs together
    ("All runs, seed 42",      "L*_T*_S42.parquet",     "Lmulti_Tmulti_S42"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate all standard plots with caching",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass figure cache check and regenerate all plots",
    )
    parser.add_argument(
        "--plots", type=str, nargs="+", default=None,
        choices=PLOT_TYPES,
        help="Which plot types to generate (default: all)",
    )
    parser.add_argument(
        "--downsample", type=int, default=100,
        help="Downsample factor for time series (default: 100)",
    )
    return parser.parse_args()


def resolve_inputs(pattern: str) -> list[Path]:
    """Glob for input parquet files matching a pattern under RUNS_DIR."""
    full_pattern = str(RUNS_DIR / pattern)
    return sorted(Path(p) for p in glob.glob(full_pattern))


def expected_figures(prefix: str, plot_types: list[str]) -> list[Path]:
    """Return the list of figure paths that plot.py would produce for a slice."""
    return [FIGURES_DIR / f"{prefix}_{pt}.png" for pt in plot_types]


def is_figures_cached(inputs: list[Path], outputs: list[Path]) -> bool:
    """Check if all output figures exist and are newer than all inputs."""
    if not outputs:
        return False
    for out in outputs:
        if not out.exists():
            return False
    newest_input = max(p.stat().st_mtime for p in inputs)
    oldest_output = min(p.stat().st_mtime for p in outputs)
    return oldest_output > newest_input


def load_run_bundle(
    path: Path,
    params: dict,
    label: str,
    needs_analysis: bool,
    analysis_cache: dict[str, dict],
) -> RunBundle:
    """Load a RunBundle, reusing analysis from the shared cache."""
    df = pd.read_parquet(path)
    exp = df[df.phase == "experiment"].reset_index(drop=True)

    analysis = None
    if needs_analysis:
        key = str(path)
        if key not in analysis_cache:
            analysis_cache[key] = analyze_run(path, params["L"])
        analysis = analysis_cache[key]

    return RunBundle(path=path, params=params, label=label, exp=exp, analysis=analysis)


def plot_slice(
    runs: list[RunBundle],
    prefix: str,
    title_ctx: str,
    plot_types: list[str],
    downsample: int,
) -> None:
    """Generate all requested plot types for a slice of runs."""
    ensure_figures_dir()

    dispatch = {
        "entropy": (plot_entropy_timeseries, "Entropy over time"),
        "compressibility": (plot_compressibility_timeseries, "Compressibility over time"),
        "phase": (plot_phase_portrait, "Phase portrait"),
        "temporal": (plot_temporal_phase, "Temporal phase portrait"),
        "violin": (plot_violin, "Distribution over time"),
    }

    for pt in plot_types:
        fn, title_base = dispatch[pt]
        fn(
            runs,
            title=f"{title_base} ({title_ctx})",
            output_name=f"{prefix}_{pt}.png",
            downsample=downsample,
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    active_plots = args.plots if args.plots is not None else PLOT_TYPES
    needs_analysis = bool(set(active_plots) & NEEDS_ANALYSIS)

    # Shared analysis cache: analyze each parquet at most once
    analysis_cache: dict[str, dict] = {}

    # Pre-parse all available runs so labels are consistent per-slice
    all_run_info: dict[str, tuple[Path, dict]] = {}
    for pattern in {s[1] for s in SLICES}:
        for path in resolve_inputs(pattern):
            key = str(path)
            if key not in all_run_info:
                all_run_info[key] = (path, parse_run_name(path))

    plotted = 0
    skipped = 0

    for label, pattern, prefix in SLICES:
        inputs = resolve_inputs(pattern)
        if not inputs:
            log.info("Skipping: %s (no files match %s)", label, pattern)
            skipped += 1
            continue

        if not args.force:
            figures = expected_figures(prefix, active_plots)
            if is_figures_cached(inputs, figures):
                log.info("Cached:   %s", label)
                skipped += 1
                continue

        # Build labels for this slice's runs
        slice_params = [all_run_info[str(p)][1] for p in inputs]
        labels = [make_label(p, slice_params) for p in slice_params]

        # Build title context
        ls = sorted({p["L"] for p in slice_params})
        ts = sorted({p["T"] for p in slice_params})
        ss = sorted({p["S"] for p in slice_params})
        fixed = []
        if len(ls) == 1:
            fixed.append(f"L={ls[0]}")
        if len(ts) == 1:
            fixed.append(f"T={ts[0]:.2f}")
        if len(ss) == 1:
            fixed.append(f"seed={ss[0]}")
        title_ctx = ", ".join(fixed) if fixed else "all runs"

        log.info("Plotting: %s", label)
        runs = [
            load_run_bundle(path, params, lbl, needs_analysis, analysis_cache)
            for path, params, lbl in zip(inputs, slice_params, labels)
        ]

        plot_slice(runs, prefix, title_ctx, active_plots, args.downsample)
        plotted += 1

    log.info("Done. %d plotted, %d skipped.", plotted, skipped)


if __name__ == "__main__":
    main()

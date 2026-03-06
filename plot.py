"""Visualization for autoloop analysis.

Generates plots from generation data and derived metrics.
All plots saved to data/figures/.

Usage:
    python plot.py --runs data/runs/L0064_T*.parquet
    python plot.py --runs data/runs/L*_T0.50_S42.parquet
    python plot.py --runs data/runs/L0064_T0.50_S42.parquet data/runs/L0256_T0.50_S42.parquet
    python plot.py --runs data/runs/L0064_T*_S42.parquet --plots entropy phase
    python plot.py --runs data/runs/L0064_T*_S42.parquet --downsample 50
"""

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import analyze_run

log = logging.getLogger(__name__)

FIGURES_DIR = Path("data/figures")

PLOT_TYPES = ["entropy", "compressibility", "phase", "temporal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot autoloop analysis results")
    parser.add_argument(
        "--runs", type=str, nargs="+", required=True,
        help="Parquet files to plot (supports shell globs)",
    )
    parser.add_argument(
        "--plots", type=str, nargs="+", default=PLOT_TYPES,
        choices=PLOT_TYPES,
        help=f"Which plots to generate (default: all). Choices: {PLOT_TYPES}",
    )
    parser.add_argument(
        "--downsample", type=int, default=100,
        help="Downsample factor for time series (default: 100)",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Optional suffix for output filenames",
    )
    return parser.parse_args()


def parse_run_name(path: Path) -> dict:
    """Extract L, T, S from a run filename like L0064_T0.50_S42."""
    m = re.match(r"L(\d+)_T([\d.]+)_S(\d+)", path.stem)
    if not m:
        raise ValueError(f"Cannot parse run name: {path.stem}")
    return {
        "L": int(m.group(1)),
        "T": float(m.group(2)),
        "S": int(m.group(3)),
    }


def make_label(params: dict, all_params: list[dict]) -> str:
    """Build a concise label showing only the dimensions that vary across runs."""
    parts = []
    if len({p["L"] for p in all_params}) > 1:
        parts.append(f"L={params['L']}")
    if len({p["T"] for p in all_params}) > 1:
        parts.append(f"T={params['T']:.2f}")
    if len({p["S"] for p in all_params}) > 1:
        parts.append(f"S={params['S']}")
    return " ".join(parts) if parts else f"L={params['L']} T={params['T']:.2f} S={params['S']}"


def make_output_prefix(all_params: list[dict], suffix: str) -> str:
    """Build a descriptive output filename prefix from the set of runs."""
    ls = sorted({p["L"] for p in all_params})
    ts = sorted({p["T"] for p in all_params})
    ss = sorted({p["S"] for p in all_params})

    parts = []
    if len(ls) == 1:
        parts.append(f"L{ls[0]:04d}")
    else:
        parts.append("Lmulti")
    if len(ts) == 1:
        parts.append(f"T{ts[0]:.2f}")
    else:
        parts.append("Tmulti")
    if len(ss) == 1:
        parts.append(f"S{ss[0]}")
    else:
        parts.append("Smulti")

    prefix = "_".join(parts)
    if suffix:
        prefix += f"_{suffix}"
    return prefix


def ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def plot_entropy_timeseries(
    run_paths: list[Path],
    labels: list[str],
    title: str,
    output_name: str,
    downsample: int = 100,
) -> Path:
    """Plot entropy over time for multiple runs, downsampled for readability."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for path, label in zip(run_paths, labels):
        df = pd.read_parquet(path)
        exp = df[df.phase == "experiment"].reset_index(drop=True)
        n = len(exp) // downsample * downsample
        entropy = exp.entropy.to_numpy()[:n].reshape(-1, downsample).mean(axis=1)
        steps = np.arange(downsample // 2, n, downsample)
        ax.plot(steps, entropy, label=label, linewidth=0.8)

    ax.set_xlabel("Step (experiment phase)")
    ax.set_ylabel("Softmax entropy (nats)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = ensure_figures_dir() / output_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)
    return out


def plot_compressibility_timeseries(
    run_paths: list[Path],
    context_lengths: list[int],
    labels: list[str],
    title: str,
    output_name: str,
    downsample: int = 100,
) -> Path:
    """Plot compressibility (W=L) over time for multiple runs."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for path, context_length, label in zip(run_paths, context_lengths, labels):
        result = analyze_run(path, context_length)
        comp = result["compressibility_primary"]
        valid = comp[~np.isnan(comp)]
        n = len(valid) // downsample * downsample
        comp_ds = valid[:n].reshape(-1, downsample).mean(axis=1)
        steps = np.arange(downsample // 2, n, downsample)
        ax.plot(steps, comp_ds, label=label, linewidth=0.8)

    ax.set_xlabel("Step (experiment phase)")
    ax.set_ylabel("Compressibility (W=L)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = ensure_figures_dir() / output_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)
    return out


def plot_phase_portrait(
    run_paths: list[Path],
    context_lengths: list[int],
    labels: list[str],
    title: str,
    output_name: str,
    downsample: int = 100,
) -> Path:
    """2D scatter: entropy vs compressibility for multiple runs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for path, context_length, label in zip(run_paths, context_lengths, labels):
        df = pd.read_parquet(path)
        exp = df[df.phase == "experiment"].reset_index(drop=True)
        result = analyze_run(path, context_length)
        comp = result["compressibility_primary"]

        valid_mask = ~np.isnan(comp)
        entropy = exp.entropy.to_numpy()[valid_mask]
        comp_valid = comp[valid_mask]

        n = len(entropy) // downsample * downsample
        entropy_ds = entropy[:n].reshape(-1, downsample).mean(axis=1)
        comp_ds = comp_valid[:n].reshape(-1, downsample).mean(axis=1)

        ax.scatter(entropy_ds, comp_ds, label=label, s=5, alpha=0.5)

    ax.set_xlabel("Softmax entropy (nats)")
    ax.set_ylabel("Compressibility (W=L)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = ensure_figures_dir() / output_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)
    return out


def plot_temporal_phase(
    run_paths: list[Path],
    context_lengths: list[int],
    labels: list[str],
    title: str,
    output_name: str,
    downsample: int = 100,
) -> Path:
    """Phase portrait with color = time, one subplot per run."""
    n_runs = len(run_paths)
    cols = min(n_runs, 3)
    rows = (n_runs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    # Collect global axis limits across all runs for consistent scales
    all_entropy = []
    all_comp = []
    run_data = []

    for path, context_length in zip(run_paths, context_lengths):
        df = pd.read_parquet(path)
        exp = df[df.phase == "experiment"].reset_index(drop=True)
        result = analyze_run(path, context_length)
        comp = result["compressibility_primary"]

        valid_mask = ~np.isnan(comp)
        entropy = exp.entropy.to_numpy()[valid_mask]
        comp_valid = comp[valid_mask]

        n = len(entropy) // downsample * downsample
        entropy_ds = entropy[:n].reshape(-1, downsample).mean(axis=1)
        comp_ds = comp_valid[:n].reshape(-1, downsample).mean(axis=1)
        time_ds = np.arange(len(entropy_ds))

        all_entropy.extend(entropy_ds)
        all_comp.extend(comp_ds)
        run_data.append((entropy_ds, comp_ds, time_ds))

    e_min, e_max = min(all_entropy), max(all_entropy)
    c_min, c_max = min(all_comp), max(all_comp)
    e_pad = (e_max - e_min) * 0.05
    c_pad = (c_max - c_min) * 0.05

    for idx, (label, (entropy_ds, comp_ds, time_ds)) in enumerate(zip(labels, run_data)):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        sc = ax.scatter(
            entropy_ds, comp_ds, c=time_ds, cmap="RdYlGn_r",
            s=8, alpha=0.6, edgecolors="none",
        )
        ax.set_xlim(e_min - e_pad, e_max + e_pad)
        ax.set_ylim(c_min - c_pad, c_max + c_pad)
        ax.set_title(label)
        ax.set_xlabel("Softmax entropy (nats)")
        ax.set_ylabel("Compressibility (W=L)")
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f"Step (×{downsample})")

    # Hide unused subplots
    for idx in range(n_runs, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()

    out = ensure_figures_dir() / output_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)
    return out


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    paths = sorted(Path(p) for p in args.runs)
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Run file not found: {p}")

    all_params = [parse_run_name(p) for p in paths]
    labels = [make_label(p, all_params) for p in all_params]
    context_lengths = [p["L"] for p in all_params]
    prefix = make_output_prefix(all_params, args.suffix)

    # Build title from varying dimensions
    fixed_parts = []
    ls = sorted({p["L"] for p in all_params})
    ts = sorted({p["T"] for p in all_params})
    ss = sorted({p["S"] for p in all_params})
    if len(ls) == 1:
        fixed_parts.append(f"L={ls[0]}")
    if len(ts) == 1:
        fixed_parts.append(f"T={ts[0]:.2f}")
    if len(ss) == 1:
        fixed_parts.append(f"seed={ss[0]}")
    title_ctx = ", ".join(fixed_parts) if fixed_parts else "all runs"

    log.info("Plotting %d runs: %s", len(paths), [p.stem for p in paths])

    if "entropy" in args.plots:
        plot_entropy_timeseries(
            paths, labels,
            title=f"Entropy over time ({title_ctx})",
            output_name=f"{prefix}_entropy.png",
            downsample=args.downsample,
        )

    if "compressibility" in args.plots:
        plot_compressibility_timeseries(
            paths, context_lengths, labels,
            title=f"Compressibility over time ({title_ctx})",
            output_name=f"{prefix}_compressibility.png",
            downsample=args.downsample,
        )

    if "phase" in args.plots:
        plot_phase_portrait(
            paths, context_lengths, labels,
            title=f"Phase portrait ({title_ctx})",
            output_name=f"{prefix}_phase.png",
            downsample=args.downsample,
        )

    if "temporal" in args.plots:
        plot_temporal_phase(
            paths, context_lengths, labels,
            title=f"Temporal phase portrait ({title_ctx})",
            output_name=f"{prefix}_temporal.png",
            downsample=args.downsample,
        )


if __name__ == "__main__":
    main()

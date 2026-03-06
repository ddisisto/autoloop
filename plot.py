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
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import analyze_run
from utils import eos_ema

log = logging.getLogger(__name__)

FIGURES_DIR = Path("data/figures")

PLOT_TYPES = ["entropy", "compressibility", "phase", "temporal", "violin"]

NEEDS_ANALYSIS = {"compressibility", "phase", "temporal", "violin"}

EOS_EMA_SPAN = 1000


@dataclass
class RunBundle:
    """Pre-loaded data for a single run, shared across plot functions."""
    path: Path
    params: dict
    label: str
    exp: pd.DataFrame
    analysis: dict | None  # Result of analyze_run; None if no plot needs it


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
    runs: list[RunBundle],
    title: str,
    output_name: str,
    downsample: int = 100,
) -> Path:
    """Plot entropy over time with EOS rate EMA overlay on secondary axis."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax_eos = ax.twinx()

    for run in runs:
        n = len(run.exp) // downsample * downsample
        entropy = run.exp.entropy.to_numpy()[:n].reshape(-1, downsample).mean(axis=1)
        steps = np.arange(downsample // 2, n, downsample)
        ax.plot(steps, entropy, label=run.label, linewidth=0.8)

        # EOS rate EMA, downsampled to match
        ema = eos_ema(run.exp.eos.to_numpy()[:n].astype(float), EOS_EMA_SPAN)
        ema_ds = ema.reshape(-1, downsample).mean(axis=1)
        ax_eos.plot(steps, ema_ds, linewidth=0.6, alpha=0.4, linestyle="--")

    ax.set_xlabel("Step (experiment phase)")
    ax.set_ylabel("Softmax entropy (nats)")
    ax_eos.set_ylabel("EOS rate (EMA)", alpha=0.5)
    ax_eos.tick_params(axis="y", colors=(0.5, 0.5, 0.5))
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = ensure_figures_dir() / output_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)
    return out


def default_window_sizes(L: int) -> list[int]:
    """Default window sizes for a given context length."""
    return [L, max(L // 4, 16)]


def _comp(run: RunBundle, w: int) -> np.ndarray:
    """Get compressibility array for window size w from a run's analysis."""
    return run.analysis["compressibility"][w]


def plot_compressibility_timeseries(
    runs: list[RunBundle],
    title: str,
    output_name: str,
    downsample: int = 100,
) -> Path:
    """Plot compressibility (W=L) over time for multiple runs."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for run in runs:
        comp = _comp(run, run.params["L"])
        valid = comp[~np.isnan(comp)]
        n = len(valid) // downsample * downsample
        comp_ds = valid[:n].reshape(-1, downsample).mean(axis=1)
        steps = np.arange(downsample // 2, n, downsample)
        ax.plot(steps, comp_ds, label=run.label, linewidth=0.8)

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
    runs: list[RunBundle],
    title: str,
    output_name: str,
    downsample: int = 100,
) -> Path:
    """2D scatter: entropy vs compressibility for multiple runs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for run in runs:
        comp = _comp(run, run.params["L"])
        valid_mask = ~np.isnan(comp)
        entropy = run.exp.entropy.to_numpy()[valid_mask]
        comp_valid = comp[valid_mask]

        n = len(entropy) // downsample * downsample
        entropy_ds = entropy[:n].reshape(-1, downsample).mean(axis=1)
        comp_ds = comp_valid[:n].reshape(-1, downsample).mean(axis=1)

        ax.scatter(entropy_ds, comp_ds, label=run.label, s=5, alpha=0.5)

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
    runs: list[RunBundle],
    title: str,
    output_name: str,
    downsample: int = 100,
) -> Path:
    """Phase portrait with color = time, one subplot per run."""
    n_runs = len(runs)
    cols = min(n_runs, 3)
    rows = (n_runs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)

    # Collect global axis limits across all runs for consistent scales
    all_entropy = []
    all_comp = []
    run_data = []

    for run in runs:
        comp = _comp(run, run.params["L"])
        valid_mask = ~np.isnan(comp)
        entropy = run.exp.entropy.to_numpy()[valid_mask]
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

    for idx, (run, (entropy_ds, comp_ds, time_ds)) in enumerate(zip(runs, run_data)):
        row, col = divmod(idx, cols)
        ax = axes[row][col]
        sc = ax.scatter(
            entropy_ds, comp_ds, c=time_ds, cmap="cividis",
            s=8, alpha=0.6, edgecolors="none",
        )
        ax.set_xlim(e_min - e_pad, e_max + e_pad)
        ax.set_ylim(c_min - c_pad, c_max + c_pad)
        ax.set_title(run.label)
        ax.set_xlabel("Softmax entropy (nats)")
        ax.set_ylabel("Compressibility (W=L)")
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(f"Step (x{downsample})")

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


def _draw_half_violin(ax: plt.Axes, data: np.ndarray, y_center: float,
                      side: str, color: str, width: float = 0.35,
                      alpha: float = 0.7, label: str | None = None) -> None:
    """Draw a half-violin (KDE) on one side of y_center."""
    from scipy.stats import gaussian_kde

    if len(data) < 5 or np.std(data) < 1e-10:
        # Degenerate: draw a thin line at the mean
        xval = np.mean(data) if len(data) > 0 else 0
        ax.plot([xval, xval], [y_center - width * 0.3, y_center + width * 0.3],
                color=color, linewidth=1.5, alpha=alpha, label=label)
        return

    kde = gaussian_kde(data, bw_method="scott")
    x_grid = np.linspace(np.percentile(data, 1), np.percentile(data, 99), 200)
    density = kde(x_grid)
    density = density / density.max() * width  # normalize to max width

    if side == "upper":
        ax.fill_between(x_grid, y_center, y_center + density,
                        color=color, alpha=alpha, label=label)
        ax.plot(x_grid, y_center + density, color=color, linewidth=0.5, alpha=0.9)
    else:
        ax.fill_between(x_grid, y_center - density, y_center,
                        color=color, alpha=alpha, label=label)
        ax.plot(x_grid, y_center - density, color=color, linewidth=0.5, alpha=0.9)


def plot_violin(
    runs: list[RunBundle],
    title: str,
    output_name: str,
    n_blocks: int = 10,
    downsample: int = 100,
) -> Path:
    """Split violin: entropy (upper) vs inverted compressibility (lower) per time block.

    One column per run. Upper half = entropy distribution (blue),
    lower half = 1/compression_ratio distribution (red = W=L, lighter = W=L/4).
    Higher compressibility = more structured/repetitive.
    """
    n_runs = len(runs)
    fig, axes = plt.subplots(1, n_runs, figsize=(5 * n_runs, 5), squeeze=False)

    for run_idx, run in enumerate(runs):
        ax = axes[0][run_idx]

        L = run.params["L"]
        comp_primary = _comp(run, L)
        comp_secondary = _comp(run, max(L // 4, 16))

        block_size = len(run.exp) // n_blocks

        # Twin axis: entropy on top x-axis, compressibility on bottom
        ax_top = ax.twiny()

        for b in range(n_blocks):
            sl = slice(b * block_size, (b + 1) * block_size)
            y = b

            # Entropy -- upper half (blue)
            ent_block = run.exp.entropy.to_numpy()[sl]
            _draw_half_violin(ax_top, ent_block, y, "upper", "#2166ac")

            # Inverted compressibility W=L -- lower half (red)
            comp_block = comp_primary[sl]
            comp_valid = comp_block[~np.isnan(comp_block)]
            if len(comp_valid) > 0:
                _draw_half_violin(ax, 1.0 / comp_valid, y, "lower", "#b2182b")

            # Inverted compressibility W=L/4 -- lower half (lighter red)
            comp_s_block = comp_secondary[sl]
            comp_s_valid = comp_s_block[~np.isnan(comp_s_block)]
            if len(comp_s_valid) > 0:
                _draw_half_violin(ax, 1.0 / comp_s_valid, y, "lower", "#b2182b",
                                  alpha=0.25)

            ax.axhline(y, color="grey", linewidth=0.3, alpha=0.5)

        block_k = block_size // 1000
        yticks = range(n_blocks)
        yticklabels = [f"{b * block_k}-{(b + 1) * block_k}k" for b in range(n_blocks)]
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.grid(True, axis="x", alpha=0.3)

        ax.set_xlabel("1 / compression ratio (↑ = more structured)", fontsize=8)
        ax_top.set_xlabel("Entropy (nats)", fontsize=8)
        ax.set_title(run.label)

    axes[0][0].set_ylabel("Time block")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2166ac", alpha=0.7, label="Entropy (upper)"),
        Patch(facecolor="#b2182b", alpha=0.7, label="1/compr. W=L (lower)"),
        Patch(facecolor="#b2182b", alpha=0.25, label="1/compr. W=L/4 (lower)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

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

    # Determine whether any requested plot needs analysis
    needs_analysis = bool(set(args.plots) & NEEDS_ANALYSIS)

    # Pre-load all run data: parquet once, analyze_run once (if needed)
    runs: list[RunBundle] = []
    for path, params, label in zip(paths, all_params, labels):
        log.info("Loading %s", path.stem)
        df = pd.read_parquet(path)
        exp = df[df.phase == "experiment"].reset_index(drop=True)

        analysis = None
        if needs_analysis:
            ws = default_window_sizes(params["L"])
            log.info("Analyzing %s (W=%s)", path.stem, ws)
            analysis = analyze_run(path, ws)

        runs.append(RunBundle(
            path=path, params=params, label=label, exp=exp, analysis=analysis,
        ))

    if "entropy" in args.plots:
        plot_entropy_timeseries(
            runs,
            title=f"Entropy over time ({title_ctx})",
            output_name=f"{prefix}_entropy.png",
            downsample=args.downsample,
        )

    if "compressibility" in args.plots:
        plot_compressibility_timeseries(
            runs,
            title=f"Compressibility over time ({title_ctx})",
            output_name=f"{prefix}_compressibility.png",
            downsample=args.downsample,
        )

    if "phase" in args.plots:
        plot_phase_portrait(
            runs,
            title=f"Phase portrait ({title_ctx})",
            output_name=f"{prefix}_phase.png",
            downsample=args.downsample,
        )

    if "temporal" in args.plots:
        plot_temporal_phase(
            runs,
            title=f"Temporal phase portrait ({title_ctx})",
            output_name=f"{prefix}_temporal.png",
            downsample=args.downsample,
        )

    if "violin" in args.plots:
        plot_violin(
            runs,
            title=f"Distribution over time ({title_ctx})",
            output_name=f"{prefix}_violin.png",
            downsample=args.downsample,
        )


if __name__ == "__main__":
    main()

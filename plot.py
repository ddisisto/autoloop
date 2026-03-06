"""Visualization for autoloop analysis.

Generates plots from generation data and derived metrics.
All plots saved to data/figures/.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import analyze_run

log = logging.getLogger(__name__)

FIGURES_DIR = Path("data/figures")


def ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR


def plot_entropy_timeseries(
    run_paths: list[Path],
    context_length: int,
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
        # Downsample by taking block means
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
    context_length: int,
    labels: list[str],
    title: str,
    output_name: str,
    downsample: int = 100,
) -> Path:
    """Plot compressibility (W=L) over time for multiple runs."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for path, label in zip(run_paths, labels):
        result = analyze_run(path, context_length)
        comp = result["compressibility_primary"]
        valid = comp[~np.isnan(comp)]
        n = len(valid) // downsample * downsample
        comp_ds = valid[:n].reshape(-1, downsample).mean(axis=1)
        steps = np.arange(downsample // 2, n, downsample)
        ax.plot(steps, comp_ds, label=label, linewidth=0.8)

    ax.set_xlabel("Step (experiment phase)")
    ax.set_ylabel(f"Compressibility (W={context_length})")
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
    context_length: int,
    labels: list[str],
    title: str,
    output_name: str,
    downsample: int = 100,
) -> Path:
    """2D scatter: entropy vs compressibility for multiple runs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for path, label in zip(run_paths, labels):
        df = pd.read_parquet(path)
        exp = df[df.phase == "experiment"].reset_index(drop=True)
        result = analyze_run(path, context_length)
        comp = result["compressibility_primary"]

        # Align: compressibility starts at index context_length-1
        valid_mask = ~np.isnan(comp)
        entropy = exp.entropy.to_numpy()[valid_mask]
        comp_valid = comp[valid_mask]

        # Downsample both together
        n = len(entropy) // downsample * downsample
        entropy_ds = entropy[:n].reshape(-1, downsample).mean(axis=1)
        comp_ds = comp_valid[:n].reshape(-1, downsample).mean(axis=1)

        ax.scatter(entropy_ds, comp_ds, label=label, s=5, alpha=0.5)

    ax.set_xlabel("Softmax entropy (nats)")
    ax.set_ylabel(f"Compressibility (W={context_length})")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = ensure_figures_dir() / output_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Plot all available L=64 seed=42 runs
    runs_dir = Path("data/runs")
    paths = sorted(runs_dir.glob("L0064_T*_S42.parquet"))
    if not paths:
        log.info("No L=64 runs found")
        raise SystemExit(1)

    labels = [p.stem.split("_")[1] for p in paths]  # e.g. "T0.50"

    plot_entropy_timeseries(
        paths, 64, labels,
        title="Entropy over time (L=64, seed=42)",
        output_name="L0064_S42_entropy_timeseries.png",
    )

    plot_compressibility_timeseries(
        paths, 64, labels,
        title="Compressibility over time (L=64, W=64, seed=42)",
        output_name="L0064_S42_compressibility_timeseries.png",
    )

    plot_phase_portrait(
        paths, 64, labels,
        title="Phase portrait (L=64, W=64, seed=42)",
        output_name="L0064_S42_phase_portrait.png",
    )

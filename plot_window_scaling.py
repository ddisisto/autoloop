"""Plot compressibility as a function of L and W.

Explores how measurement window size interacts with context length.
Reads standard-W analysis caches produced by analyze_windows.py.

Usage:
    python plot_window_scaling.py
    python plot_window_scaling.py --temps 0.50 1.00
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze import analyze_run
from analyze_windows import STANDARD_WINDOWS
from plot import FIGURES_DIR, ensure_figures_dir, parse_run_name

log = logging.getLogger(__name__)


def load_mean_compressibility(parquet_path: Path, window_sizes: list[int]) -> dict[int, float]:
    """Load analysis and return mean compressibility per window size."""
    analysis = analyze_run(parquet_path, window_sizes)
    result = {}
    for w in window_sizes:
        arr = analysis["compressibility"][w]
        valid = arr[~np.isnan(arr)]
        result[w] = float(valid.mean()) if len(valid) > 0 else np.nan
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temps", type=float, nargs="*", help="Filter to specific temperatures")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    paths = sorted(Path("data/runs").glob("*.parquet"))
    runs = []
    for p in paths:
        params = parse_run_name(p)
        if args.temps and params["T"] not in args.temps:
            continue
        runs.append((p, params))

    # Group by temperature
    temps = sorted({p["T"] for _, p in runs})
    ls = sorted({p["L"] for _, p in runs})

    log.info("Runs: %d, L=%s, T=%s", len(runs), ls, temps)

    ensure_figures_dir()

    # --- Plot 1: Compressibility(W=64) vs L, one curve per T ---
    # Fixed measurement window, varying context length
    W_FIXED = 64
    fig, ax = plt.subplots(figsize=(8, 5))

    for t in temps:
        t_runs = [(p, pr) for p, pr in runs if pr["T"] == t]
        x_l = []
        y_comp = []
        for p, pr in sorted(t_runs, key=lambda x: x[1]["L"]):
            mc = load_mean_compressibility(p, STANDARD_WINDOWS)
            x_l.append(pr["L"])
            y_comp.append(mc[W_FIXED])
        ax.plot(x_l, y_comp, marker="o", label=f"T={t:.2f}", linewidth=1.5)

    ax.set_xlabel("Context length L")
    ax.set_ylabel(f"Mean compressibility (W={W_FIXED})")
    ax.set_title(f"Attractor depth at fixed measurement scale (W={W_FIXED})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "scaling_comp_vs_L_W64.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)

    # --- Plot 2: Compressibility vs W, one curve per (L, T) ---
    # How measurement scale affects what we see
    fig, axes = plt.subplots(1, len(temps), figsize=(5 * len(temps), 5), squeeze=False)

    for t_idx, t in enumerate(temps):
        ax = axes[0][t_idx]
        t_runs = [(p, pr) for p, pr in runs if pr["T"] == t]

        for p, pr in sorted(t_runs, key=lambda x: x[1]["L"]):
            mc = load_mean_compressibility(p, STANDARD_WINDOWS)
            ws = sorted(mc.keys())
            # Only plot W values <= L (larger W than context doesn't make sense)
            ws_valid = [w for w in ws if w <= pr["L"]]
            ax.plot(ws_valid, [mc[w] for w in ws_valid],
                    marker="o", label=f"L={pr['L']}", linewidth=1.2)

        ax.set_xlabel("Window size W")
        ax.set_ylabel("Mean compressibility")
        ax.set_title(f"T={t:.2f}")
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Compressibility vs measurement scale", y=1.02)
    fig.tight_layout()

    out = FIGURES_DIR / "scaling_comp_vs_W.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)

    # --- Plot 3: Heatmap — mean compressibility as f(L, W) per temperature ---
    for t in temps:
        t_runs = [(p, pr) for p, pr in runs if pr["T"] == t]
        t_ls = sorted({pr["L"] for _, pr in t_runs})

        grid = np.full((len(STANDARD_WINDOWS), len(t_ls)), np.nan)
        for l_idx, l in enumerate(t_ls):
            matching = [(p, pr) for p, pr in t_runs if pr["L"] == l]
            if not matching:
                continue
            p, pr = matching[0]
            mc = load_mean_compressibility(p, STANDARD_WINDOWS)
            for w_idx, w in enumerate(STANDARD_WINDOWS):
                if w <= l:
                    grid[w_idx, l_idx] = mc[w]

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(t_ls)))
        ax.set_xticklabels(t_ls)
        ax.set_yticks(range(len(STANDARD_WINDOWS)))
        ax.set_yticklabels(STANDARD_WINDOWS)
        ax.set_xlabel("Context length L")
        ax.set_ylabel("Window size W")
        ax.set_title(f"Mean compressibility (T={t:.2f})")

        # Annotate cells
        for w_idx in range(len(STANDARD_WINDOWS)):
            for l_idx in range(len(t_ls)):
                val = grid[w_idx, l_idx]
                if not np.isnan(val):
                    ax.text(l_idx, w_idx, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if val < 0.5 else "black")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        out = FIGURES_DIR / f"scaling_heatmap_T{t:.2f}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        log.info("Saved %s", out)


if __name__ == "__main__":
    main()

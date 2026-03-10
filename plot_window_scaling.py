"""Cross-run analysis plots: window scaling, transfer functions, autocorrelation.

Reads analysis caches and summary stats across the full run grid.
Seed-aware: averages across seeds with error bars where multiple seeds exist.

Usage:
    python plot_window_scaling.py
    python plot_window_scaling.py --temps 0.50 1.00
    python plot_window_scaling.py --temps 0.50 --ldense     # L-dense detail plot
    python plot_window_scaling.py --transfer                 # T→H and T→C curves
    python plot_window_scaling.py --autocorr                 # entropy autocorrelation
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze import analyze_run, comp_stats
from analyze_windows import STANDARD_WINDOWS
from plot import FIGURES_DIR, ensure_figures_dir, parse_run_name

log = logging.getLogger(__name__)


_analysis_cache: dict[Path, dict] = {}


def _get_analysis(parquet_path: Path, window_sizes: list[int]) -> dict:
    """Load analysis with in-memory caching across calls."""
    if parquet_path not in _analysis_cache:
        _analysis_cache[parquet_path] = analyze_run(parquet_path, window_sizes)
    return _analysis_cache[parquet_path]


def load_mean_compressibility(parquet_path: Path, window_sizes: list[int]) -> dict[int, float]:
    """Load analysis and return mean compressibility per window size."""
    analysis = _get_analysis(parquet_path, window_sizes)
    return {w: comp_stats(analysis, w)["mean"] for w in window_sizes}


def load_summary(parquet_path: Path, window_sizes: list[int]) -> dict:
    """Load analysis and return summary stats."""
    return _get_analysis(parquet_path, window_sizes)["summary"]


def group_by_LT(runs: list[tuple[Path, dict]]) -> dict[tuple[int, float], list[tuple[Path, dict]]]:
    """Group runs by (L, T), collecting seeds together."""
    groups: dict[tuple[int, float], list[tuple[Path, dict]]] = defaultdict(list)
    for p, pr in runs:
        groups[(pr["L"], pr["T"])].append((p, pr))
    return groups


def seed_stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, std) from a list of per-seed values, ignoring NaN."""
    clean = [v for v in values if not np.isnan(v)]
    if not clean:
        return np.nan, np.nan
    return float(np.mean(clean)), float(np.std(clean))


def _plot_transfer(
    groups: dict[tuple[int, float], list[tuple[Path, dict]]],
    ls: list[int],
    temps: list[float],
) -> None:
    """Transfer function plots: T→entropy and T→compressibility at each L."""
    # Filter to core L values (skip L-dense intermediates for clarity)
    core_ls = [l for l in ls if l in (64, 128, 192, 256)]
    if not core_ls:
        core_ls = ls

    fig, (ax_h, ax_c, ax_e) = plt.subplots(1, 3, figsize=(15, 5))

    for l in core_ls:
        t_vals, h_mean, h_std = [], [], []
        c_mean, c_std = [], []
        e_mean, e_std = [], []

        for t in temps:
            if (l, t) not in groups:
                continue
            seed_h, seed_c, seed_e = [], [], []
            for p, pr in groups[(l, t)]:
                summary = load_summary(p, STANDARD_WINDOWS)
                mc = load_mean_compressibility(p, STANDARD_WINDOWS)
                seed_h.append(summary["entropy_mean"])
                seed_c.append(mc[64])
                seed_e.append(summary["eos_rate"])

            hm, hs = seed_stats(seed_h)
            cm, cs = seed_stats(seed_c)
            em, es = seed_stats(seed_e)
            t_vals.append(t)
            h_mean.append(hm)
            h_std.append(hs)
            c_mean.append(cm)
            c_std.append(cs)
            e_mean.append(em)
            e_std.append(es)

        h_arr, hs_arr = np.array(h_mean), np.array(h_std)
        c_arr, cs_arr = np.array(c_mean), np.array(c_std)
        e_arr, es_arr = np.array(e_mean), np.array(e_std)

        ax_h.plot(t_vals, h_arr, marker="o", label=f"L={l}", linewidth=1.5)
        if any(s > 0 for s in h_std):
            ax_h.fill_between(t_vals, h_arr - hs_arr, h_arr + hs_arr, alpha=0.15)

        ax_c.plot(t_vals, c_arr, marker="o", label=f"L={l}", linewidth=1.5)
        if any(s > 0 for s in c_std):
            ax_c.fill_between(t_vals, c_arr - cs_arr, c_arr + cs_arr, alpha=0.15)

        ax_e.plot(t_vals, e_arr, marker="o", label=f"L={l}", linewidth=1.5)
        if any(s > 0 for s in e_std):
            ax_e.fill_between(t_vals, e_arr - es_arr, e_arr + es_arr, alpha=0.15)

    ax_h.set_xlabel("Temperature T")
    ax_h.set_ylabel("Mean entropy (nats)")
    ax_h.set_title("T → Entropy")
    ax_h.legend()
    ax_h.grid(True, alpha=0.3)

    ax_c.set_xlabel("Temperature T")
    ax_c.set_ylabel("Mean compressibility (W=64)")
    ax_c.set_title("T → Compressibility")
    ax_c.legend()
    ax_c.grid(True, alpha=0.3)

    ax_e.set_xlabel("Temperature T")
    ax_e.set_ylabel("EOS rate")
    ax_e.set_title("T → EOS Rate")
    ax_e.legend()
    ax_e.grid(True, alpha=0.3)

    fig.suptitle("Transfer functions: temperature to observables", fontsize=13)
    fig.tight_layout()

    out = FIGURES_DIR / "transfer_functions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


def _plot_autocorr(
    groups: dict[tuple[int, float], list[tuple[Path, dict]]],
    ls: list[int],
    temps: list[float],
) -> None:
    """Entropy autocorrelation plots: one subplot per T, one curve per L."""
    core_ls = [l for l in ls if l in (64, 128, 192, 256)]
    if not core_ls:
        core_ls = ls

    fig, axes = plt.subplots(1, len(temps), figsize=(5 * len(temps), 4), squeeze=False)

    for t_idx, t in enumerate(temps):
        ax = axes[0][t_idx]
        for l in core_ls:
            if (l, t) not in groups:
                continue
            # Average ACF across seeds
            acfs = []
            for p, pr in groups[(l, t)]:
                analysis = _get_analysis(p, STANDARD_WINDOWS)
                if "entropy_autocorrelation" in analysis:
                    acfs.append(analysis["entropy_autocorrelation"])
            if not acfs:
                continue
            min_len = min(len(a) for a in acfs)
            acf_mean = np.mean([a[:min_len] for a in acfs], axis=0)
            lags = np.arange(min_len)
            ax.plot(lags, acf_mean, label=f"L={l}", linewidth=1.2, alpha=0.8)

        ax.set_xlabel("Lag (steps)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"T={t:.2f}")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 500)

    fig.suptitle("Entropy autocorrelation", y=1.02, fontsize=13)
    fig.tight_layout()

    out = FIGURES_DIR / "entropy_autocorrelation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temps", type=float, nargs="*", help="Filter to specific temperatures")
    parser.add_argument("--ldense", action="store_true",
                        help="Generate L-densification detail plot (T=0.50 focused)")
    parser.add_argument("--transfer", action="store_true",
                        help="Generate transfer function plots (T→H, T→C, T→EOS)")
    parser.add_argument("--autocorr", action="store_true",
                        help="Generate entropy autocorrelation plots")
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

    temps = sorted({p["T"] for _, p in runs})
    ls = sorted({p["L"] for _, p in runs})
    groups = group_by_LT(runs)

    log.info("Runs: %d, L=%s, T=%s", len(runs), ls, temps)

    ensure_figures_dir()

    # --- Plot 1: Compressibility(W=64) vs L, one curve per T, seed-averaged ---
    W_FIXED = 64
    fig, ax = plt.subplots(figsize=(8, 5))

    for t in temps:
        t_ls = sorted(L for (L, T) in groups if T == t)
        x_l, y_mean, y_std = [], [], []
        for l in t_ls:
            seed_vals = []
            for p, pr in groups[(l, t)]:
                mc = load_mean_compressibility(p, STANDARD_WINDOWS)
                seed_vals.append(mc[W_FIXED])
            m, s = seed_stats(seed_vals)
            x_l.append(l)
            y_mean.append(m)
            y_std.append(s)

        y_mean_arr = np.array(y_mean)
        y_std_arr = np.array(y_std)
        ax.plot(x_l, y_mean_arr, marker="o", label=f"T={t:.2f}", linewidth=1.5)
        if any(s > 0 for s in y_std):
            ax.fill_between(x_l, y_mean_arr - y_std_arr, y_mean_arr + y_std_arr, alpha=0.15)

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

    # --- Plot 2: Compressibility vs W, one curve per L, seed-averaged ---
    fig, axes = plt.subplots(1, len(temps), figsize=(5 * len(temps), 5), squeeze=False)

    for t_idx, t in enumerate(temps):
        ax = axes[0][t_idx]
        t_ls = sorted(L for (L, T) in groups if T == t)

        for l in t_ls:
            seed_means = defaultdict(list)
            for p, pr in groups[(l, t)]:
                mc = load_mean_compressibility(p, STANDARD_WINDOWS)
                for w in STANDARD_WINDOWS:
                    seed_means[w].append(mc[w])

            ws_valid = sorted(seed_means.keys())
            y_vals = [np.mean(seed_means[w]) for w in ws_valid]
            ax.plot(ws_valid, y_vals, marker="o", label=f"L={l}", linewidth=1.2)

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
        t_ls = sorted(L for (L, T) in groups if T == t)

        grid = np.full((len(STANDARD_WINDOWS), len(t_ls)), np.nan)
        for l_idx, l in enumerate(t_ls):
            seed_vals_per_w = defaultdict(list)
            for p, pr in groups[(l, t)]:
                mc = load_mean_compressibility(p, STANDARD_WINDOWS)
                for w_idx, w in enumerate(STANDARD_WINDOWS):
                    seed_vals_per_w[w_idx].append(mc[w])
            for w_idx, vals in seed_vals_per_w.items():
                grid[w_idx, l_idx] = np.mean(vals)

        fig, ax = plt.subplots(figsize=(max(6, len(t_ls) * 0.8 + 2), 4))
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(t_ls)))
        ax.set_xticklabels(t_ls, rotation=45 if len(t_ls) > 6 else 0)
        ax.set_yticks(range(len(STANDARD_WINDOWS)))
        ax.set_yticklabels(STANDARD_WINDOWS)
        ax.set_xlabel("Context length L")
        ax.set_ylabel("Window size W")
        ax.set_title(f"Mean compressibility (T={t:.2f}, seed-averaged)")

        for w_idx in range(len(STANDARD_WINDOWS)):
            for l_idx in range(len(t_ls)):
                val = grid[w_idx, l_idx]
                if not np.isnan(val):
                    ax.text(l_idx, w_idx, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if val < 0.5 else "black")

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        out = FIGURES_DIR / f"scaling_heatmap_T{t:.2f}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        log.info("Saved %s", out)

    # --- Optional plots ---
    if args.ldense:
        _plot_ldense(runs, groups)
    if args.transfer:
        _plot_transfer(groups, ls, temps)
    if args.autocorr:
        _plot_autocorr(groups, ls, temps)


def _plot_ldense(
    runs: list[tuple[Path, dict]],
    groups: dict[tuple[int, float], list[tuple[Path, dict]]],
) -> None:
    """L-densification detail: per-seed points + mean, compressibility + entropy."""
    T_FOCUS = 0.50
    t_ls = sorted(L for (L, T) in groups if T == T_FOCUS)
    if not t_ls:
        log.warning("No runs at T=%.2f for --ldense", T_FOCUS)
        return

    seed_markers = {42: "o", 123: "s", 7: "^"}
    seed_colors = {42: "#1f77b4", 123: "#ff7f0e", 7: "#2ca02c"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for col, w in enumerate([64, 128]):
        # Top row: compressibility
        ax = axes[0][col]
        # Bottom row: entropy
        ax_ent = axes[1][col] if col == 0 else None

        mean_x, mean_y, mean_std = [], [], []
        for l in t_ls:
            seed_vals = []
            for p, pr in groups[(l, T_FOCUS)]:
                mc = load_mean_compressibility(p, STANDARD_WINDOWS)
                val = mc[w]
                seed = pr["S"]
                ax.scatter(l, val, marker=seed_markers.get(seed, "x"),
                           color=seed_colors.get(seed, "gray"), s=50, alpha=0.7,
                           zorder=3)
                seed_vals.append(val)

            m, s = seed_stats(seed_vals)
            mean_x.append(l)
            mean_y.append(m)
            mean_std.append(s)

        y_arr = np.array(mean_y)
        s_arr = np.array(mean_std)
        ax.plot(mean_x, y_arr, "k-", linewidth=2, alpha=0.6, zorder=2)
        ax.fill_between(mean_x, y_arr - s_arr, y_arr + s_arr,
                        color="gray", alpha=0.15, zorder=1)
        ax.set_xlabel("Context length L")
        ax.set_ylabel(f"Mean compressibility (W={w})")
        ax.set_title(f"T={T_FOCUS} — Compressibility (W={w})")
        ax.grid(True, alpha=0.3)

        # Add seed legend on first panel
        if col == 0:
            for seed in sorted(seed_markers):
                ax.scatter([], [], marker=seed_markers[seed],
                           color=seed_colors[seed], s=50, label=f"S={seed}")
            ax.legend(fontsize=8)

    # Bottom-left: entropy
    ax = axes[1][0]
    mean_x, mean_y, mean_std = [], [], []
    for l in t_ls:
        seed_vals = []
        for p, pr in groups[(l, T_FOCUS)]:
            summary = load_summary(p, STANDARD_WINDOWS)
            val = summary["entropy_mean"]
            seed = pr["S"]
            ax.scatter(l, val, marker=seed_markers.get(seed, "x"),
                       color=seed_colors.get(seed, "gray"), s=50, alpha=0.7,
                       zorder=3)
            seed_vals.append(val)

        m, s = seed_stats(seed_vals)
        mean_x.append(l)
        mean_y.append(m)
        mean_std.append(s)

    y_arr = np.array(mean_y)
    s_arr = np.array(mean_std)
    ax.plot(mean_x, y_arr, "k-", linewidth=2, alpha=0.6, zorder=2)
    ax.fill_between(mean_x, y_arr - s_arr, y_arr + s_arr,
                    color="gray", alpha=0.15, zorder=1)
    ax.set_xlabel("Context length L")
    ax.set_ylabel("Mean entropy")
    ax.set_title(f"T={T_FOCUS} — Entropy")
    ax.grid(True, alpha=0.3)
    for seed in sorted(seed_markers):
        ax.scatter([], [], marker=seed_markers[seed],
                   color=seed_colors[seed], s=50, label=f"S={seed}")
    ax.legend(fontsize=8)

    # Bottom-right: EOS rate
    ax = axes[1][1]
    mean_x, mean_y, mean_std = [], [], []
    for l in t_ls:
        seed_vals = []
        for p, pr in groups[(l, T_FOCUS)]:
            summary = load_summary(p, STANDARD_WINDOWS)
            val = summary["eos_rate"]
            seed = pr["S"]
            ax.scatter(l, val, marker=seed_markers.get(seed, "x"),
                       color=seed_colors.get(seed, "gray"), s=50, alpha=0.7,
                       zorder=3)
            seed_vals.append(val)

        m, s = seed_stats(seed_vals)
        mean_x.append(l)
        mean_y.append(m)
        mean_std.append(s)

    y_arr = np.array(mean_y)
    s_arr = np.array(mean_std)
    ax.plot(mean_x, y_arr, "k-", linewidth=2, alpha=0.6, zorder=2)
    ax.fill_between(mean_x, y_arr - s_arr, y_arr + s_arr,
                    color="gray", alpha=0.15, zorder=1)
    ax.set_xlabel("Context length L")
    ax.set_ylabel("EOS rate")
    ax.set_title(f"T={T_FOCUS} — EOS Rate")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"L-densification at T={T_FOCUS} (per-seed + mean±std)", fontsize=13)
    fig.tight_layout()

    out = FIGURES_DIR / f"ldense_T{T_FOCUS:.2f}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()

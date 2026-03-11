"""Unified sweep runner for autoloop generation experiments.

Replaces pilot_sweep.py, crossover_sweep.py, seed_sweep.py, ldense_sweep.py,
l256_sweep.py. Same properties: idempotent, crash-resilient, file logging.

Usage:
    # Run a named preset
    python sweep.py pilot
    python sweep.py crossover --dry-run

    # Ad-hoc grid from CLI
    python sweep.py --L 256 --T 0.60 0.70 0.80 0.90 --seed 42

    # Show status of all data on disk
    python sweep.py --status

    # Show status for a preset
    python sweep.py --status pilot

    # List available presets
    python sweep.py --list
"""

import argparse
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pandas as pd

import runlib

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
SWEEP_DIR = runlib.SWEEP_DIR
MODEL_DIR = REPO_ROOT / "data" / "model" / "SmolLM-135M"

NUM_TOKENS = 100_000
DEVICE = "cuda"


# --- Presets ---
# Each preset records a sweep that was (or will be) run, with rationale.
# Grid is the cross-product of L × T × seeds.

class Preset(TypedDict):
    description: str
    L: list[int]
    T: list[float]
    seeds: list[int]


PRESETS: dict[str, Preset] = {
    "pilot": {
        "description": "Phase 0 pilot grid: coarse L×T at seed=42",
        "L": [64, 128, 192, 256],
        "T": [0.50, 1.00, 1.50],
        "seeds": [42],
    },
    "crossover": {
        "description": "T-densification in crossover region (T=0.60–0.90)",
        "L": [64, 128, 192],
        "T": [0.60, 0.70, 0.80, 0.90],
        "seeds": [42],
    },
    "seed": {
        "description": "Seed replication across full pilot grid",
        "L": [64, 128, 192, 256],
        "T": [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.50],
        "seeds": [123, 7],
    },
    "ldense": {
        "description": "L-densification around L=192 anomaly at T=0.50",
        "L": [160, 176, 192, 208, 224],
        "T": [0.50],
        "seeds": [42, 123, 7],
    },
    "l256-crossover": {
        "description": "Fill L=256 crossover gap (T=0.60–0.90)",
        "L": [256],
        "T": [0.60, 0.70, 0.80, 0.90],
        "seeds": [42],
    },
}


def expand_grid(L: list[int], T: list[float], seeds: list[int]) -> list[tuple[int, float, int]]:
    return [(l, t, s) for l in L for t in T for s in seeds]


def parquet_path(L: int, T: float, seed: int) -> Path:
    return SWEEP_DIR / f"L{L:04d}_T{T:.2f}_S{seed}.parquet"


def is_complete(L: int, T: float, seed: int) -> bool:
    path = parquet_path(L, T, seed)
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, columns=["phase"])
        n_experiment = (df.phase == "experiment").sum()
        return n_experiment >= NUM_TOKENS
    except Exception:
        return False


def run_condition(L: int, T: float, seed: int) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(REPO_ROOT / "generate.py"),
        "--context-length", str(L),
        "--temperature", str(T),
        "--seed", str(seed),
        "--num-tokens", str(NUM_TOKENS),
        "--model-dir", str(MODEL_DIR),
        "--output-dir", str(SWEEP_DIR),
        "--device", DEVICE,
    ]
    return subprocess.run(cmd)


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def scan_runs() -> dict[tuple[int, float, int], bool]:
    """Scan data/runs/ for all parquet files and check completeness."""
    results: dict[tuple[int, float, int], bool] = {}
    if not SWEEP_DIR.exists():
        return results
    pattern = re.compile(r"L(\d{4})_T(\d+\.\d+)_S(\d+)\.parquet")
    for path in sorted(SWEEP_DIR.glob("L*_T*_S*.parquet")):
        m = pattern.match(path.name)
        if m:
            L, T, seed = int(m.group(1)), float(m.group(2)), int(m.group(3))
            results[(L, T, seed)] = is_complete(L, T, seed)
    return results


def print_status(preset_name: str | None = None) -> None:
    """Print a grid table of all runs on disk (or for a specific preset)."""
    runs = scan_runs()
    if not runs:
        print("No runs found in", SWEEP_DIR)
        return

    if preset_name:
        preset = PRESETS[preset_name]
        grid = set(expand_grid(preset["L"], preset["T"], preset["seeds"]))
        # Filter to preset's grid, but also show any runs that match
        runs = {k: v for k, v in runs.items() if k in grid}
        if not runs:
            print(f"No runs found for preset '{preset_name}'")
            return

    # Collect all L, T, seed values
    all_L = sorted({k[0] for k in runs})
    all_T = sorted({k[1] for k in runs})
    all_seeds = sorted({k[2] for k in runs})

    # Build grid table: cells show seeds that are complete
    print(f"\nRuns on disk: {len(runs)} files ({sum(runs.values())} complete, "
          f"{sum(1 for v in runs.values() if not v)} incomplete)")
    if preset_name:
        preset = PRESETS[preset_name]
        grid = set(expand_grid(preset["L"], preset["T"], preset["seeds"]))
        missing = grid - set(runs.keys())
        if missing:
            print(f"Missing from preset '{preset_name}': {len(missing)} runs")
        else:
            print(f"Preset '{preset_name}': all {len(grid)} runs complete")
    print()

    # Header
    t_strs = [f"{t:.2f}" for t in all_T]
    header = "L \\ T | " + " | ".join(f"{s:>10s}" for s in t_strs) + " |"
    sep = "------|" + "|".join("-" * 12 for _ in all_T) + "|"
    print(header)
    print(sep)

    for L in all_L:
        cells = []
        for T in all_T:
            seeds_here = []
            for s in all_seeds:
                if (L, T, s) in runs:
                    if runs[(L, T, s)]:
                        seeds_here.append(str(s))
                    else:
                        seeds_here.append(f"({s})")
            cell = ",".join(seeds_here) if seeds_here else ""
            cells.append(f"{cell:>10s}")
        print(f"{L:>5d} | " + " | ".join(cells) + " |")

    print()
    print("Legend: seed = complete, (seed) = incomplete/in-progress")


def print_presets() -> None:
    """List available presets with descriptions and run counts."""
    print("\nAvailable presets:\n")
    for name, preset in PRESETS.items():
        grid = expand_grid(preset["L"], preset["T"], preset["seeds"])
        done = sum(1 for cond in grid if is_complete(*cond))
        print(f"  {name:20s}  {done}/{len(grid)} done  — {preset['description']}")
    print()


def run_sweep(grid: list[tuple[int, float, int]], label: str, dry_run: bool) -> None:
    """Run a sweep over the given grid conditions."""
    pending = []
    for L, T, seed in grid:
        status = "done" if is_complete(L, T, seed) else "run"
        if dry_run:
            log.info("[%s] L=%d T=%.2f S=%d", status.upper(), L, T, seed)
        if status == "run":
            pending.append((L, T, seed))

    if dry_run:
        log.info("%d pending, %d already done", len(pending), len(grid) - len(pending))
        return

    if not pending:
        log.info("All %d conditions complete. Nothing to do.", len(grid))
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO_ROOT / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"sweep_{label}_{timestamp}.txt"
    logging.getLogger().addHandler(logging.FileHandler(log_file))

    log.info("%s: %d conditions pending, %d already done",
             label, len(pending), len(grid) - len(pending))
    log.info("Log file: %s", log_file)

    done = 0
    failed = 0
    failures = []

    for i, (L, T, seed) in enumerate(pending, 1):
        header = f"=== [{i}/{len(pending)}] L={L} T={T:.2f} S={seed}"
        log.info("%s — started %s ===", header, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        t0 = time.monotonic()
        try:
            result = run_condition(L, T, seed)
            elapsed = time.monotonic() - t0

            if result.returncode == 0:
                done += 1
                log.info("%s — done in %s (%d done, %d failed, %d remaining) ===",
                         header, format_duration(elapsed), done, failed,
                         len(pending) - done - failed)
            elif result.returncode == 130 or result.returncode == -2:
                # SIGINT — generate.py saved checkpoint, stop sweep
                log.info("%s — interrupted after %s (checkpoint saved). Stopping sweep.",
                         header, format_duration(elapsed))
                break
            else:
                failed += 1
                failures.append((L, T, seed, f"exit code {result.returncode}"))
                log.error("%s — FAILED (exit %d) after %s (%d done, %d failed, %d remaining) ===",
                          header, result.returncode, format_duration(elapsed),
                          done, failed, len(pending) - done - failed)
        except KeyboardInterrupt:
            elapsed = time.monotonic() - t0
            log.info("%s — sweep interrupted after %s. Subprocess checkpoint may be saved.",
                     header, format_duration(elapsed))
            break
        except Exception as e:
            elapsed = time.monotonic() - t0
            failed += 1
            failures.append((L, T, seed, str(e)))
            log.error("%s — EXCEPTION after %s: %s ===", header, format_duration(elapsed), e)

    log.info("=" * 60)
    log.info("%s finished. %d done, %d failed, %d skipped (already done).",
             label, done, failed, len(grid) - len(pending))
    if failures:
        log.error("Failed runs:")
        for L, T, seed, reason in failures:
            log.error("  L=%d T=%.2f S=%d: %s", L, T, seed, reason)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified sweep runner for autoloop experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python sweep.py pilot              # run the pilot preset\n"
               "  python sweep.py --status            # show all data on disk\n"
               "  python sweep.py --status pilot      # show status for pilot preset\n"
               "  python sweep.py --list              # list presets\n"
               "  python sweep.py --L 256 --T 0.6 0.7 --seed 42  # ad-hoc grid\n",
    )
    parser.add_argument("preset", nargs="?", default=None,
                        help="Named preset to run (see --list)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--status", nargs="?", const="__all__", default=None,
                        metavar="PRESET",
                        help="Show grid status (optionally filtered to a preset)")
    parser.add_argument("--list", action="store_true",
                        help="List available presets")
    parser.add_argument("--L", type=int, nargs="+", metavar="N",
                        help="Context lengths for ad-hoc grid")
    parser.add_argument("--T", type=float, nargs="+", metavar="F",
                        help="Temperatures for ad-hoc grid")
    parser.add_argument("--seed", type=int, nargs="+", default=[42],
                        help="Seeds for ad-hoc grid (default: 42)")

    args = parser.parse_args()

    # --list
    if args.list:
        print_presets()
        return

    # --status
    if args.status is not None:
        preset_name = None if args.status == "__all__" else args.status
        if preset_name and preset_name not in PRESETS:
            parser.error(f"Unknown preset '{preset_name}'. Use --list to see available presets.")
        print_status(preset_name)
        return

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    # Determine grid
    if args.preset:
        if args.preset not in PRESETS:
            parser.error(f"Unknown preset '{args.preset}'. Use --list to see available presets.")
        preset = PRESETS[args.preset]
        grid = expand_grid(preset["L"], preset["T"], preset["seeds"])
        label = args.preset
    elif args.L and args.T:
        grid = expand_grid(args.L, args.T, args.seed)
        parts = [f"L{'_'.join(str(l) for l in args.L)}",
                 f"T{'_'.join(f'{t:.2f}' for t in args.T)}"]
        label = "_".join(parts)
    else:
        parser.error("Provide a preset name or --L and --T for an ad-hoc grid. "
                     "Use --list to see presets, --status to see data on disk.")
        return  # unreachable, for type checker

    run_sweep(grid, label, args.dry_run)


if __name__ == "__main__":
    main()

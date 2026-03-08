"""Batch runner for seed replication across existing grid.

Priority order: T=0.50 all L (attractor anomaly verification),
then crossover T=0.60-0.90, then T=1.00/1.50.

Usage:
    python seed_sweep.py
    python seed_sweep.py --dry-run
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "data" / "runs"
MODEL_DIR = REPO_ROOT / "data" / "model" / "SmolLM-135M"

NUM_TOKENS = 100_000
DEVICE = "cuda"

SEEDS = [123, 7]

# Priority order: collapse regime first (anomaly check), then crossover, then rest.
GRID: list[tuple[int, float, int]] = []

# 1. T=0.50 all L — verify L=192 anomaly
for L in [64, 128, 192, 256]:
    for seed in SEEDS:
        GRID.append((L, 0.50, seed))

# 2. Crossover region — verify T-dependence
for T in [0.60, 0.70, 0.80, 0.90]:
    for L in [64, 128, 192]:
        for seed in SEEDS:
            GRID.append((L, T, seed))

# 3. Rest — T=1.00 and T=1.50
for T in [1.00, 1.50]:
    for L in [64, 128, 192, 256]:
        for seed in SEEDS:
            GRID.append((L, T, seed))


def parquet_path(L: int, T: float, seed: int) -> Path:
    return RUNS_DIR / f"L{L:04d}_T{T:.2f}_S{seed}.parquet"


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
        "--output-dir", str(RUNS_DIR),
        "--device", DEVICE,
    ]
    return subprocess.run(cmd)


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run seed replication sweep")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO_ROOT / "data"
    log_dir.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler()]
    if not args.dry_run:
        log_file = log_dir / f"seed_sweep_log_{timestamp}.txt"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    pending = []
    for L, T, seed in GRID:
        status = "done" if is_complete(L, T, seed) else "run"
        if args.dry_run:
            log.info("[%s] L=%d T=%.2f S=%d", status.upper(), L, T, seed)
        if status == "run":
            pending.append((L, T, seed))

    if args.dry_run:
        log.info("%d pending, %d already done", len(pending), len(GRID) - len(pending))
        return

    if not pending:
        log.info("All %d conditions complete. Nothing to do.", len(GRID))
        return

    log.info("Seed sweep: %d conditions pending, %d already done",
             len(pending), len(GRID) - len(pending))
    if not args.dry_run:
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
            else:
                failed += 1
                failures.append((L, T, seed, f"exit code {result.returncode}"))
                log.error("%s — FAILED (exit %d) after %s (%d done, %d failed, %d remaining) ===",
                          header, result.returncode, format_duration(elapsed),
                          done, failed, len(pending) - done - failed)
        except Exception as e:
            elapsed = time.monotonic() - t0
            failed += 1
            failures.append((L, T, seed, str(e)))
            log.error("%s — EXCEPTION after %s: %s ===", header, format_duration(elapsed), e)

    log.info("=" * 60)
    log.info("Seed sweep complete. %d done, %d failed, %d skipped (already done).",
             done, failed, len(GRID) - len(pending))
    if failures:
        log.error("Failed runs:")
        for L, T, seed, reason in failures:
            log.error("  L=%d T=%.2f S=%d: %s", L, T, seed, reason)


if __name__ == "__main__":
    main()

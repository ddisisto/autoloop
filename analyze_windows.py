"""Recompute analysis at standard window sizes for all runs.

Produces cached .analysis.pkl files for the standard W grid.
Existing caches are reused if valid.

Usage:
    python analyze_windows.py
    python analyze_windows.py --runs data/runs/L0064_T1.00_S42.parquet
"""

import argparse
import logging
from pathlib import Path

from analyze import analyze_run, default_window_sizes

log = logging.getLogger(__name__)

STANDARD_WINDOWS = default_window_sizes(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute analysis at standard W grid")
    parser.add_argument(
        "--runs", type=str, nargs="*",
        help="Specific parquet files (default: all in data/runs/)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.runs:
        paths = sorted(Path(p) for p in args.runs)
    else:
        paths = sorted(Path("data/runs").glob("*.parquet"))

    log.info("Analyzing %d runs at W=%s", len(paths), STANDARD_WINDOWS)

    for path in paths:
        log.info("Processing %s", path.stem)
        analyze_run(path, STANDARD_WINDOWS)

    log.info("Done.")


if __name__ == "__main__":
    main()

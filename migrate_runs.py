"""One-shot migration: move files from flat data/runs/ into organized subdirectories.

Usage:
    python migrate_runs.py              # dry-run (default) — shows what would happen
    python migrate_runs.py --execute    # actually move files
"""

import argparse
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from runlib import RUNS_ROOT, SURVEY_DIR, SCHEDULE_DIR, classify_run

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Files/dirs in RUNS_ROOT to skip entirely
SKIP_NAMES: set[str] = {"_old_caches", "index.db"}

# Known sidecar extensions (including .parquet itself)
SIDECAR_EXTS: set[str] = {".parquet", ".meta.json", ".analysis.pkl", ".ckpt", ".decisions.json"}


def extract_stem(path: Path) -> str | None:
    """Extract the logical run stem from a sidecar filename.

    Returns None if the file doesn't match a known sidecar extension.
    """
    name = path.name
    for ext in sorted(SIDECAR_EXTS, key=len, reverse=True):
        if name.endswith(ext):
            return name[: -len(ext)]
    return None


def gather_files(runs_root: Path) -> dict[str, list[Path]]:
    """Group files in runs_root by their run stem.

    Only considers files directly in runs_root (not in subdirectories).
    Skips entries listed in SKIP_NAMES.

    Returns:
        Mapping from stem to list of file paths.
    """
    groups: dict[str, list[Path]] = defaultdict(list)

    for entry in sorted(runs_root.iterdir()):
        if entry.name in SKIP_NAMES:
            logger.info("Skipping: %s", entry.name)
            continue
        if entry.is_dir():
            logger.info("Skipping subdirectory: %s", entry.name)
            continue
        if not entry.is_file():
            continue

        stem = extract_stem(entry)
        if stem is None:
            logger.warning("Unrecognized file (no known extension): %s", entry.name)
            continue

        groups[stem].append(entry)

    return dict(groups)


def plan_moves(groups: dict[str, list[Path]], runs_root: Path) -> list[tuple[Path, Path]]:
    """Determine (source, destination) pairs for all files.

    Uses classify_run() to determine the target subdirectory for each stem.

    Returns:
        List of (src, dst) tuples.

    Raises:
        ValueError: If a stem cannot be classified.
    """
    moves: list[tuple[Path, Path]] = []

    for stem in sorted(groups):
        parquet_name = stem + ".parquet"
        run_type = classify_run(parquet_name)
        target_dir = runs_root / run_type

        for src in groups[stem]:
            dst = target_dir / src.name
            moves.append((src, dst))

    return moves


def execute_moves(moves: list[tuple[Path, Path]]) -> dict[str, int]:
    """Move files and return counts per subdirectory."""
    counts: dict[str, int] = defaultdict(int)

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        subdir = dst.parent.name
        counts[subdir] += 1

    return dict(counts)


def print_plan(moves: list[tuple[Path, Path]], runs_root: Path) -> dict[str, int]:
    """Print the planned moves and return counts per subdirectory."""
    counts: dict[str, int] = defaultdict(int)

    for src, dst in moves:
        rel_src = src.relative_to(runs_root)
        rel_dst = dst.relative_to(runs_root)
        logger.info("  %s -> %s", rel_src, rel_dst)
        subdir = dst.parent.name
        counts[subdir] += 1

    return dict(counts)


def print_summary(counts: dict[str, int], dry_run: bool) -> None:
    """Print a summary of file counts per subdirectory."""
    action = "Would move" if dry_run else "Moved"
    total = sum(counts.values())
    logger.info("")
    logger.info("Summary:")
    for subdir in sorted(counts):
        logger.info("  %s/: %d files", subdir, counts[subdir])
    logger.info("  Total: %d files %s", total, action.lower())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate data/runs/ from flat layout to organized subdirectories."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files (default is dry-run).",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=RUNS_ROOT,
        help="Root runs directory (default: data/runs).",
    )
    args = parser.parse_args()

    runs_root: Path = args.runs_root.resolve()
    dry_run: bool = not args.execute

    if not runs_root.is_dir():
        logger.error("Runs root does not exist: %s", runs_root)
        sys.exit(1)

    if dry_run:
        logger.info("DRY RUN — no files will be moved. Use --execute to apply.\n")
    else:
        logger.info("EXECUTING — files will be moved.\n")

    # Gather and classify
    groups = gather_files(runs_root)
    if not groups:
        logger.info("No files to migrate.")
        return

    logger.info("Found %d run stems (%d files)\n", len(groups), sum(len(v) for v in groups.values()))

    # Plan moves
    try:
        moves = plan_moves(groups, runs_root)
    except ValueError as e:
        logger.error("Classification failed: %s", e)
        sys.exit(1)

    # Execute or print plan
    if dry_run:
        counts = print_plan(moves, runs_root)
    else:
        counts = execute_moves(moves)

    # Create empty dirs for future use
    future_dirs = [SURVEY_DIR, SCHEDULE_DIR]
    for d in future_dirs:
        resolved = runs_root / d.name
        if not resolved.is_dir():
            if dry_run:
                logger.info("\nWould create empty directory: %s/", d.name)
            else:
                resolved.mkdir(parents=True, exist_ok=True)
                logger.info("\nCreated empty directory: %s/", d.name)

    print_summary(counts, dry_run)

    if not dry_run:
        logger.info("\nReminder: run `python runindex.py build` to rebuild the index.")


if __name__ == "__main__":
    main()

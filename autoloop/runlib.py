"""Shared module for run discovery and classification.

Provides path constants, filename classification by experiment type,
and directory-based run discovery for the organized data/runs/ layout.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Directory constants ──────────────────────────────────────────────

RUNS_ROOT = Path("data/runs")

SWEEP_DIR = RUNS_ROOT / "sweep"
CONTROLLER_DIR = RUNS_ROOT / "controller"
ANNEAL_DIR = RUNS_ROOT / "anneal"
PROBE_DIR = RUNS_ROOT / "probe"
SURVEY_DIR = RUNS_ROOT / "survey"
SCHEDULE_DIR = RUNS_ROOT / "schedule"

_RUN_TYPES: dict[str, Path] = {
    "sweep": SWEEP_DIR,
    "controller": CONTROLLER_DIR,
    "anneal": ANNEAL_DIR,
    "probe": PROBE_DIR,
    "survey": SURVEY_DIR,
    "schedule": SCHEDULE_DIR,
}

# ── Classification patterns ──────────────────────────────────────────
# Order matters: more specific patterns first.

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^ctrld?_"), "controller"),
    (re.compile(r"^anneal_"), "anneal"),
    (re.compile(r"^probe_"), "probe"),
    (re.compile(r"^survey_"), "survey"),
    (re.compile(r"^sched_"), "schedule"),
    (re.compile(r"^L\d{4}_T[\d.]+_S\d+"), "sweep"),
]


def classify_run(filename: str) -> str:
    """Classify a parquet filename into its run type.

    Args:
        filename: Parquet filename (basename only, not a full path).

    Returns:
        Run type string ("sweep", "controller", "anneal", "probe",
        "survey", "schedule").

    Raises:
        ValueError: If the filename does not match any known pattern.
    """
    stem = Path(filename).stem
    for pattern, run_type in _PATTERNS:
        if pattern.search(stem):
            return run_type
    raise ValueError(f"Unrecognized run filename: {filename!r}")


def run_subdir(run_type: str) -> Path:
    """Return the subdirectory path for a given run type.

    Args:
        run_type: One of "sweep", "controller", "anneal", "probe",
                  "survey", "schedule".

    Returns:
        Path to the run type's subdirectory (relative to repo root).

    Raises:
        ValueError: If run_type is not recognized.
    """
    if run_type not in _RUN_TYPES:
        raise ValueError(
            f"Unknown run type: {run_type!r}. "
            f"Valid types: {sorted(_RUN_TYPES)}"
        )
    return _RUN_TYPES[run_type]


def discover_runs(
    root: Path | None = None, run_type: str | None = None
) -> list[Path]:
    """Discover parquet run files under the organized directory structure.

    Args:
        root: Base runs directory. Defaults to RUNS_ROOT.
        run_type: If given, only scan that run type's subdirectory.
                  Must be a valid run type name.

    Returns:
        Sorted list of absolute paths to .parquet files.

    Raises:
        ValueError: If run_type is not recognized.
    """
    root = root or RUNS_ROOT
    if run_type is not None:
        if run_type not in _RUN_TYPES:
            raise ValueError(
                f"Unknown run type: {run_type!r}. "
                f"Valid types: {sorted(_RUN_TYPES)}"
            )
        dirs = [root / run_type]
    else:
        dirs = [root / name for name in sorted(_RUN_TYPES)]

    results: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            logger.debug("Skipping missing directory: %s", d)
            continue
        results.extend(d.glob("*.parquet"))

    return sorted(p.resolve() for p in results)

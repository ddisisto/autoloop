"""Run resolution: ID lookup and filter-based queries against the SQLite index."""

import argparse
import logging
import sys
from pathlib import Path

from .runindex import DB_PATH, RUNS_ROOT

log = logging.getLogger(__name__)


def resolve_runs(
    run_ids: list[str] | None = None,
    run_type: str | None = None,
    L: int | None = None,
    T: float | None = None,
    seed: int | None = None,
    regime: str | None = None,
    db_path: Path = DB_PATH,
) -> list[Path]:
    """Resolve run identifiers and/or filter flags to parquet paths.

    If run_ids are given, they take precedence over filters. Each run_id
    is a parquet stem (e.g. "L0064_T0.50_S42"). Filters are applied only
    when no run_ids are provided.

    Args:
        run_ids: Parquet stems to look up directly.
        run_type: Filter by run type (sweep, controller, etc.).
        L: Filter by context length.
        T: Filter by temperature.
        seed: Filter by random seed.
        regime: Filter by regime classification.
        db_path: Path to the SQLite index database.

    Returns:
        List of resolved parquet Paths (absolute).

    Raises:
        SystemExit: If the index doesn't exist or a run_id is not found.
    """
    if not db_path.exists():
        log.error("No index at %s — run 'loop index build' first.", db_path)
        sys.exit(1)

    from .runindex import create_db, query_runs

    conn = create_db(db_path)

    if run_ids:
        paths: list[Path] = []
        for rid in run_ids:
            rows = conn.execute(
                "SELECT parquet_path FROM runs WHERE run_id = ?", (rid,)
            ).fetchall()
            if not rows:
                conn.close()
                log.error("Run '%s' not found in index.", rid)
                sys.exit(1)
            paths.append((RUNS_ROOT / rows[0]["parquet_path"]).resolve())
        conn.close()
        return paths

    # Filter mode
    filters: dict[str, float | int | str] = {}
    if L is not None:
        filters["L"] = L
    if T is not None:
        filters["T"] = T
    if seed is not None:
        filters["seed"] = seed
    if regime is not None:
        filters["regime"] = regime

    runs = query_runs(conn, run_type=run_type, **filters)
    conn.close()

    if not runs:
        log.error("No runs match the given filters.")
        sys.exit(1)

    return [(RUNS_ROOT / r["parquet_path"]).resolve() for r in runs]


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    """Add common run-filter flags to a subparser."""
    parser.add_argument("run_ids", nargs="*", default=None,
                        help="Run IDs (parquet stems, e.g. L0064_T0.50_S42)")
    parser.add_argument("--type", dest="run_type",
                        help="Filter by run type (sweep, controller, etc.)")
    parser.add_argument("--L", type=int, dest="filter_L",
                        help="Filter by context length")
    parser.add_argument("--T", type=float, dest="filter_T",
                        help="Filter by temperature")
    parser.add_argument("--seed", type=int, dest="filter_seed",
                        help="Filter by seed")
    parser.add_argument("--regime", dest="filter_regime",
                        help="Filter by regime classification")


def resolve_from_args(args: argparse.Namespace) -> list[Path]:
    """Call resolve_runs using parsed argparse namespace."""
    ids = args.run_ids if args.run_ids else None
    return resolve_runs(
        run_ids=ids,
        run_type=getattr(args, "run_type", None),
        L=getattr(args, "filter_L", None),
        T=getattr(args, "filter_T", None),
        seed=getattr(args, "filter_seed", None),
        regime=getattr(args, "filter_regime", None),
    )


def auto_index_run(parquet_path: Path) -> None:
    """Index a single run into the database after completion."""
    from .runindex import create_db, index_run
    conn = create_db(DB_PATH)
    index_run(conn, parquet_path)
    conn.commit()
    conn.close()
    log.info("Indexed %s", parquet_path.stem)

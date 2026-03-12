"""SQLite index for run metadata with CLI for building and querying.

Indexes all parquet runs under data/runs/ into a single SQLite database,
combining metadata from .meta.json sidecars and .analysis.pkl caches.

Basin data is ingested from two sources:
- data/basins/centroids.json  → basin_types table
- data/runs/survey/*.basins.pkl → basin_captures table

Usage:
    python runindex.py build                           # full reindex
    python runindex.py query                           # list all runs
    python runindex.py query --type sweep --T 0.50     # filtered
    python runindex.py query --type controller         # just controller runs
"""

import argparse
import json
import logging
import pickle
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .runlib import RUNS_ROOT, SURVEY_DIR, classify_run
from .schema import (
    BASIN_CAPTURES_COLUMNS, BASIN_TYPES_COLUMNS, init_db,
)

logger = logging.getLogger(__name__)

DB_PATH = RUNS_ROOT / "index.db"
BASINS_DIR = Path("data/basins")

# ── Filename parsing ────────────────────────────────────────────────

_SWEEP_RE = re.compile(r"^L(\d{4})_T([\d.]+)_S(\d+)$")
_CTRL_RE = re.compile(r"^ctrld?_S(\d+)_(\d+)_([\d.]+)$")
_ANNEAL_RE = re.compile(r"^anneal_L(\d+)_\w+_S(\d+)$")
_PROBE_RE = re.compile(r"^probe_L(\d+)_T(\d+)$")
_SCHED_RE = re.compile(r"^sched_S(\d+)_")


def _parse_filename(stem: str) -> dict:
    """Extract L, T, seed from filename where possible."""
    result: dict = {}
    m = _SWEEP_RE.match(stem)
    if m:
        result["L"] = int(m.group(1))
        result["T"] = float(m.group(2))
        result["seed"] = int(m.group(3))
        return result
    m = _CTRL_RE.match(stem)
    if m:
        result["seed"] = int(m.group(1))
        result["L"] = int(m.group(2))
        result["T"] = float(m.group(3))
        return result
    m = _ANNEAL_RE.match(stem)
    if m:
        result["L"] = int(m.group(1))
        result["seed"] = int(m.group(2))
        return result
    m = _PROBE_RE.match(stem)
    if m:
        result["L"] = int(m.group(1))
        # Probe T is encoded as integer (e.g., T060 = 0.60)
        result["T"] = int(m.group(2)) / 100.0
        return result
    m = _SCHED_RE.match(stem)
    if m:
        result["seed"] = int(m.group(1))
        return result
    return result


# ── Core functions ──────────────────────────────────────────────────

def create_db(db_path: Path) -> sqlite3.Connection:
    """Create tables if not exist, return connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    init_db(conn)
    return conn


def _load_meta(parquet_path: Path) -> dict | None:
    """Load .meta.json sidecar for a parquet file."""
    meta_path = parquet_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)


def _load_analysis(parquet_path: Path) -> dict | None:
    """Load .analysis.pkl cache for a parquet file."""
    pkl_path = parquet_path.with_suffix(".analysis.pkl")
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _extract_analysis_metrics(cache: dict) -> dict:
    """Extract quick metrics from analysis cache."""
    from .analyze.summary import comp_stats

    metrics: dict = {}

    summary = cache.get("summary", {})
    if summary:
        metrics["entropy_mean"] = summary.get("entropy_mean")
        metrics["entropy_std"] = summary.get("entropy_std")
        metrics["eos_rate"] = summary.get("eos_rate")

    # comp_W64 mean
    cs = comp_stats(cache, 64)
    if not np.isnan(cs["mean"]):
        metrics["comp_W64_mean"] = cs["mean"]

    return metrics


def _detect_varying_params(meta: dict) -> dict:
    """Detect if L or T vary, and their ranges, from meta.json."""
    result: dict = {"L_varies": False, "T_varies": False}

    # Controller runs: check start vs final
    if meta.get("controller"):
        start_L = meta.get("start_L")
        final_L = meta.get("final_L")
        start_T = meta.get("start_T")
        final_T = meta.get("final_T")
        if start_L is not None and final_L is not None and start_L != final_L:
            result["L_varies"] = True
            result["L_min"] = min(start_L, final_L)
            result["L_max"] = max(start_L, final_L)
        if start_T is not None and final_T is not None and abs(start_T - final_T) > 0.001:
            result["T_varies"] = True
            result["T_min"] = min(start_T, final_T)
            result["T_max"] = max(start_T, final_T)
        return result

    # Schedule runs: check if segments have different L or T
    schedule = meta.get("schedule")
    if schedule and len(schedule) > 1:
        ls = {seg.get("L") for seg in schedule if seg.get("L") is not None}
        ts = {seg.get("T") for seg in schedule if seg.get("T") is not None}
        if len(ls) > 1:
            result["L_varies"] = True
            result["L_min"] = min(ls)
            result["L_max"] = max(ls)
        if len(ts) > 1:
            result["T_varies"] = True
            result["T_min"] = min(ts)
            result["T_max"] = max(ts)

    return result


def index_run(conn: sqlite3.Connection, parquet_path: Path) -> None:
    """Read .meta.json sidecar + .analysis.pkl, upsert into runs table.

    Skips re-indexing if parquet_mtime hasn't changed.
    """
    parquet_path = parquet_path.resolve()
    stem = parquet_path.stem
    pq_mtime = parquet_path.stat().st_mtime

    # Staleness check
    existing = conn.execute(
        "SELECT parquet_mtime FROM runs WHERE run_id = ?", (stem,)
    ).fetchone()
    if existing and existing["parquet_mtime"] == pq_mtime:
        logger.debug("Skipping %s (up to date)", stem)
        return

    # Classify run type
    try:
        run_type = classify_run(parquet_path.name)
    except ValueError:
        logger.warning("Cannot classify %s, skipping", stem)
        return

    # Parse filename for L/T/seed
    parsed = _parse_filename(stem)

    # Load meta.json
    meta = _load_meta(parquet_path) or {}

    # Load analysis.pkl
    analysis = _load_analysis(parquet_path)
    analysis_metrics = _extract_analysis_metrics(analysis) if analysis else {}

    # Determine varying params
    varying = _detect_varying_params(meta) if meta else {}

    # Build L, T, seed — prefer meta.json, fall back to filename parsing
    L = meta.get("context_length") or meta.get("start_L") or parsed.get("L")
    T = meta.get("temperature") or meta.get("start_T") or parsed.get("T")
    seed = meta.get("seed") or parsed.get("seed")

    # Schedule spec as JSON string
    schedule = meta.get("schedule")
    schedule_spec = json.dumps(schedule) if schedule else None

    # Relative path from RUNS_ROOT
    try:
        rel_path = str(parquet_path.relative_to(RUNS_ROOT.resolve()))
    except ValueError:
        rel_path = str(parquet_path)

    row = {
        "run_id": stem,
        "run_type": run_type,
        "parquet_path": rel_path,
        "L": L,
        "T": T,
        "seed": seed,
        "L_varies": varying.get("L_varies", False),
        "T_varies": varying.get("T_varies", False),
        "L_min": varying.get("L_min"),
        "L_max": varying.get("L_max"),
        "T_min": varying.get("T_min"),
        "T_max": varying.get("T_max"),
        "total_tokens": meta.get("num_tokens") or meta.get("total_steps"),
        "prefill_text": meta.get("prefill_text"),
        "model_dir": meta.get("model_dir"),
        "elapsed_seconds": meta.get("elapsed_seconds"),
        "tokens_per_sec": meta.get("tokens_per_second"),
        "controller": meta.get("controller", False),
        "beta_target": meta.get("beta_target"),
        "drift": meta.get("drift"),
        "n_rollbacks": meta.get("n_rollbacks"),
        "schedule_spec": schedule_spec,
        "parquet_mtime": pq_mtime,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "entropy_mean": analysis_metrics.get("entropy_mean"),
        "entropy_std": analysis_metrics.get("entropy_std"),
        "heaps_beta": analysis_metrics.get("heaps_beta"),
        "comp_W64_mean": analysis_metrics.get("comp_W64_mean"),
        "eos_rate": analysis_metrics.get("eos_rate"),
        "regime": None,  # populated by precollapse analysis separately
    }

    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    updates = ", ".join(f"{k}=excluded.{k}" for k in row if k != "run_id")
    sql = (
        f"INSERT INTO runs ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT(run_id) DO UPDATE SET {updates}"
    )
    conn.execute(sql, list(row.values()))
    logger.info("Indexed %s (%s)", stem, run_type)


# ── Basin ingestion ────────────────────────────────────────────────

# Fields from .basins.pkl that map to basin_captures columns.
# 'embedding' key is stripped (stays in pkl only).
_CAPTURE_DB_FIELDS: set[str] = {
    col[0] for col in BASIN_CAPTURES_COLUMNS
}

# Fields from centroids.json that map to basin_types columns.
_TYPE_DB_FIELDS: set[str] = {
    col[0] for col in BASIN_TYPES_COLUMNS
}


def _upsert_sql(table: str, row: dict, pk: str) -> tuple[str, list]:
    """Build an INSERT ... ON CONFLICT DO UPDATE statement."""
    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    updates = ", ".join(
        f"{k}=excluded.{k}" for k in row if k != pk
    )
    sql = (
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT({pk}) DO UPDATE SET {updates}"
    )
    return sql, list(row.values())


def index_basin_types(conn: sqlite3.Connection, basins_dir: Path) -> int:
    """Ingest basin types from centroids.json into basin_types table.

    centroids.json: list of dicts, one per type. Each dict's keys are a
    subset of BASIN_TYPES_COLUMNS. Row order aligns with centroids.npy.

    Returns number of types upserted.
    """
    meta_path = basins_dir / "centroids.json"
    if not meta_path.exists():
        return 0

    with open(meta_path) as f:
        types = json.load(f)

    count = 0
    for t in types:
        row = {k: v for k, v in t.items() if k in _TYPE_DB_FIELDS}
        if "type_id" not in row:
            logger.warning("Basin type missing type_id, skipping: %s", t)
            continue
        sql, params = _upsert_sql("basin_types", row, "type_id")
        conn.execute(sql, params)
        count += 1

    conn.commit()
    return count


def index_basin_captures(
    conn: sqlite3.Connection, survey_dir: Path,
) -> int:
    """Ingest basin captures from .basins.pkl sidecars.

    Each .basins.pkl is a list of capture dicts. The 'embedding' key
    (numpy array) is stripped — it stays in the pkl file only.
    Scalar fields matching BASIN_CAPTURES_COLUMNS are upserted.

    Returns total number of captures upserted.
    """
    if not survey_dir.is_dir():
        return 0

    pkl_files = sorted(survey_dir.glob("*.basins.pkl"))
    if not pkl_files:
        return 0

    count = 0
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            captures = pickle.load(f)

        for cap in captures:
            row = {
                k: v for k, v in cap.items()
                if k in _CAPTURE_DB_FIELDS and k != "embedding"
            }
            if "capture_id" not in row:
                logger.warning(
                    "Capture missing capture_id in %s, skipping", pkl_path.name
                )
                continue
            # Verify run_id exists in runs table
            run_id = row.get("run_id")
            if run_id:
                exists = conn.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if not exists:
                    logger.warning(
                        "Capture %s references unknown run %s, skipping",
                        row["capture_id"], run_id,
                    )
                    continue
            sql, params = _upsert_sql("basin_captures", row, "capture_id")
            conn.execute(sql, params)
            count += 1

        logger.info("Ingested %d captures from %s", len(captures), pkl_path.name)

    conn.commit()
    return count


def reindex_all(conn: sqlite3.Connection, root: Path) -> None:
    """Index all parquet runs, delete rows for files that no longer exist."""
    from .runlib import discover_runs

    root = root.resolve()
    all_parquets = discover_runs(root)

    logger.info("Found %d parquet files under %s", len(all_parquets), root)

    for pq in all_parquets:
        index_run(conn, pq)

    conn.commit()

    # Delete rows for parquets that no longer exist
    indexed_ids = {row["run_id"] for row in conn.execute("SELECT run_id FROM runs").fetchall()}
    disk_ids = {p.stem for p in all_parquets}
    stale = indexed_ids - disk_ids
    if stale:
        logger.info("Removing %d stale entries: %s", len(stale), sorted(stale))
        for run_id in stale:
            conn.execute("DELETE FROM basin_captures WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        conn.commit()

    # Ingest basin data
    n_types = index_basin_types(conn, BASINS_DIR)
    n_captures = index_basin_captures(conn, root / "survey")
    if n_types or n_captures:
        logger.info("Basin data: %d types, %d captures", n_types, n_captures)


def query_runs(
    conn: sqlite3.Connection,
    run_type: str | None = None,
    **filters: float | int | str,
) -> list[dict]:
    """Query runs with optional filtering.

    Supported filters: L, T, seed, regime (matched as equality).
    """
    clauses: list[str] = []
    params: list = []

    if run_type is not None:
        clauses.append("run_type = ?")
        params.append(run_type)

    for key, val in filters.items():
        if key in ("L", "T", "seed", "regime") and val is not None:
            clauses.append(f"{key} = ?")
            params.append(val)

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"SELECT * FROM runs{where} ORDER BY run_type, L, T, seed"

    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


# ── CLI ─────────────────────────────────────────────────────────────

def _format_table(runs: list[dict]) -> str:
    """Format query results as an aligned table."""
    if not runs:
        return "No runs found."

    # Column definitions: (header, key, width, fmt)
    columns = [
        ("run_id", "run_id", 36, "s"),
        ("type", "run_type", 12, "s"),
        ("L", "L", 5, "d"),
        ("T", "T", 5, ".2f"),
        ("seed", "seed", 5, "d"),
        ("tokens", "total_tokens", 8, "d"),
        ("H_mean", "entropy_mean", 7, ".3f"),
        ("H_std", "entropy_std", 6, ".3f"),
        ("comp64", "comp_W64_mean", 7, ".4f"),
        ("eos", "eos_rate", 7, ".4f"),
    ]

    header = "  ".join(f"{name:<{w}s}" for name, _, w, _ in columns)
    sep = "  ".join("-" * w for _, _, w, _ in columns)

    lines = [header, sep]
    for run in runs:
        parts = []
        for _, key, w, fmt in columns:
            val = run.get(key)
            if val is None:
                parts.append(f"{'':>{w}s}")
            elif fmt == "s":
                parts.append(f"{str(val):<{w}s}")
            elif fmt == "d":
                parts.append(f"{int(val):>{w}d}")
            else:
                parts.append(f"{float(val):>{w}{fmt}}")
        lines.append("  ".join(parts))

    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="SQLite index for autoloop run metadata."
    )
    sub = parser.add_subparsers(dest="command")

    # build
    build_p = sub.add_parser("build", help="Full reindex of all runs")
    build_p.add_argument(
        "--root", type=Path, default=RUNS_ROOT,
        help="Runs directory (default: data/runs)",
    )
    build_p.add_argument(
        "--db", type=Path, default=DB_PATH,
        help="Database path (default: data/runs/index.db)",
    )

    # query
    query_p = sub.add_parser("query", help="Query indexed runs")
    query_p.add_argument("--type", dest="run_type", help="Filter by run type")
    query_p.add_argument("--L", type=int, help="Filter by context length")
    query_p.add_argument("--T", type=float, help="Filter by temperature")
    query_p.add_argument("--seed", type=int, help="Filter by seed")
    query_p.add_argument("--regime", help="Filter by regime")
    query_p.add_argument(
        "--db", type=Path, default=DB_PATH,
        help="Database path (default: data/runs/index.db)",
    )
    query_p.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Output as JSON instead of table",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "build":
        conn = create_db(args.db)
        reindex_all(conn, args.root)
        total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        logger.info("Index complete: %d runs in %s", total, args.db)
        conn.close()

    elif args.command == "query":
        if not args.db.exists():
            logger.error("No index at %s — run 'python runindex.py build' first", args.db)
            sys.exit(1)
        conn = create_db(args.db)
        filters = {}
        if args.L is not None:
            filters["L"] = args.L
        if args.T is not None:
            filters["T"] = args.T
        if args.seed is not None:
            filters["seed"] = args.seed
        if args.regime is not None:
            filters["regime"] = args.regime
        runs = query_runs(conn, run_type=args.run_type, **filters)
        if args.as_json:
            # Clean up non-serializable types
            for r in runs:
                for k, v in r.items():
                    if isinstance(v, (float,)) and np.isnan(v):
                        r[k] = None
            print(json.dumps(runs, indent=2))
        else:
            print(_format_table(runs))
        conn.close()


if __name__ == "__main__":
    main()

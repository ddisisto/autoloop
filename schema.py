"""Schema definitions for the autoloop run index database.

Column definitions as data structures (inspectable without SQL parsing),
plus SQL generation and DB initialization.

Storage tiers for basin data:
- SQLite (basin_types + basin_captures): scalar fields, queryable
- .basins.pkl per survey run: full capture data including 576-dim embeddings
- data/basins/centroids.npy: (N_types, 576) float32, loaded at survey startup
- data/basins/centroids.json: type metadata aligned with centroids.npy rows
"""

import logging
import sqlite3

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# ── Column definitions ─────────────────────────────────────────────
# Each column: (name, sql_type, description)

RUNS_COLUMNS: list[tuple[str, str, str]] = [
    ("run_id", "TEXT PRIMARY KEY", "parquet stem"),
    ("run_type", "TEXT NOT NULL", "sweep/controller/anneal/probe/survey/schedule"),
    ("parquet_path", "TEXT NOT NULL", "path relative to RUNS_ROOT"),
    ("L", "INTEGER", "context length (initial for varying runs)"),
    ("T", "REAL", "temperature (initial for varying runs)"),
    ("seed", "INTEGER", "random seed"),
    ("L_varies", "BOOLEAN DEFAULT 0", "whether L changes during run"),
    ("T_varies", "BOOLEAN DEFAULT 0", "whether T changes during run"),
    ("L_min", "INTEGER", "minimum L if varying"),
    ("L_max", "INTEGER", "maximum L if varying"),
    ("T_min", "REAL", "minimum T if varying"),
    ("T_max", "REAL", "maximum T if varying"),
    ("total_tokens", "INTEGER", "total generated tokens"),
    ("prefill_text", "TEXT", "initial context text"),
    ("model_dir", "TEXT", "model directory path"),
    ("elapsed_seconds", "REAL", "wall-clock run time"),
    ("tokens_per_sec", "REAL", "generation throughput"),
    ("controller", "BOOLEAN", "whether run used controller"),
    ("beta_target", "REAL", "controller beta target"),
    ("drift", "BOOLEAN", "whether controller used drift mode"),
    ("n_rollbacks", "INTEGER", "number of controller rollbacks"),
    ("schedule_spec", "TEXT", "schedule JSON for scheduled runs"),
    ("parquet_mtime", "REAL NOT NULL", "parquet file modification time"),
    ("indexed_at", "TEXT NOT NULL", "UTC ISO timestamp of indexing"),
    ("entropy_mean", "REAL", "mean surprisal over run"),
    ("entropy_std", "REAL", "surprisal standard deviation"),
    ("heaps_beta", "REAL", "Heaps law exponent"),
    ("comp_W64_mean", "REAL", "mean compressibility at W=64"),
    ("eos_rate", "REAL", "end-of-sequence token rate"),
    ("regime", "TEXT", "regime classification from precollapse"),
]

# Basin types: the taxonomy. One row per distinct attractor type.
# Centroid embedding stored in data/basins/centroids.npy (row-aligned by type_id).
BASIN_TYPES_COLUMNS: list[tuple[str, str, str]] = [
    ("type_id", "INTEGER PRIMARY KEY", "auto-increment, aligns with centroids.npy row"),
    ("hit_count", "INTEGER NOT NULL DEFAULT 1", "number of captures assigned to this type"),
    ("first_seen_run", "TEXT NOT NULL", "run_id of first capture"),
    ("first_seen_step", "INTEGER NOT NULL", "step of first capture"),
    ("last_seen_run", "TEXT", "run_id of most recent capture"),
    ("last_seen_step", "INTEGER", "step of most recent capture"),
    ("min_L", "INTEGER", "smallest L where this type appears"),
    ("max_L", "INTEGER", "largest L where this type appears"),
    ("comp_W16", "REAL", "centroid compressibility at W=16"),
    ("comp_W32", "REAL", "centroid compressibility at W=32"),
    ("comp_W64", "REAL", "centroid compressibility at W=64"),
    ("comp_W128", "REAL", "centroid compressibility at W=128"),
    ("comp_W256", "REAL", "centroid compressibility at W=256"),
    ("W_star", "INTEGER", "characteristic window size (best compression)"),
    ("entropy_mean", "REAL", "centroid entropy mean"),
    ("entropy_std", "REAL", "centroid entropy std"),
    ("entropy_floor", "REAL", "centroid entropy floor"),
    ("heaps_beta", "REAL", "centroid Heaps law exponent"),
    ("representative_text", "TEXT", "attractor text from most typical capture"),
    ("label", "TEXT", "optional human-readable label"),
]

# Basin captures: every observation of the system in a basin.
# Each capture is linked to a basin type. Multiple captures of the
# same type at different (L, T) are valuable — depth measurements,
# transition edges, and operating-point coverage.
BASIN_CAPTURES_COLUMNS: list[tuple[str, str, str]] = [
    ("capture_id", "TEXT PRIMARY KEY", "run_id:capture_step"),
    ("run_id", "TEXT NOT NULL REFERENCES runs(run_id)", "parent survey run"),
    ("type_id", "INTEGER REFERENCES basin_types(type_id)", "assigned basin type"),
    ("capture_step", "INTEGER NOT NULL", "step when capture detected"),
    ("record_step", "INTEGER", "step of basin record point"),
    ("L", "INTEGER NOT NULL", "context length at capture"),
    ("T_survey", "REAL NOT NULL", "survey temperature at capture"),
    ("comp_W16", "REAL", "compressibility at W=16"),
    ("comp_W32", "REAL", "compressibility at W=32"),
    ("comp_W64", "REAL", "compressibility at W=64"),
    ("comp_W128", "REAL", "compressibility at W=128"),
    ("comp_W256", "REAL", "compressibility at W=256"),
    ("W_star", "INTEGER", "characteristic window size"),
    ("entropy_mean", "REAL", "mean surprisal in basin"),
    ("entropy_std", "REAL", "surprisal std in basin"),
    ("entropy_floor", "REAL", "minimum entropy in basin"),
    ("heaps_beta", "REAL", "Heaps law exponent in basin"),
    ("decorrelation_lag", "INTEGER", "autocorrelation decorrelation lag"),
    ("eos_rate", "REAL", "EOS rate in basin"),
    ("depth_score", "REAL", "basin depth metric from fork probing"),
    ("escape_T", "REAL", "temperature that triggered escape"),
    ("escape_steps", "INTEGER", "steps from T_heat to escape"),
    ("attractor_text", "TEXT", "representative attractor content"),
    ("attractor_period", "INTEGER", "attractor repetition period"),
    ("novelty_distance", "REAL", "cosine distance to nearest type at capture time"),
    ("prev_capture_id", "TEXT", "previous capture in this run's trajectory"),
    ("next_capture_id", "TEXT", "next capture in this run's trajectory"),
]

INDEX_DEFINITIONS: list[tuple[str, str, list[str]]] = [
    # runs
    ("idx_runs_type", "runs", ["run_type"]),
    ("idx_runs_L_T", "runs", ["L", "T"]),
    ("idx_runs_seed", "runs", ["seed"]),
    # basin_types
    ("idx_btypes_L", "basin_types", ["min_L"]),
    ("idx_btypes_hits", "basin_types", ["hit_count"]),
    # basin_captures
    ("idx_bcap_run", "basin_captures", ["run_id"]),
    ("idx_bcap_type", "basin_captures", ["type_id"]),
    ("idx_bcap_L_T", "basin_captures", ["L", "T_survey"]),
]

# ── Derived column lists ───────────────────────────────────────────

RUNS_PARAM_COLUMNS: list[str] = [
    "L", "T", "seed", "L_varies", "T_varies",
    "L_min", "L_max", "T_min", "T_max",
]

RUNS_METRIC_COLUMNS: list[str] = [
    "entropy_mean", "entropy_std", "heaps_beta", "comp_W64_mean", "eos_rate",
]


# ── SQL generation ─────────────────────────────────────────────────

def _create_table_sql(table_name: str, columns: list[tuple[str, str, str]]) -> str:
    """Generate a CREATE TABLE IF NOT EXISTS statement from column definitions."""
    col_defs = ",\n    ".join(f"{name:<20s} {sql_type}" for name, sql_type, _ in columns)
    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {col_defs}\n);"


def _create_index_sql(index_name: str, table_name: str, columns: list[str]) -> str:
    """Generate a CREATE INDEX IF NOT EXISTS statement."""
    cols = ", ".join(columns)
    return f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({cols});"


def create_tables_sql() -> list[str]:
    """Return all CREATE TABLE and CREATE INDEX statements."""
    stmts: list[str] = [
        _create_table_sql("runs", RUNS_COLUMNS),
        _create_table_sql("basin_types", BASIN_TYPES_COLUMNS),
        _create_table_sql("basin_captures", BASIN_CAPTURES_COLUMNS),
    ]
    for index_name, table_name, columns in INDEX_DEFINITIONS:
        stmts.append(_create_index_sql(index_name, table_name, columns))
    return stmts


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Migrate from v1 (single basins table) to v2 (types + captures).

    The v1 basins table was always empty, so migration is just drop + create.
    """
    row = conn.execute(
        "SELECT COUNT(*) FROM basins"
    ).fetchone()
    if row[0] > 0:
        raise RuntimeError(
            "Cannot auto-migrate: v1 basins table has data. "
            "Back up the database and migrate manually."
        )
    conn.execute("DROP TABLE IF EXISTS basins")
    conn.execute("DROP INDEX IF EXISTS idx_basins_run")
    conn.execute("DROP INDEX IF EXISTS idx_basins_fp")
    log.info("Migrated schema v1 → v2: dropped empty basins table, "
             "creating basin_types + basin_captures")


def init_db(conn: sqlite3.Connection) -> None:
    """Execute schema creation, set user_version, check version mismatch."""
    current_version = conn.execute("PRAGMA user_version").fetchone()[0]

    if current_version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {current_version} is newer than "
            f"code schema version {SCHEMA_VERSION}. Update your code."
        )

    if current_version == 1:
        _migrate_v1_to_v2(conn)

    for stmt in create_tables_sql():
        conn.executescript(stmt)

    if current_version < SCHEMA_VERSION:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    conn.commit()

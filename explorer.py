"""Interactive explorer backend for autoloop run data.

FastAPI server that serves run metadata, metric registries, and data slices.
Frontend discovers everything from the API and builds UI dynamically.

Run with: uvicorn explorer:app --reload --port 8000
"""

import fnmatch
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from analyze import analyze_run, default_window_sizes

log = logging.getLogger(__name__)

RUNS_DIR = Path("data/runs")
DEFAULT_DOWNSAMPLE = 500

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

# Step-level metric definitions (from parquet columns)
STEP_METRIC_DEFS: dict[str, dict] = {
    "entropy": {
        "name": "Softmax Entropy",
        "description": "Shannon entropy of the softmax distribution (nats)",
        "column": "entropy",
    },
    "log_prob": {
        "name": "Log Probability",
        "description": "Log probability of the chosen token",
        "column": "log_prob",
    },
    "eos": {
        "name": "EOS Flag",
        "description": "Whether the generated token was EOS (0/1)",
        "column": "eos",
    },
}

# Block-level metric prefix (compressibility at various W)
BLOCK_METRIC_PREFIX = "compressibility_W"


# ---------------------------------------------------------------------------
# In-memory caches
# ---------------------------------------------------------------------------

class RunInfo:
    """Metadata for a single run, extracted from filename and JSON sidecar."""

    def __init__(self, parquet_path: Path) -> None:
        self.path = parquet_path
        self.id = parquet_path.stem  # e.g. L0064_T0.50_S42

        m = re.match(r"L(\d+)_T([\d.]+)_S(\d+)", self.id)
        if not m:
            raise ValueError(f"Cannot parse run name: {self.id}")
        self.L = int(m.group(1))
        self.T = float(m.group(2))
        self.seed = int(m.group(3))

        # Read JSON sidecar if present
        sidecar = parquet_path.with_suffix(".meta.json")
        self.meta: dict = {}
        if sidecar.exists():
            with open(sidecar) as f:
                self.meta = json.load(f)

        self.tokens = self.meta.get("num_tokens", 0)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "L": self.L,
            "T": self.T,
            "seed": self.seed,
            "tokens": self.tokens,
            "path": self.path.name,
        }


class RunCache:
    """Lazily-loaded and cached run data (experiment-phase DataFrame + analysis)."""

    def __init__(self) -> None:
        self._dataframes: dict[str, pd.DataFrame] = {}
        self._analyses: dict[str, dict] = {}

    def get_experiment_df(self, info: RunInfo) -> pd.DataFrame:
        """Load experiment-phase rows, cached in memory."""
        if info.id not in self._dataframes:
            log.info("Loading parquet for %s", info.id)
            df = pd.read_parquet(info.path)
            exp = df[df.phase == "experiment"].reset_index(drop=True)
            self._dataframes[info.id] = exp
        return self._dataframes[info.id]

    def get_analysis(self, info: RunInfo) -> dict:
        """Get analysis results, triggering computation if needed.

        Passes the already-loaded experiment DataFrame to avoid double-loading.
        """
        if info.id not in self._analyses:
            ws = default_window_sizes(info.L)
            exp = self.get_experiment_df(info)
            log.info("Analyzing %s (W=%s)", info.id, ws)
            self._analyses[info.id] = analyze_run(info.path, ws, exp=exp)
        return self._analyses[info.id]


# ---------------------------------------------------------------------------
# Run index
# ---------------------------------------------------------------------------

class RunIndex:
    """Scans data/runs/ for parquet files, builds metadata index."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self.runs: dict[str, RunInfo] = {}
        self._scan()

    def _scan(self) -> None:
        """Scan directory for parquet files and index them."""
        if not self.runs_dir.exists():
            log.warning("Runs directory does not exist: %s", self.runs_dir)
            return

        parquets = sorted(self.runs_dir.glob("*.parquet"))
        for p in parquets:
            try:
                info = RunInfo(p)
                self.runs[info.id] = info
                log.debug("Indexed run: %s", info.id)
            except ValueError as e:
                log.warning("Skipping %s: %s", p.name, e)

        log.info("Indexed %d runs", len(self.runs))

    def resolve_glob(self, pattern: str) -> list[str]:
        """Resolve a glob/fnmatch pattern against run IDs."""
        return sorted(
            rid for rid in self.runs
            if fnmatch.fnmatch(rid, pattern)
        )

    def resolve_patterns(self, patterns: list[str]) -> list[str]:
        """Resolve multiple patterns (may contain globs), return unique sorted IDs."""
        result: set[str] = set()
        for pattern in patterns:
            if any(c in pattern for c in "*?[]"):
                result.update(self.resolve_glob(pattern))
            else:
                if pattern in self.runs:
                    result.add(pattern)
        return sorted(result)


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

def build_metric_registry(
    index: RunIndex,
    cache: RunCache,
) -> list[dict]:
    """Build the metric registry by inspecting actual data.

    Checks parquet columns for step-level metrics and analysis caches
    for block-level metrics. Only includes metrics that exist in at
    least one run.
    """
    metrics: list[dict] = []

    # Check step-level metrics from the first available run
    step_columns_found: set[str] = set()
    if index.runs:
        sample_info = next(iter(index.runs.values()))
        sample_df = cache.get_experiment_df(sample_info)
        step_columns_found = set(sample_df.columns)

    for metric_id, defn in STEP_METRIC_DEFS.items():
        if defn["column"] in step_columns_found:
            metrics.append({
                "id": metric_id,
                "name": defn["name"],
                "resolution": "step",
                "source": "parquet",
                "column": defn["column"],
                "description": defn["description"],
            })

    # Check block-level metrics from analysis caches on disk
    # Scan for .analysis.pkl files to discover window sizes without
    # loading full parquets or triggering expensive analysis
    window_sizes_found: set[int] = set()
    for p in index.runs_dir.glob("*.analysis.pkl"):
        m = re.findall(r"W(\d+)", p.stem)
        for w_str in m:
            window_sizes_found.add(int(w_str))

    for w in sorted(window_sizes_found):
        metrics.append({
            "id": f"compressibility_W{w}",
            "name": f"Compressibility (W={w})",
            "resolution": "block",
            "source": "analysis",
            "window_size": w,
            "description": f"Gzip compressibility over {w}-token sliding windows",
        })

    # EOS rate EMA (derived step-level metric, always available if eos exists)
    if "eos" in step_columns_found:
        metrics.append({
            "id": "eos_ema",
            "name": "EOS Rate (EMA)",
            "resolution": "step",
            "source": "derived",
            "description": "Exponential moving average of EOS flag (span=1000)",
        })

    return metrics


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

EOS_EMA_SPAN = 1000


def extract_metric(
    metric_id: str,
    info: RunInfo,
    cache: RunCache,
    downsample: int,
) -> dict[str, list] | None:
    """Extract x/y data for a single metric from a single run.

    Returns {"x": [...], "y": [...]} or None if metric unavailable.
    """
    # Step-level metrics from parquet
    if metric_id in STEP_METRIC_DEFS:
        defn = STEP_METRIC_DEFS[metric_id]
        exp = cache.get_experiment_df(info)
        col = defn["column"]
        if col not in exp.columns:
            return None

        values = exp[col].to_numpy(dtype=np.float64)
        n = len(values)

        # Downsample by striding
        stride = max(1, n // downsample)
        if stride > 1:
            # Average within windows for cleaner signal
            n_trimmed = (n // stride) * stride
            values_ds = values[:n_trimmed].reshape(-1, stride).mean(axis=1)
            steps_ds = np.arange(stride // 2, n_trimmed, stride)
        else:
            values_ds = values
            steps_ds = np.arange(n)

        return {
            "x": steps_ds.tolist(),
            "y": values_ds.tolist(),
        }

    # EOS EMA (derived step-level)
    if metric_id == "eos_ema":
        from utils import eos_ema
        exp = cache.get_experiment_df(info)
        if "eos" not in exp.columns:
            return None

        ema = eos_ema(exp.eos.to_numpy().astype(float), EOS_EMA_SPAN)
        n = len(ema)

        stride = max(1, n // downsample)
        if stride > 1:
            n_trimmed = (n // stride) * stride
            ema_ds = ema[:n_trimmed].reshape(-1, stride).mean(axis=1)
            steps_ds = np.arange(stride // 2, n_trimmed, stride)
        else:
            ema_ds = ema
            steps_ds = np.arange(n)

        return {
            "x": steps_ds.tolist(),
            "y": ema_ds.tolist(),
        }

    # Block-level compressibility metrics
    m = re.match(r"compressibility_W(\d+)$", metric_id)
    if m:
        w = int(m.group(1))
        analysis = cache.get_analysis(info)
        comp_dict = analysis.get("compressibility", {})
        if w not in comp_dict:
            return None

        comp = comp_dict[w]
        valid_mask = ~np.isnan(comp)
        valid_indices = np.where(valid_mask)[0]
        valid_values = comp[valid_mask]

        # Block-level data is already sparser than step-level, but
        # can still be large (100k points minus window warmup).
        # Apply same downsampling logic.
        n = len(valid_values)
        stride = max(1, n // downsample)
        if stride > 1:
            n_trimmed = (n // stride) * stride
            values_ds = valid_values[:n_trimmed].reshape(-1, stride).mean(axis=1)
            indices_ds = valid_indices[:n_trimmed].reshape(-1, stride).mean(axis=1).astype(int)
        else:
            values_ds = valid_values
            indices_ds = valid_indices

        return {
            "x": indices_ds.tolist(),
            "y": values_ds.tolist(),
        }

    return None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="autoloop explorer", version="0.1.0")

# Global state, initialized on startup
run_index: RunIndex | None = None
run_cache: RunCache | None = None
metric_registry: list[dict] | None = None


@app.on_event("startup")
def startup() -> None:
    global run_index, run_cache, metric_registry
    run_index = RunIndex(RUNS_DIR)
    run_cache = RunCache()
    metric_registry = build_metric_registry(run_index, run_cache)
    log.info("Metric registry: %d metrics", len(metric_registry))


@app.get("/api/runs")
def get_runs() -> list[dict]:
    """Return metadata for all indexed runs."""
    assert run_index is not None
    return [info.to_dict() for info in run_index.runs.values()]


@app.get("/api/metrics")
def get_metrics() -> list[dict]:
    """Return the metric registry."""
    assert metric_registry is not None
    return metric_registry


@app.get("/api/resolve")
def resolve_pattern(
    pattern: str = Query(..., description="Glob pattern to resolve against run IDs"),
) -> list[str]:
    """Resolve a glob pattern to a list of matching run IDs."""
    assert run_index is not None
    return run_index.resolve_glob(pattern)


@app.get("/api/data")
def get_data(
    runs: str = Query(..., description="Comma-separated run IDs or glob patterns"),
    metrics: str = Query(..., description="Comma-separated metric IDs"),
    downsample: int = Query(DEFAULT_DOWNSAMPLE, description="Target number of points per series"),
) -> JSONResponse:
    """Return metric data for selected runs.

    Supports glob patterns in run IDs. Step-level data is downsampled
    by averaging within stride windows. Block-level data uses the same
    approach but is typically sparser.
    """
    assert run_index is not None
    assert run_cache is not None

    # Parse and resolve run patterns
    run_patterns = [r.strip() for r in runs.split(",") if r.strip()]
    resolved_ids = run_index.resolve_patterns(run_patterns)

    if not resolved_ids:
        return JSONResponse(
            content={"error": f"No runs matched patterns: {run_patterns}"},
            status_code=404,
        )

    metric_ids = [m.strip() for m in metrics.split(",") if m.strip()]
    if not metric_ids:
        return JSONResponse(
            content={"error": "No metrics specified"},
            status_code=400,
        )

    result: dict[str, dict] = {}

    for run_id in resolved_ids:
        info = run_index.runs[run_id]
        run_data: dict[str, dict] = {}

        for metric_id in metric_ids:
            extracted = extract_metric(metric_id, info, run_cache, downsample)
            if extracted is not None:
                run_data[metric_id] = extracted

        if run_data:
            result[run_id] = run_data

    return JSONResponse(content=result)


@app.get("/api/step_range")
def get_step_range(
    run: str = Query(..., description="Run ID (e.g. L0064_T0.70_S42)"),
) -> JSONResponse:
    """Return step range and all EOS positions for a run."""
    assert run_index is not None
    assert run_cache is not None

    if run not in run_index.runs:
        return JSONResponse(
            content={"error": f"Run not found: {run}"},
            status_code=404,
        )

    info = run_index.runs[run]
    exp = run_cache.get_experiment_df(info)

    eos_steps = exp.index[exp.eos.astype(bool)].tolist()

    return JSONResponse(content={
        "run_id": info.id,
        "L": info.L,
        "min_step": int(exp.index[0]) if len(exp) > 0 else 0,
        "max_step": int(exp.index[-1]) if len(exp) > 0 else 0,
        "total_eos": len(eos_steps),
        "eos_steps": eos_steps,
    })


@app.get("/api/context")
def get_context(
    run: str = Query(..., description="Run ID (e.g. L0064_T0.70_S42)"),
    step: int = Query(..., description="Step number to center the context window on"),
    window: int | None = Query(None, description="Number of tokens to return (default: run's L value)"),
) -> JSONResponse:
    """Return the tokens in the model's context window at a given step."""
    assert run_index is not None
    assert run_cache is not None

    if run not in run_index.runs:
        return JSONResponse(
            content={"error": f"Run not found: {run}"},
            status_code=404,
        )

    info = run_index.runs[run]
    exp = run_cache.get_experiment_df(info)

    w = window if window is not None else info.L

    # Context window at step N: the w tokens ending at step N
    window_end = step
    window_start = step - w + 1

    # Clamp to data range
    window_start = max(window_start, 0)
    window_end = min(window_end, len(exp) - 1)

    # Slice the window from the experiment dataframe
    window_df = exp.iloc[window_start:window_end + 1]

    tokens = []
    eos_positions = []
    for idx, row in window_df.iterrows():
        s = int(idx)
        tok = {
            "step": s,
            "token_id": int(row["token_id"]),
            "text": row["decoded_text"],
            "entropy": float(row["entropy"]),
            "log_prob": float(row["log_prob"]),
            "eos": bool(row["eos"]),
        }
        tokens.append(tok)
        if tok["eos"]:
            eos_positions.append(s)

    # Find prev_eos: nearest EOS before window_start
    eos_mask = exp.eos.astype(bool)
    eos_before = exp.index[eos_mask & (exp.index < window_start)]
    prev_eos: int | None = int(eos_before[-1]) if len(eos_before) > 0 else None

    # Find next_eos: nearest EOS after window_end
    eos_after = exp.index[eos_mask & (exp.index > window_end)]
    next_eos: int | None = int(eos_after[0]) if len(eos_after) > 0 else None

    return JSONResponse(content={
        "run_id": info.id,
        "L": info.L,
        "step": step,
        "window_start": int(window_start),
        "window_end": int(window_end),
        "tokens": tokens,
        "eos_positions": eos_positions,
        "prev_eos": prev_eos,
        "next_eos": next_eos,
    })


# Static file serving: mount at root, must be last
STATIC_DIR = Path("static")
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

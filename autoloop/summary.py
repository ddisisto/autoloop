"""Cross-condition summary table for autoloop runs.

Walks all parquet files in the runs directory, extracts per-run scalar
metrics from analysis caches, and returns a sorted DataFrame.
"""

import logging
from pathlib import Path

import pandas as pd

from .analyze import analyze_run, comp_stats, load_experiment_df
from .analyze.scalars import run_scalars
from .plot import parse_run_name

log = logging.getLogger(__name__)


def build_summary(runs_dir: Path) -> pd.DataFrame:
    """Build summary DataFrame from all parquet files in runs_dir."""
    from .runlib import discover_runs
    parquets = discover_runs(runs_dir, run_type="sweep")
    log.info("Found %d parquet files in %s", len(parquets), runs_dir)

    rows: list[dict] = []
    for pq in parquets:
        log.info("Processing %s", pq.name)
        params = parse_run_name(pq)

        exp = load_experiment_df(pq)
        analysis = analyze_run(pq, window_sizes=[64], exp=exp)

        scalars = run_scalars(exp, analysis)

        comp_w64_mean = comp_stats(analysis, 64)["mean"]

        row = {
            "L": params["L"],
            "T": params["T"],
            "S": params["S"],
            **scalars,
            "comp_W64_mean": comp_w64_mean,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.sort_values(["T", "L", "S"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

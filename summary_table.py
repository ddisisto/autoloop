"""Cross-condition summary table for autoloop runs.

Walks all parquet files in the runs directory, extracts per-run scalar
metrics from analysis caches, and emits a sorted CSV.

Usage:
    python summary_table.py                         # print to stdout
    python summary_table.py --out data/summary.csv  # write to file
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from autoloop.analyze import analyze_run, comp_stats, load_experiment_df
from autoloop.analyze.metrics import run_scalars
from autoloop.plot import parse_run_name

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit cross-condition summary CSV from autoloop runs"
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("data/runs"),
        help="Directory containing parquet files (default: data/runs)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output CSV path. If omitted, print to stdout.",
    )
    return parser.parse_args()


def build_summary(runs_dir: Path) -> pd.DataFrame:
    """Build summary DataFrame from all parquet files in runs_dir."""
    from runlib import discover_runs
    parquets = discover_runs(runs_dir, run_type="sweep")
    log.info("Found %d parquet files in %s", len(parquets), runs_dir)

    rows: list[dict] = []
    for pq in parquets:
        log.info("Processing %s", pq.name)
        params = parse_run_name(pq)

        exp = load_experiment_df(pq)
        analysis = analyze_run(pq, window_sizes=[64], exp=exp)
        acf = analysis["entropy_autocorrelation"]

        scalars = run_scalars(exp, acf)

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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()

    df = build_summary(args.runs_dir)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        log.info("Wrote %d rows to %s", len(df), args.out)
    else:
        sys.stdout.write(df.to_csv(index=False))


if __name__ == "__main__":
    main()

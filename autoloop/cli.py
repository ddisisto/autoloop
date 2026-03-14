"""Unified CLI entry point for autoloop.

All subcommands: run, sweep, index, explore, plot, analyze, grep,
semantic, precollapse, summary.
"""

import argparse
import logging
import sys
from pathlib import Path

from .resolve import (
    add_filter_args,
    auto_index_run,
    resolve_from_args,
    resolve_runs,
)
from .runindex import DB_PATH, RUNS_ROOT

log = logging.getLogger(__name__)


def cmd_run(args: argparse.Namespace) -> None:
    """Run an experiment (fixed, schedule, or beta)."""
    from .engine import StepEngine, load_model
    from .experiment import (
        BetaController,
        FixedController,
        ScheduleController,
        run_experiment,
    )
    from autoloop import runlib

    model, tokenizer = load_model(args.model_dir, args.device)
    engine = StepEngine(model, tokenizer, args.device, args.seed)

    mode_dirs = {
        "fixed": runlib.SWEEP_DIR,
        "schedule": runlib.SCHEDULE_DIR,
        "beta": runlib.CONTROLLER_DIR,
    }
    output_dir = Path(args.output_dir) if args.output_dir else mode_dirs[args.run_mode]

    if args.run_mode == "fixed":
        ctrl = FixedController(args.L, args.T)
        name = args.run_name or f"L{args.L:04d}_T{args.T:.2f}_S{args.seed}"
        extra = {"controller": False, "context_length": args.L,
                 "temperature": args.T}
        run_experiment(
            engine, ctrl, args.total_steps, args.segment_steps,
            name, output_dir, args.L, args.T,
            save_every=args.save_every, prefill_text=args.prefill_text,
            dry_run=args.dry_run, extra_meta=extra,
        )
        auto_index_run(output_dir / f"{name}.parquet")

    elif args.run_mode == "schedule":
        segments = []
        for part in args.spec.split(","):
            tokens = part.strip().split(":")
            steps = int(tokens[0])
            L = int(tokens[1].lstrip("L"))
            T = float(tokens[2].lstrip("T"))
            segments.append((steps, L, T))
        ctrl = ScheduleController(segments)
        total = sum(s[0] for s in segments)
        start_L, start_T = segments[0][1], segments[0][2]
        import hashlib
        h = hashlib.sha256(args.spec.encode()).hexdigest()[:8]
        name = args.run_name or f"sched_S{args.seed}_{h}"
        extra = {"controller": False, "schedule": args.spec}
        run_experiment(
            engine, ctrl, total, args.segment_steps,
            name, output_dir, start_L, start_T,
            save_every=args.save_every, prefill_text=args.prefill_text,
            dry_run=args.dry_run, extra_meta=extra,
        )
        auto_index_run(output_dir / f"{name}.parquet")

    elif args.run_mode == "beta":
        suffix = "d" if args.drift else ""
        name = (args.run_name or
                f"ctrl{suffix}_S{args.seed}_{args.start_L}_{args.start_T:.2f}")
        ctrl = BetaController(target=args.target, drift=args.drift)
        extra = {"controller": True, "beta_target": args.target,
                 "drift": args.drift}
        run_experiment(
            engine, ctrl, args.total_steps, args.segment_steps,
            name, output_dir, args.start_L, args.start_T,
            save_every=args.save_every, prefill_text=args.prefill_text,
            dry_run=args.dry_run, extra_meta=extra,
        )
        auto_index_run(output_dir / f"{name}.parquet")


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run or inspect sweeps."""
    from .sweep import (
        PRESETS,
        expand_grid,
        print_presets,
        print_status,
        run_sweep,
    )

    if args.list:
        print_presets()
        return

    if args.status is not None:
        preset_name = None if args.status == "__all__" else args.status
        if preset_name and preset_name not in PRESETS:
            log.error("Unknown preset '%s'. Use 'loop sweep --list'.", preset_name)
            sys.exit(1)
        print_status(preset_name)
        return

    # Determine grid
    if args.preset:
        if args.preset not in PRESETS:
            log.error("Unknown preset '%s'. Use 'loop sweep --list'.", args.preset)
            sys.exit(1)
        preset = PRESETS[args.preset]
        grid = expand_grid(preset["L"], preset["T"], preset["seeds"])
        label = args.preset
    elif args.L and args.T:
        grid = expand_grid(args.L, args.T, args.seed)
        parts = [f"L{'_'.join(str(l) for l in args.L)}",
                 f"T{'_'.join(f'{t:.2f}' for t in args.T)}"]
        label = "_".join(parts)
    else:
        log.error("Provide a preset name or --L and --T for an ad-hoc grid. "
                  "Use 'loop sweep --list' to see presets.")
        sys.exit(1)

    run_sweep(grid, label, args.dry_run)

    # Rebuild index after sweep
    if not args.dry_run:
        log.info("Rebuilding index after sweep...")
        from .runindex import create_db, reindex_all
        conn = create_db(DB_PATH)
        reindex_all(conn, RUNS_ROOT)
        conn.close()


def cmd_index(args: argparse.Namespace) -> None:
    """Build or query the run index."""
    from .runindex import create_db, query_runs, reindex_all, _format_table

    if args.index_cmd == "build":
        root = Path(args.root) if args.root else RUNS_ROOT
        db = Path(args.db) if args.db else DB_PATH
        conn = create_db(db)
        reindex_all(conn, root)
        total = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        log.info("Index complete: %d runs in %s", total, db)
        conn.close()

    elif args.index_cmd == "query":
        db = Path(args.db) if args.db else DB_PATH
        if not db.exists():
            log.error("No index at %s — run 'loop index build' first.", db)
            sys.exit(1)
        conn = create_db(db)
        filters: dict[str, float | int | str] = {}
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
            import json
            import numpy as np
            for r in runs:
                for k, v in r.items():
                    if isinstance(v, float) and np.isnan(v):
                        r[k] = None
            print(json.dumps(runs, indent=2))
        else:
            print(_format_table(runs))
        conn.close()

def cmd_explore(args: argparse.Namespace) -> None:
    """Start the interactive explorer."""
    import uvicorn
    log.info("Starting explorer on port %d...", args.port)
    uvicorn.run("autoloop.explorer:app", host="0.0.0.0", port=args.port, reload=True)


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate plots for resolved runs."""
    from .plot import plot_runs

    paths = resolve_from_args(args)
    log.info("Plotting %d runs", len(paths))
    plot_runs(
        paths,
        plots=args.plots,
        downsample=getattr(args, "downsample", 100),
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    """Recompute analysis caches for resolved runs."""
    from .analyze import analyze_run, default_window_sizes

    paths = resolve_from_args(args)
    window_sizes = default_window_sizes(0)
    log.info("Analyzing %d runs at W=%s", len(paths), window_sizes)

    for path in paths:
        log.info("Processing %s", path.stem)
        analyze_run(path, window_sizes)

    log.info("Done — analyzed %d runs.", len(paths))


def cmd_grep(args: argparse.Namespace) -> None:
    """Search decoded text in resolved runs."""
    import re
    from .grep_text import grep_run, format_match

    paths = resolve_from_args(args)
    log.info("Searching %d runs for '%s'", len(paths), args.pattern)

    flags = re.IGNORECASE if args.ignore_case else 0
    if getattr(args, "regex", False):
        pat = re.compile(args.pattern, flags)
    else:
        pat = re.compile(re.escape(args.pattern), flags)

    context_tokens = args.context if args.context > 0 else 20

    total = 0
    for path in paths:
        run_name = path.stem
        matches = grep_run(path, pat, context_tokens=context_tokens)
        total += len(matches)

        if args.count:
            if matches:
                print(f"{run_name}: {len(matches)}")
            continue

        if matches:
            print(f"\n\033[1m{run_name}\033[0m  ({len(matches)} matches)")
            for m in matches:
                print(format_match(m, run_name))

    if args.count:
        print(f"\nTotal: {total}")
    elif total == 0:
        print("No matches found.")


def cmd_semantic(args: argparse.Namespace) -> None:
    """Run semantic analysis (clouds or themes)."""
    from .semantic import _load_runs
    from .semantic_clouds import run_clouds
    from .semantic_report import run_themes

    # Semantic analysis uses its own run loading (needs full text).
    # Convert resolved paths to string file list for _load_runs.
    if hasattr(args, "run_ids") and args.run_ids:
        paths = resolve_from_args(args)
        files = [str(p) for p in paths]
    else:
        # No run IDs specified — use all runs via semantic's own discovery
        from .semantic import _discover_run_files
        files = _discover_run_files(None)

    if not files:
        log.error("No parquet files found")
        sys.exit(1)

    log.info("Loading %d runs...", len(files))
    all_runs = _load_runs(files)
    log.info("Loaded %d runs", len(all_runs))

    if args.clouds:
        csv_path = getattr(args, "csv", None)
        run_clouds(all_runs, csv_path=csv_path)
    elif args.themes:
        run_themes(
            all_runs,
            args.themes,
            context_radius=getattr(args, "context_radius", 80),
            entropy_window=getattr(args, "entropy_window", 20),
        )


def cmd_precollapse(args: argparse.Namespace) -> None:
    """Run pre-collapse trajectory analysis."""
    import pandas as pd
    from .precollapse import analyze_precollapse
    from .precollapse_report import (
        detail_report,
        print_summary,
        summary_row,
    )

    # If no run IDs or filters provided, fall back to sweep type
    ids = args.run_ids if args.run_ids else None
    has_filters = any(
        getattr(args, f, None) is not None
        for f in ("run_type", "filter_L", "filter_T", "filter_seed", "filter_regime")
    )
    if ids or has_filters:
        paths = resolve_from_args(args)
    else:
        # Default: resolve all sweep runs
        paths = resolve_runs(run_type="sweep")

    log.info("Analyzing %d runs", len(paths))

    results = []
    for p in paths:
        log.info("Processing %s", p.stem)
        ra = analyze_precollapse(p)
        results.append(ra)

        if args.detail and args.detail in ra.run_id:
            print(detail_report(ra))

    rows = [summary_row(ra) for ra in results]
    df = pd.DataFrame(rows)
    df = df.sort_values(["L", "T", "seed"]).reset_index(drop=True)

    if args.csv:
        df.to_csv(args.csv, index=False)
        log.info("Wrote %s", args.csv)
    elif not args.detail:
        print_summary(df)


def cmd_survey(args: argparse.Namespace) -> None:
    """Run a basin survey."""
    from .survey import run_survey

    output_dir = Path(args.output_dir) if args.output_dir else None
    parquet_path = run_survey(
        seed=args.seed,
        L=args.L,
        total_steps=args.total_steps,
        T_min=args.T_min,
        T_max=args.T_max,
        segment_steps=args.segment_steps,
        model_dir=args.model_dir,
        device=args.device,
        run_name=args.run_name,
        output_dir=output_dir,
        save_every=args.save_every,
    )
    auto_index_run(parquet_path)


def cmd_summary(args: argparse.Namespace) -> None:
    """Generate cross-condition summary table."""
    from .summary import build_summary

    runs_dir = Path(getattr(args, "runs_dir", "data/runs"))
    df = build_summary(runs_dir)

    out = getattr(args, "out", None)
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info("Wrote %d rows to %s", len(df), out_path)
    else:
        sys.stdout.write(df.to_csv(index=False))


def _add_run_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across all run modes."""
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--segment-steps", type=int, default=1000)
    parser.add_argument("--model-dir", type=str, default="data/model/SmolLM-135M")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-name", type=str, help="Override auto run name")
    parser.add_argument("--prefill-text", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save snapshot every N segments")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="loop",
        description="Unified CLI for autoloop experiments.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Run an experiment")
    run_sub = p_run.add_subparsers(dest="run_mode", required=True)

    # run fixed
    p_fixed = run_sub.add_parser("fixed", help="Fixed L/T run")
    _add_run_common_args(p_fixed)
    p_fixed.add_argument("-L", type=int, required=True, help="Context length")
    p_fixed.add_argument("-T", type=float, required=True, help="Temperature")

    # run schedule
    p_sched = run_sub.add_parser("schedule", help="Scheduled L/T segments")
    _add_run_common_args(p_sched)
    p_sched.add_argument("--spec", type=str, required=True,
                         help="Schedule: 'steps:L{n}:T{f},...'")

    # run beta
    p_beta = run_sub.add_parser("beta", help="Beta hill-climb controller")
    _add_run_common_args(p_beta)
    p_beta.add_argument("--start-L", type=int, default=64)
    p_beta.add_argument("--start-T", type=float, default=0.70)
    p_beta.add_argument("--target", type=float, default=0.9)
    p_beta.add_argument("--drift", action="store_true")

    # ── sweep ─────────────────────────────────────────────────────
    p_sweep = sub.add_parser("sweep", help="Run or inspect sweeps")
    p_sweep.add_argument("preset", nargs="?", default=None,
                         help="Named preset to run")
    p_sweep.add_argument("--dry-run", action="store_true")
    p_sweep.add_argument("--status", nargs="?", const="__all__", default=None,
                         metavar="PRESET",
                         help="Show grid status")
    p_sweep.add_argument("--list", action="store_true",
                         help="List available presets")
    p_sweep.add_argument("--L", type=int, nargs="+", metavar="N",
                         help="Context lengths for ad-hoc grid")
    p_sweep.add_argument("--T", type=float, nargs="+", metavar="F",
                         help="Temperatures for ad-hoc grid")
    p_sweep.add_argument("--seed", type=int, nargs="+", default=[42],
                         help="Seeds for ad-hoc grid")

    # ── index ─────────────────────────────────────────────────────
    p_index = sub.add_parser("index", help="Manage the run index")
    idx_sub = p_index.add_subparsers(dest="index_cmd", required=True)

    # index build
    p_build = idx_sub.add_parser("build", help="Full reindex of all runs")
    p_build.add_argument("--root", type=str, default=None,
                         help="Runs directory (default: data/runs)")
    p_build.add_argument("--db", type=str, default=None,
                         help="Database path (default: data/runs/index.db)")

    # index query
    p_query = idx_sub.add_parser("query", help="Query indexed runs")
    p_query.add_argument("--type", dest="run_type", help="Filter by run type")
    p_query.add_argument("--L", type=int, help="Filter by context length")
    p_query.add_argument("--T", type=float, help="Filter by temperature")
    p_query.add_argument("--seed", type=int, help="Filter by seed")
    p_query.add_argument("--regime", help="Filter by regime")
    p_query.add_argument("--db", type=str, default=None,
                         help="Database path")
    p_query.add_argument("--json", action="store_true", dest="as_json",
                         help="Output as JSON")

    # ── survey ────────────────────────────────────────────────────
    p_survey = sub.add_parser("survey", help="Run basin survey at fixed L")
    p_survey.add_argument("--seed", type=int, required=True)
    p_survey.add_argument("-L", type=int, required=True, help="Context length")
    p_survey.add_argument("--total-steps", type=int, default=100_000)
    p_survey.add_argument("--T-min", type=float, default=None,
                          help="Temperature floor for cooling (default: L-dependent)")
    p_survey.add_argument("--T-max", type=float, default=None,
                          help="Temperature ceiling for heating (default: L-dependent)")
    p_survey.add_argument("--segment-steps", type=int, default=None,
                          help="Steps per segment (default: max(L, 50))")
    p_survey.add_argument("--model-dir", type=str, default="data/model/SmolLM-135M")
    p_survey.add_argument("--device", type=str, default="cuda")
    p_survey.add_argument("--run-name", type=str, default=None)
    p_survey.add_argument("--output-dir", type=str, default=None)
    p_survey.add_argument("--save-every", type=int, default=50)

    # ── explore ───────────────────────────────────────────────────
    p_explore = sub.add_parser("explore", help="Start the interactive explorer")
    p_explore.add_argument("--port", type=int, default=8000,
                           help="Port to serve on (default: 8000)")

    # ── plot ──────────────────────────────────────────────────────
    p_plot = sub.add_parser("plot", help="Generate plots for runs")
    add_filter_args(p_plot)
    p_plot.add_argument("--plots", nargs="+",
                        help="Plot types to generate (entropy, compressibility, "
                             "phase, temporal, violin)")
    p_plot.add_argument("--downsample", type=int, default=100,
                        help="Downsample factor for time series (default: 100)")

    # ── analyze ──────────────────────────────────────────────────
    p_analyze = sub.add_parser("analyze",
                               help="Recompute analysis caches for runs")
    add_filter_args(p_analyze)

    # ── grep ─────────────────────────────────────────────────────
    p_grep = sub.add_parser("grep", help="Search decoded text in runs")
    p_grep.add_argument("pattern", help="Search pattern")
    add_filter_args(p_grep)
    p_grep.add_argument("--count", action="store_true",
                        help="Show match counts only")
    p_grep.add_argument("-i", action="store_true", dest="ignore_case",
                        help="Case-insensitive search")
    p_grep.add_argument("-C", type=int, default=0, dest="context",
                        help="Context tokens around matches")
    p_grep.add_argument("--regex", action="store_true",
                        help="Treat pattern as regex")

    # ── semantic ─────────────────────────────────────────────────
    p_semantic = sub.add_parser("semantic",
                                help="Semantic analysis (clouds or themes)")
    add_filter_args(p_semantic)
    sem_group = p_semantic.add_mutually_exclusive_group(required=True)
    sem_group.add_argument("--clouds", action="store_true",
                           help="Auto-discover themes and map basins")
    sem_group.add_argument("--themes", nargs="+", metavar="WORD",
                           help="Multi-theme density report")
    p_semantic.add_argument("--csv", type=str, default=None,
                            help="Export basin fingerprints to CSV (--clouds mode)")
    p_semantic.add_argument("--context-radius", type=int, default=80,
                            help="Characters of context around hits (default: 80)")
    p_semantic.add_argument("--entropy-window", type=int, default=20,
                            help="Token window for local entropy (default: 20)")

    # ── precollapse ──────────────────────────────────────────────
    p_pre = sub.add_parser("precollapse",
                           help="Pre-collapse trajectory analysis")
    add_filter_args(p_pre)
    p_pre.add_argument("--detail", type=str, metavar="RUN_ID",
                       help="Detailed report for a specific run")
    p_pre.add_argument("--csv", type=str, metavar="PATH",
                       help="Export metrics to CSV")

    # ── summary ──────────────────────────────────────────────────
    p_summary = sub.add_parser("summary",
                               help="Cross-condition summary table")
    p_summary.add_argument("--out", type=str, default=None,
                           help="Output CSV path (default: print to stdout)")
    p_summary.add_argument("--runs-dir", type=str, default="data/runs",
                           help="Runs directory (default: data/runs)")

    # ── basin ────────────────────────────────────────────────────
    from .basin import add_basin_subparser
    add_basin_subparser(sub)

    return parser


def cmd_basin(args: argparse.Namespace) -> None:
    """Dispatch to basin subcommands."""
    from .basin import cmd_basin as _basin_dispatch
    _basin_dispatch(args)


_DISPATCH: dict[str, callable] = {
    "run": cmd_run,
    "sweep": cmd_sweep,
    "survey": cmd_survey,
    "index": cmd_index,
    "explore": cmd_explore,
    "plot": cmd_plot,
    "analyze": cmd_analyze,
    "grep": cmd_grep,
    "semantic": cmd_semantic,
    "precollapse": cmd_precollapse,
    "summary": cmd_summary,
    "basin": cmd_basin,
}


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()

"""Annealing experiment runner.

Runs probes and full experiments from docs/annealing-experiment.md.
Sequential execution with decision gates between phases.

Usage:
    python anneal.py probes          # Phase 0: quick feasibility (5k tokens)
    python anneal.py probes --check  # Analyze probe results without running
    python anneal.py tier1           # Phase A: escape threshold (100k tokens)
    python anneal.py tier2           # Phase B: return dynamics (100k tokens)
    python anneal.py tier5           # Phase B: T vs L comparison (100k tokens)
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import runlib

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent
MODEL_DIR = REPO_ROOT / "data" / "model" / "SmolLM-135M"
ANNEAL_DIR = runlib.ANNEAL_DIR
PROBE_DIR = runlib.PROBE_DIR
DEVICE = "cuda"


def _output_dir_for(name: str) -> Path:
    """Return the correct output directory based on run name prefix."""
    if name.startswith("probe_"):
        return PROBE_DIR
    return ANNEAL_DIR


def run(name: str, *, L: int | None = None, T: float | None = None,
        N: int, prefill: str, seed: int = 42,
        schedule: str | None = None) -> subprocess.CompletedProcess:
    """Run a single generate.py invocation."""
    output_dir = _output_dir_for(name)
    cmd = [
        sys.executable, str(REPO_ROOT / "generate.py"),
        "--seed", str(seed),
        "--model-dir", str(MODEL_DIR),
        "--output-dir", str(output_dir),
        "--device", DEVICE,
        "--run-name", name,
        "--prefill-text", prefill,
    ]
    if schedule:
        cmd += ["--schedule", schedule]
    else:
        cmd += [
            "--context-length", str(L),
            "--temperature", str(T),
            "--num-tokens", str(N),
        ]
    log.info(">>> %s", name)
    return subprocess.run(cmd)


def is_done(name: str) -> bool:
    return (_output_dir_for(name) / f"{name}.parquet").exists()


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def run_batch(runs: list[dict], label: str) -> None:
    """Run a batch sequentially, skipping completed runs."""
    total = len(runs)
    pending = [r for r in runs if not is_done(r["name"])]
    skipped = total - len(pending)

    if not pending:
        log.info("All %d runs already complete.", total)
        return

    # File logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO_ROOT / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"anneal_{label}_{timestamp}.txt"
    logging.getLogger().addHandler(logging.FileHandler(log_file))

    log.info("%s: %d conditions pending, %d already done",
             label, len(pending), skipped)
    log.info("Log file: %s", log_file)

    done = 0
    failed = 0
    failures: list[tuple[str, str]] = []

    for i, r in enumerate(pending, 1):
        name = r["name"]
        header = f"=== [{i}/{len(pending)}] {name}"
        log.info("%s — started %s ===", header, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        t0 = time.monotonic()
        try:
            result = run(**r)
            elapsed = time.monotonic() - t0

            if result.returncode == 0:
                done += 1
                log.info("%s — done in %s (%d done, %d failed, %d remaining) ===",
                         header, format_duration(elapsed), done, failed,
                         len(pending) - done - failed)
            elif result.returncode in (130, -2):
                log.info("%s — interrupted after %s (checkpoint saved). Stopping.",
                         header, format_duration(elapsed))
                break
            else:
                failed += 1
                failures.append((name, f"exit code {result.returncode}"))
                log.error("%s — FAILED (exit %d) after %s (%d done, %d failed, %d remaining) ===",
                          header, result.returncode, format_duration(elapsed),
                          done, failed, len(pending) - done - failed)
        except KeyboardInterrupt:
            elapsed = time.monotonic() - t0
            log.info("%s — interrupted after %s. Subprocess checkpoint may be saved.",
                     header, format_duration(elapsed))
            break
        except Exception as e:
            elapsed = time.monotonic() - t0
            failed += 1
            failures.append((name, str(e)))
            log.error("%s — EXCEPTION after %s: %s ===", header, format_duration(elapsed), e)

    log.info("=" * 60)
    log.info("%s finished. %d done, %d failed, %d skipped (already done).",
             label, done, failed, skipped)
    if failures:
        log.error("Failed runs:")
        for name, reason in failures:
            log.error("  %s: %s", name, reason)


def check_probes() -> None:
    """Analyze probe results: entropy and compressibility of last 1k tokens."""
    from utils import compressibility as comp_fn
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    probes = [
        "probe_L002_T060", "probe_L004_T060", "probe_L008_T060",
        "probe_L016_T060", "probe_L032_T060", "probe_L064_T060",
        "probe_L064_T080", "probe_L064_T100", "probe_L256",
        "probe_young_L064",
    ]

    print(f"\n{'Probe':<22} {'Status':<8} {'Ent(last1k)':<12} {'Comp(last1k)':<13} {'Top token':<20} {'Top %':<8}")
    print("-" * 85)

    for name in probes:
        path = PROBE_DIR / f"{name}.parquet"
        if not path.exists():
            print(f"{name:<22} {'MISSING':<8}")
            continue

        df = pd.read_parquet(path)
        exp = df[df["phase"] == "experiment"]
        tail = exp.tail(1000)

        ent_mean = tail["entropy"].mean()

        tail_ids = tail["token_id"].tolist()
        tail_text = tokenizer.decode(tail_ids)
        comp = comp_fn(tail_text.encode("utf-8"))

        # Top token by frequency
        top = tail["decoded_text"].value_counts()
        top_tok = top.index[0]
        top_pct = 100 * top.iloc[0] / len(tail)

        escaped = ent_mean > 1.0 and comp < 0.8
        status = "ESCAPE" if escaped else "STUCK"

        print(f"{name:<22} {status:<8} {ent_mean:<12.3f} {comp:<13.3f} {repr(top_tok):<20} {top_pct:<8.1f}")

    print()


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

PROBES = [
    dict(name="probe_L256",       L=256, T=0.60, N=5000, prefill=" Star Wars"),
    dict(name="probe_L064_T060",  L=64,  T=0.60, N=5000, prefill=" Star Wars"),
    dict(name="probe_L064_T080",  L=64,  T=0.80, N=5000, prefill=" Star Wars"),
    dict(name="probe_L064_T100",  L=64,  T=1.00, N=5000, prefill=" Star Wars"),
    dict(name="probe_L032_T060",  L=32,  T=0.60, N=5000, prefill=" Star Wars"),
    dict(name="probe_young_L064", L=64,  T=0.60, N=5000, prefill=" young"),
]

TIER1_SEEDS = [42, 123, 7]
TIER1 = [
    dict(name=f"anneal_L{L:03d}_{tag}_S{s}", L=L, T=0.60, N=100000,
         prefill=" Star Wars", seed=s)
    for s in TIER1_SEEDS
    for L, tag in [(256, "control"), (16, "stuck"), (8, "escape"), (4, "escape")]
] + [
    # Boundary probes — pinpoint the transition (10k, seed 42 only)
    dict(name=f"anneal_L{L:03d}_probe_S42", L=L, T=0.60, N=10000,
         prefill=" Star Wars", seed=42)
    for L in [10, 12, 14]
]

TIER2 = [
    dict(name=f"anneal_cycle_short_S{s}",
         schedule="20000:L256:T0.60,10000:L8:T0.60,70000:L256:T0.60",
         N=100000, prefill=" Star Wars", seed=s)
    for s in TIER1_SEEDS
] + [
    dict(name=f"anneal_cycle_long_S{s}",
         schedule="20000:L256:T0.60,30000:L8:T0.60,50000:L256:T0.60",
         N=100000, prefill=" Star Wars", seed=s)
    for s in TIER1_SEEDS
] + [
    dict(name=f"anneal_cycle_brief_S{s}",
         schedule="20000:L256:T0.60,3000:L8:T0.60,77000:L256:T0.60",
         N=100000, prefill=" Star Wars", seed=s)
    for s in TIER1_SEEDS
] + [
    dict(name=f"anneal_no_return_S{s}",
         schedule="20000:L256:T0.60,80000:L8:T0.60",
         N=100000, prefill=" Star Wars", seed=s)
    for s in TIER1_SEEDS
]

TIER5 = [
    dict(name=f"anneal_T_escape_S{s}",
         schedule="20000:L256:T0.60,10000:L256:T1.00,70000:L256:T0.60",
         N=100000, prefill=" Star Wars", seed=s)
    for s in [42]
] + [
    dict(name=f"anneal_L_escape_S{s}",
         schedule="20000:L256:T0.60,10000:L8:T0.60,70000:L256:T0.60",
         N=100000, prefill=" Star Wars", seed=s)
    for s in [42]
] + [
    dict(name=f"anneal_both_S{s}",
         schedule="20000:L256:T0.60,10000:L8:T1.00,70000:L256:T0.60",
         N=100000, prefill=" Star Wars", seed=s)
    for s in [42]
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Annealing experiment runner")
    parser.add_argument("phase", choices=["probes", "tier1", "tier2", "tier5"],
                        help="Which phase to run")
    parser.add_argument("--check", action="store_true",
                        help="Analyze results instead of running (probes only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    phases = {
        "probes": PROBES,
        "tier1": TIER1,
        "tier2": TIER2,
        "tier5": TIER5,
    }
    runs = phases[args.phase]

    if args.check:
        if args.phase == "probes":
            check_probes()
        else:
            # Show status for any phase (same as --dry-run)
            for r in runs:
                status = "DONE" if is_done(r["name"]) else "TODO"
                sched = r.get("schedule", f"L{r.get('L')} T{r.get('T')}")
                print(f"  [{status}] {r['name']:<35} {sched}")
            done = sum(1 for r in runs if is_done(r["name"]))
            print(f"\n{done}/{len(runs)} complete")
        return

    if args.dry_run:
        for r in runs:
            status = "DONE" if is_done(r["name"]) else "TODO"
            sched = r.get("schedule", f"L{r.get('L')} T{r.get('T')}")
            print(f"  [{status}] {r['name']:<35} {sched}")
        done = sum(1 for r in runs if is_done(r["name"]))
        print(f"\n{done}/{len(runs)} complete")
        return

    run_batch(runs, label=args.phase)

    if args.phase == "probes":
        log.info("Probes complete. Run 'python anneal.py probes --check' to analyze.")


if __name__ == "__main__":
    main()

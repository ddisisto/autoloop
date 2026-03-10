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
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent
MODEL_DIR = REPO_ROOT / "data" / "model" / "SmolLM-135M"
RUNS_DIR = REPO_ROOT / "data" / "runs"
DEVICE = "cuda"


def run(name: str, *, L: int | None = None, T: float | None = None,
        N: int, prefill: str, seed: int = 42,
        schedule: str | None = None) -> subprocess.CompletedProcess:
    """Run a single generate.py invocation."""
    cmd = [
        sys.executable, str(REPO_ROOT / "generate.py"),
        "--seed", str(seed),
        "--model-dir", str(MODEL_DIR),
        "--output-dir", str(RUNS_DIR),
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
    return (RUNS_DIR / f"{name}.parquet").exists()


def run_batch(runs: list[dict]) -> None:
    """Run a batch sequentially, skipping completed runs."""
    total = len(runs)
    done = sum(1 for r in runs if is_done(r["name"]))
    if done == total:
        log.info("All %d runs already complete.", total)
        return
    log.info("%d/%d runs complete, %d remaining.", done, total, total - done)

    for i, r in enumerate(runs, 1):
        if is_done(r["name"]):
            log.info("[%d/%d] %s — already done, skipping", i, total, r["name"])
            continue
        log.info("[%d/%d] Starting %s", i, total, r["name"])
        t0 = time.monotonic()
        result = run(**r)
        elapsed = time.monotonic() - t0
        if result.returncode == 130:
            log.info("Interrupted. Exiting.")
            sys.exit(130)
        if result.returncode != 0:
            log.error("%s failed with code %d", r["name"], result.returncode)
            sys.exit(result.returncode)
        log.info("[%d/%d] %s done in %.0fs", i, total, r["name"], elapsed)


def check_probes() -> None:
    """Analyze probe results: entropy and compressibility of last 1k tokens."""
    from utils import compressibility as comp_fn
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))

    probes = [
        "probe_L256", "probe_L064_T060", "probe_L064_T080",
        "probe_L064_T100", "probe_L032_T060", "probe_young_L064",
    ]

    print(f"\n{'Probe':<22} {'Status':<8} {'Ent(last1k)':<12} {'Comp(last1k)':<13} {'Top token':<20} {'Top %':<8}")
    print("-" * 85)

    for name in probes:
        path = RUNS_DIR / f"{name}.parquet"
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
    for L, tag in [(256, "control"), (192, "stuck"), (128, "border"), (64, "escape")]
]

TIER2 = [
    dict(name=f"anneal_cycle_short_S{s}",
         schedule="20000:L256:T0.60,10000:L64:T0.60,70000:L256:T0.60",
         N=100000, prefill=" Star Wars", seed=s)
    for s in TIER1_SEEDS
] + [
    dict(name=f"anneal_cycle_long_S{s}",
         schedule="20000:L256:T0.60,30000:L64:T0.60,50000:L256:T0.60",
         N=100000, prefill=" Star Wars", seed=s)
    for s in TIER1_SEEDS
] + [
    dict(name=f"anneal_cycle_brief_S{s}",
         schedule="20000:L256:T0.60,3000:L64:T0.60,77000:L256:T0.60",
         N=100000, prefill=" Star Wars", seed=s)
    for s in TIER1_SEEDS
] + [
    dict(name=f"anneal_no_return_S{s}",
         schedule="20000:L256:T0.60,80000:L64:T0.60",
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
         schedule="20000:L256:T0.60,10000:L64:T0.60,70000:L256:T0.60",
         N=100000, prefill=" Star Wars", seed=s)
    for s in [42]
] + [
    dict(name=f"anneal_both_S{s}",
         schedule="20000:L256:T0.60,10000:L64:T1.00,70000:L256:T0.60",
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

    if args.phase == "probes" and args.check:
        check_probes()
        return

    if args.dry_run:
        for r in runs:
            status = "DONE" if is_done(r["name"]) else "TODO"
            sched = r.get("schedule", f"L{r.get('L')} T{r.get('T')}")
            print(f"  [{status}] {r['name']:<35} {sched}")
        done = sum(1 for r in runs if is_done(r["name"]))
        print(f"\n{done}/{len(runs)} complete")
        return

    run_batch(runs)

    if args.phase == "probes":
        log.info("Probes complete. Run 'python anneal.py probes --check' to analyze.")


if __name__ == "__main__":
    main()

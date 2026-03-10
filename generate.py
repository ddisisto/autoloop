"""Core generation loop for autoloop experiments.

Runs closed-loop autoregressive generation with optional pre-seeded context
and per-segment schedule control of temperature (T) and context length (L).

Every run is a schedule — fixed-parameter runs are single-segment schedules.
Checkpoints every 1k steps for resume and run extension.
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import disable_progress_bar

from utils import compressibility, fix_decoded_texts

log = logging.getLogger(__name__)

CHECKPOINT_INTERVAL = 1000


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Segment:
    steps: int
    L: int
    T: float


@dataclasses.dataclass
class Schedule:
    segments: list[Segment]

    @property
    def total_steps(self) -> int:
        return sum(seg.steps for seg in self.segments)

    def at_step(self, step: int) -> Segment:
        """Return the segment active at a 1-based experiment step."""
        remaining = step
        for seg in self.segments:
            if remaining <= seg.steps:
                return seg
            remaining -= seg.steps
        raise ValueError(f"Step {step} exceeds schedule total {self.total_steps}")

    def to_spec(self) -> str:
        """Serialize back to CLI spec format."""
        return ",".join(f"{s.steps}:L{s.L}:T{s.T:.2f}" for s in self.segments)


def parse_schedule(spec: str) -> Schedule:
    """Parse '50000:L256:T0.60,10000:L64:T0.60' into a Schedule."""
    segments = []
    for part in spec.split(","):
        tokens = part.strip().split(":")
        if len(tokens) != 3:
            raise ValueError(f"Bad segment '{part}': expected 'steps:L{{n}}:T{{f}}'")
        steps = int(tokens[0])
        L = int(tokens[1].lstrip("L"))
        T = float(tokens[2].lstrip("T"))
        segments.append(Segment(steps=steps, L=L, T=T))
    return Schedule(segments=segments)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoloop generation run")

    # Fixed-parameter mode (original interface)
    parser.add_argument("--context-length", type=int, help="Context window size L")
    parser.add_argument("--temperature", type=float, help="Sampling temperature T")
    parser.add_argument("--num-tokens", type=int, help="Number of experiment tokens N")

    # Schedule mode
    parser.add_argument("--schedule", type=str,
                        help="Schedule spec: 'steps:L{n}:T{f},...' (mutually exclusive with --context-length/--temperature/--num-tokens)")

    # Prefill
    parser.add_argument("--prefill-text", type=str,
                        help="Pre-seed context by repeating this text to fill L (skips generative prefill)")

    # Common
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to local model directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output files")
    parser.add_argument("--device", type=str, required=True, help="Torch device (e.g. cuda, cpu)")
    parser.add_argument("--run-name", type=str, help="Override auto-generated run name")

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Check mutual exclusion and required combinations."""
    fixed = [args.context_length, args.temperature, args.num_tokens]
    has_fixed = any(v is not None for v in fixed)
    has_schedule = args.schedule is not None

    if has_schedule and has_fixed:
        raise SystemExit("Error: --schedule is mutually exclusive with --context-length/--temperature/--num-tokens")
    if not has_schedule and not all(v is not None for v in fixed):
        raise SystemExit("Error: --context-length, --temperature, and --num-tokens are all required without --schedule")
    if not has_schedule and not has_fixed:
        raise SystemExit("Error: provide either --schedule or --context-length/--temperature/--num-tokens")


def build_schedule(args: argparse.Namespace) -> Schedule:
    """Build Schedule from either --schedule spec or fixed parameters."""
    if args.schedule:
        return parse_schedule(args.schedule)
    return Schedule([Segment(steps=args.num_tokens, L=args.context_length, T=args.temperature)])


def build_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    if args.schedule:
        h = hashlib.sha256(args.schedule.encode()).hexdigest()[:8]
        return f"sched_S{args.seed}_{h}"
    return f"L{args.context_length:04d}_T{args.temperature:.2f}_S{args.seed}"


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(
    checkpoint_path: Path,
    parquet_path: Path,
    step: int,
    context: torch.Tensor,
    records: list[dict],
    schedule_spec: str,
) -> None:
    torch.save({
        "step": step,
        "context": context.cpu(),
        "records": records,
        "rng_torch": torch.random.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "rng_numpy": np.random.get_state(),
        "schedule_spec": schedule_spec,
    }, checkpoint_path)
    pd.DataFrame(records).to_parquet(parquet_path, index=False)


def load_checkpoint(checkpoint_path: Path, device: str) -> dict:
    ckpt = torch.load(checkpoint_path, weights_only=False)
    torch.random.set_rng_state(ckpt["rng_torch"])
    if ckpt["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["rng_cuda"])
    np.random.set_state(ckpt["rng_numpy"])
    ckpt["context"] = ckpt["context"].to(device)
    return ckpt


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def compute_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy of the softmax distribution (in nats)."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs).item()


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def run_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    schedule: Schedule,
    seed: int,
    device: str,
    checkpoint_path: Path,
    parquet_path: Path,
    prefill_ids: list[int] | None = None,
) -> tuple[list[dict], bool]:
    """Run generation loop with schedule and optional pre-seeded context.

    Returns (records, interrupted).
    """
    # Prefill: generative (BOS → L tokens) or pre-seeded (skip prefill)
    prefill_steps = 0 if prefill_ids is not None else schedule.segments[0].L
    total_steps = prefill_steps + schedule.total_steps
    schedule_spec = schedule.to_spec()
    eos_token_id = tokenizer.eos_token_id

    # Resume from checkpoint or start fresh
    if checkpoint_path.exists():
        ckpt = load_checkpoint(checkpoint_path, device)
        saved_spec = ckpt.get("schedule_spec")
        if saved_spec is not None and saved_spec != schedule_spec:
            raise SystemExit(
                f"Error: checkpoint schedule '{saved_spec}' != current '{schedule_spec}'. "
                "Delete checkpoint to restart, or use the original schedule to resume."
            )
        start_step = ckpt["step"] + 1
        context = ckpt["context"]
        records = ckpt["records"]
        n_experiment = sum(1 for r in records if r["phase"] == "experiment")
        n_total_exp = schedule.total_steps
        log.info("Resumed from checkpoint at step %d (%d/%d experiment tokens, %.0f%% complete)",
                 ckpt["step"], n_experiment, n_total_exp, 100 * n_experiment / n_total_exp)
    elif prefill_ids is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        initial_L = schedule.segments[0].L
        ids = prefill_ids[:initial_L]
        context = torch.tensor([ids], dtype=torch.long, device=device)
        records = []
        start_step = 1
        log.info("Pre-seeded context with %d tokens (L=%d)", len(ids), initial_L)
    else:
        torch.manual_seed(seed)
        np.random.seed(seed)
        context = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
        records = []
        start_step = 1

    interval_start = time.monotonic()
    interrupted = False
    last_step = start_step - 1

    try:
        for t in range(start_step, total_steps + 1):
            if t <= prefill_steps:
                phase = "prefill"
                seg = schedule.segments[0]
            else:
                phase = "experiment"
                seg = schedule.at_step(t - prefill_steps)

            current_L = seg.L
            current_T = seg.T

            with torch.no_grad():
                outputs = model(input_ids=context)
            logits = outputs.logits[0, -1, :]

            entropy = compute_entropy(logits)

            scaled_logits = logits / current_T
            probs = torch.softmax(scaled_logits, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()

            log_prob = torch.log_softmax(scaled_logits, dim=-1)[token_id].item()

            decoded_text = tokenizer.decode([token_id])
            is_eos = token_id == eos_token_id

            records.append({
                "step": t,
                "phase": phase,
                "token_id": token_id,
                "decoded_text": decoded_text,
                "entropy": entropy,
                "log_prob": log_prob,
                "temperature": current_T,
                "context_length": current_L,
                "eos": is_eos,
            })

            new_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
            context = torch.cat([context, new_token], dim=1)
            if context.shape[1] > current_L:
                context = context[:, -current_L:]

            last_step = t

            if t % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(checkpoint_path, parquet_path, t, context, records, schedule_spec)
                now = time.monotonic()
                tok_s = CHECKPOINT_INTERVAL / (now - interval_start)
                recent = records[-current_L:]
                ent_vals = [r["entropy"] for r in recent]
                ent_mean = sum(ent_vals) / len(ent_vals)
                trail_ids = [r["token_id"] for r in recent]
                trail_text = tokenizer.decode(trail_ids)
                comp = compressibility(trail_text.encode("utf-8"))
                log.info(
                    "step %d/%d (%.0f%%) | %.1f tok/s | L=%d T=%.2f | ent=%.2f comp=%.3f",
                    t, total_steps, 100 * t / total_steps, tok_s,
                    current_L, current_T, ent_mean, comp,
                )
                interval_start = now

            log.debug(
                "step=%d phase=%s token=%d text=%r entropy=%.4f log_prob=%.4f eos=%s",
                t, phase, token_id, decoded_text, entropy, log_prob, is_eos,
            )
    except KeyboardInterrupt:
        interrupted = True
        n_experiment = sum(1 for r in records if r["phase"] == "experiment")
        n_total_exp = schedule.total_steps
        log.info("Interrupted at step %d (%d/%d experiment tokens, %.0f%% complete)",
                 last_step, n_experiment, n_total_exp, 100 * n_experiment / n_total_exp)
        log.info("Saving checkpoint...")
        save_checkpoint(checkpoint_path, parquet_path, last_step, context, records, schedule_spec)
        log.info("Checkpoint saved. Resume with the same command.")

    if not interrupted:
        all_ids = [r["token_id"] for r in records]
        all_texts = [r["decoded_text"] for r in records]
        fixed = fix_decoded_texts(tokenizer, all_ids, all_texts)
        for r, txt in zip(records, fixed):
            r["decoded_text"] = txt
        save_checkpoint(checkpoint_path, parquet_path, total_steps, context, records, schedule_spec)

    return records, interrupted


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    validate_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    schedule = build_schedule(args)
    run_name = build_run_name(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{run_name}.parquet"
    meta_path = output_dir / f"{run_name}.meta.json"
    checkpoint_path = output_dir / f"{run_name}.ckpt"

    log.info("Starting run: %s", run_name)
    log.info("Schedule: %s", schedule.to_spec())
    if args.prefill_text:
        log.info("Prefill text: %r", args.prefill_text)

    log.info("Loading model from %s", args.model_dir)
    disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32,
    ).to(args.device)
    model.eval()
    log.info("Model loaded: %s parameters", f"{sum(p.numel() for p in model.parameters()):,}")

    # Build prefill token IDs if requested
    prefill_ids = None
    if args.prefill_text:
        token_ids = tokenizer.encode(args.prefill_text, add_special_tokens=False)
        if not token_ids:
            raise SystemExit(f"Error: --prefill-text '{args.prefill_text}' encodes to zero tokens")
        initial_L = schedule.segments[0].L
        repeats = (initial_L // len(token_ids)) + 1
        prefill_ids = (token_ids * repeats)[:initial_L]
        log.info("Prefill: %d tokens repeated to fill L=%d (%r → %d token IDs)",
                 len(token_ids), initial_L, args.prefill_text, len(prefill_ids))

    t_start = time.monotonic()
    records, interrupted = run_generation(
        model=model,
        tokenizer=tokenizer,
        schedule=schedule,
        seed=args.seed,
        device=args.device,
        checkpoint_path=checkpoint_path,
        parquet_path=parquet_path,
        prefill_ids=prefill_ids,
    )
    elapsed = time.monotonic() - t_start

    if interrupted:
        sys.exit(130)

    # Metadata sidecar
    metadata = {
        "run_name": run_name,
        "seed": args.seed,
        "schedule": [{"steps": s.steps, "L": s.L, "T": s.T} for s in schedule.segments],
        "prefill_text": args.prefill_text,
        "device": args.device,
        "model_dir": args.model_dir,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round(len(records) / elapsed, 1),
        "total_steps": len(records),
    }
    # Backward-compatible fields for single-segment runs
    if len(schedule.segments) == 1:
        seg = schedule.segments[0]
        metadata["context_length"] = seg.L
        metadata["temperature"] = seg.T
        metadata["num_tokens"] = seg.steps
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    log.info("Wrote metadata to %s", meta_path)

    log.info(
        "Done: %d steps in %.1fs (%.1f tok/s)",
        len(records), elapsed, len(records) / elapsed,
    )


if __name__ == "__main__":
    main()

"""Closed-loop controller for autoregressive generation.

Generates tokens in segments, reads sensors after each segment, and adjusts
L and T to hill-climb toward a target objective (default: Heaps' β ≈ 1.0).

Architecture:
- Runs generation in-process (no subprocess overhead)
- Each segment is SEGMENT_STEPS tokens at fixed L/T
- After each segment: compute sensors, decide next L/T
- If a segment "fails" (objective regresses past threshold), rollback to
  the pre-segment checkpoint and try a different L/T
- Builds schedule dynamically, saves standard parquet + metadata

Usage:
    python controller.py --seed 42 --total-steps 100000
    python controller.py --seed 42 --total-steps 50000 --segment-steps 5000
    python controller.py --seed 42 --total-steps 100000 --start-L 128 --start-T 0.80
    python controller.py --seed 42 --total-steps 100000 --dry-run  # sensors only, no adjustment
"""

import argparse
import dataclasses
import json
import logging
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import disable_progress_bar

from generate import compute_entropy, save_checkpoint, load_checkpoint
from utils import compressibility, fix_decoded_texts
import runlib

log = logging.getLogger(__name__)

MODEL_DIR = "data/model/SmolLM-135M"
OUTPUT_DIR = runlib.CONTROLLER_DIR
DEVICE = "cuda"

# Defaults
DEFAULT_SEGMENT_STEPS = 1000
DEFAULT_TOTAL_STEPS = 100_000
DEFAULT_START_L = 64
DEFAULT_START_T = 0.70

# L/T search grid
L_MIN = 16
L_MAX = 1024
L_STEP = 16
T_MIN = 0.55
T_MAX = 1.25
T_STEP = 0.05

# Target
BETA_TARGET = 0.9

# Drift: slow pressure toward the edge when β is in the dead zone
DRIFT_T_STEP = 0.005   # cool T by this much per segment
DRIFT_L_STEP = 2       # grow L by this much per segment


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SensorReading:
    """Sensor snapshot after a segment."""
    step: int
    L: int
    T: float
    entropy_mean: float
    entropy_std: float
    comp_W64: float
    heaps_beta: float
    n_words: int
    n_unique: int


def read_sensors(
    records: list[dict],
    window: int = 0,
    segment_steps: int = 0,
) -> SensorReading:
    """Compute sensor values from the last `window` records.

    If window is 0, defaults to 5× segment_steps (or 2000 if segment_steps is also 0).
    Uses a trailing window so sensors reflect recent dynamics, not full-run average.
    """
    if window == 0:
        window = max(5 * segment_steps, 500) if segment_steps > 0 else 2000
    tail = records[-window:] if len(records) > window else records
    exp_tail = [r for r in tail if r["phase"] == "experiment"]
    if not exp_tail:
        exp_tail = tail

    # Entropy
    ent = [r["entropy"] for r in exp_tail]
    ent_mean = sum(ent) / len(ent)
    ent_std = (sum((e - ent_mean) ** 2 for e in ent) / len(ent)) ** 0.5

    # Compressibility (W=64 from trailing text)
    texts = [r["decoded_text"] for r in exp_tail]
    chunk = "".join(texts[-64:]) if len(texts) >= 64 else "".join(texts)
    comp_w64 = compressibility(chunk.encode("utf-8")) if len(chunk) > 10 else 0.0

    # Heaps' β from trailing window
    text = "".join(texts)
    words = [w.lower() for w in text.split() if len(w) > 1]
    n_words = len(words)

    beta = 0.0
    n_unique = 0
    if n_words >= 50:
        seen: set[str] = set()
        checkpoints = 20
        step_size = max(1, n_words // checkpoints)
        ns, vs = [], []
        for i, w in enumerate(words):
            seen.add(w)
            if (i + 1) % step_size == 0:
                ns.append(i + 1)
                vs.append(len(seen))
        n_unique = len(seen)

        if len(ns) >= 3:
            log_n = np.log(np.array(ns, dtype=float))
            log_v = np.log(np.array(vs, dtype=float))
            n_pts = len(log_n)
            sum_x = log_n.sum()
            sum_y = log_v.sum()
            sum_xy = (log_n * log_v).sum()
            sum_x2 = (log_n ** 2).sum()
            denom = n_pts * sum_x2 - sum_x ** 2
            if abs(denom) > 1e-12:
                beta = float((n_pts * sum_xy - sum_x * sum_y) / denom)

    last = exp_tail[-1] if exp_tail else records[-1]
    return SensorReading(
        step=last["step"],
        L=last["context_length"],
        T=last["temperature"],
        entropy_mean=ent_mean,
        entropy_std=ent_std,
        comp_W64=comp_w64,
        heaps_beta=beta,
        n_words=n_words,
        n_unique=n_unique,
    )


# ---------------------------------------------------------------------------
# Controller logic
# ---------------------------------------------------------------------------

def decide_next(
    current_L: int,
    current_T: float,
    sensors: SensorReading,
    history: list[SensorReading],
    drift: bool = False,
) -> tuple[int, float, str]:
    """Decide next L/T based on sensor readings. Returns (L, T, reason).

    Strategy: T is the fast control (adjust every segment), L is the slow
    control (adjust when T is saturated or β is persistently off-target).

    β < target → need more novelty → raise T, then raise L if T-capped
    β > target → too much novelty → lower T, then lower L if T-floored

    With drift=True: when β is in the dead zone, apply slow pressure toward
    the edge. Entropy relative to its rolling mean decides which lever:
      high entropy → grow L (system has thermal energy, give it more memory)
      low entropy  → cool T (system is already structured, tighten further)
    """
    beta = sensors.heaps_beta
    err = beta - BETA_TARGET  # negative = need more novelty

    # Dead zone: β within ±0.15 of target
    if abs(err) < 0.15:
        if not drift or len(history) < 3:
            return current_L, current_T, f"β={beta:.2f} in zone, hold"

        # Drift: compare current entropy to recent rolling mean (last 20 segments)
        recent = history[-20:]
        ent_mean = sum(h.entropy_mean for h in recent) / len(recent)
        if sensors.entropy_mean >= ent_mean:
            # High entropy → system has energy → grow L
            new_L = min(current_L + DRIFT_L_STEP, L_MAX)
            if new_L != current_L:
                return new_L, current_T, (
                    f"β={beta:.2f} in zone, drift L↑{new_L} "
                    f"(ent={sensors.entropy_mean:.2f}≥avg={ent_mean:.2f})")
            # L maxed — fall through to cool T
        # Low entropy (or L maxed) → cool T
        new_T = max(current_T - DRIFT_T_STEP, T_MIN)
        if abs(new_T - current_T) > 1e-6:
            return current_L, new_T, (
                f"β={beta:.2f} in zone, drift T↓{new_T:.3f} "
                f"(ent={sensors.entropy_mean:.2f}<avg={ent_mean:.2f})")
        return current_L, current_T, f"β={beta:.2f} in zone, drift stalled (T@min)"

    new_T = current_T
    new_L = current_L
    reason_parts = []

    t_at_ceiling = current_T >= T_MAX - 1e-6
    t_at_floor = current_T <= T_MIN + 1e-6

    if err < -0.15:
        # β too low — need more novelty
        if not t_at_ceiling:
            new_T = min(current_T + T_STEP, T_MAX)
            reason_parts.append(f"β={beta:.2f}<zone, T↑{new_T:.2f}")
        else:
            # T maxed out — increase L to lift β ceiling
            new_L = min(current_L + L_STEP, L_MAX)
            if new_L != current_L:
                reason_parts.append(f"β={beta:.2f}<zone, T@max, L↑{new_L}")
            else:
                reason_parts.append(f"β={beta:.2f}<zone, T@max L@max, stuck")
    elif err > 0.15:
        # β too high — too much novelty
        if not t_at_floor:
            new_T = max(current_T - T_STEP, T_MIN)
            reason_parts.append(f"β={beta:.2f}>zone, T↓{new_T:.2f}")
        else:
            # T floored — decrease L to constrain
            new_L = max(current_L - L_STEP, L_MIN)
            if new_L != current_L:
                reason_parts.append(f"β={beta:.2f}>zone, T@min, L↓{new_L}")
            else:
                reason_parts.append(f"β={beta:.2f}>zone, T@min L@min, stuck")

    reason = "; ".join(reason_parts) if reason_parts else f"β={beta:.2f}, no adjustment"
    return new_L, new_T, reason


def should_rollback(
    sensors: SensorReading,
    prev_sensors: SensorReading | None,
) -> bool:
    """Check if current segment should be rolled back.

    Rollback if entropy collapsed (deep attractor) or β crashed.
    """
    if prev_sensors is None:
        return False
    # Entropy crash: dropped below 1.0 (deep collapse)
    if sensors.entropy_mean < 1.0 and prev_sensors.entropy_mean > 2.0:
        return True
    # β crash: dropped from healthy to near-zero
    if sensors.heaps_beta < 0.2 and prev_sensors.heaps_beta > 0.5:
        return True
    return False


# ---------------------------------------------------------------------------
# Generation segment
# ---------------------------------------------------------------------------

def run_segment(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: torch.Tensor,
    records: list[dict],
    L: int,
    T: float,
    n_steps: int,
    device: str,
    start_step: int,
) -> torch.Tensor:
    """Generate n_steps tokens at L/T, appending to records. Returns updated context."""
    eos_token_id = tokenizer.eos_token_id

    for i in range(n_steps):
        step = start_step + i

        with torch.no_grad():
            outputs = model(input_ids=context)
        logits = outputs.logits[0, -1, :]

        entropy = compute_entropy(logits)
        scaled_logits = logits / T
        probs = torch.softmax(scaled_logits, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).item()
        log_prob = torch.log_softmax(scaled_logits, dim=-1)[token_id].item()
        decoded_text = tokenizer.decode([token_id])
        is_eos = token_id == eos_token_id

        records.append({
            "step": step,
            "phase": "experiment",
            "token_id": token_id,
            "decoded_text": decoded_text,
            "entropy": entropy,
            "log_prob": log_prob,
            "temperature": T,
            "context_length": L,
            "eos": is_eos,
        })

        new_token = torch.tensor([[token_id]], dtype=torch.long, device=device)
        context = torch.cat([context, new_token], dim=1)
        if context.shape[1] > L:
            context = context[:, -L:]

    return context


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_controller(
    seed: int,
    total_steps: int,
    segment_steps: int,
    start_L: int,
    start_T: float,
    dry_run: bool = False,
    drift: bool = False,
) -> None:
    suffix = "d" if drift else ""
    run_name = f"ctrl{suffix}_S{seed}_{start_L}_{start_T:.2f}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_DIR / f"{run_name}.parquet"
    meta_path = OUTPUT_DIR / f"{run_name}.meta.json"
    checkpoint_path = OUTPUT_DIR / f"{run_name}.ckpt"

    log.info("Controller run: %s", run_name)
    log.info("Target: β ≈ %.1f | Segments: %d steps | Total: %d steps | Drift: %s",
             BETA_TARGET, segment_steps, total_steps, drift)

    # Load model
    log.info("Loading model from %s", MODEL_DIR)
    disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32,
    ).to(DEVICE)
    model.eval()
    log.info("Model loaded: %s parameters",
             f"{sum(p.numel() for p in model.parameters()):,}")

    # Init
    torch.manual_seed(seed)
    np.random.seed(seed)
    context = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=DEVICE)
    records: list[dict] = []
    sensor_history: list[SensorReading] = []


    current_L = start_L
    current_T = start_T
    current_step = 1

    # Periodic save interval (in segments)
    SAVE_EVERY = 50  # ~50k steps at default segment_steps=1000

    def save_snapshot(final: bool = False) -> None:
        """Write parquet and metadata to disk."""
        all_ids = [r["token_id"] for r in records]
        all_texts = [r["decoded_text"] for r in records]
        fixed = fix_decoded_texts(tokenizer, all_ids, all_texts)
        snap_records = [dict(r) for r in records]
        for r, txt in zip(snap_records, fixed):
            r["decoded_text"] = txt

        snap_df = pd.DataFrame(snap_records)
        snap_df.to_parquet(parquet_path, index=False)

        elapsed = time.monotonic() - t_start if t_start else 0
        metadata = {
            "run_name": run_name,
            "controller": True,
            "seed": seed,
            "total_steps": total_steps,
            "segment_steps": segment_steps,
            "start_L": start_L,
            "start_T": start_T,
            "beta_target": BETA_TARGET,
            "drift": drift,
            "n_segments": len(sensor_history),
            "n_rollbacks": n_rollbacks,
            "dry_run": dry_run,
            "final_L": current_L,
            "final_T": current_T,
            "final_beta": sensor_history[-1].heaps_beta if sensor_history else None,
            "final_entropy": sensor_history[-1].entropy_mean if sensor_history else None,
            "device": DEVICE,
            "model_dir": MODEL_DIR,
            "torch_version": torch.__version__,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(len(records) / elapsed, 1) if elapsed > 0 else 0,
            "num_tokens": experiment_steps,
            "complete": final,

        }
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

        label = "Final" if final else "Snapshot"
        log.info("%s save: %s (%d records)",
                 label, parquet_path.name, len(records))

    # Prefill: generate L tokens at current T to fill the context
    log.info("Prefill: %d tokens at L=%d T=%.2f", start_L, start_L, start_T)
    context = run_segment(
        model, tokenizer, context, records,
        L=start_L, T=start_T, n_steps=start_L,
        device=DEVICE, start_step=current_step,
    )
    # Mark prefill records
    for r in records:
        r["phase"] = "prefill"
    current_step += start_L

    t_start = time.monotonic()
    experiment_steps = 0
    n_rollbacks = 0

    while experiment_steps < total_steps:
        seg_size = min(segment_steps, total_steps - experiment_steps)

        # Save pre-segment state for potential rollback
        pre_records_len = len(records)
        pre_context = context.clone()
        pre_rng_torch = torch.random.get_rng_state()
        pre_rng_cuda = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        pre_rng_numpy = np.random.get_state()

        # Run segment
        seg_start = time.monotonic()
        context = run_segment(
            model, tokenizer, context, records,
            L=current_L, T=current_T, n_steps=seg_size,
            device=DEVICE, start_step=current_step,
        )
        seg_elapsed = time.monotonic() - seg_start
        tok_s = seg_size / seg_elapsed

        experiment_steps += seg_size
        current_step += seg_size

        # Read sensors
        sensors = read_sensors(records, segment_steps=segment_steps)
        sensor_history.append(sensors)

        # Check rollback
        prev_sensors = sensor_history[-2] if len(sensor_history) >= 2 else None
        if not dry_run and should_rollback(sensors, prev_sensors):
            log.info(
                "ROLLBACK at step %d: β=%.2f ent=%.1f (was β=%.2f ent=%.1f)",
                current_step, sensors.heaps_beta, sensors.entropy_mean,
                prev_sensors.heaps_beta, prev_sensors.entropy_mean,
            )
            # Restore state
            records[:] = records[:pre_records_len]
            context = pre_context
            torch.random.set_rng_state(pre_rng_torch)
            if pre_rng_cuda is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(pre_rng_cuda)
            np.random.set_state(pre_rng_numpy)
            experiment_steps -= seg_size
            current_step -= seg_size
            sensor_history.pop()
            n_rollbacks += 1

            # After rollback: more aggressive T increase
            current_T = min(current_T + T_STEP * 2, max(T_OPTIONS))
            reason = f"rollback, T→{current_T:.2f}"
            log.info("  → %s", reason)
            continue

        # Decide next L/T
        if dry_run:
            new_L, new_T, reason = current_L, current_T, "dry-run, no adjustment"
        else:
            new_L, new_T, reason = decide_next(
                current_L, current_T, sensors, sensor_history, drift=drift
            )

        pct = 100 * experiment_steps / total_steps
        log.info(
            "step %d/%d (%.0f%%) | %.0f tok/s | L=%d T=%.2f | "
            "ent=%.2f β=%.2f comp=%.3f | %s",
            experiment_steps, total_steps, pct, tok_s,
            current_L, current_T,
            sensors.entropy_mean, sensors.heaps_beta, sensors.comp_W64,
            reason,
        )

        current_L = new_L
        current_T = new_T

        # Checkpoint every 10 segments, snapshot every SAVE_EVERY segments
        n_seg = len(sensor_history)
        if n_seg % 10 == 0:
            save_checkpoint(
                checkpoint_path, parquet_path,
                current_step, context, records,
                f"controller:{run_name}",
            )
        if n_seg % SAVE_EVERY == 0:
            save_snapshot()

    # Final save
    save_snapshot(final=True)
    elapsed = time.monotonic() - t_start
    log.info("Done: %d steps in %.1fs (%.1f tok/s), %d rollbacks",
             len(records), elapsed, len(records) / elapsed, n_rollbacks)
    log.info("Final: L=%d T=%.2f β=%.2f ent=%.2f",
             current_L, current_T,
             sensor_history[-1].heaps_beta, sensor_history[-1].entropy_mean)
    log.info("Output: %s", parquet_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop controller")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--segment-steps", type=int, default=DEFAULT_SEGMENT_STEPS)
    parser.add_argument("--start-L", type=int, default=DEFAULT_START_L)
    parser.add_argument("--start-T", type=float, default=DEFAULT_START_T)
    parser.add_argument("--dry-run", action="store_true",
                        help="Run sensors but don't adjust L/T")
    parser.add_argument("--drift", action="store_true",
                        help="Slow pressure when in dead zone: grow L (high entropy) or cool T (low entropy)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    run_controller(
        seed=args.seed,
        total_steps=args.total_steps,
        segment_steps=args.segment_steps,
        start_L=args.start_L,
        start_T=args.start_T,
        dry_run=args.dry_run,
        drift=args.drift,
    )


if __name__ == "__main__":
    main()

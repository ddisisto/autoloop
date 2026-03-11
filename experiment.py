"""Experiment framework: controllers + universal run loop.

An experiment is a controller (decision function) driven by a universal loop.
The loop generates segments, reads sensors, and asks the controller what to
do next.

Controllers range from trivial (hold L/T fixed) to complex (state machines
with sensor-driven transitions).

Usage:
    # Fixed-parameter run
    python experiment.py fixed --seed 42 -L 64 -T 0.50 --total-steps 100000

    # Schedule run
    python experiment.py schedule --seed 42 --spec "50000:L256:T0.60,50000:L64:T0.80"

    # Beta controller (replaces controller.py)
    python experiment.py beta --seed 42 --start-L 8 --start-T 1.00 --drift

    # Dry run (sensors only, no control)
    python experiment.py beta --seed 42 --dry-run
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch

from engine import SensorReading, StepEngine, load_model
import runlib

log = logging.getLogger(__name__)

MODEL_DIR = "data/model/SmolLM-135M"
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Action: what a controller returns
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """Controller decision after a segment."""
    L: int
    T: float
    reason: str
    rollback: bool = False


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

class FixedController:
    """Hold L/T constant. Used for fixed-parameter and schedule runs."""

    def __init__(self, L: int, T: float):
        self.L = L
        self.T = T

    def decide(
        self, sensors: SensorReading, history: list[SensorReading],
        experiment_steps: int,
    ) -> Action:
        return Action(L=self.L, T=self.T, reason="fixed")


class ScheduleController:
    """Follow a predefined schedule of (steps, L, T) segments."""

    def __init__(self, segments: list[tuple[int, int, float]]):
        """segments: list of (n_steps, L, T) — not cumulative."""
        self._breakpoints: list[tuple[int, int, float]] = []
        cumulative = 0
        for steps, L, T in segments:
            cumulative += steps
            self._breakpoints.append((cumulative, L, T))

    def decide(
        self, sensors: SensorReading, history: list[SensorReading],
        experiment_steps: int,
    ) -> Action:
        for end, L, T in self._breakpoints:
            if experiment_steps <= end:
                return Action(L=L, T=T, reason=f"schedule L={L} T={T:.2f}")
        # Past end — hold last
        _, L, T = self._breakpoints[-1]
        return Action(L=L, T=T, reason="schedule (past end)")


class BetaController:
    """Hill-climb L/T toward target Heaps' beta.

    T is the fast control (adjust every segment), L is the slow control
    (adjust when T is saturated). With drift=True, applies slow pressure
    when beta is in the dead zone.
    """

    def __init__(
        self,
        target: float = 0.9,
        dead_zone: float = 0.15,
        drift: bool = False,
        L_min: int = 16, L_max: int = 1024, L_step: int = 16,
        T_min: float = 0.55, T_max: float = 1.25, T_step: float = 0.05,
        drift_T_step: float = 0.005, drift_L_step: int = 2,
    ):
        self.target = target
        self.dead_zone = dead_zone
        self.drift = drift
        self.L_min, self.L_max, self.L_step = L_min, L_max, L_step
        self.T_min, self.T_max, self.T_step = T_min, T_max, T_step
        self.drift_T_step = drift_T_step
        self.drift_L_step = drift_L_step

    def _should_rollback(
        self, sensors: SensorReading, history: list[SensorReading],
    ) -> bool:
        if len(history) < 2:
            return False
        prev = history[-2]
        if sensors.entropy_mean < 1.0 and prev.entropy_mean > 2.0:
            return True
        if sensors.heaps_beta < 0.2 and prev.heaps_beta > 0.5:
            return True
        return False

    def decide(
        self, sensors: SensorReading, history: list[SensorReading],
        experiment_steps: int,
    ) -> Action:
        beta = sensors.heaps_beta
        current_L, current_T = sensors.L, sensors.T
        err = beta - self.target

        # Rollback check
        if self._should_rollback(sensors, history):
            new_T = min(current_T + self.T_step * 2, self.T_max)
            return Action(L=current_L, T=new_T,
                          reason=f"rollback β={beta:.2f} ent={sensors.entropy_mean:.1f}, "
                                 f"T→{new_T:.2f}",
                          rollback=True)

        # Dead zone
        if abs(err) < self.dead_zone:
            if not self.drift or len(history) < 3:
                return Action(L=current_L, T=current_T,
                              reason=f"β={beta:.2f} in zone, hold")
            # Drift logic
            recent = history[-20:]
            ent_mean = sum(h.entropy_mean for h in recent) / len(recent)
            if sensors.entropy_mean >= ent_mean:
                new_L = min(current_L + self.drift_L_step, self.L_max)
                if new_L != current_L:
                    return Action(
                        L=new_L, T=current_T,
                        reason=f"β={beta:.2f} in zone, drift L↑{new_L} "
                               f"(ent={sensors.entropy_mean:.2f}≥avg={ent_mean:.2f})")
            new_T = max(current_T - self.drift_T_step, self.T_min)
            if abs(new_T - current_T) > 1e-6:
                return Action(
                    L=current_L, T=new_T,
                    reason=f"β={beta:.2f} in zone, drift T↓{new_T:.3f} "
                           f"(ent={sensors.entropy_mean:.2f}<avg={ent_mean:.2f})")
            return Action(L=current_L, T=current_T,
                          reason=f"β={beta:.2f} in zone, drift stalled (T@min)")

        # Active control
        new_T, new_L = current_T, current_L
        t_at_ceiling = current_T >= self.T_max - 1e-6
        t_at_floor = current_T <= self.T_min + 1e-6

        if err < -self.dead_zone:
            if not t_at_ceiling:
                new_T = min(current_T + self.T_step, self.T_max)
                reason = f"β={beta:.2f}<zone, T↑{new_T:.2f}"
            else:
                new_L = min(current_L + self.L_step, self.L_max)
                reason = (f"β={beta:.2f}<zone, T@max, L↑{new_L}" if new_L != current_L
                          else f"β={beta:.2f}<zone, T@max L@max, stuck")
        else:
            if not t_at_floor:
                new_T = max(current_T - self.T_step, self.T_min)
                reason = f"β={beta:.2f}>zone, T↓{new_T:.2f}"
            else:
                new_L = max(current_L - self.L_step, self.L_min)
                reason = (f"β={beta:.2f}>zone, T@min, L↓{new_L}" if new_L != current_L
                          else f"β={beta:.2f}>zone, T@min L@min, stuck")

        return Action(L=new_L, T=new_T, reason=reason)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """Sensor-driven state transition."""
    target: str
    condition: Callable[[SensorReading, list[SensorReading]], bool]
    reason: str = ""


@dataclass
class MachineState:
    """A state in a state machine experiment."""
    name: str
    controller: object  # anything with .decide(sensors, history, exp_steps)
    transitions: list[Transition] = field(default_factory=list)


class StateMachine:
    """Controller composed of named states with sensor-driven transitions.

    Each state has its own sub-controller for L/T decisions. Transitions
    are checked after each segment; the first matching transition fires.
    """

    def __init__(self, states: dict[str, MachineState], initial: str):
        self.states = states
        self.current = initial
        self.transition_log: list[tuple[int, str, str, str]] = []

    def decide(
        self, sensors: SensorReading, history: list[SensorReading],
        experiment_steps: int,
    ) -> Action:
        state = self.states[self.current]

        # Check transitions
        for t in state.transitions:
            if t.condition(sensors, history):
                old = self.current
                self.current = t.target
                self.transition_log.append(
                    (sensors.step, old, t.target, t.reason))
                log.info("State: %s → %s (%s)", old, t.target, t.reason)
                new_state = self.states[self.current]
                action = new_state.controller.decide(
                    sensors, history, experiment_steps)
                action.reason = f"[{old}→{t.target}] {action.reason}"
                return action

        # No transition — decide within current state
        action = state.controller.decide(sensors, history, experiment_steps)
        action.reason = f"[{self.current}] {action.reason}"
        return action


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run_experiment(
    engine: StepEngine,
    controller: object,
    total_steps: int,
    segment_steps: int,
    run_name: str,
    output_dir: Path,
    start_L: int,
    start_T: float,
    save_every: int = 50,
    prefill_text: str | None = None,
    dry_run: bool = False,
    extra_meta: dict | None = None,
) -> None:
    """Universal experiment loop.

    Generates tokens in segments, reads sensors after each, calls
    controller.decide() to get the next L/T (or rollback).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{run_name}.parquet"
    meta_path = output_dir / f"{run_name}.meta.json"
    checkpoint_path = output_dir / f"{run_name}.ckpt"
    log.info("Experiment: %s | %d steps, %d/segment", run_name, total_steps, segment_steps)

    current_L = start_L
    current_T = start_T

    # Prefill
    if prefill_text:
        token_ids = engine.tokenizer.encode(prefill_text, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Prefill text '{prefill_text}' encodes to zero tokens")
        repeats = (start_L // len(token_ids)) + 1
        prefill_ids = (token_ids * repeats)[:start_L]
        engine.set_context(prefill_ids)
        log.info("Pre-seeded context: %d tokens (%r)", len(prefill_ids), prefill_text)
    else:
        log.info("Prefill: %d tokens at L=%d T=%.2f", start_L, start_L, start_T)
        engine.run_segment(start_L, start_T, start_L, phase="prefill")

    sensor_history: list[SensorReading] = []
    n_rollbacks = 0
    experiment_steps = 0

    t_start = time.monotonic()

    while experiment_steps < total_steps:
        seg_size = min(segment_steps, total_steps - experiment_steps)

        # Snapshot for rollback
        snap = engine.snapshot()

        # Generate segment
        seg_t0 = time.monotonic()
        engine.run_segment(current_L, current_T, seg_size)
        tok_s = seg_size / (time.monotonic() - seg_t0) if seg_size > 0 else 0

        experiment_steps += seg_size

        # Read sensors
        sensors = engine.read_sensors(segment_steps=segment_steps)
        sensor_history.append(sensors)

        # Decide
        if dry_run:
            action = Action(L=current_L, T=current_T, reason="dry-run, hold")
        else:
            action = controller.decide(sensors, sensor_history, experiment_steps)

        # Rollback
        if action.rollback:
            log.info("ROLLBACK at step %d: %s", engine.current_step, action.reason)
            engine.restore(snap)
            experiment_steps -= seg_size
            sensor_history.pop()
            n_rollbacks += 1
            current_L, current_T = action.L, action.T
            continue

        pct = 100 * experiment_steps / total_steps
        log.info(
            "step %d/%d (%.0f%%) | %.0f tok/s | L=%d T=%.2f | "
            "ent=%.2f β=%.2f comp=%.3f | %s",
            experiment_steps, total_steps, pct, tok_s,
            current_L, current_T,
            sensors.entropy_mean, sensors.heaps_beta, sensors.comp_W64,
            action.reason,
        )

        current_L, current_T = action.L, action.T

        # Periodic saves
        n_seg = len(sensor_history)
        if n_seg % 10 == 0:
            engine.save_checkpoint(checkpoint_path, parquet_path)
        if n_seg % save_every == 0:
            _write_outputs(
                engine, parquet_path, meta_path,
                run_name, total_steps, segment_steps, start_L, start_T,
                current_L, current_T, experiment_steps, n_rollbacks,
                sensor_history, t_start, dry_run, extra_meta,
            )

    # Final save
    engine.fix_texts()
    _write_outputs(
        engine, parquet_path, meta_path,
        run_name, total_steps, segment_steps, start_L, start_T,
        current_L, current_T, experiment_steps, n_rollbacks,
        sensor_history, t_start, dry_run, extra_meta, final=True,
    )

    elapsed = time.monotonic() - t_start
    n_records = len(engine.records)
    log.info("Done: %d steps in %.1fs (%.1f tok/s), %d rollbacks",
             n_records, elapsed,
             n_records / elapsed if elapsed > 0 else 0, n_rollbacks)
    if sensor_history:
        log.info("Final: L=%d T=%.2f β=%.2f ent=%.2f",
                 current_L, current_T,
                 sensor_history[-1].heaps_beta, sensor_history[-1].entropy_mean)
    log.info("Output: %s", parquet_path)


def _write_outputs(
    engine: StepEngine,
    parquet_path: Path,
    meta_path: Path,
    run_name: str,
    total_steps: int,
    segment_steps: int,
    start_L: int,
    start_T: float,
    current_L: int,
    current_T: float,
    experiment_steps: int,
    n_rollbacks: int,
    sensor_history: list[SensorReading],
    t_start: float,
    dry_run: bool,
    extra_meta: dict | None,
    final: bool = False,
) -> None:
    """Write parquet and metadata."""
    engine.save_parquet(parquet_path)

    elapsed = time.monotonic() - t_start
    metadata = {
        "run_name": run_name,
        "seed": engine.seed,
        "total_steps": total_steps,
        "segment_steps": segment_steps,
        "start_L": start_L,
        "start_T": start_T,
        "final_L": current_L,
        "final_T": current_T,
        "final_beta": sensor_history[-1].heaps_beta if sensor_history else None,
        "final_entropy": sensor_history[-1].entropy_mean if sensor_history else None,
        "n_segments": len(sensor_history),
        "n_rollbacks": n_rollbacks,
        "dry_run": dry_run,
        "device": engine.device,
        "torch_version": torch.__version__,
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": (round(len(engine.records) / elapsed, 1)
                              if elapsed > 0 else 0),
        "num_tokens": experiment_steps,
        "complete": final,
    }
    if extra_meta:
        metadata.update(extra_meta)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    label = "Final" if final else "Snapshot"
    log.info("%s save: %s (%d records)",
             label, parquet_path.name, len(engine.records))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--segment-steps", type=int, default=1000)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory (default: per-mode subdir)")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--run-name", type=str, help="Override auto run name")
    parser.add_argument("--prefill-text", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save snapshot every N segments")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiments")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Fixed
    p_fixed = sub.add_parser("fixed", help="Fixed L/T run")
    _add_common_args(p_fixed)
    p_fixed.add_argument("-L", type=int, required=True, help="Context length")
    p_fixed.add_argument("-T", type=float, required=True, help="Temperature")

    # Schedule
    p_sched = sub.add_parser("schedule", help="Scheduled L/T segments")
    _add_common_args(p_sched)
    p_sched.add_argument("--spec", type=str, required=True,
                         help="Schedule: 'steps:L{n}:T{f},...'")

    # Beta controller
    p_beta = sub.add_parser("beta", help="Beta hill-climb controller")
    _add_common_args(p_beta)
    p_beta.add_argument("--start-L", type=int, default=64)
    p_beta.add_argument("--start-T", type=float, default=0.70)
    p_beta.add_argument("--target", type=float, default=0.9)
    p_beta.add_argument("--drift", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load model
    model, tokenizer = load_model(args.model_dir, args.device)
    engine = StepEngine(model, tokenizer, args.device, args.seed)

    # Per-mode default output directories
    _mode_dirs = {
        "fixed": runlib.SWEEP_DIR,
        "schedule": runlib.SCHEDULE_DIR,
        "beta": runlib.CONTROLLER_DIR,
    }
    output_dir = Path(args.output_dir) if args.output_dir else _mode_dirs[args.mode]

    if args.mode == "fixed":
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

    elif args.mode == "schedule":
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

    elif args.mode == "beta":
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


if __name__ == "__main__":
    main()

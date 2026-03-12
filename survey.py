"""Basin survey: systematic basin discovery via cooling/heating ramps.

State machine experiment that cycles through:
  COOLING  -> CAPTURED -> HEATING -> TRANSIT -> COOLING ...

Temperature ramps down during COOLING (finding basins at successively
lower T) and up during HEATING (probing escape). Each cycle captures
one basin with its accessibility temperature (T at capture) and escape
temperature (T at escape). Captures are accumulated in a .basins.pkl
sidecar. Novel basins (by embedding distance) extend the centroid
catalogue.

Usage:
    loop survey --seed 42 -L 8 --total-steps 100000
    loop survey --seed 42 -L 64 --T-min 0.50 --T-max 0.80
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from engine import SensorReading, StepEngine, load_model
from experiment import (
    Action,
    FixedController,
    MachineState,
    StateMachine,
    Transition,
    run_experiment,
)
import runlib

log = logging.getLogger(__name__)

# ── L-dependent defaults ──────────────────────────────────────────

# (L_max, T_min, T_max) — first match wins
_L_DEFAULTS: list[tuple[int, float, float]] = [
    (32, 0.10, 0.70),
    (96, 0.30, 0.80),
    (192, 0.40, 0.90),
    (256, 0.50, 1.00),
    (9999, 0.55, 1.05),
]


def l_defaults(L: int) -> tuple[float, float]:
    """Return (T_min, T_max) for a given L."""
    for l_max, t_min, t_max in _L_DEFAULTS:
        if L <= l_max:
            return t_min, t_max
    return _L_DEFAULTS[-1][1], _L_DEFAULTS[-1][2]


# ── Centroid catalogue ────────────────────────────────────────────

BASINS_DIR = Path("data/basins")


@dataclass
class CentroidCatalogue:
    """In-memory centroid store for online novelty detection."""
    centroids: np.ndarray  # (N, D) float32, may be empty (0, D)
    metadata: list[dict]   # one dict per centroid row
    threshold: float = 0.3
    dirty: bool = False

    @staticmethod
    def load(basins_dir: Path, embed_dim: int = 576) -> "CentroidCatalogue":
        """Load existing centroids or create empty catalogue."""
        npy_path = basins_dir / "centroids.npy"
        json_path = basins_dir / "centroids.json"

        if npy_path.exists() and json_path.exists():
            centroids = np.load(npy_path)
            with open(json_path) as f:
                metadata = json.load(f)
            log.info("Loaded %d existing basin types", len(metadata))
            return CentroidCatalogue(centroids=centroids, metadata=metadata)

        return CentroidCatalogue(
            centroids=np.empty((0, embed_dim), dtype=np.float32),
            metadata=[],
        )

    def match(self, embedding: np.ndarray) -> tuple[int | None, float]:
        """Find nearest centroid. Returns (type_id, distance).

        Returns (None, inf) if catalogue is empty.
        """
        if len(self.centroids) == 0:
            return None, float("inf")

        # Cosine distance
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
        cent_norms = self.centroids / (
            np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-10
        )
        similarities = cent_norms @ emb_norm
        best_idx = int(np.argmax(similarities))
        distance = float(1.0 - similarities[best_idx])
        type_id = self.metadata[best_idx]["type_id"]
        return type_id, distance

    def add_type(
        self, embedding: np.ndarray, capture: dict, run_id: str,
    ) -> int:
        """Register a new basin type. Returns the new type_id."""
        type_id = len(self.metadata)
        self.centroids = np.vstack([
            self.centroids,
            embedding.reshape(1, -1).astype(np.float32),
        ])
        meta = {
            "type_id": type_id,
            "hit_count": 1,
            "first_seen_run": run_id,
            "first_seen_step": capture["capture_step"],
            "last_seen_run": run_id,
            "last_seen_step": capture["capture_step"],
            "min_L": capture["L"],
            "max_L": capture["L"],
        }
        # Copy spectrum fields
        for w in [16, 32, 64, 128, 256]:
            key = f"comp_W{w}"
            if key in capture:
                meta[key] = capture[key]
        if "W_star" in capture:
            meta["W_star"] = capture["W_star"]
        for k in ("entropy_mean", "entropy_std", "entropy_floor", "heaps_beta"):
            if k in capture:
                meta[k] = capture[k]
        if "attractor_text" in capture:
            meta["representative_text"] = capture["attractor_text"]

        self.metadata.append(meta)
        self.dirty = True
        return type_id

    def update_type(
        self, type_id: int, capture: dict, run_id: str,
    ) -> None:
        """Update an existing type's stats with a new capture."""
        meta = self.metadata[type_id]
        meta["hit_count"] = meta.get("hit_count", 0) + 1
        meta["last_seen_run"] = run_id
        meta["last_seen_step"] = capture["capture_step"]
        L = capture["L"]
        meta["min_L"] = min(meta.get("min_L", L), L)
        meta["max_L"] = max(meta.get("max_L", L), L)
        self.dirty = True

    def save(self, basins_dir: Path) -> None:
        """Write centroids.npy + centroids.json."""
        if not self.dirty:
            return
        basins_dir.mkdir(parents=True, exist_ok=True)
        np.save(basins_dir / "centroids.npy", self.centroids)
        with open(basins_dir / "centroids.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        log.info("Saved %d basin types to %s", len(self.metadata), basins_dir)
        self.dirty = False


# ── Survey controller ─────────────────────────────────────────────

# Temperature ramp: relative step per segment (multiplicative).
# Cooling multiplies by (1 - DT_FRAC), heating by (1 + DT_FRAC).
DT_FRAC = 0.05

# Stability detection: entropy_mean must change by less than this between
# consecutive sensor readings for N consecutive readings to declare capture.
ENTROPY_DELTA_THRESHOLD = 0.1
STABILITY_COUNT = 3
# Minimum segments in COOLING before capture can trigger (let system settle).
MIN_COOLING_SEGMENTS = 5

# Escape detection: entropy must rise this much above the basin floor.
ESCAPE_ENTROPY_RISE = 1.0


@dataclass
class SurveyState:
    """Mutable state tracked across the survey run."""
    L: int
    T_min: float
    T_max: float
    dT_frac: float
    run_id: str
    current_T: float = 0.0  # set in __post_init__
    captures: list[dict] = field(default_factory=list)
    stability_streak: int = 0
    cooling_segments: int = 0
    basin_entropy_floor: float = float("inf")
    transit_entry_step: int = 0
    cycle_count: int = 0
    novel_count: int = 0
    consecutive_known: int = 0

    def __post_init__(self) -> None:
        if self.current_T == 0.0:
            self.current_T = self.T_max

    def cool(self) -> float:
        """Step T down. Returns new T."""
        self.current_T = max(self.T_min, self.current_T * (1 - self.dT_frac))
        return self.current_T

    def heat(self) -> float:
        """Step T up. Returns new T."""
        self.current_T = min(self.T_max, self.current_T * (1 + self.dT_frac))
        return self.current_T


class SurveyController:
    """Basin survey with gradient temperature ramps.

    Manages T directly: ramps down during COOLING, up during HEATING.
    Uses a StateMachine for state/transition logic only — the per-state
    FixedControllers are placeholders (T is overridden by decide()).
    """

    def __init__(
        self,
        engine: StepEngine,
        catalogue: CentroidCatalogue,
        survey_state: SurveyState,
    ):
        self.engine = engine
        self.catalogue = catalogue
        self.ss = survey_state
        self._machine = self._build_machine()

    def _build_machine(self) -> StateMachine:
        ss = self.ss
        # Placeholder controllers — T is overridden in decide()
        placeholder = FixedController(ss.L, ss.current_T)

        states = {
            "COOLING": MachineState(
                name="COOLING",
                controller=placeholder,
                transitions=[
                    Transition(
                        target="CAPTURED",
                        condition=self._cooling_captured,
                        reason="entropy stabilized",
                    ),
                ],
            ),
            "CAPTURED": MachineState(
                name="CAPTURED",
                controller=placeholder,
                transitions=[
                    Transition(
                        target="HEATING",
                        condition=self._captured_heating,
                        reason="characterization complete",
                    ),
                ],
            ),
            "HEATING": MachineState(
                name="HEATING",
                controller=placeholder,
                transitions=[
                    Transition(
                        target="CAPTURED",
                        condition=self._heating_deeper,
                        reason="deeper basin found",
                    ),
                    Transition(
                        target="TRANSIT",
                        condition=self._heating_escaped,
                        reason="escaped basin",
                    ),
                ],
            ),
            "TRANSIT": MachineState(
                name="TRANSIT",
                controller=placeholder,
                transitions=[
                    Transition(
                        target="COOLING",
                        condition=self._transit_flushed,
                        reason="context flushed",
                    ),
                ],
            ),
        }
        return StateMachine(states, initial="COOLING")

    # ── Transition conditions ─────────────────────────────────────

    def _cooling_captured(
        self, sensors: SensorReading, history: list[SensorReading],
    ) -> bool:
        self.ss.cooling_segments += 1
        if self.ss.cooling_segments < MIN_COOLING_SEGMENTS:
            return False
        if len(history) < 2:
            return False
        prev = history[-2]
        delta = abs(sensors.entropy_mean - prev.entropy_mean)
        if delta < ENTROPY_DELTA_THRESHOLD:
            self.ss.stability_streak += 1
        else:
            self.ss.stability_streak = 0

        if self.ss.stability_streak >= STABILITY_COUNT:
            self.ss.basin_entropy_floor = sensors.entropy_mean
            self.ss.stability_streak = 0
            return True
        return False

    def _captured_heating(
        self, sensors: SensorReading, history: list[SensorReading],
    ) -> bool:
        # Transient state: characterize and immediately transition.
        self._record_capture(sensors)
        return True

    def _heating_deeper(
        self, sensors: SensorReading, history: list[SensorReading],
    ) -> bool:
        # Entropy dropped to a new floor during heating
        if len(history) < 2:
            return False
        prev = history[-2]
        delta = abs(sensors.entropy_mean - prev.entropy_mean)
        if (sensors.entropy_mean < self.ss.basin_entropy_floor * 0.8
                and delta < ENTROPY_DELTA_THRESHOLD
                and prev.entropy_mean > sensors.entropy_mean):
            self.ss.basin_entropy_floor = sensors.entropy_mean
            return True
        return False

    def _heating_escaped(
        self, sensors: SensorReading, history: list[SensorReading],
    ) -> bool:
        rise = sensors.entropy_mean - self.ss.basin_entropy_floor
        if rise > ESCAPE_ENTROPY_RISE:
            self.ss.transit_entry_step = sensors.step
            return True
        return False

    def _transit_flushed(
        self, sensors: SensorReading, history: list[SensorReading],
    ) -> bool:
        tokens_in_transit = sensors.step - self.ss.transit_entry_step
        if tokens_in_transit >= self.ss.L:
            self.ss.basin_entropy_floor = float("inf")
            self.ss.cooling_segments = 0
            return True
        return False

    # ── Capture logic ─────────────────────────────────────────────

    def _record_capture(self, sensors: SensorReading) -> None:
        """Record a basin capture: spectrum, embedding, novelty check."""
        step = self.engine.current_step

        # Compression spectrum
        spectrum = self.engine.comp_spectrum()
        w_star = min(spectrum, key=spectrum.get)

        # Embedding
        embedding = self.engine.embed_context()

        # Attractor text (trailing W* tokens decoded)
        exp_records = [r for r in self.engine.records if r["phase"] == "experiment"]
        tail = exp_records[-w_star:] if len(exp_records) >= w_star else exp_records
        attractor_text = "".join(r["decoded_text"] for r in tail)

        # Build capture dict
        capture: dict = {
            "capture_id": f"{self.ss.run_id}:{step}",
            "run_id": self.ss.run_id,
            "capture_step": step,
            "record_step": step,
            "L": self.ss.L,
            "T_capture": self.ss.current_T,
            "W_star": w_star,
            "entropy_mean": sensors.entropy_mean,
            "entropy_std": sensors.entropy_std,
            "entropy_floor": sensors.entropy_mean,
            "heaps_beta": sensors.heaps_beta,
            "eos_rate": 0.0,
            "attractor_text": attractor_text[:2000],
            "embedding": embedding,
        }
        for w, val in spectrum.items():
            capture[f"comp_W{w}"] = val

        # Novelty detection
        type_id, distance = self.catalogue.match(embedding)
        capture["novelty_distance"] = distance

        if type_id is None or distance > self.catalogue.threshold:
            type_id = self.catalogue.add_type(embedding, capture, self.ss.run_id)
            self.ss.novel_count += 1
            self.ss.consecutive_known = 0
            log.info(
                "NOVEL basin type %d (dist=%.3f) at step %d T=%.3f | "
                "W*=%d ent=%.2f beta=%.2f",
                type_id, distance, step, self.ss.current_T, w_star,
                sensors.entropy_mean, sensors.heaps_beta,
            )
        else:
            self.catalogue.update_type(type_id, capture, self.ss.run_id)
            self.ss.consecutive_known += 1
            log.info(
                "KNOWN basin type %d (dist=%.3f) at step %d T=%.3f | "
                "W*=%d ent=%.2f beta=%.2f | streak=%d",
                type_id, distance, step, self.ss.current_T, w_star,
                sensors.entropy_mean, sensors.heaps_beta,
                self.ss.consecutive_known,
            )

        capture["type_id"] = type_id

        # Link to previous capture
        if self.ss.captures:
            prev = self.ss.captures[-1]
            capture["prev_capture_id"] = prev["capture_id"]
            prev["next_capture_id"] = capture["capture_id"]

        self.ss.captures.append(capture)
        self.ss.cycle_count += 1

    # ── Controller interface ──────────────────────────────────────

    def decide(
        self, sensors: SensorReading, history: list[SensorReading],
        experiment_steps: int,
    ) -> Action:
        # Let the state machine handle transitions
        old_state = self._machine.current
        action = self._machine.decide(sensors, history, experiment_steps)
        new_state = self._machine.current

        # Ramp T based on current state
        if new_state == "COOLING":
            self.ss.cool()
        elif new_state == "HEATING":
            self.ss.heat()
        # TRANSIT holds T at escape level; CAPTURED is transient

        # Override action's T with our managed T
        action.T = self.ss.current_T
        action.L = self.ss.L

        if old_state != new_state:
            log.debug("T=%.3f after %s→%s", self.ss.current_T, old_state, new_state)

        return action


# ── Run entry point ───────────────────────────────────────────────

def run_survey(
    seed: int,
    L: int,
    total_steps: int,
    T_min: float | None = None,
    T_max: float | None = None,
    dT_frac: float = DT_FRAC,
    segment_steps: int | None = None,
    model_dir: str = "data/model/SmolLM-135M",
    device: str = "cuda",
    run_name: str | None = None,
    output_dir: Path | None = None,
    save_every: int = 50,
    novelty_threshold: float = 0.3,
) -> Path:
    """Run a basin survey at fixed L.

    Returns path to the output parquet.
    """
    # Defaults
    default_t_min, default_t_max = l_defaults(L)
    if T_min is None:
        T_min = default_t_min
    if T_max is None:
        T_max = default_t_max
    if segment_steps is None:
        segment_steps = max(L, 50)
    if output_dir is None:
        output_dir = runlib.SURVEY_DIR
    if run_name is None:
        run_name = f"survey_L{L:04d}_S{seed}"

    log.info(
        "Basin survey: L=%d T_min=%.2f T_max=%.2f dT=%.0f%% seed=%d segments=%d",
        L, T_min, T_max, dT_frac * 100, seed, segment_steps,
    )

    # Load model
    model, tokenizer = load_model(model_dir, device)
    engine = StepEngine(model, tokenizer, device, seed)

    # Load centroid catalogue
    catalogue = CentroidCatalogue.load(BASINS_DIR, embed_dim=576)
    catalogue.threshold = novelty_threshold

    # Build survey controller
    survey_state = SurveyState(
        L=L, T_min=T_min, T_max=T_max, dT_frac=dT_frac, run_id=run_name,
    )
    controller = SurveyController(engine, catalogue, survey_state)

    # Extra metadata for the sidecar
    extra_meta = {
        "controller": True,
        "survey": True,
        "context_length": L,
        "T_min": T_min,
        "T_max": T_max,
        "dT_frac": dT_frac,
        "novelty_threshold": novelty_threshold,
    }

    # Run — start at T_max, cooling will ramp down
    run_experiment(
        engine=engine,
        controller=controller,
        total_steps=total_steps,
        segment_steps=segment_steps,
        run_name=run_name,
        output_dir=output_dir,
        start_L=L,
        start_T=T_max,
        save_every=save_every,
        extra_meta=extra_meta,
    )

    # Save basin data
    parquet_path = output_dir / f"{run_name}.parquet"
    _save_basin_data(survey_state, catalogue, output_dir, run_name)

    # Summary
    log.info(
        "Survey complete: %d cycles, %d novel types, %d total types",
        survey_state.cycle_count,
        survey_state.novel_count,
        len(catalogue.metadata),
    )

    return parquet_path


def _save_basin_data(
    ss: SurveyState,
    catalogue: CentroidCatalogue,
    output_dir: Path,
    run_name: str,
) -> None:
    """Save .basins.pkl sidecar and updated centroid catalogue."""
    # .basins.pkl — full capture data including embeddings
    pkl_path = output_dir / f"{run_name}.basins.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(ss.captures, f)
    log.info("Saved %d captures to %s", len(ss.captures), pkl_path.name)

    # Update global centroids
    catalogue.save(BASINS_DIR)

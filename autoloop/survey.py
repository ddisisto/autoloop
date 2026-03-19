"""Basin survey: systematic basin discovery via cooling/heating ramps.

State machine experiment that cycles through:
  COOLING -> HEATING -> TRANSIT -> COOLING ...

Temperature ramps down during COOLING (finding basins at successively
lower T) and up during HEATING (probing escape). Each cycle captures
one basin with its accessibility temperature (T at capture) and escape
temperature (T at escape). Captures are accumulated in a .basins.pkl
sidecar. Novel basins (by cluster membership in the fitted HDBSCAN
feature space) extend the cluster catalogue.

Usage:
    loop survey --seed 42 -L 8 --total-steps 100000
    loop survey --seed 42 -L 64 --T-min 0.50 --T-max 0.80
"""

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .clustering import (
    MODELS_DIR,
    load_cluster_centroids,
    load_models,
)
from .engine import SensorReading, StepEngine, load_model
from .experiment import Action, run_experiment
from . import runlib

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


# ── Cluster catalogue ─────────────────────────────────────────────

BASINS_DIR = Path("data/basins")

# Generous margin on cluster radius for matching new captures.
# 1.5x means a capture up to 50% further than the furthest training
# member still counts as belonging to that cluster.
RADIUS_MARGIN = 1.5


@dataclass
class ClusterCatalogue:
    """HDBSCAN-based novelty detection for online basin survey.

    Uses fitted PCA + StandardScaler from the clustering pipeline to
    project new captures into the 14-dim feature space, then matches
    against precomputed cluster centroids by Euclidean distance.

    Falls back gracefully when no fitted models exist (everything
    is novel).
    """
    pca: object | None             # fitted PCA, or None if no models
    scaler: object | None          # fitted StandardScaler, or None
    cluster_centroids: dict        # cluster_id -> {"centroid", "max_radius"}
    provisional: list[dict]        # unmatched captures awaiting future clustering
    dirty: bool = False

    @staticmethod
    def load(models_dir: Path | None = None) -> "ClusterCatalogue":
        """Load fitted models and cluster centroids, or create empty catalogue."""
        if models_dir is None:
            models_dir = MODELS_DIR

        pca = None
        scaler = None
        cluster_centroids: dict = {}

        feature_models_path = models_dir / "feature_models.pkl"
        centroids_path = models_dir / "cluster_centroids.pkl"

        if feature_models_path.exists() and centroids_path.exists():
            models = load_models(models_dir)
            pca = models["pca"]
            scaler = models["scaler"]
            cluster_centroids = load_cluster_centroids(models_dir)
            log.info(
                "ClusterCatalogue: loaded %d clusters from %s",
                len(cluster_centroids), models_dir,
            )
        else:
            log.info(
                "ClusterCatalogue: no fitted models at %s, "
                "all captures will be novel",
                models_dir,
            )

        return ClusterCatalogue(
            pca=pca,
            scaler=scaler,
            cluster_centroids=cluster_centroids,
            provisional=[],
        )

    def _build_feature_vector(
        self, embedding: np.ndarray, capture: dict,
    ) -> np.ndarray:
        """Build the 8-dim feature vector for a single capture.

        Mirrors the batch construction in clustering.build_feature_matrix():
          - PCA-project embedding (8 dims)
          - StandardScaler to unit variance
        """
        pca_features = self.pca.transform(embedding.reshape(1, -1))  # (1, 8)
        scaled = self.scaler.transform(pca_features)  # (1, 8)
        return scaled[0]  # (8,)

    def match(
        self, embedding: np.ndarray, capture: dict,
    ) -> tuple[int | None, float]:
        """Find nearest cluster centroid. Returns (cluster_id, distance).

        Returns (None, inf) if no fitted models or no clusters.
        """
        if self.pca is None or not self.cluster_centroids:
            return None, float("inf")

        fv = self._build_feature_vector(embedding, capture)

        best_id = None
        best_dist = float("inf")
        for cid, info in self.cluster_centroids.items():
            dist = float(np.linalg.norm(fv - info["centroid"]))
            if dist < best_dist:
                best_dist = dist
                best_id = cid

        # Check against radius threshold
        if best_id is not None:
            max_radius = self.cluster_centroids[best_id]["max_radius"]
            if best_dist > max_radius * RADIUS_MARGIN:
                return None, best_dist

        return best_id, best_dist

    def add_type(
        self, embedding: np.ndarray, capture: dict, run_id: str,
    ) -> int:
        """Record an unmatched capture with a provisional type_id.

        Does not create a new cluster online — provisional captures
        get swept into clusters on the next clustering run.
        Returns the provisional type_id.
        """
        # Provisional IDs start after the highest existing cluster ID
        existing_max = max(self.cluster_centroids.keys()) if self.cluster_centroids else -1
        provisional_offset = existing_max + 1 + len(self.provisional)
        type_id = provisional_offset

        self.provisional.append({
            "type_id": type_id,
            "run_id": run_id,
            "capture_step": capture["capture_step"],
            "L": capture["L"],
        })
        self.dirty = True
        return type_id

    def save(self, models_dir: Path | None = None) -> None:
        """Persist provisional captures for future clustering."""
        if not self.dirty:
            return
        if models_dir is None:
            models_dir = MODELS_DIR
        models_dir.mkdir(parents=True, exist_ok=True)

        save_path = models_dir / "provisional_captures.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(self.provisional, f)
        log.info(
            "Saved %d provisional captures to %s",
            len(self.provisional), save_path,
        )
        self.dirty = False


# ── Survey controller ─────────────────────────────────────────────

# Temperature ramp: relative step per segment (multiplicative).
# Cooling multiplies by (1 - DT_FRAC), heating by (1 + DT_FRAC).
DT_FRAC = 0.05

# Capture detection: two independent gates, either sufficient.
#
# Gate 1 — β < 0.40 (when measurable, i.e. n_words >= 50).
#   Clean collapse wall from regime analysis (50 runs, d=3.4, zero false positives).
#
# Gate 2 — compressibility < 0.45.
#   Catches short-cycle basins (e.g. "1.1.1." at L=8) where tokens produce
#   no words >1 char so β is unmeasurable, and entropy stays high (~4.5)
#   because the output distribution is broad even though sampling is locked.
#
# Either gate fires capture. Both require MIN_COOLING_SEGMENTS.
CAPTURE_BETA_THRESHOLD = 0.40
CAPTURE_COMP_THRESHOLD = 0.45
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
    Direct state tracking with _check_transitions() — no StateMachine wrapper.
    """

    def __init__(
        self,
        engine: StepEngine,
        catalogue: ClusterCatalogue,
        survey_state: SurveyState,
    ):
        self.engine = engine
        self.catalogue = catalogue
        self.ss = survey_state
        self.state = "COOLING"

    # ── Transition checks ─────────────────────────────────────────

    def _check_transitions(
        self, sensors: SensorReading, history: list[SensorReading],
    ) -> str | None:
        """Check transition conditions for current state. Returns new state or None."""
        if self.state == "COOLING":
            self.ss.cooling_segments += 1
            if self.ss.cooling_segments < MIN_COOLING_SEGMENTS:
                return None
            # Two independent capture gates — either sufficient:
            # 1) β < 0.40 (when measurable): vocabulary has died
            # 2) comp < 0.45: output is highly repetitive/compressible
            beta_captured = (sensors.n_words >= 50
                             and sensors.heaps_beta < CAPTURE_BETA_THRESHOLD)
            comp_captured = sensors.comp_W64 < CAPTURE_COMP_THRESHOLD
            if beta_captured or comp_captured:
                self.ss.basin_entropy_floor = sensors.entropy_mean
                self._record_capture(sensors)
                return "HEATING"
            return None

        if self.state == "HEATING":
            # Check deeper basin: same gates as capture, plus deeper than current
            beta_captured = (sensors.n_words >= 50
                             and sensors.heaps_beta < CAPTURE_BETA_THRESHOLD)
            comp_captured = sensors.comp_W64 < CAPTURE_COMP_THRESHOLD
            if ((beta_captured or comp_captured)
                    and sensors.entropy_mean < self.ss.basin_entropy_floor * 0.8):
                self.ss.basin_entropy_floor = sensors.entropy_mean
                self._record_capture(sensors)
            # Check escape
            rise = sensors.entropy_mean - self.ss.basin_entropy_floor
            at_ceiling = self.ss.current_T >= self.ss.T_max - 1e-6
            if rise > ESCAPE_ENTROPY_RISE or at_ceiling:
                self.ss.transit_entry_step = sensors.step
                return "TRANSIT"
            return None

        if self.state == "TRANSIT":
            tokens_in_transit = sensors.step - self.ss.transit_entry_step
            if tokens_in_transit >= self.ss.L:
                self.ss.basin_entropy_floor = float("inf")
                self.ss.cooling_segments = 0
                return "COOLING"
            return None

        return None

    # ── Capture logic ─────────────────────────────────────────────

    def _record_capture(self, sensors: SensorReading) -> None:
        """Record a basin capture: spectrum, embedding, novelty check."""
        step = self.engine.current_step

        # Compression spectrum
        spectrum = self.engine.comp_spectrum()
        w_star = min(spectrum, key=spectrum.get)

        # Embedding
        embedding = self.engine.embed_context()

        # Attractor text (trailing W* tokens for full collapse context)
        exp_records = [r for r in self.engine.records if r["phase"] == "experiment"]
        tail = exp_records[-w_star:] if len(exp_records) >= w_star else exp_records
        attractor_text = "".join(r["decoded_text"] for r in tail)

        # Context window text (L tokens — matches embedding, pure basin content)
        context_text = self.engine.tokenizer.decode(self.engine.context[0])

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
            "context_text": context_text,
            "embedding": embedding,
        }
        for w, val in spectrum.items():
            capture[f"comp_W{w}"] = val

        # Novelty detection
        type_id, distance = self.catalogue.match(embedding, capture)
        capture["novelty_distance"] = distance

        if type_id is None:
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
        old_state = self.state
        new_state = self._check_transitions(sensors, history)
        if new_state is not None:
            self.state = new_state
            log.info("State: %s → %s", old_state, new_state)

        # Ramp T based on current state
        if self.state == "COOLING":
            self.ss.cool()
        elif self.state == "HEATING":
            self.ss.heat()
        # TRANSIT holds T at escape level

        if old_state != self.state:
            log.debug("T=%.3f after %s→%s", self.ss.current_T, old_state, self.state)

        return Action(
            L=self.ss.L,
            T=self.ss.current_T,
            reason=f"[{self.state}]",
        )


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
        segment_steps = 2 * L
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

    # Load cluster catalogue
    catalogue = ClusterCatalogue.load()

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
    n_clusters = len(catalogue.cluster_centroids)
    n_provisional = len(catalogue.provisional)
    log.info(
        "Survey complete: %d cycles, %d novel types, "
        "%d known clusters + %d provisional",
        survey_state.cycle_count,
        survey_state.novel_count,
        n_clusters,
        n_provisional,
    )

    return parquet_path


def _save_basin_data(
    ss: SurveyState,
    catalogue: ClusterCatalogue,
    output_dir: Path,
    run_name: str,
) -> None:
    """Save .basins.pkl sidecar and updated cluster catalogue."""
    # .basins.pkl — full capture data including embeddings
    pkl_path = output_dir / f"{run_name}.basins.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(ss.captures, f)
    log.info("Saved %d captures to %s", len(ss.captures), pkl_path.name)

    # Save provisional captures
    catalogue.save()

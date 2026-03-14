"""Basin clustering: feature extraction and dimensionality reduction.

Loads basin captures from .basins.pkl files, builds a joint feature matrix
(PCA-reduced embeddings + normalized compression spectrum + L), and
normalizes for downstream clustering (HDBSCAN).

Feature vector per capture (14 dims):
  - PCA(576 → 8) on embeddings (8 dims)
  - Normalized comp_W{16,32,64,128,256} (5 dims)
  - L / 512 (1 dim)

Entropy and heaps_beta are excluded from clustering features — they
reflect observation-time depth (where on the deepening trajectory a
capture was taken), not basin identity. They remain available as
per-capture metadata for display and analysis.

Usage:
    from autoloop.clustering import build_feature_matrix
    result = build_feature_matrix()
    # result.features: (N, 14) ndarray, unit-variance scaled
    # result.captures: list of capture dicts (with metadata)
    # result.pca: fitted PCA model
    # result.scaler: fitted StandardScaler
"""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import runlib
from .utils import compressibility_baseline

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────

EMBED_DIM = 576
PCA_COMPONENTS = 8
COMP_WINDOWS = [16, 32, 64, 128, 256]
MAX_L = 512
FEATURE_DIM = PCA_COMPONENTS + len(COMP_WINDOWS) + 1  # 14

MODELS_DIR = Path("data/basins/clustering")


# ── Data structures ───────────────────────────────────────────────

@dataclass
class FeatureResult:
    """Output of build_feature_matrix()."""
    features: np.ndarray       # (N, FEATURE_DIM) float64, unit-variance scaled
    captures: list[dict]       # original capture dicts (unmodified)
    pca: PCA                   # fitted PCA model
    scaler: StandardScaler     # fitted StandardScaler
    raw_features: np.ndarray   # (N, FEATURE_DIM) before scaling


@dataclass
class ClusterResult:
    """Output of cluster()."""
    labels: np.ndarray         # (N,) int, -1 for noise
    probabilities: np.ndarray  # (N,) float, cluster membership probability
    n_clusters: int            # number of clusters found (excluding noise)
    n_noise: int               # number of noise points (label == -1)
    model: Any                 # fitted HDBSCAN instance


# ── Loading ───────────────────────────────────────────────────────

def discover_basin_pkls(survey_dir: Path | None = None) -> list[Path]:
    """Find all .basins.pkl files in the survey directory."""
    if survey_dir is None:
        survey_dir = runlib.SURVEY_DIR
    if not survey_dir.is_dir():
        return []
    pkls = sorted(survey_dir.glob("*.basins.pkl"))
    log.info("Found %d .basins.pkl files in %s", len(pkls), survey_dir)
    return pkls


def load_all_captures(survey_dir: Path | None = None) -> list[dict]:
    """Load and concatenate all captures from .basins.pkl files.

    Returns a flat list of capture dicts from all survey runs.
    """
    pkls = discover_basin_pkls(survey_dir)
    if not pkls:
        raise FileNotFoundError(
            f"No .basins.pkl files found in {survey_dir or runlib.SURVEY_DIR}"
        )

    all_captures: list[dict] = []
    for pkl_path in pkls:
        with open(pkl_path, "rb") as f:
            captures = pickle.load(f)
        log.info("Loaded %d captures from %s", len(captures), pkl_path.name)
        all_captures.extend(captures)

    log.info("Total captures: %d from %d files", len(all_captures), len(pkls))
    return all_captures


# ── Feature extraction ────────────────────────────────────────────

def _extract_embeddings(captures: list[dict]) -> np.ndarray:
    """Stack embeddings from captures into (N, EMBED_DIM) array."""
    embeddings = np.stack([c["embedding"] for c in captures])
    if embeddings.shape[1] != EMBED_DIM:
        raise ValueError(
            f"Expected embedding dim {EMBED_DIM}, got {embeddings.shape[1]}"
        )
    return embeddings.astype(np.float64)


def _normalize_comp_spectrum(captures: list[dict]) -> np.ndarray:
    """Normalize raw comp_W values against incompressible baselines.

    For each capture and each window size W, estimates the byte length
    of the W-token window as len(attractor_text.encode()) * W / L,
    then divides the raw compression ratio by the baseline at that
    byte length.

    Returns (N, 5) array of normalized compression values.
    """
    n = len(captures)
    normed = np.empty((n, len(COMP_WINDOWS)), dtype=np.float64)

    for i, cap in enumerate(captures):
        text_bytes = len(cap["attractor_text"].encode("utf-8"))
        L = cap["L"]

        for j, w in enumerate(COMP_WINDOWS):
            raw = cap[f"comp_W{w}"]
            # Estimate byte length for this window size
            estimated_bytes = int(text_bytes * w / L)
            if estimated_bytes <= 0:
                estimated_bytes = 1
            baseline = compressibility_baseline(estimated_bytes)
            normed[i, j] = raw / baseline

    return normed


def _extract_L_normalized(captures: list[dict]) -> np.ndarray:
    """Extract L normalized to [0, 1] as (N, 1) array."""
    L_vals = np.array([c["L"] for c in captures], dtype=np.float64)
    return (L_vals / MAX_L).reshape(-1, 1)


# ── Main entry point ─────────────────────────────────────────────

def build_feature_matrix(
    survey_dir: Path | None = None,
) -> FeatureResult:
    """Build the 14-dim feature matrix from all basin captures.

    Steps:
      1. Load all captures from .basins.pkl files
      2. PCA(576 → 8) on embeddings
      3. Normalize compression spectrum against baselines
      4. Normalize L to [0, 1]
      5. Concatenate into (N, 14) matrix
      6. StandardScaler to unit variance

    Entropy and heaps_beta are excluded — they describe observation
    depth, not basin identity. Same attractor at different lock-in
    stages should cluster together.

    Returns a FeatureResult with the scaled feature matrix, capture
    metadata, and fitted PCA/scaler for reuse.
    """
    captures = load_all_captures(survey_dir)
    n = len(captures)
    log.info("Building feature matrix for %d captures", n)

    # 1. Embeddings → PCA
    embeddings = _extract_embeddings(captures)
    pca = PCA(n_components=PCA_COMPONENTS)
    pca_features = pca.fit_transform(embeddings)
    explained = sum(pca.explained_variance_ratio_)
    log.info(
        "PCA: %d → %d components, %.1f%% variance explained",
        EMBED_DIM, PCA_COMPONENTS, explained * 100,
    )

    # 2. Normalized compression spectrum
    comp_features = _normalize_comp_spectrum(captures)

    # 3. Normalized L
    L_features = _extract_L_normalized(captures)

    # Concatenate
    raw_features = np.hstack([
        pca_features,       # 8 dims
        comp_features,      # 5 dims
        L_features,         # 1 dim
    ])
    assert raw_features.shape == (n, FEATURE_DIM), (
        f"Expected ({n}, {FEATURE_DIM}), got {raw_features.shape}"
    )

    # 5. Scale to unit variance
    scaler = StandardScaler()
    features = scaler.fit_transform(raw_features)

    log.info("Feature matrix: %s, scaled to unit variance", features.shape)

    return FeatureResult(
        features=features,
        captures=captures,
        pca=pca,
        scaler=scaler,
        raw_features=raw_features,
    )


# ── Clustering ────────────────────────────────────────────────────

def cluster(result: FeatureResult, min_cluster_size: int = 3) -> ClusterResult:
    """Run HDBSCAN on the scaled feature matrix.

    Uses sklearn.cluster.HDBSCAN with Euclidean metric on the
    unit-variance-scaled features from build_feature_matrix().

    Returns a ClusterResult with labels, probabilities, and the
    fitted model.
    """
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, copy=True)
    hdb.fit(result.features)

    labels = hdb.labels_
    probabilities = hdb.probabilities_
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    log.info(
        "HDBSCAN: %d clusters, %d noise points (of %d total)",
        n_clusters, n_noise, len(labels),
    )

    return ClusterResult(
        labels=labels,
        probabilities=probabilities,
        n_clusters=n_clusters,
        n_noise=n_noise,
        model=hdb,
    )


def save_models(result: FeatureResult, models_dir: Path | None = None) -> Path:
    """Save fitted PCA and scaler for future use.

    Saves a single pickle with both models to models_dir.
    Returns the path to the saved file.
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    save_path = models_dir / "feature_models.pkl"
    payload = {
        "pca": result.pca,
        "scaler": result.scaler,
        "feature_dim": FEATURE_DIM,
        "pca_components": PCA_COMPONENTS,
        "comp_windows": COMP_WINDOWS,
        "max_L": MAX_L,
    }
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)
    log.info("Saved PCA + scaler to %s", save_path)
    return save_path


def load_models(models_dir: Path | None = None) -> dict:
    """Load previously saved PCA and scaler.

    Returns dict with 'pca' and 'scaler' keys (plus metadata).
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    load_path = models_dir / "feature_models.pkl"
    with open(load_path, "rb") as f:
        payload = pickle.load(f)
    log.info("Loaded PCA + scaler from %s", load_path)
    return payload


# ── Cluster centroids for online matching ─────────────────────────

def compute_cluster_centroids(
    result: FeatureResult,
    cr: ClusterResult,
    models_dir: Path | None = None,
) -> dict[int, dict]:
    """Compute per-cluster centroids and max radii in scaled feature space.

    For each cluster, computes the mean (centroid) and the maximum distance
    from centroid to any member. Saves the result alongside the feature
    models for use by ClusterCatalogue during surveys.

    Returns dict mapping cluster_id -> {"centroid": ndarray, "max_radius": float}.
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    cluster_info: dict[int, dict] = {}
    for cid in range(cr.n_clusters):
        mask = cr.labels == cid
        members = result.features[mask]
        centroid = members.mean(axis=0)
        distances = np.linalg.norm(members - centroid, axis=1)
        max_radius = float(distances.max())
        cluster_info[cid] = {
            "centroid": centroid,
            "max_radius": max_radius,
        }

    # Save
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "cluster_centroids.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(cluster_info, f)
    log.info(
        "Saved %d cluster centroids to %s", len(cluster_info), save_path,
    )
    return cluster_info


def load_cluster_centroids(models_dir: Path | None = None) -> dict[int, dict]:
    """Load precomputed cluster centroids and radii.

    Returns dict mapping cluster_id -> {"centroid": ndarray, "max_radius": float}.
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    load_path = models_dir / "cluster_centroids.pkl"
    with open(load_path, "rb") as f:
        centroids = pickle.load(f)
    log.info("Loaded %d cluster centroids from %s", len(centroids), load_path)
    return centroids

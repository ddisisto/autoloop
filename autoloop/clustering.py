"""Basin clustering: feature extraction and dimensionality reduction.

Loads basin captures from .basins.pkl files, builds a feature matrix
from PCA-reduced embeddings, and clusters with HDBSCAN.

Feature vector per capture (8 dims):
  - PCA(576 → 8) on model embeddings

The embedding is the model's own representation of the L-token context
window at capture time. It already encodes structural and semantic
properties of the basin. Compression spectrum, entropy, heaps_beta,
and L are excluded from clustering features — they describe observation
conditions, not basin identity. They remain available as per-capture
metadata for display and analysis.

Usage:
    from autoloop.clustering import build_feature_matrix
    result = build_feature_matrix()
    # result.features: (N, 8) ndarray, unit-variance scaled
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

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────

EMBED_DIM = 576
PCA_COMPONENTS = 8
FEATURE_DIM = PCA_COMPONENTS  # 8

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


# ── Main entry point ─────────────────────────────────────────────

def build_feature_matrix(
    survey_dir: Path | None = None,
) -> FeatureResult:
    """Build the 8-dim feature matrix from all basin captures.

    Steps:
      1. Load all captures from .basins.pkl files
      2. PCA(576 → 8) on embeddings
      3. StandardScaler to unit variance

    Returns a FeatureResult with the scaled feature matrix, capture
    metadata, and fitted PCA/scaler for reuse.
    """
    captures = load_all_captures(survey_dir)
    n = len(captures)
    log.info("Building feature matrix for %d captures", n)

    # Embeddings → PCA
    embeddings = _extract_embeddings(captures)
    pca = PCA(n_components=PCA_COMPONENTS)
    raw_features = pca.fit_transform(embeddings)
    explained = sum(pca.explained_variance_ratio_)
    log.info(
        "PCA: %d → %d components, %.1f%% variance explained",
        EMBED_DIM, PCA_COMPONENTS, explained * 100,
    )

    assert raw_features.shape == (n, FEATURE_DIM), (
        f"Expected ({n}, {FEATURE_DIM}), got {raw_features.shape}"
    )

    # Scale to unit variance
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

MERGE_THRESHOLD = 0.5
MERGE_MAX_DIAMETER = 7.0


def _merge_close_clusters(
    labels: np.ndarray,
    features: np.ndarray,
    threshold: float = MERGE_THRESHOLD,
    max_diameter: float = MERGE_MAX_DIAMETER,
) -> np.ndarray:
    """Agglomerative merge of clusters whose centroids are within threshold.

    Repeatedly finds the closest centroid pair, merges them (reassigning
    labels and recomputing the centroid), and repeats until all centroid
    distances exceed the threshold.

    After all merges, checks that no merged cluster has a diameter (max
    pairwise distance between members) exceeding max_diameter. Raises
    ValueError if so — this means the threshold is chaining unrelated
    clusters and needs investigation.

    Returns a new labels array with contiguous cluster IDs.
    """
    from scipy.spatial.distance import pdist

    labels = labels.copy()
    cluster_ids = sorted(set(labels) - {-1})
    if len(cluster_ids) < 2:
        return labels

    def _centroid(cid: int) -> np.ndarray:
        return features[labels == cid].mean(axis=0)

    merges = 0
    while True:
        cluster_ids = sorted(set(labels) - {-1})
        if len(cluster_ids) < 2:
            break

        centroids = {cid: _centroid(cid) for cid in cluster_ids}

        best_dist = float("inf")
        best_pair = (-1, -1)
        for i, ci in enumerate(cluster_ids):
            for cj in cluster_ids[i + 1:]:
                d = float(np.linalg.norm(centroids[ci] - centroids[cj]))
                if d < best_dist:
                    best_dist = d
                    best_pair = (ci, cj)

        if best_dist > threshold:
            break

        keep, drop = best_pair
        labels[labels == drop] = keep
        merges += 1
        log.debug("Merged cluster %d into %d (dist=%.4f)", drop, keep, best_dist)

    if merges > 0:
        log.info("Post-hoc merge: %d merges at threshold=%.2f", merges, threshold)

    # Sanity check: no merged cluster should be too spread out
    for cid in sorted(set(labels) - {-1}):
        members = features[labels == cid]
        if len(members) < 2:
            continue
        diameter = float(pdist(members).max())
        if diameter > max_diameter:
            raise ValueError(
                f"Cluster {cid} has diameter {diameter:.2f} after merge, "
                f"exceeding max_diameter={max_diameter:.2f}. "
                f"Merge threshold {threshold} may be chaining unrelated clusters."
            )

    # Re-label contiguously (0, 1, 2, ...)
    old_ids = sorted(set(labels) - {-1})
    remap = {old: new for new, old in enumerate(old_ids)}
    remap[-1] = -1
    labels = np.array([remap[l] for l in labels])

    return labels


def cluster(result: FeatureResult, min_cluster_size: int = 3) -> ClusterResult:
    """Run HDBSCAN on the scaled feature matrix, then merge near-duplicates.

    Uses sklearn.cluster.HDBSCAN with Euclidean metric on the
    unit-variance-scaled features from build_feature_matrix().
    Post-hoc merges clusters whose centroids are within MERGE_THRESHOLD.

    Returns a ClusterResult with labels, probabilities, and the
    fitted model.
    """
    hdb = HDBSCAN(min_cluster_size=min_cluster_size, copy=True)
    hdb.fit(result.features)

    labels = hdb.labels_
    probabilities = hdb.probabilities_
    n_raw = len(set(labels) - {-1})

    log.info(
        "HDBSCAN: %d clusters, %d noise points (of %d total)",
        n_raw, int(np.sum(labels == -1)), len(labels),
    )

    labels = _merge_close_clusters(labels, result.features)
    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))

    if n_clusters < n_raw:
        log.info(
            "After merge: %d clusters (%d merged away)",
            n_clusters, n_raw - n_clusters,
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

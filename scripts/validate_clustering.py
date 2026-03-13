"""Validate HDBSCAN clustering against basin captures.

Runs build_feature_matrix() and cluster(), then prints diagnostic
output for manual assessment: cluster sizes, scalar distributions,
representative text, and grab-bag detection via pairwise cosine
distance on raw embeddings.

Usage:
    python scripts/validate_clustering.py [--min-cluster-size N]
"""

import argparse
import logging
import sys
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from autoloop.clustering import (
    ClusterResult,
    FeatureResult,
    build_feature_matrix,
    cluster,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

COSINE_GRAB_BAG_THRESHOLD = 0.4


def _embeddings_array(captures: list[dict], indices: list[int]) -> np.ndarray:
    """Stack raw embeddings for given capture indices."""
    return np.stack([captures[i]["embedding"] for i in indices]).astype(np.float64)


def _max_pairwise_cosine(embeddings: np.ndarray) -> float:
    """Max pairwise cosine distance within a set of embeddings."""
    if len(embeddings) < 2:
        return 0.0
    dists = cosine_distances(embeddings)
    np.fill_diagonal(dists, 0.0)
    return float(np.max(dists))


def _representative_texts(captures: list[dict], indices: list[int], n: int = 3) -> list[str]:
    """Pick up to n representative attractor_text snippets (first 80 chars)."""
    # Pick evenly spaced indices for variety
    step = max(1, len(indices) // n)
    picked = indices[::step][:n]
    return [captures[i]["attractor_text"][:80].replace("\n", " ") for i in picked]


def print_cluster_summary(
    result: FeatureResult,
    cr: ClusterResult,
) -> None:
    """Print per-cluster diagnostics."""
    captures = result.captures
    labels = cr.labels

    print("\n" + "=" * 72)
    print(f"HDBSCAN CLUSTERING SUMMARY")
    print(f"  Total captures: {len(labels)}")
    print(f"  Clusters found: {cr.n_clusters}")
    print(f"  Noise points:   {cr.n_noise}")
    print("=" * 72)

    cluster_ids = sorted(set(labels) - {-1})
    grab_bags: list[int] = []

    for cid in cluster_ids:
        indices = [i for i, lb in enumerate(labels) if lb == cid]
        n = len(indices)

        entropies = [captures[i]["entropy_mean"] for i in indices]
        betas = [captures[i]["heaps_beta"] for i in indices]

        embs = _embeddings_array(captures, indices)
        max_cos = _max_pairwise_cosine(embs)
        is_grab_bag = max_cos > COSINE_GRAB_BAG_THRESHOLD

        if is_grab_bag:
            grab_bags.append(cid)

        texts = _representative_texts(captures, indices)
        old_types = Counter(captures[i].get("type_id", "?") for i in indices)

        print(f"\n── Cluster {cid}  (size={n}) {'*** GRAB-BAG ***' if is_grab_bag else ''}")
        print(f"   Entropy:  mean={np.mean(entropies):.3f}  std={np.std(entropies):.3f}  "
              f"range=[{np.min(entropies):.3f}, {np.max(entropies):.3f}]")
        print(f"   Beta:     mean={np.mean(betas):.3f}  std={np.std(betas):.3f}  "
              f"range=[{np.min(betas):.3f}, {np.max(betas):.3f}]")
        print(f"   Max pairwise cosine dist: {max_cos:.3f}")
        print(f"   Old type_ids: {dict(old_types)}")
        print(f"   Representative texts:")
        for t in texts:
            print(f"     | {t}")

    # Noise / outlier captures
    noise_indices = [i for i, lb in enumerate(labels) if lb == -1]
    if noise_indices:
        print(f"\n── Noise / Outliers  (n={len(noise_indices)})")
        for i in noise_indices:
            cap = captures[i]
            text = cap["attractor_text"][:80].replace("\n", " ")
            print(f"   [{cap.get('capture_id', '?'):>3}] ent={cap['entropy_mean']:.3f} "
                  f"beta={cap['heaps_beta']:.3f} old_type={cap.get('type_id', '?')} "
                  f"| {text}")

    # Summary
    print("\n" + "=" * 72)
    print("ASSESSMENT")
    if grab_bags:
        print(f"  Potential grab-bags (max cosine > {COSINE_GRAB_BAG_THRESHOLD}): "
              f"clusters {[int(c) for c in grab_bags]}")
    else:
        print(f"  No grab-bags detected (all clusters max cosine <= {COSINE_GRAB_BAG_THRESHOLD})")
    print(f"  Cluster size distribution: {sorted(Counter(labels[labels >= 0]).values(), reverse=True)}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HDBSCAN basin clustering")
    parser.add_argument("--min-cluster-size", type=int, default=3,
                        help="HDBSCAN min_cluster_size (default: 3)")
    args = parser.parse_args()

    log.info("Building feature matrix...")
    result = build_feature_matrix()

    log.info("Running HDBSCAN (min_cluster_size=%d)...", args.min_cluster_size)
    cr = cluster(result, min_cluster_size=args.min_cluster_size)

    print_cluster_summary(result, cr)

    # Save labels back into captures for inspection
    for i, cap in enumerate(result.captures):
        cap["hdbscan_cluster"] = int(cr.labels[i])
        cap["hdbscan_probability"] = float(cr.probabilities[i])

    log.info("Cluster labels attached to captures in memory. Done.")


if __name__ == "__main__":
    main()

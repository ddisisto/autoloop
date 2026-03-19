"""Basin catalogue CLI: explore clustered basin captures.

Presentation layer for basin data. Loads captures from .basins.pkl,
runs the clustering pipeline (build_feature_matrix + cluster), and
formats results for terminal display.

Subcommands:
    loop basin list                          # all clusters with hit counts, scalars, representative text
    loop basin show <cluster_id>             # all captures for a cluster
    loop basin compare <id1> <id2>           # side-by-side scalars + cosine distance
    loop basin matrix                        # cross-cluster centroid cosine distance matrix
    loop basin matrix --within <cluster_id>  # within-cluster pairwise distances
    loop basin captures [--type <id>] [--seed <s>]  # list captures with filters
"""

import argparse
import logging
import sys
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from .clustering import (
    ClusterResult,
    FeatureResult,
    build_feature_matrix,
    cluster,
)

log = logging.getLogger(__name__)


# ── Cached pipeline ──────────────────────────────────────────────────

class PipelineResult(NamedTuple):
    """Cached output of the full clustering pipeline."""
    features: FeatureResult
    clusters: ClusterResult
    captures: list[dict]
    embeddings: np.ndarray  # (N, 576) raw embeddings


@lru_cache(maxsize=1)
def _run_pipeline(min_cluster_size: int = 3) -> PipelineResult:
    """Run feature extraction + clustering, cached for the session."""
    feat = build_feature_matrix()
    clust = cluster(feat, min_cluster_size=min_cluster_size)

    # Extract raw embeddings for cosine distance calculations
    embeddings = np.stack([c["embedding"] for c in feat.captures])

    return PipelineResult(
        features=feat,
        clusters=clust,
        captures=feat.captures,
        embeddings=embeddings,
    )


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors (1 - cosine_similarity)."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return float(1.0 - dot / norm)


def _cluster_indices(labels: np.ndarray, cluster_id: int) -> list[int]:
    """Return indices of captures belonging to a cluster."""
    return [i for i, l in enumerate(labels) if l == cluster_id]


def _cluster_centroid(embeddings: np.ndarray, indices: list[int]) -> np.ndarray:
    """Mean embedding for a cluster."""
    return embeddings[indices].mean(axis=0)


def _truncate_text(text: str, width: int) -> str:
    """Truncate text to width, adding ellipsis if needed."""
    text = text.replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return text[:width - 3] + "..."


def _extract_seed(capture: dict) -> int | None:
    """Extract seed from run_id (e.g., survey_L0008_S42 -> 42)."""
    run_id = capture.get("run_id", "")
    for part in run_id.split("_"):
        if part.startswith("S") and part[1:].isdigit():
            return int(part[1:])
    return None


# ── Formatting helpers ───────────────────────────────────────────────

def _format_aligned_table(
    headers: list[str],
    rows: list[list[str]],
    alignments: list[str] | None = None,
) -> str:
    """Format rows into an aligned table.

    alignments: list of '<' (left) or '>' (right) per column.
    Defaults to left for all columns.
    """
    if not rows:
        return "No data."
    n_cols = len(headers)
    if alignments is None:
        alignments = ["<"] * n_cols

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells: list[str]) -> str:
        parts = []
        for cell, w, align in zip(cells, widths, alignments):
            if align == ">":
                parts.append(f"{cell:>{w}s}")
            else:
                parts.append(f"{cell:<{w}s}")
        return "  ".join(parts)

    lines = [_fmt_row(headers), "  ".join("-" * w for w in widths)]
    for row in rows:
        lines.append(_fmt_row(row))
    return "\n".join(lines)


# ── Subcommand implementations ───────────────────────────────────────

def cmd_basin_list(args: argparse.Namespace) -> None:
    """List all clusters with hit counts, scalars, and representative text."""
    pipe = _run_pipeline(min_cluster_size=args.min_cluster_size)
    labels = pipe.clusters.labels
    captures = pipe.captures
    embeddings = pipe.embeddings

    cluster_ids = sorted(set(labels))

    headers = ["cluster", "size", "ent_mean", "beta_mean", "comp64_mean", "representative_text"]
    alignments = [">", ">", ">", ">", ">", "<"]
    rows: list[list[str]] = []

    for cid in cluster_ids:
        label_str = str(cid) if cid >= 0 else "noise"
        indices = _cluster_indices(labels, cid)
        size = len(indices)

        ent_vals = [captures[i]["entropy_mean"] for i in indices]
        beta_vals = [captures[i]["heaps_beta"] for i in indices]
        comp64_vals = [captures[i].get("comp_W64", float("nan")) for i in indices]

        ent_mean = np.nanmean(ent_vals)
        beta_mean = np.nanmean(beta_vals)
        comp64_mean = np.nanmean(comp64_vals)

        # Representative text: capture closest to median entropy
        median_ent = np.nanmedian(ent_vals)
        best_idx = min(indices, key=lambda i: abs(captures[i]["entropy_mean"] - median_ent))
        text = _truncate_text(captures[best_idx].get("context_text", captures[best_idx].get("attractor_text", "")), 60)

        rows.append([
            label_str,
            str(size),
            f"{ent_mean:.3f}",
            f"{beta_mean:.3f}",
            f"{comp64_mean:.4f}",
            text,
        ])

    print(f"\n{pipe.clusters.n_clusters} clusters, "
          f"{pipe.clusters.n_noise} noise points, "
          f"{len(captures)} total captures\n")
    print(_format_aligned_table(headers, rows, alignments))


def cmd_basin_show(args: argparse.Namespace) -> None:
    """Show all captures for a given cluster."""
    pipe = _run_pipeline(min_cluster_size=args.min_cluster_size)
    labels = pipe.clusters.labels
    captures = pipe.captures
    embeddings = pipe.embeddings

    cluster_id = args.type_id
    indices = _cluster_indices(labels, cluster_id)

    if not indices:
        print(f"No captures found for cluster {cluster_id}.")
        sys.exit(1)

    centroid = _cluster_centroid(embeddings, indices)

    print(f"\nCluster {cluster_id}: {len(indices)} captures\n")

    headers = ["capture_id", "seed", "step", "T", "entropy", "beta", "comp64", "dist", "text"]
    alignments = [">", ">", ">", ">", ">", ">", ">", ">", "<"]
    rows: list[list[str]] = []

    for i in sorted(indices, key=lambda i: captures[i]["entropy_mean"]):
        cap = captures[i]
        dist = _cosine_distance(embeddings[i], centroid)
        seed = _extract_seed(cap)
        rows.append([
            cap.get("capture_id", "?"),
            str(seed) if seed is not None else "?",
            str(cap.get("capture_step", "?")),
            f"{cap.get('T_capture', 0.0):.2f}",
            f"{cap['entropy_mean']:.3f}",
            f"{cap['heaps_beta']:.3f}",
            f"{cap.get('comp_W64', float('nan')):.4f}",
            f"{dist:.3f}",
            _truncate_text(cap.get("context_text", cap.get("attractor_text", "")), 50),
        ])

    print(_format_aligned_table(headers, rows, alignments))


def cmd_basin_compare(args: argparse.Namespace) -> None:
    """Compare two clusters side-by-side."""
    pipe = _run_pipeline(min_cluster_size=args.min_cluster_size)
    labels = pipe.clusters.labels
    captures = pipe.captures
    embeddings = pipe.embeddings

    id1, id2 = args.id1, args.id2
    idx1 = _cluster_indices(labels, id1)
    idx2 = _cluster_indices(labels, id2)

    if not idx1:
        print(f"No captures found for cluster {id1}.")
        sys.exit(1)
    if not idx2:
        print(f"No captures found for cluster {id2}.")
        sys.exit(1)

    centroid1 = _cluster_centroid(embeddings, idx1)
    centroid2 = _cluster_centroid(embeddings, idx2)
    cos_dist = _cosine_distance(centroid1, centroid2)

    # Scalar summaries
    def _scalar_summary(indices: list[int]) -> dict[str, float]:
        ent = [captures[i]["entropy_mean"] for i in indices]
        beta = [captures[i]["heaps_beta"] for i in indices]
        comp64 = [captures[i].get("comp_W64", float("nan")) for i in indices]
        return {
            "size": len(indices),
            "entropy_mean": float(np.nanmean(ent)),
            "entropy_std": float(np.nanstd(ent)),
            "beta_mean": float(np.nanmean(beta)),
            "beta_std": float(np.nanstd(beta)),
            "comp64_mean": float(np.nanmean(comp64)),
            "comp64_std": float(np.nanstd(comp64)),
        }

    s1 = _scalar_summary(idx1)
    s2 = _scalar_summary(idx2)

    # Representative text for each
    median_ent1 = np.nanmedian([captures[i]["entropy_mean"] for i in idx1])
    median_ent2 = np.nanmedian([captures[i]["entropy_mean"] for i in idx2])
    rep1 = min(idx1, key=lambda i: abs(captures[i]["entropy_mean"] - median_ent1))
    rep2 = min(idx2, key=lambda i: abs(captures[i]["entropy_mean"] - median_ent2))

    print(f"\nComparing cluster {id1} vs cluster {id2}")
    print(f"Centroid cosine distance: {cos_dist:.4f}\n")

    headers = ["metric", f"cluster {id1}", f"cluster {id2}"]
    alignments = ["<", ">", ">"]
    rows = [
        ["size", str(int(s1["size"])), str(int(s2["size"]))],
        ["entropy_mean", f"{s1['entropy_mean']:.3f}", f"{s2['entropy_mean']:.3f}"],
        ["entropy_std", f"{s1['entropy_std']:.3f}", f"{s2['entropy_std']:.3f}"],
        ["beta_mean", f"{s1['beta_mean']:.3f}", f"{s2['beta_mean']:.3f}"],
        ["beta_std", f"{s1['beta_std']:.3f}", f"{s2['beta_std']:.3f}"],
        ["comp64_mean", f"{s1['comp64_mean']:.4f}", f"{s2['comp64_mean']:.4f}"],
        ["comp64_std", f"{s1['comp64_std']:.4f}", f"{s2['comp64_std']:.4f}"],
    ]
    print(_format_aligned_table(headers, rows, alignments))

    print(f"\nRepresentative text (cluster {id1}):")
    print(f"  {_truncate_text(captures[rep1].get('context_text', captures[rep1].get('attractor_text', '')), 120)}")
    print(f"\nRepresentative text (cluster {id2}):")
    print(f"  {_truncate_text(captures[rep2].get('context_text', captures[rep2].get('attractor_text', '')), 120)}")


def cmd_basin_matrix(args: argparse.Namespace) -> None:
    """Print cosine distance matrix (centroid-to-centroid or within-cluster)."""
    pipe = _run_pipeline(min_cluster_size=args.min_cluster_size)
    labels = pipe.clusters.labels
    captures = pipe.captures
    embeddings = pipe.embeddings

    if args.within is not None:
        _matrix_within(labels, captures, embeddings, args.within)
    else:
        _matrix_cross(labels, captures, embeddings)


def _matrix_cross(
    labels: np.ndarray,
    captures: list[dict],
    embeddings: np.ndarray,
) -> None:
    """Cross-cluster centroid cosine distance matrix."""
    cluster_ids = sorted(set(labels) - {-1})
    if not cluster_ids:
        print("No clusters found.")
        return

    centroids = {}
    for cid in cluster_ids:
        indices = _cluster_indices(labels, cid)
        centroids[cid] = _cluster_centroid(embeddings, indices)

    n = len(cluster_ids)
    matrix = np.zeros((n, n))
    for i, cid_i in enumerate(cluster_ids):
        for j, cid_j in enumerate(cluster_ids):
            matrix[i, j] = _cosine_distance(centroids[cid_i], centroids[cid_j])

    # Format as table
    col_width = 7
    id_width = max(4, max(len(str(c)) for c in cluster_ids))

    header = " " * id_width + "  " + "  ".join(f"{c:>{col_width}d}" for c in cluster_ids)
    sep = "-" * len(header)

    print(f"\nCross-cluster centroid cosine distances ({n} clusters)\n")
    print(header)
    print(sep)
    for i, cid_i in enumerate(cluster_ids):
        row_parts = [f"{cid_i:>{id_width}d}"]
        for j in range(n):
            row_parts.append(f"{matrix[i, j]:>{col_width}.4f}")
        print("  ".join(row_parts))


def _matrix_within(
    labels: np.ndarray,
    captures: list[dict],
    embeddings: np.ndarray,
    cluster_id: int,
) -> None:
    """Within-cluster pairwise cosine distance matrix."""
    indices = _cluster_indices(labels, cluster_id)
    if not indices:
        print(f"No captures found for cluster {cluster_id}.")
        sys.exit(1)

    n = len(indices)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = _cosine_distance(embeddings[indices[i]], embeddings[indices[j]])

    # Use capture_id suffixes as labels
    cap_labels = []
    for idx in indices:
        cid = captures[idx].get("capture_id", "?")
        # Shorten: take step part after ':'
        short = cid.split(":")[-1] if ":" in cid else cid[-8:]
        cap_labels.append(short)

    label_width = max(len(l) for l in cap_labels)
    col_width = max(7, label_width)

    print(f"\nWithin-cluster pairwise distances for cluster {cluster_id} ({n} captures)\n")

    header = " " * label_width + "  " + "  ".join(f"{l:>{col_width}s}" for l in cap_labels)
    print(header)
    print("-" * len(header))
    for i in range(n):
        row_parts = [f"{cap_labels[i]:>{label_width}s}"]
        for j in range(n):
            row_parts.append(f"{matrix[i, j]:>{col_width}.4f}")
        print("  ".join(row_parts))

    # Summary stats
    triu = matrix[np.triu_indices(n, k=1)]
    if len(triu) > 0:
        print(f"\nMean pairwise distance: {triu.mean():.4f}")
        print(f"Max pairwise distance:  {triu.max():.4f}")
        print(f"Min pairwise distance:  {triu.min():.4f}")


def cmd_basin_captures(args: argparse.Namespace) -> None:
    """List individual captures with optional filters."""
    pipe = _run_pipeline(min_cluster_size=args.min_cluster_size)
    labels = pipe.clusters.labels
    captures = pipe.captures

    # Apply filters
    filtered: list[tuple[int, dict, int]] = []  # (index, capture, cluster_label)
    for i, cap in enumerate(captures):
        if args.type is not None and labels[i] != args.type:
            continue
        if args.seed is not None:
            seed = _extract_seed(cap)
            if seed != args.seed:
                continue
        filtered.append((i, cap, int(labels[i])))

    if not filtered:
        print("No captures match the given filters.")
        return

    print(f"\n{len(filtered)} captures" +
          (f" (filtered from {len(captures)} total)" if len(filtered) != len(captures) else "") +
          "\n")

    headers = ["capture_id", "cluster", "seed", "step", "T", "entropy", "beta", "comp64", "text"]
    alignments = [">", ">", ">", ">", ">", ">", ">", ">", "<"]
    rows: list[list[str]] = []

    for i, cap, cid in sorted(filtered, key=lambda x: (x[2], x[1]["entropy_mean"])):
        seed = _extract_seed(cap)
        cluster_str = str(cid) if cid >= 0 else "noise"
        rows.append([
            cap.get("capture_id", "?"),
            cluster_str,
            str(seed) if seed is not None else "?",
            str(cap.get("capture_step", "?")),
            f"{cap.get('T_capture', 0.0):.2f}",
            f"{cap['entropy_mean']:.3f}",
            f"{cap['heaps_beta']:.3f}",
            f"{cap.get('comp_W64', float('nan')):.4f}",
            _truncate_text(cap.get("context_text", cap.get("attractor_text", "")), 50),
        ])

    print(_format_aligned_table(headers, rows, alignments))


# ── Parser construction ──────────────────────────────────────────────

def add_basin_subparser(parent_subparsers: argparse._SubParsersAction) -> None:
    """Register `loop basin` subcommand group with all subcommands."""
    p_basin = parent_subparsers.add_parser("basin", help="Explore basin catalogue")
    basin_sub = p_basin.add_subparsers(dest="basin_cmd")

    # Shared argument for HDBSCAN min_cluster_size
    def _add_cluster_arg(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--min-cluster-size", type=int, default=3,
            help="HDBSCAN min_cluster_size (default: 3)",
        )

    # basin list
    p_list = basin_sub.add_parser("list", help="All clusters with scalars and representative text")
    _add_cluster_arg(p_list)

    # basin show <type_id>
    p_show = basin_sub.add_parser("show", help="All captures for a cluster")
    p_show.add_argument("type_id", type=int, help="Cluster ID to inspect")
    _add_cluster_arg(p_show)

    # basin compare <id1> <id2>
    p_compare = basin_sub.add_parser("compare", help="Side-by-side cluster comparison")
    p_compare.add_argument("id1", type=int, help="First cluster ID")
    p_compare.add_argument("id2", type=int, help="Second cluster ID")
    _add_cluster_arg(p_compare)

    # basin matrix [--within <type_id>]
    p_matrix = basin_sub.add_parser("matrix", help="Cosine distance matrix")
    p_matrix.add_argument(
        "--within", type=int, default=None, metavar="TYPE_ID",
        help="Within-cluster pairwise distances instead of cross-cluster",
    )
    _add_cluster_arg(p_matrix)

    # basin captures [--type <id>] [--seed <s>]
    p_captures = basin_sub.add_parser("captures", help="List individual captures with filters")
    p_captures.add_argument("--type", type=int, default=None, help="Filter by cluster ID")
    p_captures.add_argument("--seed", type=int, default=None, help="Filter by seed")
    _add_cluster_arg(p_captures)


_BASIN_DISPATCH: dict[str, callable] = {
    "list": cmd_basin_list,
    "show": cmd_basin_show,
    "compare": cmd_basin_compare,
    "matrix": cmd_basin_matrix,
    "captures": cmd_basin_captures,
}


def cmd_basin(args: argparse.Namespace) -> None:
    """Dispatch basin subcommands."""
    if not hasattr(args, "basin_cmd") or args.basin_cmd is None:
        # Print help for basin subcommand
        print("usage: loop basin {list,show,compare,matrix,captures} ...")
        print("\nSubcommands:")
        print("  list       All clusters with scalars and representative text")
        print("  show       All captures for a cluster")
        print("  compare    Side-by-side cluster comparison")
        print("  matrix     Cosine distance matrix")
        print("  captures   List individual captures with filters")
        sys.exit(1)

    handler = _BASIN_DISPATCH.get(args.basin_cmd)
    if handler is None:
        log.error("Unknown basin subcommand: %s", args.basin_cmd)
        sys.exit(1)

    handler(args)

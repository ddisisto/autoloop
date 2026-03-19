# Observations — 2026-03-19

## L=8 Basin Taxonomy: Cross-seed Structure and Deepening

### Context

L=8 recollection produced 327 captures across 3 seeds (42/123/7), each 100k steps. Clustering simplified to PCA-only features (576→8 dims) with post-hoc centroid merge. HDBSCAN + merge yields 30 clusters + 69 noise (21%). This analysis completes the taxonomy handoff from 2026-03-14.

### Finding: 30 basin types at L=8, two dominant

Final clustering: 30 clusters from 327 captures. Two clusters dominate:

| Cluster | Size | Description |
|---------|------|-------------|
| 16 | 64 | Zeros/numbers (`00000000`) |
| 7 | 49 | Decimal loops (`1.1.1.1.`) |

Together these account for 113/327 (35%) of all captures. The remaining 28 clusters range from 3–19 captures each. Notable types:

- **Python code** (cluster 24, 19 caps): import loops and `self.x = x` attribute patterns
- **Sentence fragments** (cluster 21, 9 caps): "The sentence is incomplete" self-referential loops
- **Medical/health lists** (clusters 23, 27, 28, 29): symptom and condition lists
- **LaTeX** (clusters 13, 14): `\text{}` and `\frac{}` loops
- **Character-level** (cluster 0, 6 caps): `r e r e`, `s e r s`
- **Institutional text** (clusters 2, 8): US government agencies, historical text
- **CJK/Korean** (clusters 9, 15): hash with CJK characters, Korean text
- **Bullet/whitespace** (clusters 3, 5): structural formatting loops

### Finding: Within-basin deepening confirmed (71%)

For consecutive same-cluster captures within a seed (sorted by step), entropy decreases 71% of the time:

- Deeper: 61/86 (71%)
- Shallower: 22/86 (26%)
- Same (±0.01): 3/86 (3%)

Weaker than the pilot finding (86% from cosine-threshold types) but still dominant. The effect is real: once the system enters a basin, small perturbations from heating/cooling cycles tend to tighten the attractor rather than loosen it.

### Finding: Basin discovery not saturating

Last novel cluster discovered at:

| Seed | Captures | Last novel | Position |
|------|----------|------------|----------|
| 42 | 123 | #122 | 99% |
| 7 | 91 | #86 | 95% |
| 123 | 113 | #96 | 85% |

Every seed is still finding new basin types near the end of its run. The long tail of rare types continues — 100k steps per seed is not enough to exhaust L=8's repertoire.

### Finding: Cross-seed structure — 9 universal basins

| Category | Count | Examples |
|----------|-------|---------|
| Universal (3/3 seeds) | 9 | zeros, decimal loops, Python code, sentence fragments, health advice |
| Two-seed | 12 | US institutions, LaTeX, numeric arrays, medical lists |
| Seed-specific | 9 | character loops (S42), bullets (S123), Boolean loops (S7) |

The dominant attractors (zeros, decimal loops) are universal — every seed finds them. Smaller basins are more stochastic: 9/30 appear in only one seed. Noise is evenly distributed across seeds (26/22/21), suggesting it reflects genuine heterogeneity rather than seed-specific artifacts.

### Finding: No grab-bags after merge

The previous 28-cluster solution (before post-hoc centroid merge) flagged 4 grab-bag clusters. The current 30-cluster solution with merge has no obvious grab-bags:

- Cluster 24 (Python code, 19 caps): two sub-populations (imports, self.x patterns) but both are Python structural attractors — coherent
- Cluster 21 (sentence fragments, 9 caps): 8/9 unique text prefixes but all variations on "the sentence is incomplete" — coherent
- Cluster 16 (zeros, 64 caps): large but homogeneous (`00000000`)
- Cluster 7 (decimal loops, 49 caps): large but homogeneous (`1.1.1.1.`)

### Summary

L=8 basin cartography is complete for this data. The model has at least 30 distinct attractor types at L=8, dominated by numeric/structural patterns. Basin deepening is a real effect (71%), discovery is not saturating, and ~30% of basins are universal across seeds. The remaining 21% noise likely contains additional rare types below the min_cluster_size=3 threshold.

### Reproduction

```python
from autoloop.clustering import build_feature_matrix, cluster
result = build_feature_matrix()
cr = cluster(result)
# cr.labels: cluster assignments, -1 = noise
# result.captures: list of dicts with metadata
```

```bash
loop basin list          # overview of all clusters
loop basin show <id>     # detailed view of one cluster
loop basin captures      # all captures with assignments
```

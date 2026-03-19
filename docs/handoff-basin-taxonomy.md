# Handoff: L=8 Basin Taxonomy & Interpretation

Working doc for tagging and interpreting the recollected L=8 basin clusters. Delete when complete.

## Context

L=8 recollection complete (2026-03-14). 327 captures across 3 seeds (42/123/7), each 100k steps. HDBSCAN produced 28 clusters + 85 noise points. Fitted models saved in `data/basins/clustering/`. Pilot data archived in `data/runs/survey/pilot_archive/`.

### What changed from pilot
- Pilot clustering models and data discarded — fresh start with no inherited biases
- Seed 42 ran with no catalogue (everything novel), then clustered to produce initial 9 clusters
- Seeds 123 and 7 ran against those 9 clusters — mostly KNOWN (7-9% novel rate)
- Final recluster on all 327 captures → 28 clusters
- Per-segment logging throttled to every 1000 steps (experiment.py change)

## Step 1: Tag clusters

Use `loop basin list` and `loop basin show <id>` to examine each of the 28 clusters. Assign a short descriptive tag to each. A simple tagging system — no deep interpretation yet, just enough to identify what each cluster *is*.

Grab-bags (clusters 10, 16, 17, 21) need closer inspection — they may contain distinct sub-populations that HDBSCAN couldn't separate, or they may be genuinely heterogeneous basins.

### Cluster summary from validation (for reference)

| Cluster | Size | Entropy | Content sketch |
|---------|------|---------|----------------|
| 0 | 5 | ~1.8 | Character-level loops (`r e r e`, `s e r s`) |
| 1 | 4 | ~3.5 | LaTeX fraction loops (`\frac{1}{2}`) |
| 2 | 4 | ~3.7 | Size/area repetition |
| 3 | 5 | ~1.2 | Bullet loops (`•`) |
| 4 | 5 | ~3.6 | US institutional text |
| 5 | 3 | ~2.7 | Hash/pound loops (`# #`) |
| 6 | 3 | ~3.4 | Hash with CJK text |
| 7 | 7 | ~2.5 | Medical list attractors (cancer, dental) |
| 8 | 3 | ~3.7 | Management/department lists |
| 9 | 3 | ~1.5 | Bullet/whitespace loops (beta=0) |
| 10 | 14 | ~2.3 | **grab-bag** — Python code / imports |
| 11 | 3 | ~2.2 | LaTeX `\text{}` loops |
| 12 | 6 | ~2.1 | Numeric arrays / code with zeros |
| 13 | 5 | ~1.4 | Numeric zeros with Korean text |
| 14 | 3 | ~3.9 | Decimal version loops (`2.1.1.1.`) |
| 15 | 3 | ~3.3 | Mixed — LaTeX/PyCharm/math |
| 16 | 25 | ~3.3 | **grab-bag** — Python code (self., sklearn) |
| 17 | 7 | ~3.6 | **grab-bag** — US history / soul / Bible |
| 18 | 4 | ~3.7 | Decimal loops (`1.1.1.`) variant A |
| 19 | 6 | ~3.7 | Decimal loops (`###` prefix) |
| 20 | 21 | ~3.8 | Decimal loops (canonical `1.1.1.`) |
| 21 | 25 | ~3.7 | **grab-bag** — Instructional/grammar text |
| 22 | 3 | ~3.9 | Decimal loops (pure `.1.1.1.`) |
| 23 | 4 | ~3.6 | Decimal loops (beta>1.0 variant) |
| 24 | 14 | ~3.6 | Health/advice list attractors |
| 25 | 3 | ~3.2 | Medical symptom lists (chest pain, breath) |
| 26 | 50 | ~3.7 | Zeros/numbers (dominant basin) |
| 27 | 4 | ~3.6 | Numbered zeros (`1. 1000...`) |

## Step 2: Analyse cross-seed structure

- Which clusters appear in all 3 seeds? Which are seed-specific?
- Capture frequency per cluster per seed — are some basins more "attractive" than others?
- Does the within-basin deepening finding hold in the new data?

## Step 3: Assess clustering quality

- The 7 decimal-loop sub-clusters (14, 18, 19, 20, 22, 23, and parts of others) — are these genuinely distinct basins or over-splitting?
- The 4 grab-bags — can they be improved by adjusting HDBSCAN params, or are they inherent heterogeneity?
- 85 noise points (26%) — is this acceptable, or should min_cluster_size be lowered?

## Step 4: Re-examine pending findings

With tagged clusters, revisit:
- Within-basin deepening: do consecutive same-cluster captures go deeper?
- Discovery saturation: plot cumulative unique clusters over time
- Universal vs rare: which clusters appear in 1/2/3 seeds?

## Step 5: Update docs

- Record confirmed findings in observations.md
- Update CLAUDE.md current state
- Delete this handoff doc

## Key files

- `autoloop/survey.py` — survey controller with ClusterCatalogue
- `autoloop/clustering.py` — feature extraction, HDBSCAN, centroid computation
- `autoloop/basin.py` — `loop basin` CLI for exploring results
- `scripts/validate_clustering.py` — clustering validation script
- `docs/basin-mapping.md` — consolidated roadmap
- `data/basins/clustering/` — fitted PCA, scaler, cluster centroids
- `data/runs/survey/pilot_archive/` — archived pilot data (do not use for new analysis)

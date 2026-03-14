# Handoff: L=8 Recollection & Interpretation

Working doc for recollecting L=8 survey data and interpreting the results. Delete when complete.

## Context

HDBSCAN clustering pipeline is built and validated on the original L=8 pilot data (201 captures → 22 clusters, 59 noise). The survey code has two fixes applied since the pilot:

1. **Record at gate-fire time** — captures now use the same sensors that triggered detection (eliminates the record-vs-detect mismatch from the pilot)
2. **Segment size = 2*L** — at L=8 that's 16 steps (~2 context rotations) instead of the old 50 (~6.25 rotations). Finer temporal resolution.

Online novelty detection now uses `ClusterCatalogue` (HDBSCAN-based, 14-dim feature space) instead of the old cosine-threshold `CentroidCatalogue`.

## Step 1: Clean up old data

Archive (don't delete yet) the existing survey runs in `data/runs/survey/`:
```
survey_L0008_S42.{parquet,basins.pkl,meta.json,ckpt}
survey_L0008_S123.{parquet,basins.pkl,meta.json,ckpt}
survey_L0008_S7.{parquet,basins.pkl,meta.json,ckpt}
```

Move to `data/runs/survey/pilot_archive/` so they're available for comparison but won't be picked up by the clustering pipeline (`discover_basin_pkls` globs `*.basins.pkl` in the survey dir).

## Step 2: Validate tooling

Before committing to 3x 100k-step runs, smoke test the full pipeline:

```bash
loop survey --seed 42 -L 8 --total-steps 2000
```

Check:
- [ ] Capture fires correctly (beta < 0.40 or comp_W64 < 0.45 after MIN_COOLING_SEGMENTS=5)
- [ ] `ClusterCatalogue.load()` picks up the fitted models from `data/basins/clustering/`
- [ ] Known basins from the pilot are recognized (KNOWN in log), genuinely new ones are NOVEL
- [ ] Capture dict has consistent scalars (sensors match at least one gate)
- [ ] .basins.pkl written correctly with all expected fields

If smoke test passes, delete the smoke test output before proceeding.

## Step 3: Recollect

```bash
loop survey --seed 42 -L 8 --total-steps 100000
loop survey --seed 123 -L 8 --total-steps 100000
loop survey --seed 7 -L 8 --total-steps 100000
```

These can run sequentially (each ~20-30 min on GPU). Monitor the logs for:
- Capture rate (expect 50-200 captures per run at L=8)
- Novel vs known ratio (most should be KNOWN against the pilot clusters, with some NOVEL)
- Any unexpected behavior from the finer segment size

## Step 4: Recluster on fresh data

After recollection, rebuild the clustering pipeline on the new data:

```python
from autoloop.clustering import build_feature_matrix, cluster, save_models, compute_cluster_centroids

result = build_feature_matrix()
cr = cluster(result)
save_models(result)
compute_cluster_centroids(result, cr)
```

Then validate:
```bash
python scripts/validate_clustering.py
loop basin list
```

## Step 5: Interpret

Compare recollected clusters to pilot clusters:
- Did the record-vs-detect fix change capture quality? (Check: are scalars now always consistent with gate thresholds?)
- Did the finer segment size change capture count or timing?
- Do the same 22 cluster types reappear, or does the cleaner data produce a different clustering?
- Are the pending findings confirmed?
  - Within-basin deepening (25/29 recaptures go deeper)
  - Discovery not saturating (long tail of rare types)
  - Universal vs rare types (4 shared, 8 unique)

## Step 6: Update docs

- Record confirmed findings in observations.md
- Update CLAUDE.md current state
- Delete this handoff doc

## Key files

- `autoloop/survey.py` — survey controller with ClusterCatalogue
- `autoloop/clustering.py` — feature extraction, HDBSCAN, centroid computation
- `autoloop/basin.py` — `loop basin` CLI for exploring results
- `autoloop/utils.py` — compressibility normalization
- `scripts/validate_clustering.py` — clustering validation script
- `docs/basin-mapping.md` — consolidated roadmap (pending work section)
- `data/basins/clustering/` — fitted PCA, scaler, cluster centroids

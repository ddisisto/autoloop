# Basin Clustering & Tooling Plan

Working plan for replacing threshold-based novelty detection with proper clustering, and building CLI tools to explore the basin catalogue.

## Context

L=8 pilot survey complete: 3 seeds x 100k steps, 201 captures, 17 types via cosine threshold (0.3). The threshold approach has clear problems:

- Unrelated captures grouped into same type (type 4: "is the one who", "a trigger is a trigger", "U.S. EPA" — max within-type distance 0.476)
- Types that are grab-bags of whatever landed near the centroid (types 1, 6)
- Numeric repetition types (3, 7) absorb everything with low token diversity regardless of structure
- Some types are tight and real (type 2: list-loop structure, mean distance 0.108)

Root cause: 576-dim embeddings are overparameterized for L=8 (only 8 tokens of real information), and cosine distance on raw embeddings doesn't separate structural variants well.

## Phase 0: Compressibility normalization

Gzip has fixed overhead (~20B header + Huffman table) that inflates compression ratios at short byte lengths. At W=16 this pushes ratios above 1.0; even at W=32 it's a measurable bias. Since the compression spectrum is 5 of 16 clustering dimensions, this length-dependent artifact would distort cluster distances.

**Fix:** normalize raw compression ratios against an incompressible baseline at matched byte length. For each byte length N, compress random bytes to get the baseline ratio, then divide through: `normalized = raw_ratio / baseline_ratio`. This maps incompressible content to ~1.0 regardless of W, removing the overhead bias.

Implementation: `normalized_compressibility()` and `compressibility_baseline()` in `autoloop/utils.py`. Baselines are cached per byte length (deterministic enough after averaging 8 random samples).

**Applied at feature-extraction time, not capture time.** Raw comp_W values stay in `.basins.pkl` and SQLite — normalization happens when building the clustering feature matrix. This keeps existing data valid and lets the normalization be adjusted without re-running surveys.

## Phase 1: Recluster L=8

### Feature engineering

Joint feature vector per capture (16 dims):

1. **Embedding projection**: PCA(576 → 8) on all L=8 capture embeddings. 8 dims to match context length — captures the real structure, discards hidden-state noise. Save PCA model for projecting future L=8 captures.
2. **Compression spectrum**: normalized comp_W16, comp_W32, comp_W64, comp_W128, comp_W256 (5 dims). Normalized against incompressible baseline at matched byte length (see Phase 0). Fingerprints cycle structure at multiple scales without gzip overhead bias.
3. **Scalar metrics**: entropy_mean, heaps_beta (2 dims). Basin depth indicators.
4. **Context length**: L normalized to [0,1] (1 dim). Constant for L=8-only data (zero impact on clustering), but ready for cross-L integration when multi-L data arrives. Cheap to include; provides gentle separation pressure at different L without dominating the vector.

Normalize each feature to unit variance before clustering.

### Clustering

HDBSCAN with min_cluster_size=3 (or tune). Properties we want:
- Variable-density clusters (tight numeric basins and loose structural types can coexist)
- Explicit noise/outlier labels for rare one-off captures
- No forced assignment — singletons stay as singletons

### Capture selection

The deepening sequences (same basin, progressively lower entropy across consecutive captures) mean one basin visit produces 3-5 captures at different lock-in stages. Options:

- **Cluster all captures**: depth variation becomes a feature, not noise. Deepening sequences should form elongated clusters.
- **Cluster on first-per-visit only**: cleaner entry-point comparison across seeds. Deduplicates depth stages.

Start with all captures. If depth dominates, revisit.

### Validation

- Do the "good" types (type 2 list-loops) stay together?
- Do the "bad" types (type 4 grab-bag) get split?
- Does "1.1.1." vs "000000" cluster together or apart? (Answer is whatever the model's geometry says — we map, not interpret.)
- Within-cluster text inspection: do members share structural identity?

### Outputs

- New `basin_types` table (or version bump) with cluster labels
- Per-capture cluster assignment stored in basins.pkl or SQLite
- PCA model + scaler saved for online use during future surveys

## Phase 2: CLI tooling (`loop basin ...`)

Subcommands for exploring the catalogue:

```
loop basin list                          # all types with hit counts, scalars, representative text
loop basin show <type_id>                # all captures for a type: scalars, text, embedding distances
loop basin compare <id1> <id2>           # side-by-side: scalars, text, cosine distance, shared features
loop basin matrix                        # cross-type cosine distance matrix (centroid-to-centroid)
loop basin matrix --within <type_id>     # within-type pairwise distances
loop basin captures [--type <id>] [--seed <s>]  # list individual captures with filters
```

Design notes:
- Operates on whatever clustering is current (threshold or HDBSCAN)
- Text output, not plots (terminal-first)
- Could add `--json` for programmatic use

## Phase 3: Recollect L=8

Survey code fixes already applied:
- **Record-vs-detect mismatch fixed**: capture now recorded immediately when the gate fires (same sensors that triggered detection). CAPTURED state eliminated — state machine is COOLING → HEATING → TRANSIT → COOLING.
- **Segment size**: `max(L, 50)` → `2 * L`. At L=8 that's 16 steps (~2 context rotations) instead of 50 (~6.25 rotations). Finer temporal resolution for capture detection.

Steps:
1. Smoke test with new segment size — verify MIN_COOLING_SEGMENTS=5 still reasonable (80 steps = 10 rotations at L=8)
2. Check capture gate thresholds still appropriate at finer resolution (β < 0.40, comp_W64 < 0.45)
3. Recollect all 3 seeds (42, 123, 7) x 100k steps
4. Verify mismatch is gone: all captures should have scalars consistent with at least one gate

### Online clustering integration (future)

Once HDBSCAN model exists, the survey can use it for online novelty detection instead of cosine threshold:
- Project new capture embedding through saved PCA
- Build joint feature vector
- Approximate cluster membership (nearest cluster centroid in feature space, or HDBSCAN.approximate_predict)
- Novel if no cluster assignment or distance exceeds learned cluster boundary

This replaces the `CentroidCatalogue` with something more principled.

### Adaptive heating rate (future)

Currently `dT_frac` is fixed at 5%. For novel basins, slower heating would map the deepening trajectory and escape temperature more precisely. For known basins, faster heating saves compute. Extension: after `_record_capture`, adjust `dT_frac` based on novelty — halve for novel, double for known.

## Phase 4: Cross-L (deferred)

Each L gets its own PCA projection and clustering (PCA dims = L). The L dimension in the feature vector provides gentle cross-L separation pressure. Cross-L basin correspondence is a separate analysis problem — approaches:
- Scalar features that are L-comparable (comp spectrum shape, entropy range, beta range)
- Raw 576-dim embeddings across all L values (same model hidden state space) with UMAP/HDBSCAN
- Multi-scale PCA projection (e.g., project L=64 data to 8, 64, 256 dims) to look for cross-scale structure — hypothesis that alignment is constrained, not coincidental, at sufficient N

## Findings to record (pending confirmation after recollection)

- **Within-basin deepening**: 25/29 consecutive same-type recaptures go deeper (mean delta_ent = -0.37). Small perturbation (5% dT heating) consistently tightens the cycle. Mechanistic — attractor dynamics.
- **Discovery not saturating at L=8**: 8 types (seed 42) + 3 new (seed 123) + 6 new (seed 7). Last novel type at 98% through third run. Long tail of rare basins.
- **4 universal types** (shared all 3 seeds): 0, 3, 5, 7. These are the dominant attractors any seed will find. 8 types unique to one seed — rare basins that depend on initial conditions.

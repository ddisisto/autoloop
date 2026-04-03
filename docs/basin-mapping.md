# Basin Mapping

## Overview

The phase diagram maps regimes. Basin mapping maps *content*. Each attractor basin is a mode the model "knows how to do" — a register, topic, format, or genre strong enough to capture the system at a given (T, L) operating point. The set of all recoverable basins is an empirical map of the model's behavioral repertoire, extracted from output dynamics without inspecting weights.

## System

SmolLM-135M, autoregressive free generation. No external input after initial seed.

**Measured T_escape(L):**

| L | T_escape | Source |
|---|----------|--------|
| 8 | ~0.50 | controller runs |
| 16 | ~0.50 | controller runs |
| 64 | 0.55 | sweep |
| 128 | 0.57 | sweep |
| 192 | 0.67 | sweep |
| 256 | 0.87 | sweep |
| 512 | ~0.90 | sweep |

## Survey Protocol

The SurveyController runs COOLING→HEATING→TRANSIT cycles at fixed L, sweeping temperature down until a basin is captured, then heating to escape.

**Current gate:** LZ_W64 + Heaps' β (dual gate, either fires). See `docs/observations-2026-03-19b.md` for LZ validation.

**Planned redesign:** Replace with surprisal-only gate. Mean segment surprisal < threshold detects degeneration directly — the point where enrichment fraction drops to zero and the system enters complete self-prediction. No windowing, no L-dependent parameters, no additional computation. See `docs/survey-redesign.md` for full rationale and validation plan.

- **Escape detection:** entropy rises 1.0 nat above basin floor, or T hits ceiling
- **Segment size:** 2×L tokens per segment (10+ context rotations at MIN_COOLING_SEGMENTS=5)
- **Per-capture data:** 576-dim embedding, LZ spectrum, gzip comp spectrum, context text, attractor text, scalar metrics

## Clustering Pipeline

### Feature extraction
PCA(576→8) on model embeddings only. Compression spectrum, entropy, beta, and L are excluded — they describe observation conditions, not basin identity. The embedding already encodes structural and semantic properties of the basin.

### Clustering
HDBSCAN (min_cluster_size=3) on unit-variance-scaled 8-dim features, followed by post-hoc agglomerative merge of clusters whose centroids are within a threshold distance (MERGE_THRESHOLD=0.5). Merge prevents over-splitting without forcing assignment.

### Online novelty detection
ClusterCatalogue projects new captures through saved PCA+scaler into the 8-dim feature space, matches against precomputed cluster centroids with per-cluster radius thresholds (1.5× max training radius). Novel captures get provisional IDs and are saved for the next recluster.

### Persistence
- Fitted PCA + StandardScaler: `data/basins/clustering/feature_models.pkl`
- Cluster centroids + radii: `data/basins/clustering/cluster_centroids.pkl`
- Provisional captures: `data/basins/clustering/provisional_captures.pkl`
- Per-run captures with embeddings: `data/runs/survey/*.basins.pkl`

## Collected Data

### L=8 — Complete (2026-03-14)

3 seeds (42/123/7) × 100k steps. 327 captures, 30 clusters + 69 noise (21%).

- Two dominant attractors: zeros/numbers (64 caps), decimal loops (49 caps) — 35% of all captures
- 9 universal basins (all 3 seeds), 12 two-seed, 9 seed-specific
- Within-basin deepening confirmed: 71% of consecutive same-cluster recaptures go deeper
- Discovery not saturating: last novel type at 85-99% through each seed's run
- Full analysis: [observations-2026-03-19.md](observations-2026-03-19.md)

### L=12 — Collected (2026-03-19), analysis pending

3 seeds (42/123/7) × 100k steps. 498 captures (151 + 194 + 153). First survey with LZ capture gates.

L=12-alone clustering: 50 clusters, 155 noise (31%). More types than L=8 — longer context window supports more diverse template attractors (list cycling, phrase repetition) alongside the tight verbatim loops.

### Preliminary cross-L findings (L=8 + L=12 joint)

Joint clustering on all 825 captures (PCA refitted on combined embeddings): 75 clusters, 290 noise (35%).

| Category | Count |
|----------|-------|
| Mixed (both L=8 and L=12) | 32 |
| L=8 only | 16 |
| L=12 only | 27 |

**Basins that persist across L:** decimal loops (L8=49, L12=34), zeros (L8=61, L12=11), Python code, sentence incompleteness, health advice, LaTeX fractions. These are real attractors in the model's dynamics, not artifacts of a specific context length.

**Basins that disappear at L=12:** character-level loops (`r e r e`, `Boolean,`), single-token patterns, Korean text. These need the tight 8-token window; at 12 tokens the pattern can't fill enough of the context to self-sustain.

**Basins that emerge at L=12:** `#include` loops, longer medical lists, counting sequences, XPath patterns. These need the extra context — the template period exceeds 8 tokens.

**Distribution shift:** zeros dominate at L=8 (64 caps) but recede at L=12 (11 caps). Health advice blooms from 1 to 19 caps. Template/list attractors replace tight verbatim loops as the dominant basin type. This is consistent with the lock-in threshold (~4-8 copies of the cycle in context): at L=12 a 3-token phrase gets only 4 copies, barely enough to lock; at L=8 it gets only 2-3 copies and can't sustain.

These findings are preliminary — the joint clustering has not been validated or persisted. Proper cross-L analysis is pending.

## L-Ladder

Progress through L values, collecting 3 seeds × 100k steps at each depth before advancing.

**L sequence (each step ≤50% increase):** 8 ✓, 12 ✓, 16, 24, 32, 48, 64, 96, 128, 192, 256

**L-dependent parameters:**

| L | T_min | T_escape | T_max | Expected cycles/100k |
|---|-------|----------|-------|---------------------|
| 8–32 | 0.10 | ~0.50 | 0.70 | 50–200 |
| 48–96 | 0.30 | 0.55–0.60 | 0.80 | 10–50 |
| 128–192 | 0.40 | 0.57–0.67 | 0.90 | 5–20 |
| 256 | 0.50 | ~0.87 | 1.00 | 5–15 |

**Saturation criterion:** advance when the last N captures all match existing types. Not yet reached at L=8 or L=12 — discovery is still active.

## Tooling Roadmap

### Persistent clustering (next)

The basin CLI (`loop basin`) currently re-runs the full clustering pipeline per session. As capture counts grow, this needs persistent results:

- `loop basin recluster` — run full pipeline (load captures → PCA → HDBSCAN → merge → save), persist cluster labels and assignment to disk alongside models/centroids
- `loop basin list/show/compare` — load from persisted results, no re-clustering
- Support L-filtered and joint clustering modes

### Cross-L analysis

Joint clustering across L values uses the same 576-dim embedding space (same model hidden states regardless of L). The current approach — refit PCA on the combined set, run HDBSCAN — works at small scale. Needs:

- Per-L and joint clustering as first-class operations
- Basin correspondence tracking: which L=8 clusters map to which L=12 clusters
- Minimum L for each basin type: the shortest context that can sustain it

### LZ complexity optimization

`lz76_complexity()` is O(W²) per call due to Python set operations on tuples. Acceptable at current scale (~500 captures per L), but will become a bottleneck in batch analysis (`sliding_lz_complexity` over 100k steps at W=256). Options:

- C extension or Cython implementation (10-100x speedup)
- Numpy-based phrase counting avoiding set/tuple overhead
- Pre-compute and cache LZ arrays per run (already done via analysis cache)

### Adaptive heating rate

Currently `dT_frac` is fixed at 5%. Adjust based on novelty after capture: halve for novel basins (map deepening trajectory more precisely), double for known basins (save compute). Not yet implemented.

## Analysis Targets

1. **Basin census per L.** Does type count scale with L, saturate, or peak?
2. **Cross-L correspondence.** Minimum L for each basin type = minimum context to express that mode. This connects to Framework prediction 7: in-context learning fails when the required regularity type is absent from the weight geometry. Each basin's minimum L is the minimum context for that regularity to be expressible.
3. **Transition graph.** Directed graph: nodes = types, edges = observed transitions. Identify hubs, dead ends, connected components.
4. **Basin characterization spectrum.** LZ spectrum shape, surprisal profile, and gap dynamics per basin type. These describe the *kind* of attractor — structural vs semantic, tight vs template — without controlling gating.
5. **Block entropy scaling (future).** Entropy rate as function of block length, computed per basin. Excess entropy approximates statistical complexity — Framework's central quantity. This would directly test prediction 8: do basins with higher statistical complexity correspond to richer content-generation capabilities?

## Framework Alignment

This project is the empirical counterpart to `../framework`. The mapping:

| Framework concept | autoloop operationalisation |
|---|---|
| Degeneration (enrichment → 0) | Surprisal gate (surprisal → 0 = degenerate) |
| Compressive novelty | Entropy-surprisal gap (per-token) |
| Statistical complexity | Block entropy scaling (future, per-basin) |
| Capability as discrete mode | Basin = a content-generation mode the model possesses |
| Capability threshold | Basin existence boundary in (L, T) space |
| In-context learning failure | Basin's minimum L — context too short to express the regularity |

The basin catalogue is an empirical inventory of the model's representational modes, discovered from output dynamics. Each basin is a "thing the model knows how to do." The L-ladder maps which capabilities survive, emerge, or disappear as context grows. The transition graph maps how they connect.

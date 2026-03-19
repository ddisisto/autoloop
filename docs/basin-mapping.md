# Basin Mapping

## Overview

The phase diagram maps regimes. Basin mapping maps *content*. Each attractor basin is a mode the model "knows how to do" — a register, topic, format, or genre strong enough to capture the system at a given (T, L) operating point. The set of all recoverable basins is an empirical map of the model's behavioral repertoire, extracted from output dynamics without inspecting weights.

Previous work treated basins as obstacles to avoid. This reframes them as the primary object of study. The controller problem (navigate *between* basins) depends on first solving the cartography problem (what basins exist, where are they, what connects to what).

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

## Pending Work

### ~~Recollect L=8~~ DONE

Recollected 2026-03-14 with gate-fire recording and segment_size=2*L. Pilot data archived in `data/runs/survey/pilot_archive/`, old clustering models cleared. Fresh start.

- Seed 42: 123 captures (all novel — no catalogue yet). Clustered → 9 clusters + 27 noise
- Seed 123: 113 captures (8 novel, 105 known against seed 42 clusters)
- Seed 7: 91 captures (8 novel, 83 known)
- Full recluster on all 327 captures → 30 clusters + 69 noise (after post-hoc centroid merge)
- Per-segment logging throttled to every 1000 steps (transition/capture events still log immediately)

### Adaptive heating rate

Currently `dT_frac` is fixed at 5%. Adjust based on novelty after `_record_capture`: halve for novel basins (map deepening trajectory and escape T more precisely), double for known basins (save compute).

### ~~Basin taxonomy~~ DONE

Full analysis in [observations-2026-03-19.md](observations-2026-03-19.md). Summary:

- 30 clusters from 327 captures (PCA-only features + post-hoc centroid merge)
- Two dominant: zeros/numbers (64 caps), decimal loops (49 caps) — 35% of all captures
- 9 universal basins (all 3 seeds), 12 two-seed, 9 seed-specific
- Within-basin deepening confirmed: 71% of consecutive same-cluster recaptures go deeper
- Discovery not saturating: last novel type at 85-99% through each seed's run
- No grab-bags after merge — all clusters internally coherent
- 21% noise (69 points), likely containing additional rare types below min_cluster_size=3

### Adaptive L-Ladder

Progress through L values adaptively. Only advance when basin discovery saturates at the current depth.

**L sequence (each step ≤50% increase):** 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256

**Saturation criterion:** if the last N captures all match existing types, the current L is exhausted. Advance and check which basins survive, disappear, or emerge.

**L-dependent parameters:**

| L | T_min | T_escape | T_max | Expected cycles/100k |
|---|-------|----------|-------|---------------------|
| 8–32 | 0.10 | ~0.50 | 0.70 | 50–200 |
| 48–96 | 0.30 | 0.55–0.60 | 0.80 | 10–50 |
| 128–192 | 0.40 | 0.57–0.67 | 0.90 | 5–20 |
| 256 | 0.50 | ~0.87 | 1.00 | 5–15 |

### Cross-L clustering

Each L gets its own PCA projection and clustering (PCA dims = L). Cross-L basin correspondence is a separate analysis problem. Approaches:

- Scalar features that are L-comparable (comp spectrum shape, entropy range, beta range)
- Raw 576-dim embeddings across all L values (same hidden state space) with UMAP/HDBSCAN
- Multi-scale PCA projection to look for cross-scale structure

### Learned controller

Train a small model on (sensor_state → action) pairs from existing controller runs. Input is ~10D sensor state, output is 2D (delta_T, delta_L).

**Objectives (in order of data readiness):**
- **Beta-tracking:** minimize |beta - target| over time. Trainable today on existing data.
- **Exploration:** maximize unique basin fingerprints per unit time. Requires survey data.
- **Exploitation:** reach a target basin from arbitrary starting point. Requires basin catalogue.

### Analysis targets

1. **Basin census.** Unique fingerprints at each L. Does count scale with L, saturate, or peak?
2. **Transition graph.** Directed graph: nodes = types, edges = observed transitions. Identify hubs, dead ends, connected components.
3. **Depth vs spectrum.** Can depth be predicted from compression spectrum alone?
4. **Cross-L correspondence.** Minimum L for each basin type = minimum context to express that mode.
5. **Residue network.** Gzip dictionary overlap across transitions — the transition kernel's content structure.

## Architecture

### Layer 1 — Terrain (SmolLM-135M)
The model generates tokens autoregressively. It falls into basins. It does not know what it is doing.

### Layer 2 — Perception (sensors)
Compression spectrum, entropy, Heaps' beta, and EOS rate transform the raw token stream into a ~10-dimensional state vector. Mechanistic, fully automatable. Basins cluster naturally in this space. Built and operational.

### Layer 3 — Navigation (controller)
**Input:** sensor state vector. **Output:** (delta_T, delta_L).

Three tiers: rule-based (built — BetaController, SurveyController), learned policy (next — train on sensor→action pairs), adaptive (future — online learning during generation with bandit-style exploration/exploitation).

## Key Design Decisions

**8-dim feature vector.** PCA(576→8) on model embeddings only. Compression spectrum, entropy, beta, and L are excluded — they describe observation conditions, not basin identity. The embedding already encodes structural and semantic properties of the basin.

**HDBSCAN + post-hoc centroid merge.** HDBSCAN for variable-density clusters with explicit noise labels. Post-hoc agglomerative merge consolidates clusters whose centroids are within a threshold distance, preventing over-splitting. 30 clusters from 327 L=8 captures.

**Online novelty detection.** ClusterCatalogue projects new captures through saved PCA+scaler into the 8-dim feature space, matches against precomputed cluster centroids with per-cluster radius thresholds.

## Compression as Identity

The gzip compression dictionary at the window size of best compression (W*) *is* the basin's mechanistic identity. The repeated byte sequences gzip finds are not measurements of the attractor — they are the attractor's constituent structure.

**W*** is the characteristic scale of the attractor:
- Verbatim 12-token loop: W* ≈ 24 (two repetitions for gzip to find the match)
- Template attractor (header → definition → header): W* ≈ template period
- Self-referential attractor: W* → L (structure visible only at context scale)

**The compressibility-vs-W spectrum** is the basin's signature. The shape — which W dominates, where the ratio is minimized, how steeply it falls off — distinguishes attractor types mechanistically. This replaces manual classification with an empirical, continuous, clusterable representation.

**Basin identity criterion:** Two captures represent the same basin iff they produce the same gzip dictionary content at the same W*. Mechanistic identity, not semantic. The dictionary doesn't know what "Star Wars" means. It knows it repeats. That's sufficient.

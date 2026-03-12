# Basin Mapping

## Motivation

The phase diagram maps regimes. Basin mapping maps *content*. Each attractor basin is a mode the model "knows how to do" — a register, topic, format, or genre strong enough to capture the system at a given (T, L) operating point. The set of all recoverable basins is an empirical map of the model's behavioral repertoire, extracted from output dynamics without inspecting weights.

Previous work treated basins as obstacles to avoid. This reframes them as the primary object of study. The controller problem (navigate *between* basins) depends on first solving the cartography problem (what basins exist, where are they, what connects to what).

## System

SmolLM-135M, autoregressive free generation. No external input after initial seed. PRNG checkpointed at every step.

**Parameter ranges:**
- T (temperature): 0.50–1.50
- L (context length): 8–1024
- W (measurement window): {16, 32, 64, 128, 256}

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

## Infrastructure

**Built:**
- `StepEngine`: single token loop, trailing-window sensors (entropy, β, comp), `snapshot()`/`restore()` rollback, checkpoint persistence
- `SurveyController`: direct state tracking (COOLING → CAPTURED → HEATING → TRANSIT) with gradient T ramps
- `CentroidCatalogue`: load/save centroids, cosine novelty detection, type registration
- `BetaController`: hill-climb controller, proves the sensor→action loop works
- `loop survey`: CLI entry point for basin survey runs
- `loop grep`: decoded text search across runs — motif hunting across the corpus

## Compression as Identity

The gzip compression dictionary at the window size of best compression (W*) *is* the basin's mechanistic identity. The repeated byte sequences gzip finds are not measurements of the attractor — they are the attractor's constituent structure.

**W*** (the window size at which compressibility is minimized / compression is best) is the characteristic scale of the attractor:
- Verbatim 12-token loop: W* ≈ 24 (two repetitions for gzip to find the match)
- Template attractor (header → definition → header): W* ≈ template period
- Self-referential attractor: W* → L (structure visible only at context scale)

**The compressibility-vs-W spectrum** is the basin's signature. The shape — which W dominates, where the ratio is minimized, how steeply it falls off — distinguishes attractor types mechanistically. This replaces manual attractor type classification (repetition / template / self-referential / paraphrase) with an empirical, continuous, clusterable representation.

**Basin identity criterion:** Two basin captures represent the same basin iff they produce the same gzip dictionary content at the same W*. Mechanistic identity, not semantic. The dictionary doesn't know what " Star Wars" means. It knows it repeats. That's sufficient.

**Implication:** The entire taxonomy is automatable. Cluster basins by their compressibility-vs-W profiles. The clusters *are* the types. Human-readable labels ("mathematics," "content-mill," "astronomy") are a gloss on what the compression spectrum already encodes.

## Survey Protocol

### State Machine

```
COOLING  ──[basin_detected]──→  CAPTURED
CAPTURED ──[characterised]───→  HEATING
HEATING  ──[escaped]─────────→  TRANSIT
HEATING  ──[deeper_basin]────→  CAPTURED
TRANSIT  ──[context_flushed]─→  COOLING
```

### COOLING (basin capture)

Ramp T down from T_max toward T_min (multiplicative steps, `dT_frac` per segment). Generate until sensors indicate capture:
- Entropy delta between consecutive segments drops below threshold
- Stability persists for N consecutive readings
- Minimum cooling segments before capture can trigger (let system settle)

**Transition → CAPTURED:** entropy stabilized (N consecutive low-delta segments).

### CAPTURED (characterisation)

Transient state — record capture and immediately transition. At the basin record point, record:

**Compression spectrum (primary identity):**
- comp_W at W ∈ {16, 32, 64, 128, 256}
- W*: window of best compression
- Decoupling index: comp_W64 − comp_W256

**Sensor profile:**
- Entropy: mean, std, floor
- EOS rate
- Heaps' β (trailing window)

**Embedding:**
- Mean-pooled last-layer hidden state (576-dim) from attractor text
- Cosine distance to all known centroids for novelty detection

**Deferred — depth profile (via checkpoint forking):**
- `engine.snapshot()` at basin record point
- Fork branches with perturbed T and L
- Measure steps before collapse or escape in each branch
- `engine.restore()` after each probe
- Depth score = mean elaboration steps across forks

**Transition → HEATING:** characterisation complete (all metrics recorded).

### HEATING (escape)

Ramp T up toward T_max (multiplicative steps). Record:
- Steps to escape (entropy rises above floor + ESCAPE_ENTROPY_RISE)
- T at which escape occurs

**Transition → TRANSIT:** entropy rise detected, or T reaches T_max ceiling.
**Transition → CAPTURED:** entropy drops to new floor (deeper basin found during heating).

### TRANSIT (context flush)

Hold T at escape level for ≥L tokens (context turnover). Then begin cooling again.

**Transition → COOLING:** L tokens generated since escape.

### Cycle Yield

One cycle = one basin record. Yield depends on L: shallow basins cycle fast, deep basins cycle slow.

## Survey Design

### Adaptive L-Ladder

Instead of a fixed grid, the survey progresses through L values adaptively. Start at L=8 where cycling is fast and cheap. Only advance to the next L when basin discovery saturates at the current depth.

**L sequence (each step ≤50% increase):** 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256

**Saturation criterion:** if the last N captures all match existing basin types (by embedding distance — see below), the current L is exhausted. Advance to the next L and check which basins survive, which disappear, and which new ones appear.

**Why adaptive:**
- L=8 is cheap (~50–200 cycles per 100k tokens). Build the clustering pipeline on abundant, fast data before committing to expensive deep runs.
- Spend compute where the diversity is. Maybe L=8 has 5 types found in 20k tokens. Maybe L=64 has 30 and needs 200k.
- Cross-L correspondence builds incrementally: at each new L, you immediately see which previous basins persist and which are new.
- Fine L-steps reveal where basin types appear or vanish. If L=16 and L=24 have wildly different structure, that transition boundary is interesting.

### L-Dependent Parameters

| L | T_min | T_escape | T_max | Expected cycles/100k |
|---|-------|----------|-------|---------------------|
| 8–32 | 0.10 | ~0.50 | 0.70 | 50–200 (shallow, fast cycling) |
| 48–96 | 0.30 | 0.55–0.60 | 0.80 | 10–50 (core range) |
| 128–192 | 0.40 | 0.57–0.67 | 0.90 | 5–20 |
| 256 | 0.50 | ~0.87 | 1.00 | 5–15 (deep, rich characterisation) |

### Basin Identity: Dual Distance

Two distance metrics for basin matching:

**Embedding distance (primary, online).** SmolLM-135M is already loaded for generation. At capture time, embed the attractor text (trailing W* tokens) through the model and extract the mean hidden state. Cosine distance between embeddings gives semantic similarity. This is essentially free — no additional model load, just a forward pass on text already in memory. New captures are compared to all existing centroids; if min distance > threshold, it's a novel basin type.

**Compression spectrum distance (secondary, mechanistic).** The 5-element comp_W vector is a low-dimensional mechanistic fingerprint. Two basins with similar embeddings but different spectra are the same topic with different structure. Two basins with similar spectra but different embeddings are different topics with the same structure. Both cases are informative.

**When they agree:** solid basin identity, high confidence in type assignment.
**When they disagree:** interesting — flag for analysis. Semantic-structural decoupling may reveal how the model organizes its modes.

### Seeding Strategy

Initial context determines which basin neighborhood the system enters first:
- **Null seeds**: BOS only. Reveals highest-probability basins — modes strong enough to emerge from zero context
- **Domain seeds**: short text fragments (LaTeX, code, news, recipe). Biases toward domain basins
- **Cross-domain seeds**: juxtaposed fragments. Tests hybrid basin existence

## Basin Catalogue Schema

Two-level structure: **types** (the taxonomy) and **captures** (observations).

### basin_types (SQLite)
Each row is a distinct attractor type. Centroid embedding stored externally in `data/basins/centroids.npy` (row-aligned by type_id).

```
type_id            # integer PK, aligns with centroids.npy row
hit_count          # number of captures assigned to this type
first_seen_run/step, last_seen_run/step
min_L, max_L       # L range where this type appears
comp_W{16,32,64,128,256}  # centroid compression spectrum
W_star             # characteristic window size
entropy_mean/std/floor, heaps_beta
representative_text # from most typical capture
label              # optional human-readable name
```

### basin_captures (SQLite)
Every observation of the system in a basin. Multiple captures of the same type are valuable — each is a new depth measurement, operating point, and transition edge.

```
capture_id         # run_id:capture_step
run_id             # parent survey run (FK → runs)
type_id            # assigned basin type (FK → basin_types)
capture_step, record_step
L, T_capture
comp_W{16,32,64,128,256}, W_star
entropy_mean/std/floor, heaps_beta, eos_rate
depth_score        # mean elaboration steps under perturbation (deferred)
escape_T, escape_steps
attractor_text, attractor_period
novelty_distance   # cosine distance to nearest type at capture time
prev_capture_id, next_capture_id
```

### Storage tiers
| What | Where | Why |
|------|-------|-----|
| Per-capture embeddings (576-dim) | `.basins.pkl` per survey run | Full fidelity, reanalysis |
| Type centroids (N_types × 576) | `data/basins/centroids.npy` + `.json` | Fast load at survey startup |
| Scalar summaries | SQLite `basin_types` + `basin_captures` | Queryable via `loop index` |

## Analysis Targets

1. **Basin census.** Unique fingerprints at each L. Does count scale with L, saturate, or peak?
2. **Basin type clustering.** Cluster on comp_spectrum shape. Do clusters correspond to manually identified types?
3. **Transition graph.** Directed graph: nodes = fingerprints, edges = observed transitions. Identify hubs, dead ends, connected components. This is the model's associative topology.
4. **Depth vs. spectrum.** Can depth be predicted from compression spectrum alone? If yes, the controller can estimate depth without forking.
5. **Cross-L correspondence.** Which basins exist at L=8? Which require L≥256? Minimum L for a basin = minimum context to express that mode.
6. **Residue network.** Gzip dictionary overlap across transitions. The overlap matrix is the transition kernel's content structure.

## Architecture

### Layer 1 — Terrain (SmolLM-135M)
The model generates tokens autoregressively. It falls into basins. It does not know what it is doing.

### Layer 2 — Perception (sensors)
Compression spectrum, entropy, Heaps' β, and EOS rate transform the raw token stream into a ~10-dimensional state vector. Mechanistic, fully automatable. Basins cluster naturally in this space. **Status: built.** The engine computes these in real-time after each segment.

### Layer 3 — Navigation (controller)
**Input:** sensor state vector. **Output:** (ΔT, ΔL).

Three tiers of increasing sophistication:

**Tier A — Rule-based (built).** `BetaController` for β-targeting and `SurveyController` for basin survey. Fixed rules: ramp T based on state, detect transitions via sensor thresholds. Good enough for systematic survey and β-targeting.

**Tier B — Learned policy (next).** Train a small model on (sensor_state → action) pairs from existing controller runs. The input space is ~10D, output is 2D (ΔT, ΔL). A linear model or small MLP may suffice — the state space is well-structured.

Training data comes from per-step sensor readings in parquet files. Controller and survey runs record temperature, context_length, entropy, and other signals at every token.

Candidate objectives:
- **β-tracking:** minimize |β − target| over time (supervised, regression on existing data)
- **Exploration:** maximize unique basin fingerprints per unit time (requires survey runs)
- **Exploitation:** reach a target basin from arbitrary starting point (requires basin catalogue)

The β-tracking objective can be trained today with zero new data. Exploration and exploitation objectives require the survey protocol to generate training signal.

**Tier C — Adaptive (future).** Online learning during generation. The controller updates its policy as it discovers new basins. Bandit-style exploration/exploitation tradeoff over the basin landscape.

## Scaling Considerations

The framework — compression-based state representation, learned controller, basin catalogue — is model-agnostic. SmolLM-135M is the training ground. The basins found here are modes strong enough to survive 135M parameters of compression — the skeleton of the training distribution.

Larger models: more basins, deeper basins, richer depth profiles. The compression spectrum still works. The controller architecture transfers; the policy needs retraining.

Frontier models with tool use: action space grows beyond (T, L). State space gains tool-use indicators. The principle is identical.

## Roadmap

### Phase 1 — Pilot at L=8
- ~~Implement basin survey (`loop survey`)~~  done
- ~~Embedding extraction at capture time~~  done
- ~~Online novelty detection: cosine distance to existing centroids~~  done
- Tune capture detection thresholds (see [basin-mapping-phase1.md](basin-mapping-phase1.md))
- L=8 null seed, run until saturation
- Validate: embedding clusters and compression spectrum clusters agree

### Phase 2 — Adaptive L-Ladder
- Progress through L=12, 16, 24, 32, 48, 64, 96, 128, 192, 256
- At each L: run until saturation, then advance
- Track which basin types survive, disappear, or emerge at each L step
- Cross-L correspondence builds incrementally
- Domain seeds at selected L values to probe seeded basins

### Phase 3 — Learned Controller
- Train β-tracking model on sensor data from existing runs
- Compare to rule-based BetaController on held-out runs
- If effective: train exploration-objective model on survey data (maximize novel basins per unit time)
- Plug learned controller in as a new controller type

### Phase 4 — Topology and Analysis
- Basin census: count vs L curve, saturation point
- Type clustering: embedding clusters vs compression spectrum clusters
- Transition graph: directed graph of basin-to-basin transitions, hubs, dead ends
- Depth prediction: can depth score be predicted from spectrum + embedding alone?
- Cross-L correspondence: minimum L for each basin type = minimum context to express that mode
- Residue network: embedding overlap across transitions

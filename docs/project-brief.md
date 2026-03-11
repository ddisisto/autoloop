# autoloop — Project Brief

**Title:** *Basin Topography and Taxonomy in Autoregressive Self-Play*

**Status:** Phase 0-1 complete (phase structure mapped, controller proven), pivoting to basin mapping
**Date:** March 2026 (started), revised March 2026

---

## 1. Motivation

A frozen language model generating tokens into a fixed-length sliding window is a discrete stochastic dynamical system operating on its own output. After the initial seed exits the window, the model conditions entirely on self-generated text. The dynamics are a property of the model's learned weights, the context length L, and the sampling temperature T.

This project began as a study of multi-scale complexity control -- could we use compression-based feedback to steer generation between collapse and noise? That question has been answered. The phase structure is mapped: four regimes (collapse, suppressed dynamics, rich dynamics, noise) with sharp boundaries and a saturating escape curve T_escape(L). Closed-loop control works: a simple controller holds Heaps' beta near 0.90 regardless of starting conditions.

The more interesting discovery is what happens *inside* the collapse regime. Each attractor basin is a mode the model "knows how to do" -- a topic, register, format, or genre strong enough to capture the system at a given operating point. The set of recoverable basins is an empirical map of the model's behavioral repertoire, extracted from output dynamics without inspecting weights. Attractor content is not random: it systematically features tautologies, incomplete predicates, self-perpetuating conditions, and confinement -- eigenstates where content, structure, and prediction align into zero-gradient fixed points.

The project now focuses on basin topography: what basins exist, how deep they are, how they connect, and whether a learned controller can navigate between them. The compression spectrum at the basin's characteristic scale W* is its mechanistic identity -- the gzip dictionary *is* the attractor's constituent structure.

## 2. Research Questions

### Answered

**RQ1 -- Phase structure.** Four regimes identified: collapse (T below T_escape), suppressed dynamics (structure but slow mixing), rich dynamics (T well above T_escape), noise (T >= 1.50). Crossover is sharp. T_escape(L) increases then saturates: L=64 at 0.55, L=128 at 0.57, L=192 at 0.67, L=256 at 0.87, L=512 at ~0.90. Coupling weakens above L ~ 256.

**RQ3 -- Multi-scale structure.** Answered differently than expected. Compressibility is a collapse detector, not a rich-dynamics discriminator. W > L signal is only ~0.03 below the noise floor at T=1.00. The in-context contribution at supra-L scales is minimal. Entropy and Heaps' beta are the right control signals, not multi-scale compression ratios.

**RQ4 -- Context length as control parameter.** L deepens collapse and extends the collapse regime to higher T. No single critical L -- the L-profile is a jagged continuum. Slope-flip: compressibility decreases with L in collapse, increases in rich dynamics, sign flip at T ~ 0.70-0.80. T=1.50 is a universal noise floor regardless of L. Saturation above L ~ 256.

**RQ5 -- Path dependence.** Basin escape hysteresis confirmed: exiting a pre-existing attractor requires ~0.4T more than avoiding it from BOS. Escape by semantic mutation: period-doubling route to chaos (attractor period expands until mutual-prediction lock breaks). Pre-collapse trajectories trace paths through connected semantic basins.

**RQ6 -- Closed-loop control.** Controller finds beta ~ 0.90 equilibrium regardless of starting L or T. Balance T tracks T_escape(L): T=0.70 for L=8, T=0.75 for L=16, T=0.90-0.95 for L=128/256. Small L has wide beta basin; large L oscillates at the escape boundary.

### Partially Answered

**RQ2 -- Attractor characterization.** Attractor structure is a staircase (each L has a distinct entropy floor). Basin depth depends on mutual information between cycle positions -- multi-token cycles are far deeper than single-token repeats. 21 unique attractors found across 3 seeds x 7 L values at T=0.50. Content clusters into semantic families (medical, political, social, self-referential). Dwell time distributions not yet formally analyzed.

### Open

**RQ7 -- Basin topography.** What basins exist across the (L, T) parameter space? How many unique basins can be recovered at each L? Does count scale with L, saturate, or peak? What is the transition graph -- which basins connect to which, and are there hubs and dead ends? Can basin depth be predicted from the compression spectrum alone?

**RQ8 -- Learned steering.** Can a small model learn the sensor-to-action mapping from existing controller data (~1050 decision points)? Can it outperform rule-based control for beta-tracking? With survey data: can it learn to maximize basin discovery rate or navigate to target basins?

**RQ9 -- Semantic topology.** Do basin transition paths reveal structure in the model's learned representation? Pre-collapse trajectories walk through connected semantic basins (education -> violence -> apocalypse -> cataloging -> imprisonment -> Star Wars). Is this connectivity an artifact of T=0.50 dynamics or a property of the model's semantic organization?

## 3. Key Results

For full evidence trail, see `observations.md`. Summary of principal findings:

**Phase structure.** Four regimes with sharp boundaries. T_escape(L) saturates above L ~ 256. The suppressed-dynamics zone (L=256 at T=0.70-0.80) has the strongest multi-scale decoupling and longest decorrelation lags (253-356 steps).

**Basin escape hysteresis.** Pre-seeded attractors survive ~0.4T above T_escape measured from BOS. Basin depth is a smooth function of L with a sharp lock-in transition at 4-8 copies of the cycle in context.

**Semantic eigenstates.** Attractor content describes its own dynamics: tautologies, confinement, recursive structures, incomplete predicates. These are zero-gradient fixed points where content and prediction align.

**Period-doubling escape.** At threshold L, the model escapes by mutating the attractor ("Star Wars" -> "Star Wars 2000" -> "The Old Republic" -> freedom). Period expansion dilutes mutual prediction until the lock breaks.

**Scale invariance.** Suppressed dynamics at L=16/T=0.60 (pre-seeded) matches L=256/T=0.70 (natural) in coherence and TTR. The regime depends on basin-depth / thermal-energy ratio, not absolute parameters.

**Vocabulary as regime diagnostic.** TTR spans 100x across regimes. Heaps' beta cleanly separates collapse (0.17-0.38), rich dynamics (0.75-0.85), and escape events (> 1.0).

**Controller equilibrium.** Beta ~ 0.90 is a natural equilibrium for SmolLM-135M. Balance-point texture is L-dependent: short L = topic soup through forgetting, long L = thematic orbits through within-basin exploration.

## 4. Experimental Infrastructure

### Generation

- `engine.py`: `StepEngine` class -- single token loop (`step(L, T)`), trailing-window sensors (entropy, Heaps' beta, compressibility), `snapshot()`/`restore()` rollback, checkpoint persistence
- `experiment.py`: universal run loop + controllers -- `FixedController`, `ScheduleController`, `BetaController`, `StateMachine` (composable state graph with sensor-driven transitions)
- CLI: `experiment.py fixed|schedule|beta`
- `sweep.py`: unified sweep runner with named presets, ad-hoc grids, `--status`, `--list`

### Analysis

- `analyze/`: package with incremental `.analysis.pkl` cache per run; `default_window_sizes()`, `comp_stats()` interface
- `precollapse.py`: regime classification, basin transition detection, W/L convergence profiles
- `semantic.py`: theme discovery, attractor catalog, Heaps' law, coherence, morphology analysis
- `grep_text.py`: CLI grep for decoded text in parquet runs (regex, context display)
- `metrics.py`: scalar metric extraction (surprisal stats, EOS interarrival, decorrelation lag)
- `summary_table.py`: cross-condition summary CSV

### Visualization

- `plot.py`: entropy time series, compressibility, phase portraits, temporal phase portraits, violin plots
- `plot_window_scaling.py`: window scaling plots (comp vs L, comp vs W, heatmaps)
- `reproduce_plots.py`: one-command regen of all standard plots with mtime caching
- `explorer.py` + `static/`: interactive web explorer (FastAPI + Plotly.js), buffered context viewer with scroll sync, token search

### Data

- 70 runs total: ~53 sweep + 7 controller + anneal + probes, ~1.1 GB
- One Parquet file per run + JSON sidecar + checkpoint + analysis cache
- Model: SmolLM-135M (local at `data/model/SmolLM-135M/`)
- Compute: local GPU (GTX 1070), ~24-38 tok/s at L=64

### Parameters Explored

| Parameter | Values |
|---|---|
| Context length L | 8, 16, 32, 64, 128, 160, 176, 192, 208, 224, 256, 512 |
| Temperature T | 0.50, 0.55, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 1.00, 1.10, 1.20, 1.50 |
| Seeds | 42 (all), 123 and 7 (T=0.50 + L-dense conditions) |
| Tokens per run | 100,000 (sweep), up to 1,000,000 (controller drift) |
| Sampling | Pure temperature scaling, no top-k/top-p |

## 5. Current Phase: Basin Mapping

See `docs/basin-mapping.md` for full survey design and roadmap.

### Survey Protocol

A `StateMachine` experiment cycles through states:

```
COOLING  --[basin_detected]-->  CAPTURED
CAPTURED --[characterised]--->  HEATING
HEATING  --[escaped]--------->  TRANSIT
HEATING  --[deeper_basin]---->  CAPTURED
TRANSIT  --[context_flushed]->  COOLING
```

Each cycle yields one basin record: compression spectrum, sensor profile, depth score (via checkpoint forking), and transition metadata (escape temperature, steps, dictionary overlap with previous basin).

### Compression as Identity

The gzip dictionary at the window size of best compression (W*) is the basin's mechanistic fingerprint. Two captures represent the same basin iff they produce the same dictionary content at the same W*. The compressibility-vs-W spectrum is the basin's continuous signature -- clusterable without human labeling.

### Basin Catalogue

A structured catalogue (SQLite) stores basin records with compression identity, sensor profiles, depth scores, and transition edges. Analysis targets: basin census by L, type clustering by compression spectrum, transition graph topology, depth prediction from spectrum, cross-L correspondence (minimum L to express a mode), and residue network (dictionary overlap across transitions).

### Learned Controller

Three tiers:
- **Tier A (built):** Rule-based `BetaController` and `StateMachine` in experiment.py
- **Tier B (next):** Small model trained on sensor data from parquet runs. 10D input (sensor state), 2D output (delta-T, delta-L). Beta-tracking objective first; exploration objective once survey data exists
- **Tier C (future):** Online learning during generation, bandit-style exploration/exploitation over the basin landscape

## 6. Roadmap

### Phase 1 -- Pilot Survey
- Implement basin survey as `StateMachine` experiment
- L=64 null seed, 100k tokens -- shake down the protocol
- Validate compression-spectrum clustering
- Extract gzip dictionaries (new analysis capability)

### Phase 2 -- Systematic Survey
- L in {8, 16, 32, 64, 128, 256}, null + domain seeds
- Build basin catalogue, transition graph
- L=8-32 "skeleton" survey: what survives extreme context compression?

### Phase 3 -- Learned Controller
- Train beta-tracking model on sensor data from existing runs
- Compare to rule-based BetaController on held-out runs
- If effective: train exploration-objective model on survey data
- Plug learned controller into experiment.py as a new controller type

### Phase 4 -- Topology
- Basin census analysis: count, depth, spectrum clustering
- Transition graph: hubs, dead ends, connected components
- Cross-L correspondence: basin minimum context requirements
- Residue network: semantic bleed between basins

### Phase 5 -- Scaling and Writing
- Model scaling experiments (1B, 8B): do basin types and transition topology persist?
- Paper draft, figures, supplementary materials, code and data release

## 7. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Basin count too low for meaningful taxonomy | Low | Even 10-20 distinct basins at L=64 would be informative; L=8-32 skeleton survey tests the floor |
| Gzip dictionary extraction too noisy for fingerprinting | Medium | Compression spectrum shape is a fallback identity; dictionary content is a bonus |
| Learned controller no better than rule-based | Medium | Rule-based controller already works; learned version is an efficiency gain, not a prerequisite |
| Basin topology trivial (fully connected or fully disconnected) | Low-Medium | Pre-collapse trajectories already show structured paths; even a negative result constrains the model's semantic organization |
| Results are SmolLM-135M-specific | High | This is expected and acceptable. The framework transfers; the specific basins do not. Scaling experiments (Phase 5) test generality |

## 8. References

- `observations.md` -- current model summary and evidence log index
- `docs/basin-mapping.md` -- basin survey design and roadmap
- `docs/project-brief-v1.md` -- archived original brief (Phase 0 framing)
- `docs/observations-2026-03-*.md` -- dated observation archives

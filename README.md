# autoloop

Basin topography and learnable steering in autoregressive generation.

A small language model (SmolLM-135M) generates tokens indefinitely into a fixed-length sliding context window, conditioning entirely on its own output. The resulting system is a discrete stochastic dynamical system with surprisingly rich structure: attractor basins, phase transitions, escape dynamics, and semantic eigenstates. This project maps the basin landscape, builds closed-loop controllers that navigate it, and works toward learned steering across the full topology.

> **Status:** Active research. Phase 0 (landscape mapping) and Phase 1 (closed-loop control) complete. Currently building basin cartography infrastructure. Not taking contributions, but forks and discussion are welcome.

## What we're finding

**Four regimes** emerge across temperature (T) and context length (L): repetitive collapse, suppressed dynamics (structure but slow mixing), rich dynamics, and incoherent noise. The boundary between collapse and escape is sharp and L-dependent.

**The escape boundary T_escape(L) saturates.** L=64 escapes at T~0.55, L=192 at T~0.67, L=256 at T~0.87, L=512 at T~0.90. The steep rise from L=128 to L=256 flattens out -- above L~256, context is "sufficient" and temperature alone determines the regime.

**Collapse is a staircase of attractor basins.** At T=0.50, each L settles onto a distinct entropy floor. L=256 hits zero entropy by 15k steps. L=128 sits on a meta-stable false floor for 45k steps before dropping. L=64 stays on a higher basin for the full 100k-step run. Collapse is a timescale phenomenon -- L sets how fast you descend through a hierarchy of basins.

![Entropy time series at T=0.50 showing staircase of attractor basins](data/figures/Lmulti_T0.50_S42_entropy.png)

**Attractor content describes its own dynamics.** Across 21 collapsed runs, every attractor features tautologies, incomplete predicates, self-perpetuating conditions, and confinement. These are eigenstates: configurations where content, structure, and prediction align into zero-gradient fixed points.

**Escape by semantic mutation.** At threshold L, the model doesn't jump out of an attractor -- it tunnels out by mutating it. "Star Wars" becomes "Star Wars 2000" becomes "Star Wars: The Old Republic" becomes freedom. Period-doubling as a route to chaos.

**Basin escape hysteresis.** Exiting an occupied attractor requires ~0.4T more than avoiding it from a cold start. Basin depth depends on mutual information between cycle positions -- multi-token cycles lock harder than single-token repeats.

**Closed-loop control works.** A simple controller (adjust T per segment, adjust L when T saturates) holds Heaps' beta near a target of 0.90 -- a natural equilibrium regardless of L or starting T. Balance T tracks T_escape(L): T=0.70 for L=8, T=0.75 for L=16, T=0.90-0.95 for L=128 and L=256. Small L has a wide stability basin; large L oscillates at the escape boundary.

![Phase portrait at T=1.00 showing EOS in cloud interior](data/figures/Lmulti_T1.00_S42_phase.png)

**Suppressed dynamics is scale-invariant.** A shallow basin at low temperature behaves like a deep basin at moderate temperature. The regime is defined by the ratio of basin depth to thermal energy, not absolute L or T.

**Vocabulary richness cleanly separates regimes.** Type-token ratio spans 100x across conditions. Heaps' law exponent beta separates collapse (0.17), rich dynamics (0.80), and escape events (>1.0).

See [observations.md](observations.md) for the full findings log with reproduction commands.

## Architecture

Scripts, not a package. Flat layout (except `analyze/` which is a package).

| Script | Purpose |
|--------|---------|
| `engine.py` | Token generation engine: `StepEngine` with step, sensors, snapshot/rollback, checkpoint |
| `experiment.py` | Experiment framework: controllers (`Fixed`, `Schedule`, `Beta`), `StateMachine`, universal run loop |
| `sweep.py` | Unified sweep runner: named presets, ad-hoc grids, `--status`, `--list` |
| `analyze/` | Analysis package: compressibility, stationarity, summaries; incremental `.analysis.pkl` cache |
| `plot.py` | Visualization: entropy, compressibility, phase portraits, temporal portraits, violins |
| `plot_window_scaling.py` | Window scaling plots: comp vs L, comp vs W, heatmaps |
| `precollapse.py` | Pre-collapse trajectory analysis: regime classification, basin transitions, W/L convergence |
| `semantic.py` | Semantic analysis: theme discovery, attractor catalog, Heaps' law, coherence |
| `grep_text.py` | CLI grep for decoded text in parquet runs: regex, context, step/L/T display |
| `anneal.py` | Annealing experiment runner: phased probes and tiers |
| `explorer.py` + `static/` | Interactive web explorer: FastAPI + Plotly.js, buffered context viewer, token search |
| `runlib.py` | Run discovery and path utilities |
| `runindex.py` | SQLite index for cross-run metadata queries |
| `schema.py` | Data schema definitions |
| `summary_table.py` | Cross-condition summary CSV |
| `reproduce_plots.py` | One-command regeneration of all standard figures (with caching) |
| `utils.py` | Shared primitives: compressibility, EOS EMA |
| `generate.py` | Legacy generation CLI (superseded by `experiment.py`) |
| `controller.py` | Legacy closed-loop controller (superseded by `experiment.py beta`) |

## Data

~70 runs across sweeps, controller experiments, annealing, and probes. ~1.1GB total.

- **Sweep runs:** L={64..512} x T={0.50..1.50} x seeds {42, 123, 7}
- **Controller runs:** closed-loop beta-tracking at various (L, T) starting points, including a 1M-step drift run
- **Annealing runs:** tiered cooling/heating experiments
- **Probes:** quick feasibility checks (5k tokens)

Each run produces a Parquet file (per-token entropy, log-probability, EOS flag, decoded text, per-step T and L), a JSON sidecar with full metadata, and an incremental analysis cache. Checkpoints enable resume and extension.

Data directory is organized into subdirectories by experiment type (`sweep/`, `controller/`, `anneal/`, `probe/`, `survey/`, `schedule/`) with a SQLite index for cross-run queries. Raw data is not included in the repo. Figures are tracked in `data/figures/`.

## Quick start

```bash
# Dependencies
uv sync

# Fixed-parameter run (new framework)
python experiment.py fixed --seed 42 -L 64 -T 0.50 --total-steps 100000

# Closed-loop controller with drift
python experiment.py beta --seed 42 --start-L 8 --start-T 1.00 --drift --total-steps 1000000

# Sweep a grid
python sweep.py --L 64 128 256 --T 0.50 0.70 1.00 --seed 42
python sweep.py --status    # grid table from disk
python sweep.py --list      # named presets

# Interactive explorer
uvicorn explorer:app --reload --port 8000

# Plots
python plot.py --runs data/runs/sweep/L0064_T*_S42.parquet
python reproduce_plots.py

# Semantic analysis
python semantic.py --clouds
python grep_text.py "Star Wars" --runs data/runs/sweep/*.parquet --count

# Pre-collapse analysis
python precollapse.py --detail L0256_T0.80_S42
```

Requires a local copy of SmolLM-135M weights at `data/model/SmolLM-135M/`.

## Where this is headed

**Phase 0 (complete):** Mapped the T x L landscape. Four regimes identified. T_escape(L) curve measured. Multi-scale compression framework built.

**Phase 1 (complete):** Closed-loop control. BetaController finds beta~0.90 equilibrium. Balance T tracks T_escape(L). Drift mode grows L over time. Sensor framework validated: entropy and Heaps' beta are the right control signals.

**Current work -- basin cartography:** A survey protocol (COOLING -> CAPTURED -> CHARACTERISING -> HEATING -> TRANSIT) implemented as a `StateMachine` experiment will systematically cool the system to capture basins, fingerprint them via gzip compression spectra, heat to escape, and repeat. The compression dictionary at optimal W *is* the basin's identity. Goal: a catalog of all recoverable basins across the (T, L) parameter space. See [docs/basin-mapping.md](docs/basin-mapping.md).

**Next -- learned controller:** Train a small model on existing controller decision data (~1050 examples). 10D sensor input, 2D output (delta-T, delta-L). Beta-tracking first, then exploration objective once basin survey generates training data.

**Longer term -- semantic topology:** Basin transition paths form a graph of the model's semantic space. Pre-collapse trajectories already show connected descent paths (education -> violence -> apocalypse -> cataloging -> imprisonment -> Star Wars). The full topology -- which basins connect to which, and what the transition costs are -- is a map of the model's behavioral repertoire extracted purely from output dynamics.

## Project documents

- [observations.md](observations.md) -- Findings log with current model summary
- [run-index.md](run-index.md) -- Grid status and phase planning
- [docs/project-brief.md](docs/project-brief.md) -- Full research design
- [docs/basin-mapping.md](docs/basin-mapping.md) -- Basin survey protocol and roadmap
- [docs/interaction-topology.md](docs/interaction-topology.md) -- Speculative framing: generative dynamics as interaction paradigm

## License

[MIT](LICENSE)

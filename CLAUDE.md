# CLAUDE.md — autoloop

## Project

Multi-scale complexity control in closed-loop autoregressive generation. See `docs/project-brief.md` for full design, `observations.md` for findings log.

## Repository Layout

```
generate.py          # Core generation loop, CLI entry point (schedule, prefill, checkpoint/resume)
analyze.py           # Post-hoc analysis (compressibility, stationarity, summaries; incremental .analysis.pkl cache)
plot.py              # Visualization (5 plot types + EOS markers, CLI with --runs and --plots)
utils.py             # Shared primitives (compressibility, eos_ema, fix_decoded_texts)
metrics.py           # Scalar metric extraction (surprisal stats, EOS interarrival, decorrelation lag)
summary_table.py     # Cross-condition summary CSV from all runs
reproduce_plots.py   # Regenerate all standard plots from available data (with caching)
analyze_windows.py   # Recompute analysis at standard W grid [16,32,64,128,256]
plot_window_scaling.py # Window scaling plots (comp vs L, comp vs W, heatmaps)
precollapse.py       # Pre-collapse trajectory analysis (regime classification, basin transitions, W/L convergence)
semantic.py          # Semantic analysis (theme search, attractor catalog, Heaps' law, repetition onset, coherence)
anneal.py            # Annealing experiment runner (phased: probes, tier1-5, --check, --dry-run)
sweep.py             # Unified sweep runner (presets, ad-hoc grids, --status)
explorer.py          # Interactive web explorer backend (FastAPI)
static/              # Explorer frontend
  index.html         # HTML shell
  css/explorer.css   # Styles (panels, right drawer, zoom-synced overview)
  js/app.js          # Init, event wiring, hash state, presets
  js/panels.js       # Composable chart strips, Plotly rendering, zoom sync
  js/presets.js      # Hardcoded + user-saved chart layouts
  js/context.js      # Context drawer, buffered token view, scroll sync, search, word cloud
  js/sidebar.js      # Run list, favorites
  js/state.js        # App state, color-by system, API helpers
explorer.md          # Explorer design doc
run-index.md         # Run tracker, grid overview, phase planning
observations.md      # Current model summary + evidence log index
docs/                # Longer-form documents + observation archives
  annealing-experiment.md    # Annealing experiment design (tiered, probe-first)
  observations-2026-03-*.md  # Dated observation archives (from observations.md)
  project-brief.md   # Research design document
  share.md           # Draft post/article for sharing findings
  interaction-topology.md  # Speculative framing: generative dynamics as interaction paradigm
data/                # Gitignored except figures
  model/SmolLM-135M/ # Local model weights (pre-downloaded)
  runs/              # Parquet files + JSON sidecars + checkpoints + analysis cache (.analysis.pkl)
  figures/           # Plot outputs (tracked in git)
```

Scripts, not a package. No `src/` layout. Add modules only when genuinely needed.

## Conventions

### Code Style
- Type hints on all function signatures
- `logging` module, not print — DEBUG for per-step, INFO for run progress
- No default params for per-run configuration — explicit or fail
- No hidden logic, no silent fallbacks
- No pre-emptive exception handling — let runtime errors surface naturally
- No TODOs in code — use `raise NotImplementedError` where appropriate
- Single responsibility, DRY, YAGNI, KISS
- Modules expose clean interfaces; internals stay internal

### Data
- All generated data goes under `data/` (gitignored except `data/figures/`)
- One Parquet file per run, named `L{L:04d}_T{T:.2f}_S{seed}.parquet` (fixed-param) or `sched_S{seed}_{hash8}` (schedule) or `--run-name`
- Per-step `temperature` and `context_length` columns in parquet (vary per-step in scheduled runs)
- Each run includes a JSON sidecar with full metadata (schedule, prefill_text, model revision, torch version, timing)
- Analysis cache: single `.analysis.pkl` per parquet, incremental (new window sizes merged in), invalidated by parquet mtime
- Checkpoints: same stem as parquet `.ckpt` — kept after run completion for extension; stores schedule spec for resume validation
- Model weights: `data/model/SmolLM-135M/` (local, not fetched at runtime)

### Git
- `data/figures/` tracked in git; all other data gitignored
- `uv.lock` stays committed
- Small, logical commits

### Sweeps
- `sweep.py`: unified runner with named presets and ad-hoc `--L`/`--T`/`--seed` grids
- `sweep.py --status`: auto-generated grid table from parquet files on disk (replaces manual tracking)
- `sweep.py --list`: show presets with completion counts
- Presets record historical sweeps with rationale; new sweeps can use ad-hoc grids
- Each condition runs as a subprocess of `generate.py` (crash isolation)

## Current State (Phase 0 complete, Phase 1 in progress)

### What's Built
- `generate.py`: generation loop with schedule support (per-segment L/T), pre-seeded context (`--prefill-text`), checkpoint/resume with schedule validation, per-1k-step logging, decoded_text fix for multi-byte UTF-8
- `analyze/`: analysis package (compressibility, stationarity, summaries); single incremental `.analysis.pkl` cache per run; accepts pre-loaded DataFrames; `default_window_sizes()` returns [32,64,128,256] (floor at 32, always includes W>L)
- `analyze_windows.py`: recompute analysis at standard W grid [16,32,64,128,256]
- `plot.py`: entropy time series (EOS rate EMA overlay), compressibility, phase portraits (EOS diamonds), temporal phase portraits (cividis), split violin
- `plot_window_scaling.py`: window scaling plots (comp vs L, comp vs W, heatmaps)
- `utils.py`: shared primitives — `compressibility()`, `eos_ema()`
- `reproduce_plots.py`: one-command regen of all standard plot slices (analysis + figure mtime caching)
- `explorer.py` + `static/index.html`: interactive web explorer (FastAPI + Plotly.js), buffered context viewer with scroll sync, infinite scroll, L-window visual, token search (case/word/regex)
- `precollapse.py`: pre-collapse trajectory analysis — regime classification (escaped/oscillating/collapsed/deep_collapsed), basin transition detection (escape spike vs landing depth), W/L convergence profiles, attractor content extraction
- `sweep.py`: unified sweep runner with presets (pilot, crossover, seed, ldense, l256-crossover) and ad-hoc grids

### Data Collected (run `sweep.py --status` for live grid)
- Pilot: L={64,128,192,256} × T={0.50,1.00,1.50} × S=42 (12 runs)
- Crossover: L={64,128,192} × T={0.60,0.70,0.80,0.90} × S=42 (12 runs)
- Seed replication at T=0.50: L={64,128,192} × S={42,123,7} (9 runs)
- L-densification at T=0.50: L={160,176,208,224} × S={42,123,7} (15 runs)
- L=256 crossover: T={0.60,0.70,0.80,0.90} × S=42 (4 runs, complete)
- L=512 escape boundary: T={0.90,1.00,1.10,1.20} × S=42 (4 runs, complete)
- Total: ~53 runs across all conditions

### Key Findings (see observations.md)
- Four regimes: collapse, suppressed dynamics, rich dynamics, noise
- T_escape(L) increases then saturates: L=64→0.55, L=128→0.57, L=192→0.67, L=256→0.87, L=512→~0.90
- T and L are coupled but coupling weakens at large L (saturation above L≈256)
- Suppressed zone: L=256 at T=0.70–0.80 has structure but slow mixing (decorrelation lag 253–356)
- Slope-flip pivot shifts with L: comp crossover at T≈0.75 for L=192, T≈0.95 for L=256
- Multi-scale decoupling peaks in suppressed zone (|comp_W256−comp_W64| up to 0.35)
- Concept fragmentation: temperature controls expression fidelity, not concept activation
- L-densification at T=0.50: jagged non-monotonic profile, no clean phase transition
- "Memory-depth annealing": L-reduction as escape mechanism — bounded by T_escape saturation
- Basin escape hysteresis: pre-seeded attractor at L=64/T=0.80 stays stuck (BOS T_escape=0.55). Basin exit requires ~0.4T more than basin avoidance
- L-titration of basin depth: " Star Wars" (2-token cycle) locks in at 4-8 copies (L=8 escapes, L=16 stuck). Sharp transition, not gradual
- Single-token attractors (" young") much shallower than multi-token mutual-prediction cycles — basin depth depends on mutual information between cycle positions
- Basin transitions: escape spikes >6 nats always reach shallower basins; <1 nat leads deeper 67% of time (L=256 T=0.80)
- Progressive basin deepening: floors cascade from 0.05→0.014 over a run's lifetime
- Attractor content is semantically diverse and L-dependent (shorter periods at higher L)
- W/L convergence: slope divergence fingerprints attractor approach (local compression + global expansion)
- Attractor content describes its own dynamics: tautologies, incomplete predicates, self-perpetuating conditions, confinement
- Pre-collapse trajectories trace paths through connected semantic basins (education → violence → apocalypse → cataloging → imprisonment → Star Wars)
- Vocabulary richness (TTR) spans 100x across regimes; Heaps' β cleanly separates collapse (0.17) from rich dynamics (0.80) from escape events (>1.0)
- Collapse is deterministic (all seeds collapse at T=0.50) but content is seed-dependent — 21 unique attractors across 3 seeds × 7 L values
- Escape by semantic mutation: at L=16 (threshold lock-in), attractor period expands ("Star Wars" → "Star Wars 2000" → "The Old Republic" → escape). Period-doubling route to chaos
- Suppressed dynamics is scale-invariant: L=16/T=0.60 pre-seeded ≈ L=256/T=0.70 natural (same coherence, TTR). Regime depends on basin-depth/thermal-energy ratio, not absolute L or T

### Key Parameters
- Model: SmolLM-135M (local at `data/model/SmolLM-135M/`)
- Context lengths L: 64, 128, 160, 176, 192, 208, 224, 256, 512
- Temperatures T: 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.50
- Seeds: 42 (all conditions); 123, 7 (T=0.50 + L-dense conditions)
- Tokens per run: 100,000 (post-pre-fill)
- Sampling: pure temperature scaling, no top-k/top-p

## CLI Reference

```bash
# Sweeps
python sweep.py pilot                           # run a named preset
python sweep.py --L 256 --T 0.60 0.70 --seed 42 # ad-hoc grid
python sweep.py --status                         # grid table from disk
python sweep.py --list                           # list presets
python sweep.py crossover --dry-run              # preview without running

# Single generation run (fixed parameters)
python generate.py --context-length 64 --temperature 1.0 --seed 42 \
  --num-tokens 100000 --model-dir data/model/SmolLM-135M \
  --output-dir data/runs --device cuda

# Scheduled run (L/T vary per segment)
python generate.py --schedule "50000:L256:T0.60,10000:L64:T0.60,40000:L256:T0.60" \
  --seed 42 --model-dir data/model/SmolLM-135M \
  --output-dir data/runs --device cuda

# Pre-seeded context (repeat text to fill L, skip generative prefill)
python generate.py --context-length 256 --temperature 0.60 --seed 42 \
  --num-tokens 100000 --prefill-text " Star Wars" \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

# Interactive explorer
uvicorn explorer:app --reload --port 8000   # then open http://localhost:8000

# Plots (all types by default, or select with --plots)
python plot.py --runs data/runs/L0064_T*_S42.parquet
python plot.py --runs data/runs/L*_T0.50_S42.parquet --plots violin temporal
python plot.py --runs data/runs/L0064_T*_S42.parquet --downsample 50

# Reproduce all standard plots (skips up-to-date slices)
python reproduce_plots.py
python reproduce_plots.py --force          # bypass cache
python reproduce_plots.py --plots entropy  # only specific plot types

# Pre-collapse analysis
python precollapse.py                              # summary by regime (all T<=1.2)
python precollapse.py --detail L0256_T0.80_S42     # detailed report with basin transitions
python precollapse.py --csv data/precollapse.csv   # all metrics to CSV
python precollapse.py --runs data/runs/L0256*.parquet  # specific runs

# Annealing experiments
python anneal.py probes              # Phase 0: quick feasibility (5k tokens)
python anneal.py probes --check      # analyze probe results
python anneal.py tier1               # Phase A: escape threshold (100k tokens)
python anneal.py tier2               # Phase B: return dynamics (100k tokens)
python anneal.py tier5               # Phase B: T vs L comparison (100k tokens)
python anneal.py tier1 --dry-run     # preview without running

# Semantic analysis
python semantic.py                                  # default theme "temperature", all runs
python semantic.py --seed 42                        # filter to seed 42
python semantic.py --theme "the" --seed 42          # custom theme
python semantic.py --csv data/semantic.csv          # export all metrics
python semantic.py --runs data/runs/L0256*.parquet  # specific runs

# Cross-condition summary table
python summary_table.py                         # print to stdout
python summary_table.py --out data/summary.csv  # write to file
```

## Dependencies

Managed with `uv`. Key packages: torch (CUDA 12.6), transformers, pandas, pyarrow, scipy, scikit-learn, matplotlib, fastapi, uvicorn.

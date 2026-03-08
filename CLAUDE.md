# CLAUDE.md — autoloop

## Project

Multi-scale complexity control in closed-loop autoregressive generation. See `docs/project-brief.md` for full design, `observations.md` for findings log.

## Repository Layout

```
generate.py          # Core generation loop, CLI entry point (with checkpoint/resume)
analyze.py           # Post-hoc analysis (compressibility, stationarity, summaries; incremental .analysis.pkl cache)
plot.py              # Visualization (5 plot types + EOS markers, CLI with --runs and --plots)
utils.py             # Shared primitives (compressibility, eos_ema)
reproduce_plots.py   # Regenerate all standard plots from available data (with caching)
analyze_windows.py   # Recompute analysis at standard W grid [16,32,64,128,256]
plot_window_scaling.py # Window scaling plots (comp vs L, comp vs W, heatmaps)
pilot_sweep.py       # Batch runner for pilot grid (idempotent, crash-resilient)
crossover_sweep.py   # Batch runner for T-densification in crossover region
seed_sweep.py        # Batch runner for seed replication (seeds 123, 7)
explorer.py          # Interactive web explorer backend (FastAPI)
static/index.html    # Explorer frontend (Plotly.js, single-page app)
explorer.md          # Explorer design doc
run-index.md         # Run tracker, grid overview, phase planning
observations.md      # Append-only findings log with current model summary
docs/                # Longer-form documents
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
- One Parquet file per run, named `L{L:04d}_T{T:.2f}_S{seed}.parquet`
- Each run includes a JSON sidecar with full metadata (parameters, model revision, torch version, timing)
- Analysis cache: single `.analysis.pkl` per parquet, incremental (new window sizes merged in), invalidated by parquet mtime
- Checkpoints: `L{L:04d}_T{T:.2f}_S{seed}.ckpt` — kept after run completion for extension
- Model weights: `data/model/SmolLM-135M/` (local, not fetched at runtime)

### Git
- `data/figures/` tracked in git; all other data gitignored
- `uv.lock` stays committed
- Small, logical commits

### Sweep Scripts
- `pilot_sweep.py`: batch runner for Phase 0 grid (idempotent, skips completed runs, crash-resilient)
- One sweep script per phase if needed: `phase1_sweep.py`, etc.
- Grid parameters hardcoded in each sweep script — no external config files
- Each sweep script calls `generate.py` as a subprocess per condition (crash isolation, natural per-run independence)

## Current State (Phase 0 — Pilot + Crossover Mapping)

### What's Built
- `generate.py`: generation loop with pre-fill, checkpoint/resume, per-1k-step logging
- `analyze.py`: modular metric computation (compressibility, stationarity, summaries); single incremental `.analysis.pkl` cache per run; accepts pre-loaded DataFrames; canonical home for `default_window_sizes()`
- `analyze_windows.py`: recompute analysis at standard W grid [16,32,64,128,256]
- `plot.py`: entropy time series (EOS rate EMA overlay), compressibility, phase portraits (EOS diamonds), temporal phase portraits (cividis), split violin
- `plot_window_scaling.py`: window scaling plots (comp vs L, comp vs W, heatmaps)
- `utils.py`: shared primitives — `compressibility()`, `eos_ema()`
- `reproduce_plots.py`: one-command regen of all standard plot slices (analysis + figure mtime caching)
- `explorer.py` + `static/index.html`: interactive web explorer (FastAPI + Plotly.js), context inspection
- `pilot_sweep.py`, `crossover_sweep.py`, `seed_sweep.py`: batch runners (idempotent, crash-resilient)

### Data Collected (see run-index.md for full grid)
- 24 seed=42 runs complete: L={64,128,192} × T={0.50–1.50} + L=256 × T={0.50,1.00,1.50}
- Seed replication (123, 7) in progress via seed_sweep.py
- Dense crossover coverage: T={0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.50} at L={64,128,192}

### Key Findings (see observations.md)
- Three distinct regimes: collapse (T=0.5), rich dynamics (T=1.0), noise (T=1.5)
- T and L are orthogonal actuators: T = noise floor, L = memory depth / attractor stickiness
- L=64 escapes collapse attractors; L=256 locks in permanently
- At T=1.0, L=256 shifts operating point without collapse — different equilibrium
- W=L/4 and W=L compressibility decouple in the interesting regime
- EOS rate peaks at T=1.0; L suppresses EOS dramatically in collapse regime
- Three-sensor framework: entropy, compressibility, EOS rate
- Crossover region T=0.6–0.9 now densely sampled
- "Memory-depth annealing": L-reduction as escape mechanism from stuck attractors

### Key Parameters
- Model: SmolLM-135M (local at `data/model/SmolLM-135M/`)
- Context lengths L: 64, 128, 192, 256
- Temperatures T: 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.50
- Seeds: 42 (complete); 123, 7 (planned replication)
- Tokens per run: 100,000 (post-pre-fill)
- Sampling: pure temperature scaling, no top-k/top-p

## CLI Reference

```bash
# Single generation run
python generate.py --context-length 64 --temperature 1.0 --seed 42 \
  --num-tokens 100000 --model-dir data/model/SmolLM-135M \
  --output-dir data/runs --device cuda

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
```

## Dependencies

Managed with `uv`. Key packages: torch (CUDA 12.6), transformers, pandas, pyarrow, scipy, scikit-learn, matplotlib, fastapi, uvicorn.

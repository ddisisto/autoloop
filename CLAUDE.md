# CLAUDE.md — autoloop

## Project

Multi-scale complexity control in closed-loop autoregressive generation. See `project-brief.md` for full design, `observations.md` for findings log.

## Repository Layout

```
generate.py          # Core generation loop, CLI entry point (with checkpoint/resume)
analyze.py           # Post-hoc analysis (compressibility, stationarity, summaries)
plot.py              # Visualization (5 plot types, CLI with --runs and --plots)
reproduce_plots.sh   # Regenerate all standard plots from available data
pilot-runs.md        # Run tracker: status of each pilot grid condition
observations.md      # Append-only findings log with reproduction commands
project-brief.md     # Research design document
data/                # Gitignored. All generated data lives here
  model/SmolLM-135M/ # Local model weights (pre-downloaded)
  runs/              # Parquet files + JSON sidecars + checkpoints
  figures/           # Plot outputs
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
- All generated data goes under `data/` (gitignored)
- One Parquet file per run, named `L{L:04d}_T{T:.2f}_S{seed}.parquet`
- Each run includes a JSON sidecar with full metadata (parameters, model revision, torch version, timing)
- Checkpoints: `L{L:04d}_T{T:.2f}_S{seed}.ckpt` — kept after run completion for extension
- Model weights: `data/model/SmolLM-135M/` (local, not fetched at runtime)

### Git
- No generated data, figures, or model weights in the repo
- `uv.lock` stays committed
- Small, logical commits

### Sweep Scripts
- No sweep script needed for Phase 0 — runs managed manually via `pilot-runs.md`
- One sweep script per phase if needed: `pilot_sweep.py`, then `phase1_sweep.py`, etc.
- Grid parameters hardcoded in each sweep script — no external config files
- Each sweep script calls `generate.py` as a subprocess per condition (crash isolation, natural per-run independence)

## Current State (Phase 0 Pilot)

### What's Built
- `generate.py`: generation loop with pre-fill, per-step logging, checkpoint/resume
- `analyze.py`: sliding window gzip compressibility (W=L, W=L/4), 5-block stationarity, run summaries
- `plot.py`: entropy time series, compressibility time series, phase portraits, temporal phase portraits (cividis colormap), split violin distribution plots
- `reproduce_plots.sh`: one-command regeneration of all standard plot slices

### Data Collected (see pilot-runs.md for full status)
- L=64 × T={0.5, 1.0, 1.5} × seed=42: done (100k each)
- L=256 × T=0.50 × seed=42: done/near done
- Remaining L=256 and L=1024 conditions: in progress

### Key Findings (see observations.md)
- Three distinct regimes: collapse (T=0.5), rich dynamics (T=1.0), noise (T=1.5)
- Context length L dramatically deepens collapse attractor
- W=L/4 and W=L compressibility decouple in the interesting regime
- Crossover region likely T=0.6–0.9, target for Phase 1 densification
- Multi-scale compression ratio is a natural sensor for closed-loop control (Phase 3)

## Key Parameters (Phase 0 Pilot)

- Model: SmolLM-135M (local at `data/model/SmolLM-135M/`)
- Context lengths L: 64, 256, 1024
- Temperatures T: 0.5, 1.0, 1.5
- Seeds: 42, 123, 7 (3 per condition, 27 runs total)
- Tokens per run: 100,000 (post-pre-fill)
- Sampling: pure temperature scaling, no top-k/top-p

## CLI Reference

```bash
# Single generation run
python generate.py --context-length 64 --temperature 1.0 --seed 42 \
  --num-tokens 100000 --model-dir data/model/SmolLM-135M \
  --output-dir data/runs --device cuda

# Plots (all types by default, or select with --plots)
python plot.py --runs data/runs/L0064_T*_S42.parquet
python plot.py --runs data/runs/L*_T0.50_S42.parquet --plots violin temporal
python plot.py --runs data/runs/L0064_T*_S42.parquet --downsample 50

# Reproduce all standard plots
bash reproduce_plots.sh
```

## Dependencies

Managed with `uv`. Key packages: torch (CUDA 12.6), transformers, pandas, pyarrow, scipy, scikit-learn, matplotlib.

# CLAUDE.md — autoloop

## Project

Attractor dynamics in closed-loop autoregressive generation. See `project-brief.md` for full design.

## Repository Layout

```
generate.py          # Core generation loop, CLI entry point
analyze.py           # Post-hoc analysis (compressibility, stationarity, etc.)
plot.py              # Visualization
pilot_sweep.py       # Phase 0 pilot grid runner
project-brief.md     # Research design document
data/                # Gitignored. Parquet outputs, model weights, figures
  model/SmolLM-135M/ # Local model weights (pre-downloaded)
  runs/              # Parquet files: L{L:04d}_T{T:.2f}_S{seed}.parquet
  figures/            # Plot outputs
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
- Model weights: `data/model/SmolLM-135M/` (local, not fetched at runtime)

### Git
- No generated data, figures, or model weights in the repo
- `uv.lock` stays committed
- Small, logical commits

### Sweep Scripts
- One sweep script per phase: `pilot_sweep.py`, then `phase1_sweep.py`, etc.
- Grid parameters hardcoded in each sweep script — no external config files
- Each sweep script calls `generate.py` as a subprocess per condition (crash isolation, natural per-run independence)

## Key Parameters (Phase 0 Pilot)

- Model: SmolLM-135M (local at `data/model/SmolLM-135M/`)
- Context lengths L: 64, 256, 1024
- Temperatures T: 0.5, 1.0, 1.5
- Seeds: 3 per condition (27 runs total)
- Tokens per run: 100,000 (post-pre-fill)
- Sampling: pure temperature scaling, no top-k/top-p

## Dependencies

Managed with `uv`. Key packages: torch (CUDA 12.6), transformers, pandas, pyarrow, scipy, scikit-learn, matplotlib.

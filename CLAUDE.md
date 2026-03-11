# CLAUDE.md -- autoloop

@README.md

## Project

Basin topography and learnable steering in autoregressive self-play. See `docs/project-brief.md` for full design, `observations.md` for findings log.

## Code style

- Type hints on all function signatures
- `logging` module, not print -- DEBUG for per-step, INFO for run progress
- No default params for per-run configuration -- explicit or fail
- No hidden logic, no silent fallbacks
- No pre-emptive exception handling -- let runtime errors surface naturally
- No TODOs in code -- use `raise NotImplementedError` where appropriate
- Single responsibility, DRY, YAGNI, KISS
- Modules expose clean interfaces; internals stay internal
- Scripts, not a package. No `src/` layout. Add modules only when genuinely needed
- Files should stay under ~500 lines; split when natural seams emerge

## Data internals

- All generated data under `data/` (gitignored except `data/figures/`)
- Runs organized by type under `data/runs/`: `sweep/`, `controller/`, `anneal/`, `probe/`, `schedule/`, `survey/`. `runlib.py` has path constants and classification
- `data/runs/index.db`: SQLite index (schema v2) -- `runs`, `basin_types`, `basin_captures` tables. Built by `loop index build`
- Naming: `L{L:04d}_T{T:.2f}_S{seed}` (sweep), `sched_S{seed}_{hash8}` (schedule), `ctrl[d]_S{seed}_{L}_{T}` (controller)
- Each run: one Parquet + JSON sidecar + `.analysis.pkl` cache + `.ckpt` checkpoint. Survey runs also produce `.basins.pkl` (captures with 576-dim embeddings)
- Per-step `temperature` and `context_length` columns in parquet (vary per-step in scheduled/controller runs)
- Compressibility arrays have leading NaN (first W-1 positions). Use `comp_stats(cache, W)` for scalar summaries; raw arrays only for time-series
- Analysis cache: single `.analysis.pkl` per parquet, incremental (new window sizes merged in), invalidated by parquet mtime
- `default_window_sizes()` returns [32,64,128,256] (floor at 32, always includes W>L)
- Checkpoints: same stem as parquet `.ckpt` -- kept after completion for extension
- Basin data: three-tier storage. Embeddings in `.basins.pkl`, type centroids in `data/basins/centroids.npy` + `.json`, scalar summaries in SQLite. `loop index build` ingests pkl/json into DB

## Sweep conventions

- Each condition runs as a subprocess (crash isolation)
- Presets record historical sweeps with rationale; new sweeps can use ad-hoc grids
- `data/figures/` tracked in git; all other data gitignored
- `uv.lock` stays committed; small, logical commits

## Current state

**What's built:** engine.py (StepEngine with sensors, comp_spectrum, embed_context, snapshot/rollback), experiment.py (Fixed/Schedule/Beta controllers + StateMachine), cli.py (unified `loop` CLI), analyze/ package, plot.py, explorer.py, precollapse.py, semantic.py, grep_text.py, sweep.py, runlib.py + runindex.py + schema.py (SQLite index v2 with basin_types + basin_captures).

**Data collected:** ~70 runs. Run `loop index query` for the live catalog.

**Key findings** (see observations.md for full log):
- Four regimes: collapse, suppressed dynamics, rich dynamics, noise
- T_escape(L) saturates: L=64->0.55, L=128->0.57, L=192->0.67, L=256->0.87, L=512->~0.90
- Basin escape hysteresis: exit requires ~0.4T more than avoidance
- Escape by semantic mutation: period-doubling route to chaos
- Closed-loop control finds beta~0.90 equilibrium. Balance T tracks T_escape(L)
- Compressibility is a collapse detector, not a rich-dynamics discriminator. Entropy and Heaps' beta are the right control signals
- Suppressed dynamics is scale-invariant: regime depends on basin-depth/thermal-energy ratio

**Key parameters:** SmolLM-135M, L in {64..512}, T in {0.50..1.50}, seeds {42,123,7}, 100k tokens/run, pure temperature scaling (no top-k/top-p).

## Dependencies

Managed with `uv`. Key packages: torch (CUDA 12.6), transformers, pandas, pyarrow, scipy, scikit-learn, matplotlib, fastapi, uvicorn.

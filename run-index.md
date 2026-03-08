# Run Index

## Grid Status

Run `python sweep.py --status` for the current grid table (auto-generated from parquet files on disk).

Run `python sweep.py --list` to see named presets with completion counts.

Model: SmolLM-135M | Tokens per run: 100,000 (post-pre-fill) | Sampling: pure temperature scaling

## Sweep History

### Pilot (Phase 0)

L={64,128,192,256} × T={0.50,1.00,1.50} × S=42 — 12 runs.
Coarse grid to identify interesting axes. Established three-regime structure and T/L orthogonality.

### Crossover T-densification (Phase 1)

L={64,128,192} × T={0.60,0.70,0.80,0.90} × S=42 — 12 runs.
Dense coverage through the crossover region. Revealed L-dependent collapse boundary, sharp escape at T=0.70, slope-flip in compressibility.

### Seed replication (Phase 1, partial)

L={64,128,192} × T=0.50 × S={42,123,7} — 9 runs complete.
Full grid (56 runs) paused — cross-T analysis proved more informative than more seeds.

### L-densification (Phase 1)

L={160,176,192,208,224} × T=0.50 × S={42,123,7} — 15 runs.
Maps the non-monotonic compressibility profile between L=128 and L=256. Falsified "critical L" hypothesis: jagged continuum, not bifurcation.

### L=256 crossover fill (Phase 1, in progress)

L=256 × T={0.60,0.70,0.80,0.90} × S=42 — 4 runs.
Fills the biggest gap in the T×L grid. Answers whether L=256 extends collapse further into T than L=192.

## Gaps and Decisions

- **L=1024**: Deprioritized. Interesting transition is in L=64–256 range. May revisit later.
- **T > 1.0 densification**: Only T=1.50 above 1.0. Not planned — noise regime is less interesting than crossover.
- **T-densification at L=192**: T={0.62, 0.65, 0.68} to pin exact escape temperature between T=0.60 (collapsed) and T=0.70 (escaped). Optional, low priority.
- **Longer runs**: 100k tokens may be insufficient for slow-timescale dynamics at some conditions. Extension possible via checkpoint resume.

## Command Reference

```bash
# Sweep runner
python sweep.py pilot                           # run a preset
python sweep.py --L 192 --T 0.62 0.65 --seed 42 # ad-hoc grid
python sweep.py --status                         # grid table from disk
python sweep.py --status crossover               # status for one preset
python sweep.py --list                           # list presets
python sweep.py crossover --dry-run              # preview without running

# Single run
python generate.py --context-length 64 --temperature 1.0 --seed 42 \
  --num-tokens 100000 --model-dir data/model/SmolLM-135M \
  --output-dir data/runs --device cuda

# Analysis
python analyze_windows.py      # standard W grid [16,32,64,128,256]
python reproduce_plots.py      # regen all standard plots (cached)
python plot_window_scaling.py  # scaling plots

# Explorer
uvicorn explorer:app --reload --port 8000
```

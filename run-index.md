# Run Index

## Grid Overview

Model: SmolLM-135M | Tokens per run: 100,000 (post-pre-fill) | Sampling: pure temperature scaling

### Completed Runs

**Pilot grid** (Phase 0): L={64, 128, 192, 256} × T={0.50, 1.00, 1.50} × seed=42

**Crossover densification**: L={64, 128, 192} × T={0.60, 0.70, 0.80, 0.90} × seed=42

Total: 31 runs complete.

| L \ T | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 | 1.50 |
|-------|------|------|------|------|------|------|------|
| 64    | 42,123,7 | 42 | 42 | 42 | 42 | 42 | 42 |
| 128   | 42,123,7 | 42 | 42 | 42 | 42 | 42 | 42 |
| 160   | 42   |      |      |      |      |      |      |
| 176   |      |      |      |      |      |      |      |
| 192   | 42,123,7 | 42 | 42 | 42 | 42 | 42 | 42 |
| 208   |      |      |      |      |      |      |      |
| 224   |      |      |      |      |      |      |      |
| 256   | 42   |      |      |      |      | 42   | 42   |

Cells show seed numbers for completed runs. Empty = not yet run.

### In Progress: L-densification

`ldense_sweep.py`: L={160, 176, 192, 208, 224} × S={42, 123, 7} at T=0.50. Maps the non-monotonic compressibility boundary between L=128 and L=256. ~11 hours overnight.

### Paused: Seed Replication

Seeds 123 and 7 via `seed_sweep.py` (48 runs total, 5 complete). Paused in favor of L-densification. Remaining runs cover crossover T=0.60–0.90 and T=1.00/1.50.

### Planned: L=256 Crossover Fill

L=256 × T={0.60, 0.70, 0.80, 0.90} — completes the rectangular grid. Lower priority since L=256 behavior at T=0.50 and T=1.00 already characterizes the extremes.

### Gaps and Decisions

- **L=1024**: Deprioritized. Interesting transition is in L=64–256 range. May revisit in Phase 1.
- **T > 1.0 densification**: Only T=1.50 above 1.0. Not planned — noise regime is less interesting than crossover.
- **Longer runs**: 100k tokens may be insufficient for slow-timescale dynamics at some conditions. Extension possible via checkpoint resume.

## Phase Planning

### Phase 0 — Pilot + Crossover Mapping ← CURRENT
- Core grid complete (24 runs)
- Next: seed replication, analysis of crossover data, updated observations
- Interactive explorer for richer analysis (replacing static plot pipeline)

### Phase 1 — Fixed-Temperature Characterization
- Transfer functions: T→C and T→H curves at each L (now possible with dense T spacing)
- Multi-scale compression analysis at standard W grid (already computed for existing data)
- Identify the decoupling zone and its T/L boundaries
- Complete the fixed-temperature phase map
- Refactor analyze.py: modular metric computation, better integration with explorer

### Phase 2 — Temperature Ramps
- Controlled ramps through crossover region
- Hysteresis / path dependence tests
- Design informed by Phase 1 transfer functions

### Phase 3 — Closed-Loop Control
- Joint T+L controller using multi-scale compression feedback
- Target: sustain system in decoupling zone
- Interactive control UI (builds on Phase 0 explorer)

## Command Templates

```bash
# Single run
python generate.py --context-length {L} --temperature {T} --seed {seed} \
  --num-tokens 100000 --model-dir data/model/SmolLM-135M \
  --output-dir data/runs --device cuda

# Sweep scripts
python pilot_sweep.py          # Phase 0 pilot grid
python crossover_sweep.py      # Crossover T-densification
python seed_sweep.py           # Seed replication (123, 7)
python ldense_sweep.py         # L-densification at T=0.50

# Multi-window analysis
python analyze_windows.py      # Standard W grid [16,32,64,128,256]
python plot_window_scaling.py  # Scaling plots
python plot_window_scaling.py --temps 0.50 --ldense  # L-dense detail plot
```

# Run Index

## Grid Overview

Model: SmolLM-135M | Tokens per run: 100,000 (post-pre-fill) | Sampling: pure temperature scaling

### Completed Runs

**Pilot grid** (Phase 0): L={64, 128, 192, 256} × T={0.50, 1.00, 1.50} × seed=42

**Crossover densification**: L={64, 128, 192} × T={0.60, 0.70, 0.80, 0.90} × seed=42

Total: 24 runs complete.

| L \ T | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 | 1.50 |
|-------|------|------|------|------|------|------|------|
| 64    | 42   | 42   | 42   | 42   | 42   | 42   | 42   |
| 128   | 42   | 42   | 42   | 42   | 42   | 42   | 42   |
| 192   | 42   | 42   | 42   | 42   | 42   | 42   | 42   |
| 256   | 42   |      |      |      |      | 42   | 42   |

Cells show seed numbers for completed runs. Empty = not yet run.

### Planned: Seed Replication

Seeds 123 and 7 across the full grid. Priority order TBD — may target specific (L, T) conditions where seed-dependent behavior is most informative (e.g. T=0.50 collapse boundary, T=0.7–0.9 crossover).

### Planned: L=256 Crossover Fill

L=256 × T={0.60, 0.70, 0.80, 0.90} — completes the rectangular grid. Lower priority than seed replication since L=256 behavior at T=0.50 and T=1.00 already characterizes the extremes.

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
- Multi-scale compression: W=2L, W=4L analysis on existing data
- Identify the decoupling zone and its T/L boundaries
- Complete the fixed-temperature phase map

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
```

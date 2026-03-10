# Handover — 2026-03-10e: Controller v1 & Balance Points

## What happened this session

1. **Fixed comp_stats interface** — consumers were bypassing `comp_stats()` and using raw `np.mean()` on compressibility arrays with leading NaN. Routed `plot_window_scaling.load_mean_compressibility` through `comp_stats()`, unified `STANDARD_WINDOWS` and `STANDARD_W` to defer to `default_window_sizes(0)`, removed stale `w <= l` filter. Docstrings document the NaN convention.

2. **W>L analysis across full grid** — recomputed analysis at [32,64,128,256] for all 64 runs. Key result: compressibility at W>L scales is only ~0.03 below noise floor in rich dynamics. It's a collapse detector, not a rich-dynamics discriminator. Entropy and Heaps' β are the right control signals.

3. **Built controller.py** — closed-loop generation with sensor feedback:
   - Segments of N steps at fixed L/T
   - Sensors after each segment: entropy, Heaps' β, comp_W64
   - T is fast control (±0.05), L is slow control (±16 when T saturates)
   - Rollback on collapse, dead zone ±0.15 around β target
   - Sensor window scales with segment size (5× segment_steps)

4. **Found balance points** from 4 controller runs:

   | Run | Start → Settled | β range | Notes |
   |-----|----------------|---------|-------|
   | `ctrl_S42_8_0.60` | L=8 T=0.60 → **T=0.70** | 0.85–1.07 | 30 segments held, very stable |
   | `ctrl_S42_16_0.55` | L=16 T=0.55 → **T=0.75** | 0.86–0.96 | 12 segments held |
   | `ctrl_S42_16_1.00` | L=16 T=1.00 → **no adjustment** | 0.85–0.95 | Already in zone |
   | `ctrl_S42_128_0.70` | L=128 T=0.70 → **T=0.90–0.95** | 0.85–1.27 | Oscillating, first T↓ observed |

5. **Explorer updated** (uncommitted prior work, now committed) — handles controller/anneal/schedule run names, temperature and context_length as plottable step metrics.

## Key findings

- **β ≈ 0.90 is the natural equilibrium** for SmolLM-135M regardless of L or starting T. The controller converges there from every starting point tested.
- **Balance T tracks T_escape(L)**: 0.70 (L=8), 0.75 (L=16), 0.90–0.95 (L=128).
- **Small L = wide basin, large L = sharp boundary**: L=8/16 are trivially stable; L=128 oscillates between T=0.90 and T=0.95 with β swinging 0.60–1.27.
- **Entropy tracks T linearly** at fixed β — different T values produce different entropy regimes with the same vocabulary growth rate.

## What to do next

### 1. Launch L=256 controller run
The steepest escape boundary — should actually stress the controller (rollbacks, L adjustments). Start from the suppressed zone:

```bash
python controller.py --seed 42 --total-steps 10000 --segment-steps 1000 --start-L 256 --start-T 0.70
```

Expect: T will need to ramp significantly (T_escape ≈ 0.87 for L=256). May trigger rollback if it enters collapse. May need L reduction to escape. This is where the controller becomes non-trivial.

### 2. Semantic analysis of controller runs
The L=128 run (`ctrl_S42_128_0.70`) oscillates at the escape boundary — the text should show the transition between regimes. Analyse:

```bash
# What does the text look like at balance points?
python semantic.py --runs data/runs/ctrl_S42_128_0.70.parquet
python semantic.py --runs data/runs/ctrl_S42_8_0.60.parquet

# Or manual inspection — look at text around T transitions
python -c "
import pandas as pd
df = pd.read_parquet('data/runs/ctrl_S42_128_0.70.parquet')
exp = df[df.phase == 'experiment']
# Around step 6000-7000 where β hit 1.27 then dropped
window = exp[(exp.step >= 6000) & (exp.step <= 7500)]
print(''.join(window.decoded_text.tolist()))
"
```

Questions to answer:
- What does text at β ≈ 0.90 look like? Coherent or statistical noise?
- Does the L=128 oscillation (T=0.90→0.95→0.90) produce visible regime shifts in the text?
- Is the β=1.27 spike (step 7000) an escape event visible in content?

### 3. Possible refinements
- **Proportional T steps**: not urgent (oscillation is interesting, not broken)
- **Longer runs**: 50-100k steps to see if balance is stable or drifts
- **Different seeds**: does seed 123 find the same balance T?

## Commits this session
- `bb41af6` — Route compressibility consumers through comp_stats(); unify window sizes
- `4485b63` — Controller v1: closed-loop β hill-climb with L/T adjustment

## Controller run data (preserved in data/runs/)
- `ctrl_S42_8_0.60.parquet` + `.decisions.json` + `.meta.json`
- `ctrl_S42_16_0.55.parquet` + `.decisions.json` + `.meta.json`
- `ctrl_S42_16_1.00.parquet` + `.decisions.json` + `.meta.json`
- `ctrl_S42_128_0.70.parquet` + `.decisions.json` + `.meta.json`

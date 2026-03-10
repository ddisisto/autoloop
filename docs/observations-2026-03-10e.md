# Observations — 2026-03-10e: Controller v1 & Balance Points

## comp_stats interface and W>L analysis

Fixed NaN propagation from compressibility arrays. `sliding_compressibility()` produces arrays with leading NaN (first W-1 positions lack a full window). Consumers were using ad-hoc `np.mean()` instead of `comp_stats()`, causing silent NaN cascades in aggregations.

Changes:
- `plot_window_scaling.py`: `load_mean_compressibility` now uses `comp_stats()` instead of manual NaN filtering
- `analyze_windows.STANDARD_WINDOWS` and `precollapse.STANDARD_W` now defer to `default_window_sizes(0)` — single source of truth
- Removed `w <= l` filter from plot_window_scaling (W>L is now intentional)
- Docstrings on `sliding_compressibility`, `comp_stats`, and package `__init__` document the NaN convention

### W>L analysis results

Recomputed analysis at `[32, 64, 128, 256]` across all 64 runs.

**Noise floor (T=1.50):** Identical across all L. W256=0.611 regardless of context length. Pure sampling-noise baseline.

**Rich dynamics (T=1.00):** W>L signal is weak. At L=64, comp_W128=0.639, comp_W256=0.580 — only ~0.03 below noise floor. In-context contribution at W>L scales is minimal.

**Collapse (T=0.50):** Dominates all scales. W256 as low as 0.068 at L=256.

**Key finding:** The controller's real operating range is at W≤L scales. Compressibility is a collapse detector, not a rich-dynamics discriminator. Entropy and Heaps' β are better control signals for the escape boundary.

```
# Reproduction
python -c "
from pathlib import Path
from analyze import analyze_run, comp_stats, default_window_sizes
import re
ws = default_window_sizes(0)
for p in sorted(Path('data/runs').glob('L*_T*_S42.parquet')):
    m = re.match(r'L(\d+)_T([\d.]+)_S(\d+)', p.stem)
    L, T = int(m.group(1)), float(m.group(2))
    cache = analyze_run(p, ws)
    vals = {w: comp_stats(cache, w)['mean'] for w in ws}
    print(f'L={L:3d} T={T:.2f}  ' + '  '.join(f'W{w}={v:.3f}' for w, v in vals.items()))
"
```

## Controller v1

Built `controller.py` — closed-loop generation with sensor feedback and L/T adjustment.

### Architecture
- Generates tokens in segments (configurable, default 1000 steps)
- After each segment: reads sensors (entropy, Heaps' β, compressibility)
- Hill-climbs toward β ≈ 1.0 (vocabulary growth rate = linear)
- T is fast control (±0.05 per segment), L is slow control (±16 when T saturates)
- Rollback on collapse: restores pre-segment state, tries higher T
- Sensor window scales with segment size (5× segment_steps)
- Dead zone ±0.15 around target to avoid chasing noise
- Outputs standard parquet + decisions JSON log

### Balance point discovery

Ran controller from multiple starting points. Key results:

| Run | Start | Settled at | β range | Segments held | Entropy |
|-----|-------|-----------|---------|---------------|---------|
| `ctrl_S42_8_0.60` | L=8, T=0.60 | L=8, T=0.70 | 0.85–1.07 | 30 consec | 3.5–4.0 |
| `ctrl_S42_16_0.55` | L=16, T=0.55 | L=16, T=0.75 | 0.86–0.96 | 12 consec | 2.4–3.4 |
| `ctrl_S42_16_1.00` | L=16, T=1.00 | L=16, T=1.00 | 0.85–0.95 | 19 consec (no adj) | 3.6–4.1 |
| `ctrl_S42_128_0.70` | L=128, T=0.70 | L=128, T=0.90–0.95 | 0.85–1.27 | oscillating | 1.9–3.2 |

```bash
# Reproduction
python controller.py --seed 42 --total-steps 3000 --segment-steps 80 --start-L 8 --start-T 0.60
python controller.py --seed 42 --total-steps 3000 --segment-steps 160 --start-L 16 --start-T 0.55
python controller.py --seed 42 --total-steps 3000 --segment-steps 160 --start-L 16 --start-T 1.00
python controller.py --seed 42 --total-steps 10000 --segment-steps 1000 --start-L 128 --start-T 0.70
```

### Key findings

**β ≈ 0.90 is a natural equilibrium for this model.** At small L (8, 16), β converges to 0.88–0.92 regardless of starting T (tested T=0.55 through T=1.00). The controller finds this equilibrium quickly and holds it. This appears to be a structural property of SmolLM-135M, not a control artifact.

**Balance T tracks T_escape(L).** The T at which the controller settles increases with L: T=0.70 for L=8, T=0.75 for L=16, T=0.90–0.95 for L=128. This is consistent with the T_escape(L) curve from characterization sweeps.

**Entropy tracks T linearly, independent of β.** At the same β ≈ 0.90, entropy ranges from 2.4 (L=16, T=0.75) to 4.0 (L=8, T=0.70). The controller finds different entropy regimes with the same vocabulary growth rate — different dynamics at the same "novelty temperature."

**Small L has a wide β basin; large L oscillates.** At L=8/16, β stays in [0.85, 1.07] for any T in a wide range (0.70–1.00). At L=128, β swings from 0.60 to 1.27 — the escape boundary is sharper. The controller oscillates between T=0.90 and T=0.95, straddling the boundary. This is where finer T steps or proportional control would help.

**The controller is currently a T-search, not a navigator.** At small L, it finds the T where β enters the dead zone and stops — the basin is wide enough that no further adjustment is needed. At L=128, it starts to actually navigate (the first T↓ in the dataset, from T=0.95 to T=0.90 after β=1.27). Real control only matters near the escape boundary at larger L.

### Open questions
- Is β ≈ 0.90 model-specific or a property of the generation setup? Would a different model show the same equilibrium?
- Can proportional T-steps (smaller correction when β is close to target, larger when far) stabilize the L=128 oscillation?
- What does the *text* look like at the balance point? Is L=8/T=0.70 producing anything coherent, or is it high-entropy noise that happens to have the right vocabulary growth rate?
- L=256 should show even sharper dynamics — the escape boundary is steeper there. Does the controller trigger rollbacks?

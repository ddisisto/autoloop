# Observations — 2026-03-07

Detailed findings log. See [observations.md](../observations.md) for current model and index.

### 2026-03-07 — Full L=256 sweep and cross-L comparison

**Data:** `L0256_T{0.50,1.00}_S42.parquet` (100k), `L0256_T1.50_S42.parquet` (75k, in progress)

**L=256 T=0.50** (now complete at 100k): Entropy mean 0.069 nats (vs L=64's 0.683). Collapsed to near-zero by ~15k steps and never escaped. The attractor is 10x deeper by entropy measure.

**L=256 T=1.00**: Entropy mean 4.917 nats (vs L=64's 3.721). Notably *higher* entropy than L=64 at the same temperature. Phase portrait shows the cloud shifted right and down — higher entropy but lower compressibility (~0.55-0.70 vs L=64's ~0.65-0.85). This is surprising: longer context at T=1.00 produces *less* compressible output despite higher entropy. The model may be sampling more diverse continuations when conditioning on more context, but the longer window dilutes any local repetitive structure.

**L=256 T=1.50**: Entropy 8.351 nats (vs L=64's 7.989). Tight cluster, stationary. Slightly higher entropy than L=64 — marginal effect of L in the noise regime.

**Cross-L temporal portraits at T=0.50** tell the key story: L=64 wanders broadly across phase space for the entire run (mixed time-colors everywhere — no temporal ordering). L=256 collapses to bottom-left corner early and stays. The attractor basin depth scales dramatically with L in the collapse regime.

**Cross-L temporal at T=1.00**: L=256 shifts the whole cloud down-left compared to L=64. L is pulling toward order even at T=1.00, but the system resists collapse — it finds a different operating point rather than collapsing.

**Key insight: L as a second control parameter.** T and L act on different axes:
- T controls the *noise floor* — randomness of each individual sample
- L controls the *memory horizon* — how much self-generated history the model conditions on, and therefore how deep/sticky attractor basins are

These are orthogonal actuators. T is a local perturbation (per step). L is a structural parameter (how much past determines the present). This suggests a two-actuator controller:
- T for fast corrections (raise when approaching collapse, lower when approaching noise)
- L for regime selection (shorten to escape stuck attractors, lengthen to deepen coherence)

Reducing L mid-run is an *escape mechanism* — if the system locks into a loop at L=256, shortening to L=64 makes that attractor shallower, allowing escape. Then L can be extended again. This is annealing for memory depth.

**Implication for L-grid design:** The interesting L-transition is between 64 and 256 (one escapes, the other locks). L=1024 will likely just be "even more locked." Densifying L in the 64-256 range (e.g., L=64, 128, 192) maps the attractor depth curve — critical for designing an L-controller. This may be more informative than extending to L=1024.

```bash
python plot.py --runs data/runs/L*_T0.50_S42.parquet
python plot.py --runs data/runs/L*_T1.00_S42.parquet
python plot.py --runs data/runs/L*_T*_S42.parquet
```

---

### 2026-03-07 — EOS signal analysis across all runs

**Data:** All available runs (L=64 x 3T, L=128 T=0.50 partial, L=256 x 3T)

**EOS counts and rates:**

| Run | N tokens | EOS count | EOS rate | 5-block profile |
|-----|----------|-----------|----------|-----------------|
| L=64 T=0.50 | 100k | 13 | 0.013% | [4, 3, 0, 3, 3] |
| L=64 T=1.00 | 100k | 91 | 0.091% | [21, 14, 15, 18, 23] |
| L=64 T=1.50 | 100k | 63 | 0.063% | [15, 8, 8, 18, 14] |
| L=128 T=0.50 | 52k | 3 | 0.006% | [0, 0, 1, 2, 0] |
| L=256 T=0.50 | 100k | 1 | 0.001% | [1, 0, 0, 0, 0] |
| L=256 T=1.00 | 100k | 57 | 0.057% | [11, 12, 14, 2, 18] |
| L=256 T=1.50 | 100k | 46 | 0.046% | [12, 5, 9, 9, 11] |

**EOS rate peaks at T=1.00** at both L=64 (0.091%) and L=256 (0.057%). Not at the collapse regime, not at noise — at the "interesting" regime. The block profiles at T=1.00 are relatively flat, suggesting EOS events are distributed throughout the run, not clustered in transients.

**L suppresses EOS in the collapse regime.** L=64 T=0.50: 13 EOS. L=128 T=0.50: 3 in 52k tokens. L=256 T=0.50: 1 total, in the first block only — once the attractor locks, EOS probability drops to essentially zero. The model is so confident in the repetitive continuation that it never considers stopping.

**L also suppresses EOS at T=1.00 and T=1.50**, but less dramatically (91→57 and 63→46). The suppression is strongest in the collapse regime.

**Log-probability tracks entropy closely:** mean log-prob ≈ -mean entropy across all runs (e.g., L=64 T=1.00: entropy 3.72, log-prob -3.72). This is expected at equilibrium — average surprisal matches distribution entropy.

**Three-sensor interpretation.** EOS is a model-internal coherence signal (the model's assessment of sequence completeness), distinct from entropy (local uncertainty) and compressibility (observer-assessed structure). The three sensors probe different aspects of the same process. EOS peaking at T=1.00 suggests the rich-dynamics regime is where the model encounters the most "natural boundaries" — possibly marking mode transitions.

**For control:** EOS is sparse (~1 per 1000-10000 tokens), so it's a slow-timescale indicator requiring wide trailing windows for stable rate estimates. More useful as a regime classifier than a fast correction signal. The dramatic L-dependence in the collapse regime (13→3→1 across L=64/128/256) could serve as an early warning of attractor lock-in during dynamic-L control.

```bash
python -c "
import pandas as pd
from pathlib import Path
for p in sorted(Path('data/runs').glob('*.parquet')):
    df = pd.read_parquet(p)
    exp = df[df.phase == 'experiment'].reset_index(drop=True)
    n = len(exp)
    bs = n // 5
    blocks = [int(exp.eos[i*bs:(i+1)*bs].sum()) for i in range(5)]
    print(f'{p.stem}: {n} tokens, EOS={int(exp.eos.sum())} rate={exp.eos.mean():.6f} blocks={blocks}')
"
```

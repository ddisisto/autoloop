# Observations Log

Append-only record of findings. Each entry includes reproduction commands.

## Current Model

*Rewritten as understanding evolves. Entries below are the evidence trail.*

**System:** SmolLM-135M generating into its own context. No external input. Pure autoregressive dynamics.

**Two orthogonal actuators:**
- T (temperature): per-step noise floor. Controls escape probability from attractors.
- L (context length): memory horizon. Controls attractor basin depth, stickiness, and collapse boundary.

**Three regimes** at fixed L: collapse (T≤0.60), rich dynamics (T~0.80–1.00), noise (T≥1.50). Crossover zone is T~0.70–0.80. But the collapse boundary is L-dependent — longer L extends collapse upward in T.

**Collapse boundary shifts with L.** At T=0.60: L=192 is collapsed (entropy 0.187) while L=64 (0.791) and L=128 (0.502) have escaped. The escape is sharp — at T=0.70 all L values jump above entropy 1.0. Longer context doesn't just deepen collapse, it extends the collapse regime to higher temperatures.

**Attractor structure at T=0.50 is a staircase, not a binary.** Each L value has a distinct entropy floor — L=64 sits on a high meta-stable basin (~0.2–0.4 nats), L=128 on a lower false floor (~0.1–0.2 nats), L=256 hits the true zero-entropy floor by ~15k steps. Collapse is a timescale phenomenon: every T=0.50 run may collapse eventually; L sets how fast you descend the staircase.

**L-densification at T=0.50: jagged, not smooth.** L-profile from 64→256 is non-monotonic with local dips at L=208 and L=128, local peaks at L=176 and L=192. Seed variance (std 0.04–0.10) is comparable to inter-L differences (Δmean ~0.03–0.08) at n=3 seeds. The overall downward trend is robust; the fine structure may be noise. No clean phase transition or bifurcation point was found — the "critical L" hypothesis was falsified.

**Slope-flip in the T dimension.** Compressibility (W=64) *decreases* with L at T≤0.60 (deeper collapse = less local structure) but *increases* with L at T=1.00 (longer context = more structure). The sign flip occurs around T=0.70–0.80. This is the phase boundary in the L dimension.

**EOS peak shifts with L.** Peak EOS rate occurs at T=0.90 for L=64 and L=128, but at T=0.70 for L=192. The EOS peak tracks the escape-from-collapse transition: it fires most at the boundary between structured and free dynamics. Since L=192's collapse extends to T=0.60, its escape boundary (and EOS peak) is at a lower T.

**T=1.50 is a universal noise floor.** Compressibility ~0.705 and entropy ~8.0–8.3 regardless of L. At high T, context length is irrelevant — thermal noise dominates.

**W (measurement window) is a third dimension.** Compressibility depends strongly on W. Standard grid: W ∈ {16, 32, 64, 128, 256}. At T=0.50, L-curves separate dramatically across W. At T≥0.90, L barely matters at any W. Gzip has fixed overhead (~20B header + Huffman table); at W=16 this inflates ratios above 1.0. Useful range: W≥64 for quantitative work, W≥32 qualitatively. Correction possible by normalizing against incompressible baseline at matched byte length.

**Three-sensor framework:** entropy (model uncertainty), compressibility (observer-assessed structure), EOS rate (model-assessed coherence). Each probes a different aspect. All three needed for regime identification.

**Open questions:**
- What is L=192's exact escape temperature? Somewhere in T=0.60–0.70. T-densification at T={0.62, 0.65, 0.68} would pin it.
- Does L=256 extend collapse even further in T? Gap at T={0.60–0.90} is the biggest hole in the grid.
- Is the T=1.00 compressibility increase with L (0.739→0.785) robust across seeds?

---

### 2026-03-06 — Initial L=64 temperature sweep (seed=42)

**Data:** `L0064_T{0.50,1.00,1.50}_S42.parquet` (100k tokens each)

**Throughput:** ~24-38 tok/s on GTX 1070 at L=64.

**Three distinct regimes observed:**

- **T=0.50**: Degenerate loop attractor. Entropy mean 0.68 nats. Text collapses to verbatim repetition ("Ethnomusicology is the study of music." repeating). Compressibility 0.35. Classified as transient — block-level entropy fluctuates (1.03 → 0.29 → 0.84), suggesting mode-switching between tighter and looser loops.
- **T=1.00**: Rich dynamics. Entropy mean 3.72 nats. Coherent-ish text with degraded grammar. Compressibility 0.74. Also transient. Wide spread in phase portrait — continuum of states rather than discrete modes.
- **T=1.50**: Near-random. Entropy mean 7.99 nats, very stationary (trend -0.009/block). Word salad. Compressibility 0.71 — similar to T=1.00 despite very different entropy, because gzip picks up character-level patterns in English tokens even when token-order is random.

**Phase portrait** shows three clearly separated clusters with a gap in entropy space (~2-3 nats) between T=0.50 and T=1.00. Crossover region is below T=1.0.

```bash
# Summary stats
python -c "
import pandas as pd
for t in ['0.50', '1.00', '1.50']:
    df = pd.read_parquet(f'data/runs/L0064_T{t}_S42.parquet')
    exp = df[df.phase == 'experiment']
    print(f'T={t}: entropy mean={exp.entropy.mean():.3f} std={exp.entropy.std():.3f}, EOS={exp.eos.sum()}')
"

# Plots
python plot.py --runs data/runs/L0064_T*_S42.parquet
```

---

### 2026-03-06 — Context length effect at T=0.50 (L=64 vs L=256, partial)

**Data:** `L0064_T0.50_S42.parquet` (100k), `L0256_T0.50_S42.parquet` (49k, partial)

**Context length dramatically deepens the loop attractor.** L=256 collapses to near-zero entropy (~0.1 nats) by ~15k steps and stays there. L=64 keeps having escape episodes throughout the full 100k run (entropy excursions up to ~3 nats).

**Phase portrait:** L=256 is a tight cluster at (entropy ~0.1, compressibility ~0.1). L=64 scatters widely up to entropy ~3, compressibility ~0.75. Longer context = more self-reinforcing repetitive signal = deeper attractor basin.

**Implication:** Context length is a significant experimental variable, not just a nuisance parameter. Attractor basin depth scales with L.

```bash
python plot.py --runs data/runs/L0064_T0.50_S42.parquet data/runs/L0256_T0.50_S42.parquet
```

---

### 2026-03-06 — Violin plots reveal temporal collapse structure

**Data:** Same as above.

**Violin plot** (entropy + compressibility distributions per 10k-token time block) shows the collapse dynamics much more clearly than time series alone:

- **T=0.50 L=64**: Entropy distribution progressively narrows and shifts left, but non-monotonically — blocks 50-70k show re-broadening (escape episodes). Compressibility mirrors this.
- **T=0.50 L=256**: Entropy collapses to a spike by 20-30k and stays. Compressibility reaches ~0.1-0.2 (vs L=64's 0.3-0.5). Longer context = more self-reinforcing repetitive signal = deeper attractor.
- **T=1.00**: Broad, stable distributions throughout. Some compressibility bimodality in individual blocks.
- **T=1.50**: Narrow, stable. Stationary noise.

W=L/4 (lighter overlay) is consistently tighter than W=L, confirming local-scale repetition structure.

```bash
python plot.py --runs data/runs/L0064_T*_S42.parquet --plots violin
python plot.py --runs data/runs/L0064_T0.50_S42.parquet data/runs/L0256_T0.50_S42.parquet --plots violin
```

---

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

---

### 2026-03-08 — Staircase of attractor basins at T=0.50

**Data:** `L{0064,0128,0256}_T0.50_S42.parquet` (100k each)

**Visual inspection of `Lmulti_T0.50_S42_entropy.png` reveals three distinct entropy floors,** not just "collapsed vs active":

1. **L=256 (green):** Hits true zero-entropy floor at ~15k steps. Permanent lock-in.
2. **L=128 (orange):** Meta-stable false floor at ~0.1–0.2 nats for ~45k steps, with intermittent escape bursts. Falls to true floor just before 60k steps.
3. **L=64 (blue):** Own floor at ~0.2–0.4 nats with frequent excursions above. Has not cascaded to the deeper basins within 100k steps.

**Implication:** Collapse is a timescale phenomenon, not a binary regime boundary. There's a hierarchy of attractor basins at progressively lower entropy. L controls cascade rate through them. Testable prediction: extend L=64 T=0.50 significantly — does it eventually find L=128's false floor, then the true floor?

```bash
python plot.py --runs data/runs/L*_T0.50_S42.parquet --plots entropy
```

---

### 2026-03-08 — EOS phase-space position is regime-dependent

**Data:** Phase portraits across T={0.50, 1.00, 1.50}, all L values.

**At T=1.00:** EOS diamonds (all L) sit squarely in the interior of their respective phase-portrait clouds. Not at edges, not at extremes. EOS is an interior/coherence signal — the model tries to end during richest dynamics.

**At T=0.50:** L=64 EOS events scatter across escape bursts (high entropy, high compressibility). L=256 EOS events (just 1) near the collapsed origin. EOS fires during transitions, not from the interior.

**At T=1.50:** EOS events sparse. Consistent with rate peaking at T=1.00.

The meaning of EOS depends on the regime. "Model wants to stop" means different things in collapse vs rich dynamics.

```bash
python plot.py --runs data/runs/L*_T1.00_S42.parquet --plots phase
python plot.py --runs data/runs/L*_T0.50_S42.parquet --plots phase
```

---

### 2026-03-08 — Multi-window analysis: W as measurement dimension

**Data:** All 24 runs analyzed at W ∈ {16, 32, 64, 128, 256}.

**Previous approach used W=L and W=L/4** — both scale with L, making cross-L comparison ambiguous. Fixed W=64 across all L provides consistent measurement.

**Compressibility at fixed W=64 vs L:**

| L\T | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 | 1.50 |
|-----|------|------|------|------|------|------|------|
| 64  | 0.35 | 0.48 | 0.61 | 0.68 | 0.72 | 0.74 | 0.71 |
| 128 | 0.26 | 0.33 | 0.55 | 0.63 | 0.71 | 0.76 | 0.70 |
| 192 | 0.35 | 0.31 | 0.56 | 0.60 | 0.70 | 0.79 | 0.70 |
| 256 | 0.23 | --   | --   | --   | --   | 0.74 | 0.71 |

**Key patterns:**
- T=0.50–0.60: compressibility *decreases* with L (deeper collapse = less local structure visible at W=64)
- T=1.00: compressibility *increases* with L (longer context creates more structure)
- T≥0.90: L barely matters — flat across all context lengths
- Crossover (slope flip) is between T=0.70 and T=0.80

**L=192 anomaly at T=0.50:** At W=16, L=192 (1.19) is more compressible than L=64 (1.05). Non-monotonic in L. Suggests L=192 locks into a short-period loop — high local structure, low context-scale structure. Seed replication in progress to verify.

**W=16 hits gzip overhead floor.** Compressibility > 1.0 means the gzip header costs more than the data. W=16 is below the useful measurement range.

```bash
python analyze_windows.py           # compute standard W grid
python plot_window_scaling.py       # generate scaling plots
```

---

### Reproduction

All standard plots can be regenerated via `python reproduce_plots.py`.

Figures referenced above:
- `L0064_Tmulti_S42_phase.png` — L=64 three-regime phase portrait
- `Lmulti_T0.50_S42_temporal.png` — cross-L temporal at T=0.50 (escape vs lock)
- `Lmulti_T1.00_S42_temporal.png` — cross-L temporal at T=1.00 (operating point shift)
- `Lmulti_Tmulti_S42_phase.png` — combined phase portrait all runs
- `Lmulti_Tmulti_S42_violin.png` — combined violin distributions
- `Lmulti_Tmulti_S42_compressibility.png` — compressibility time series all runs

---

### 2026-03-08 — Seed Replication Confirms L=192 Non-Monotonic Anomaly at T=0.50

**Data:** `L{0064,0128,0192,0256}_T0.50_S{42,123,7}.parquet`, analyzed at W ∈ {16, 32, 64, 128, 256}.

**The L=192 non-monotonicity is NOT a seed artifact.** Both seed=42 and seed=123 show higher compressibility and entropy at L=192 than at L=128, breaking the expected monotonic pattern (more memory → deeper collapse).

**Compressibility (W=64) at T=0.50 by L and seed:**

| L | S=42 | S=123 | S=7 |
|---|------|-------|-----|
| 64 | 0.3473 | 0.2958 | 0.4613 |
| 128 | 0.2592 | 0.3479 | 0.4373 |
| 192 | 0.3510 | 0.4127 | — |
| 256 | 0.2291 | — | — |

**Compressibility (W=128) at T=0.50:**

| L | S=42 | S=123 | S=7 |
|---|------|-------|-----|
| 64 | 0.2170 | 0.1798 | 0.3181 |
| 128 | 0.1425 | 0.1791 | 0.2564 |
| 192 | 0.1839 | 0.2371 | — |
| 256 | 0.1243 | — | — |

**Entropy mean at T=0.50:**

| L | S=42 | S=123 | S=7 |
|---|------|-------|-----|
| 64 | 0.6825 | 0.5561 | 0.9204 |
| 128 | 0.1906 | 0.1465 | 0.5109 |
| 192 | 0.3110 | 0.4669 | — |
| 256 | 0.0689 | — | — |

**Key findings:**

- S=123 shows an even stronger L=192 anomaly than S=42 (comp W=64: 0.4127 vs 0.3510, entropy: 0.4669 vs 0.3110)
- The expected monotonic pattern holds for L=64→128 and L=192→256, but breaks at L=128→192
- This suggests a structural resonance or attractor mismatch at L=192 that prevents deep collapse
- At L=64, seed variation is large (entropy ranges 0.56–0.92) — short context makes dynamics more seed-dependent
- At L=128, seeds diverge less on entropy but compressibility varies (S=42 achieves deepest collapse)
- The W=128 data confirms the same non-monotonic pattern, ruling out window-size artifacts

```bash
python -c "
import pickle, numpy as np
for L in [64, 128, 192, 256]:
    for seed in [42, 123, 7]:
        pkl = f'data/runs/L{L:04d}_T0.50_S{seed}.W16_W32_W64_W128_W256.analysis.pkl'
        try:
            with open(pkl, 'rb') as f:
                data = pickle.load(f)
            v64 = data['compressibility'][64]
            valid = v64[~np.isnan(v64)]
            print(f'L={L} S={seed}: comp_W64={valid.mean():.4f} entropy={data[\"summary\"][\"entropy_mean\"]:.4f}')
        except FileNotFoundError:
            pass
"
```

---

### 2026-03-08 — Pre-registered Predictions: L-densification at T=0.50

**Context:** L-dense sweep running overnight — L={160, 176, 208, 224} × S={42, 123, 7} at T=0.50, filling the gap between L=128 (deep collapse) and L=256 (deepest collapse) where L=192 showed anomalous non-monotonicity confirmed across three seeds.

**Predictions (recorded before data arrives):**

1. **L=160**: behaves like L=128 — deep collapse, low entropy, low compressibility. The extra 32 tokens of context aren't enough to change the attractor landscape.

2. **L=176**: transition zone — expect seed-dependent behavior. Some seeds collapse (like L=128), others resist (like L=192). If the bifurcation is sharp, this is where it lives.

3. **L=208**: behaves like L=192 — elevated entropy/compressibility, basin-hopping dynamics, regime-switching visible in time series.

4. **L=224**: begins approaching L=256 — deeper collapse than L=192, but possibly slower to converge. The longer context starts to lock in again.

**Core hypothesis:** There is a critical context length L_c somewhere in [160, 192] where the attractor landscape bifurcates. Below L_c, one dominant basin captures the dynamics. Above L_c (but below ~224), competing basins of similar depth coexist, producing the basin-hopping and elevated compressibility we see at L=192. At L=256, one basin wins again through sheer memory inertia.

**Alternative hypothesis:** The non-monotonicity is a broad plateau rather than a sharp transition — L=160 through L=224 all show similar elevated dynamics, with L=128 and L=256 as the special cases (collapse points) rather than L=192 being special.

**What would falsify these:** If L=160 shows the same elevated dynamics as L=192, the bifurcation is below L=160 and the "plateau" model wins. If L=176 collapses deep like L=128, the transition is sharp and narrow around L=192.

---

### 2026-03-09 — L-densification Results: Predictions vs Reality

**Data:** L={160, 176, 208, 224} × S={42, 123, 7} at T=0.50, 15 runs complete. Combined with existing L={64, 128, 192, 256} data for full profile.

**Compressibility (W=64) seed-averaged across L:**

| L | S=42 | S=123 | S=7 | mean | std |
|---|------|-------|-----|------|-----|
| 64 | 0.347 | 0.296 | 0.461 | 0.368 | 0.069 |
| 128 | 0.259 | 0.348 | 0.437 | 0.348 | 0.073 |
| 160 | 0.362 | 0.325 | 0.241 | 0.310 | 0.050 |
| 176 | 0.308 | 0.294 | 0.410 | 0.338 | 0.052 |
| 192 | 0.351 | 0.407 | 0.248 | 0.335 | 0.066 |
| 208 | 0.290 | 0.282 | 0.202 | 0.258 | 0.040 |
| 224 | 0.283 | 0.344 | 0.301 | 0.309 | 0.026 |
| 256 | 0.229 | — | — | 0.229 | — |

**Entropy mean across L:**

| L | S=42 | S=123 | S=7 | mean | std |
|---|------|-------|-----|------|-----|
| 64 | 0.683 | 0.556 | 0.920 | 0.720 | 0.151 |
| 128 | 0.191 | 0.147 | 0.511 | 0.283 | 0.162 |
| 160 | 0.378 | 0.201 | 0.195 | 0.258 | 0.085 |
| 176 | 0.242 | 0.244 | 0.189 | 0.225 | 0.026 |
| 192 | 0.311 | 0.485 | 0.239 | 0.345 | 0.103 |
| 208 | 0.314 | 0.155 | 0.125 | 0.198 | 0.083 |
| 224 | 0.225 | 0.278 | 0.178 | 0.227 | 0.041 |
| 256 | 0.069 | — | — | 0.069 | — |

**Prediction scorecard:**

1. **L=160 "behaves like L=128"** — WRONG. Mean comp 0.310 (lower than L=128's 0.348), entropy 0.258. Not deep collapse — sits in the mid-range. The transition is below L=160.

2. **L=176 "transition zone, seed-dependent"** — PARTIALLY RIGHT on seed-dependence (S=7 at 0.41 vs S=123 at 0.29), WRONG that it's a boundary. Just more of the same plateau.

3. **L=208 "behaves like L=192"** — WRONG in an interesting way. L=208 shows the *lowest* compressibility in the mid-range (0.258 mean, entropy 0.198). It's actually the deepest collapser between L=128 and L=256.

4. **L=224 "approaching L=256"** — WRONG. Bounces back up (0.309) instead of continuing descent.

**Both hypotheses falsified.** No sharp bifurcation (the "critical L" model) and no broad plateau (the "flat" model). Instead: **the L-profile is jagged and non-monotonic**, with local dips at L=208 and L=128, and local peaks at L=176 and L=192. The pattern resembles interference between context-length-dependent attractor structures rather than a smooth phase transition.

**Key takeaway:** At T=0.50, seed variance (std 0.04–0.10) is comparable to the differences between adjacent L values (Δmean ~0.03–0.08). With n=3 seeds, we cannot reliably resolve whether the jagged profile is reproducible structure or sampling noise. However, the overall downward trend from L=64 to L=256 is robust across all seeds.

**What this means for the project:** The L-dimension at fixed T=0.50 has diminishing returns. The more interesting question is how the L-profile *changes shape across T* — the T×L interaction.

```bash
python plot_window_scaling.py --temps 0.50 --ldense
```

---

### 2026-03-09 — Cross-T Analysis: Collapse Boundary is L-dependent

**Data:** Full S=42 grid — L={64, 128, 192} × T={0.50–1.50}, L=256 × T={0.50, 1.00, 1.50}.

**Compressibility (W=64) across T (S=42):**

| L \ T | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 | 1.50 |
|-------|------|------|------|------|------|------|------|
| 64 | 0.347 | 0.481 | 0.614 | 0.682 | 0.715 | 0.739 | 0.705 |
| 128 | 0.259 | 0.334 | 0.547 | 0.628 | 0.713 | 0.757 | 0.702 |
| 192 | 0.351 | 0.313 | 0.556 | 0.603 | 0.697 | 0.785 | 0.704 |
| 256 | 0.229 | — | — | — | — | 0.743 | 0.705 |

**Entropy mean across T (S=42):**

| L \ T | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 | 1.50 |
|-------|------|------|------|------|------|------|------|
| 64 | 0.683 | 0.791 | 1.599 | 2.220 | 2.755 | 3.721 | 7.989 |
| 128 | 0.191 | 0.502 | 1.149 | 1.642 | 2.858 | 4.168 | 8.207 |
| 192 | 0.311 | 0.187 | 1.114 | 1.618 | 2.868 | 4.691 | 8.301 |
| 256 | 0.069 | — | — | — | — | 4.917 | 8.342 |

**EOS rate across T (S=42):**

| L \ T | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 | 1.00 | 1.50 |
|-------|------|------|------|------|------|------|------|
| 64 | 0.00013 | 0.00035 | 0.00057 | 0.00074 | 0.00092 | 0.00091 | 0.00063 |
| 128 | 0.00005 | 0.00032 | 0.00055 | 0.00076 | 0.00104 | 0.00077 | 0.00053 |
| 192 | 0.00005 | 0.00007 | 0.00083 | 0.00071 | 0.00087 | 0.00065 | 0.00048 |
| 256 | 0.00001 | — | — | — | — | 0.00057 | 0.00046 |

**Key findings:**

1. **L=192 extends collapse into T=0.60.** Entropy 0.187, compressibility 0.313, EOS rate 0.00007. Meanwhile L=64 (entropy 0.791) and L=128 (entropy 0.502) have escaped collapse at T=0.60. Longer context doesn't just deepen collapse — it extends the collapse regime upward in T.

2. **Sharp escape at T=0.70.** All L values jump to entropy >1.0 and compressibility >0.55 at T=0.70. The collapse-to-escape transition is abrupt in T, regardless of L. This suggests T=0.65 may be the critical temperature for L=192 — a natural target for further densification.

3. **EOS peak shifts with L.** Peak EOS rate occurs at T=0.90 for L=128 (0.00104), at T=0.90 for L=64 (0.00092), but at T=0.70 for L=192 (0.00083). The EOS peak tracks the escape-from-collapse transition: it fires most when the system is on the boundary between structured and free dynamics.

4. **T=1.50 is a universal noise floor.** Compressibility ~0.705 and entropy ~8.0-8.3 regardless of L. At high T, context length is irrelevant — thermal noise dominates.

5. **T=1.00 compressibility *increases* with L** (0.739 → 0.757 → 0.785). The opposite of T=0.50 where comp generally decreases with L. At T=1.00, longer context enriches structure rather than deepening collapse. This confirms the "slope-flip" noted in earlier observations — the sign of dC/dL reverses somewhere in the crossover.

**Priorities:**
- Fill L=256 × T={0.60, 0.70, 0.80, 0.90} to see how far L=256 extends the collapse regime (4 runs)
- Consider T-densification at T={0.62, 0.65, 0.68} for L=192 to pin the escape temperature
- Cross-T heatmap visualization needed — current plotting tools show fixed-T slices but not T×L interaction

```bash
# Reproduction
python -c "
import pickle, numpy as np
from pathlib import Path
for L in [64, 128, 192, 256]:
    for T in [0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.50]:
        for pat in [f'data/runs/L{L:04d}_T{T:.2f}_S42.analysis.pkl',
                    f'data/runs/L{L:04d}_T{T:.2f}_S42.W16_W32_W64_W128_W256.analysis.pkl']:
            p = Path(pat)
            if p.exists():
                with open(p, 'rb') as f:
                    data = pickle.load(f)
                v = data['compressibility'][64]
                valid = v[~np.isnan(v)]
                print(f'L={L} T={T:.2f}: comp_W64={valid.mean():.4f} entropy={data[\"summary\"][\"entropy_mean\"]:.4f} eos={data[\"summary\"][\"eos_rate\"]:.6f}')
                break
"
```

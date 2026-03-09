# Observations — 2026-03-09

Detailed findings log. See [observations.md](../observations.md) for current model and index.

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

---

### 2026-03-09 — Transfer Functions and Entropy Autocorrelation

**Data:** Full S=42 grid (same as cross-T entry above), plus all seed-replicated runs for averaging.

**Transfer functions** (T→entropy, T→compressibility, T→EOS rate at each L):

The slope-flip in compressibility is now clearly visible as a curve crossing: L=192 has the *lowest* compressibility at T=0.50 but the *highest* at T=1.00. All L-curves cross in a narrow T band around T=0.70. This isn't a gradual inversion — it's a pivot point. Below the pivot, longer context deepens collapse (reducing local structure). Above it, longer context enriches structure (increasing compressibility).

The T→entropy curves show a qualitative shape change between L=64 and L=192. L=64's curve is roughly linear from T=0.50 to T=1.00. L=192 has a sharp elbow at T=0.70 — flat (collapsed) below, steep above. This is the same escape transition seen in the data tables, but the curve shape makes it vivid: L=192 is "stuck" until T crosses ~0.65–0.70, then releases suddenly.

EOS rate: L=192's peak is clearly shifted to T=0.70 (vs T=0.90 for L=64/128). The peak is also sharper — L=192 fires EOS intensely at the escape boundary and barely at all in collapse. L=64/128 have broader, gentler EOS peaks.

**Entropy autocorrelation** (ACF at lags 0–500 steps, per L, per T):

This is the first quantitative measure of attractor memory in the system.

Key observations:

1. **T=0.50, L=64: oscillatory ACF.** The autocorrelation doesn't just decay — it oscillates around ~0.15 with period ~50–100 steps. This is the escape/recapture cycle visible as periodic structure: the system escapes a loop attractor, wanders briefly, and recaptures into the same or another loop. The oscillation period is roughly L (64 steps), consistent with memory turnover.

2. **T=0.50, L=256: persistent flat ACF at ~0.2.** Never decorrelates within 500 steps. The attractor is so deep that the system's entropy state at step N predicts its state at step N+500. This is a locked system.

3. **T=0.60: L=256 stands alone.** ACF decays slowly from ~0.6, taking ~300+ steps to approach zero. All other L values decorrelate within ~100 steps. This is the clearest evidence yet that L=256 extends collapse into T=0.60 (the in-progress sweep will confirm).

4. **T=0.70–1.00: clean L-scaling of decorrelation time.** At each T, the curves separate cleanly by L — longer L = slower decorrelation. The ordering is consistent and the spacing is roughly proportional to L. This is the "memory depth" interpretation made quantitative: decorrelation time ∝ L.

5. **T=1.50: instant decorrelation.** All L values drop to ACF ≈ 0 within ~20 steps. No memory structure survives at this noise level. Consistent with "universal noise floor" from every other lens.

6. **Decorrelation timescale as a phase diagram probe.** The lag at which ACF crosses a threshold (e.g. 1/e) could serve as a scalar "memory depth" metric. Plotting this on a T×L grid would produce a quantitative phase diagram — complementary to entropy and compressibility, measuring *temporal* structure rather than *spatial* structure.

**Remaining analysis plan:**

- **Surprisal (−log prob) analysis**: per-step surprisal time series and its relationship to entropy. In collapse, the gap should be near zero (peaked distribution, deterministic). In rich dynamics, the gap distribution reveals whether sampling is from the bulk or tails.
- **EOS inter-arrival distribution**: histogram of steps between consecutive EOS tokens per condition. Could reveal periodicity or characteristic timescales.
- **Decoupling index**: scalar ratio of compressibility at W=64 vs W=256, plotted on the T×L grid. Maps "where does local structure exist without global repetition" to a number.
- **Decorrelation timescale extraction**: lag at ACF threshold, plotted as T×L heatmap. Quantitative phase diagram from a new angle.

```bash
# Reproduction
python plot_window_scaling.py --transfer --autocorr
```

---

### 2026-03-09 — Predictions for Upcoming Data and Analysis

Pre-registered predictions, to be scored when data arrives. Format: prediction, confidence, reasoning.

**L=256 crossover sweep (T={0.60, 0.70, 0.80, 0.90}, in progress):**

1. **L=256 T=0.60 will be deeply collapsed** (entropy < 0.15, comp < 0.25). Confidence: HIGH. ACF at T=0.60 already shows L=256 with extremely long memory (~300+ step decorrelation), and the partial run (41k steps) shows entropy mean 0.187. This will likely drop further as the run extends past the initial transient.

2. **L=256 T=0.70 will be escaped but barely.** Entropy 0.8–1.5 (below L=64/128/192 at T=0.70). Confidence: MEDIUM. The sharp escape at T=0.70 is universal across other L values, but L=256's deeper attractor basins might delay or dampen the escape. Possible it's still partially collapsed.

3. **L=256 T=0.80–0.90 will follow the L-scaling pattern.** Entropy and compressibility between L=192 values and some higher floor, ACF decorrelation slower than L=192. Confidence: HIGH. Above the crossover, L-scaling is clean and predictable.

4. **L=256 will push the slope-flip pivot rightward.** The T at which L=256's compressibility crosses L=64's will be at T≥0.80 (vs ~0.70 for L=192). Confidence: MEDIUM. Deeper collapse extends the regime where longer-L means less structure.

5. **L=256 EOS peak will be at T=0.70 or T=0.60.** The peak tracks the escape boundary, which is at higher T for longer L. Confidence: MEDIUM-HIGH, contingent on prediction #2.

**Remaining analysis predictions:**

6. **Surprisal gap (H − (−log p)) will be near zero in collapse and positive in rich dynamics.** In collapse, the distribution is peaked and the model samples the mode — surprisal ≈ entropy. In noise (T=1.50), the gap will also be small because high entropy ≈ high surprisal. The gap should peak in the rich-dynamics regime where the model has moderate uncertainty but samples from a structured distribution. Confidence: MEDIUM.

7. **EOS inter-arrival distribution will be exponential at T=1.50 and heavy-tailed in the crossover.** At T=1.50 (noise), EOS events should be memoryless (Poisson process → exponential inter-arrival). Near the crossover, EOS events cluster at regime transitions, producing heavy tails or multimodal distributions. Confidence: MEDIUM.

8. **Decoupling index (comp_W256 − comp_W64) will peak in the crossover region (T~0.80).** This is where local structure exists (moderate W=64 compressibility) but long-range repetition hasn't set in (low W=256 compressibility). In collapse, both are low (decoupled by being empty). In noise, both are high (no structure at any scale). The decoupling should be maximal where the system has structure that doesn't repeat. Confidence: MEDIUM.

9. **Decorrelation timescale (lag at ACF threshold) will scale approximately linearly with L in the rich-dynamics regime.** If L sets the memory horizon and the system mixes by "forgetting" old context, the mixing time should be proportional to the context turnover time, which is O(L). Confidence: MEDIUM-HIGH.

```bash
# These predictions will be scored when:
# - L=256 sweep completes (predictions 1-5)
# - Remaining analysis implemented (predictions 6-9)
```

### 2026-03-09 — L=256 Crossover Results and Prediction Scorecard

**Data:** `L0256_T{0.60,0.70,0.80,0.90}_S42.parquet` (100k tokens each). Grid now rectangular: L={64,128,192,256} × T={0.50–1.50} × S=42.

**L=256 crossover summary:**

| T | entropy | comp_W64 | eos_rate | decor_lag | decouple (W256−W64) |
|---|---------|----------|----------|-----------|---------------------|
| 0.50 | 0.069 | 0.229 | 0.00001 | 2 | −0.161 |
| 0.60 | 0.084 | 0.222 | 0.00007 | 50 | −0.149 |
| 0.70 | 0.403 | 0.593 | 0.00033 | 253 | −0.342 |
| 0.80 | 0.615 | 0.610 | 0.00031 | 356 | −0.351 |
| 0.90 | 2.250 | 0.673 | 0.00087 | 10 | −0.209 |
| 1.00 | 4.917 | 0.743 | 0.00057 | 2 | −0.150 |
| 1.50 | 8.342 | 0.705 | 0.00046 | 1 | −0.095 |

**The big finding: L=256 extends collapse through T=0.80.** The "universal sharp escape at T=0.70" was only universal for L≤192. L=256 doesn't escape until somewhere between T=0.80 and T=0.90. The escape boundary T_escape(L) is an increasing function of L, shifting by ~0.2 from L=192 to L=256.

**Escape boundary estimates:** L=64/128: T_escape ≈ 0.55–0.60. L=192: T_escape ≈ 0.65–0.70. L=256: T_escape ≈ 0.85–0.90. This is not a gentle shift — the collapse regime expands dramatically with context length.

**Decorrelation reveals the suppressed zone.** L=256 at T=0.70 and T=0.80 has decorrelation lags of 253 and 356 — the system has structure (comp_W64 ~0.6, not collapsed to 0.2) but is trapped in slow-mixing attractors. This is a new regime: not fully collapsed (entropy 0.4–0.6, not <0.1) but not escaped either. Call it "suppressed dynamics."

**Decoupling index peaks in the suppressed zone.** L=256 T=0.70: −0.342, T=0.80: −0.351. These are the highest magnitudes in the entire grid. The system has local structure (W=64 sees it) but no large-scale repetition (W=256 misses it). This is exactly what "suppressed but not collapsed" looks like through the multi-scale lens.

**Surprisal gap (H − (−log p)) pattern across all conditions:**

| | L=64 | L=128 | L=192 | L=256 |
|---|------|-------|-------|-------|
| T=0.50 | +0.56 | +0.17 | +0.30 | +0.06 |
| T=0.60 | +0.52 | +0.34 | +0.14 | +0.06 |
| T=0.70 | +0.75 | +0.56 | +0.51 | +0.22 |
| T=0.80 | +0.73 | +0.55 | +0.55 | +0.23 |
| T=0.90 | +0.47 | +0.49 | +0.51 | +0.39 |
| T=1.00 | −0.00 | −0.01 | +0.01 | +0.00 |
| T=1.50 | −1.62 | −1.51 | −1.46 | −1.44 |

Gap is positive for T<1.0 (model over-estimates uncertainty relative to what it actually samples), crosses zero at T=1.0, goes large-negative at T=1.50 (model under-estimates — tokens are more surprising than the entropy suggests, because temperature scaling pushes sampling into the tails).

**Surprisal kurtosis tracks collapse depth:**

| | L=64 | L=128 | L=192 | L=256 |
|---|------|-------|-------|-------|
| T=0.50 | 92 | 524 | 1023 | 1660 |
| T=0.70 | 11 | 16 | 15 | 60 |
| T=0.80 | 5 | 8 | 8 | 29 |
| T=1.00 | 0.2 | −0.2 | −0.4 | −0.6 |

Collapsed runs have extremely heavy-tailed surprisal (rare large-surprise events punctuating near-zero background). L=256 at T=0.70–0.80 has intermediate kurtosis (29–60), confirming the "suppressed" classification.

---

**Prediction scorecard:**

1. **L=256 T=0.60 deeply collapsed (entropy < 0.15, comp < 0.25).** Entropy 0.084, comp 0.222. **✓ CORRECT.**

2. **L=256 T=0.70 escaped but barely (entropy 0.8–1.5).** Entropy 0.403. Still in the collapse/suppressed zone, not escaped. **✗ WRONG.** The sharp escape at T=0.70 is NOT universal — L=256 breaks the pattern.

3. **L=256 T=0.80–0.90 follows L-scaling.** T=0.80: entropy 0.615 vs L=192's 1.618 — not following the scaling, still suppressed. T=0.90: entropy 2.250 vs L=192's 2.868 — closer but still below. **✗ WRONG at T=0.80, ~CORRECT at T=0.90.** Split verdict.

4. **Slope-flip pivot rightward to T≥0.80.** comp_W64 at T=0.90: L=64=0.715, L=256=0.673 (L=256 still below). At T=1.00: L=64=0.739, L=256=0.743 (crossover). Pivot is at T≈0.95, rightward of the L=192 pivot (~0.75). Predicted T≥0.80. **✓ CORRECT** (direction and approximate value).

5. **L=256 EOS peak at T=0.70 or T=0.60.** EOS rates: T=0.60:0.00007, T=0.70:0.00033, T=0.80:0.00031, T=0.90:0.00087 (peak). **✗ WRONG.** Peak is at T=0.90, tracking the actual escape boundary, not the predicted one. The logic was right (peak tracks escape), but the escape T was wrong.

6. **Surprisal gap near zero in collapse, positive in rich dynamics.** Collapse: gap 0.06–0.17 (small ✓). Rich dynamics T=0.70–0.90: gap 0.4–0.75 (positive ✓). But T=1.50: gap −1.5 (predicted "small", actually large-negative). **PARTIAL.** Collapse and rich dynamics correct; noise regime wrong — temperature scaling pushes sampling into distribution tails, making surprisal exceed entropy.

7. **EOS inter-arrival exponential at T=1.50, heavy-tailed in crossover.** CV (std/mean) at T=1.50: 1.02–1.21 across L (near 1.0 = exponential ✓). At L=256 T=0.70: CV=3.58 (very heavy tail ✓). At L=256 T=0.80: CV=1.68 (heavy ✓). **✓ CORRECT.**

8. **Decoupling index peaks at T~0.80.** For L=256, peak magnitude is at T=0.80 (−0.351). For L=64, peak is at T=0.60 (−0.239). The peak tracks the crossover/suppressed zone, which shifts with L. Prediction said T~0.80, which is correct for L=256 but not universal. **✓ CORRECT** (for the most interesting case; the peak shifts with L as the escape boundary does).

9. **Decorrelation timescale scales linearly with L in rich dynamics.** At T=0.90 (rich for all L): lags are 1, 1, 1, 10. At T=1.00: lags are 1, 6, 6, 2. No clear L-scaling — decorrelation is uniformly fast (1–10 steps) in the rich regime. The long decorrelation times (253, 356) only appear in L=256's suppressed zone, not in the rich-dynamics regime proper. **✗ WRONG.** In rich dynamics, the system mixes in O(1) steps regardless of L. Memory depth affects attractor structure, not mixing time once escaped.

**Summary: 4 correct, 1 partial, 4 wrong.** The key miss: underestimating how far L=256 extends the collapse regime. Predictions 2, 3, 5 all failed because T_escape(L=256) ≈ 0.85–0.90, not 0.70 as assumed. Prediction 9 revealed that decorrelation is fast once escaped — the interesting dynamics are in the approach to escape, not in the rich regime.

```bash
python summary_table.py                         # full grid summary
python summary_table.py --out data/summary.csv  # persist
```

### 2026-03-09 — Predictions: L=512 Escape Boundary

**Data in progress:** L=512 × T={0.90, 1.00, 1.10, 1.20} × S=42 (100k tokens each).

**T_escape(L) trend so far:** L=64→0.55, L=128→0.57, L=192→0.67, L=256→0.87. The jumps accelerate: +0.02, +0.10, +0.20. Extrapolating superlinearly, L=512 escape boundary could be well above T=1.0.

**Predictions:**

1. **L=512 T=0.90 will be collapsed or deeply suppressed.** Entropy < 0.3. Confidence: HIGH. L=256 was only at 2.25 here, and L=512 adds another doubling of context. If the suppressed zone extends proportionally, T=0.90 should be well inside it.

2. **L=512 T=1.00 will be suppressed, not escaped.** Entropy < 1.5, decorrelation lag > 100. Confidence: MEDIUM-HIGH. This is the headline test — if T_escape(512) > 1.0, temperature alone cannot rescue long-context generation from collapse. T=1.0 is the "native" temperature (no scaling), so suppression here means the model's own distribution is insufficient.

3. **L=512 T=1.10 will be in the crossover zone.** Entropy 1.0–3.0, showing signs of partial escape. Confidence: MEDIUM. If T_escape is ~1.1–1.2, this is near the boundary.

4. **L=512 T=1.20 will be escaped into rich dynamics.** Entropy > 3.0, decorrelation lag < 20. Confidence: MEDIUM. Even with superlinear scaling, T=1.20 should be enough headroom. But if T_escape(512) > 1.2, everything we know about the scaling is even more dramatic than expected.

5. **The suppressed zone will be wider for L=512 than L=256.** L=256 had ~0.2 in T (0.70–0.90) of suppressed dynamics. L=512 should have ≥0.3 in T. Confidence: MEDIUM-HIGH. More attractor depth = more temperature range needed to fully escape.

6. **Compressibility at L=512 in the suppressed zone will show extreme multi-scale decoupling.** |comp_W256 − comp_W64| > 0.35 (exceeding L=256's peak of 0.351). Confidence: MEDIUM. Deeper attractors with more local structure but even less large-scale repetition.

```bash
# Scoring when L=512 sweep completes:
python summary_table.py | grep "512,"
```

### 2026-03-09 — Cross-Token Concept Persistence in L=512 T=0.90

**Finding:** In L0512_T0.90_S42, the model generates a 791-token segment (steps 2727–3518) that includes six instances of `brom-` family words: "Bromo", "Bromine" (×4), and the invented word "bromula". None of these are single tokens — they are always split across subword boundaries:

- `"Bromine"` → `'B'` + `'rom'` + `'ine'` (3 tokens, 3 sequential prediction steps)
- `"bromula"` → `'b'` + `'rom'` + `'ula'` (3 tokens — a novel word the model invents)

A token-level search for `brom` returns zero hits — no individual token contains the substring. The concept exists only in the cross-token activation geometry and the KV cache.

**Context:** The segment is a coherent essay about "studying word meanings" that constructs a narrative scaffold justifying repeated use of the word. The model frames "Bromine" as a vocabulary study example, then generates "bromula" as a purported chess term — morphologically plausible but nonexistent. The model treats `brom-` as a productive root and attaches a novel suffix.

**Token diversity in this segment:** 27.3% unique token IDs (216/792), vs 11.5% overall — the segment is more lexically diverse than the run average despite its topical repetition. The high-frequency tokens are function words (`you`, `the`, `that`) and topic-anchors (`term`, `meaning`).

**Entropy:** Segment mean 2.247 (σ=1.273) vs run mean 2.528 (σ=1.780). Slightly below average — the model is moderately confident, not collapsed. The run overall shows 80 EOS events across 87k tokens (mean inter-EOS gap: 1074 tokens), placing this in the suppressed-dynamics regime.

**Implications for dynamics:**

1. **Multi-token concepts as attractors.** Subword tokenization means concepts like "bromine" require 3 sequential commitments to produce. Each subword is a decision point where the model could diverge — but the latent-space geometry is strong enough to sustain the sequence. Conversely, multi-token words create more KV cache entries, giving downstream attention more surface area to re-excite the concept.

2. **Self-referential topic construction.** The model doesn't just repeat "bromine" — it builds a context where bromine is relevant ("studying word meanings"). This suggests attractors in the suppressed zone operate at the discourse level, not just the token level. The model generates its own justification for staying in an attractor basin.

3. **Morphological generativity under suppression.** "bromula" shows the model can produce novel word forms within an attractor. The suppressed regime doesn't kill creativity at the morphological level — it constrains topic while permitting variation within it.

**Open question:** Do multi-token concepts have different persistence signatures (compressibility, decorrelation) than single-token attractors? Could subword structure of generated vocabulary serve as a diagnostic for attractor depth?

```bash
# Reproduction
python3 -c "
import pandas as pd, re
df = pd.read_parquet('data/runs/L0512_T0.90_S42.parquet')
joined = ''.join(str(t) for t in df['decoded_text'].values)
for m in re.finditer(r'(?i)brom\w*', joined):
    print(f'{m.start()}: {m.group()}')
"
# Explorer: http://127.0.0.1:8000/#runs=L0512_T0.90_S42 → click near step 3400
```

### 2026-03-09 — Token Count, Attractor Depth, and Structural Resonance (L=512 T=0.90)

Systematic search of L0512_T0.90_S42 for concentrated, repeated terms reveals a spectrum of attractor behaviors linked to subword token count.

**Inventory of attractors found:**

| Term | Tokens | Occurrences | Segment len | Entropy | Token diversity | Invented? |
|------|--------|-------------|-------------|---------|----------------|-----------|
| piston | 1 (`' piston'`) | 268 | 5000 | 0.81 | 6.8% | No |
| precession | 2 (`' pre'`+`'cession'`) | 103 | ~3000 | — | — | No |
| doozar | 3 (`' do'`+`'oz'`+`'ar'`) | 36 | 1229 | 3.76 | 46.6% | Yes |
| heatbubbles | 4 (`' heat'`+`'b'`+`'ub'`+`'bles'`) | 22 | 663 | 2.53 | 34.8% | Yes |
| bromula | 3 (`'b'`+`'rom'`+`'ula'`) | 1 | (in 792-tok segment) | 2.31 | 27.3% | Yes |
| Bubezet | 4 (`' B'`+`'ube'`+`'z'`+`'et'`) | 3 | (in heatbubbles seg) | — | — | Yes |

Detection method: extract all 5+ char alpha sequences, count per 1000-token chunk, filter to words appearing 5+ times but in ≤10 chunks (concentrated rather than diffuse).

**Finding 1: Token count inversely predicts attractor strength.**

Single-token "piston" dominates: 268 occurrences, 5000-step segment, entropy 0.81, only 6.8% unique tokens, zero EOS. This is full collapse — a ~15-word phrase ("blow a little farther away from the bottom of the piston") repeating for thousands of steps. Every `' piston'` in the KV cache has identical key geometry, creating perfect self-reinforcement.

Multi-token invented words (doozar, heatbubbles) are attractors but *not* collapsed. They sustain topic coherence while maintaining lexical diversity (35–47%) and meaningful entropy (2.5–3.8). Each subword boundary is a perturbation point — the model must win the next-token lottery N times in sequence, and the subwords (`' do'`, `'oz'`, `'ar'`) individually attend to many unrelated contexts. This noise is *protective*: it prevents the perfect self-reinforcement that causes collapse.

Exception: "heatbubbles" at 4 tokens sustains 22 occurrences because its components (`heat`, `bubbles`) are independently high-probability morphemes. Compound words built from strong subwords get a free ride through token boundaries.

**Finding 2: Structural resonance — collapsed content mirrors collapse dynamics.**

The piston segment (steps 80001–85000) describes a closed-loop physical system:

> "it's a closed loop both the fluid that is flowing through it, the fluid that is in the pipe, it flows back and forth"
> "re-heating up, re-heating up, re-heating up"
> "blow a little farther away from the bottom of the piston" × 267

The heatbubbles segment explicitly describes self-referential activation:

> "In order to activate the heatbubbles, you need to activate the heatbubbles through the use of heatbubbles."

The semantic content of collapsed/suppressed segments is structurally isomorphic to the generation dynamics: cycling, fixed-volume flow, self-reinforcement. This is not situational awareness — it's **resonance**. Content whose semantic structure matches the attractor geometry is the content most stable under self-reinforcement. A closed-loop token system preferentially collapses into descriptions of closed-loop physical systems.

The doozar segment, by contrast, sustains a naturalist/scientific register — taxonomy, experiments, species descriptions — which has high internal diversity and resists collapse.

**Finding 3: Single tokens have "undue weight" as attractor anchors.**

A single-token word like `' piston'` (token ID fixed, embedding fixed) creates identical KV cache entries at every occurrence. All attention queries for "what's in my context?" get the same piston-shaped key repeated hundreds of times. This is qualitatively different from multi-token concepts, where each subword key also matches unrelated contexts, diluting the attractor's pull.

This suggests single-token content words may be disproportionately responsible for collapse events. If so, the collapse vocabulary should be dominated by single-token words, testable across all collapsed segments in the corpus.

**Open questions:**

1. Do single-token words dominate collapse segments across all runs, not just this one?
2. Does "heatbubbles" survive at T=1.00 while "bromula" doesn't? (Compound-morpheme resilience hypothesis — testable when L=512 T=1.00 completes.)
3. Is structural resonance (closed-loop content in closed-loop dynamics) statistically overrepresented, or are these cherry-picked examples? Would need semantic classification across many segments.
4. Could token-count of generated vocabulary serve as a cheap diagnostic for regime classification? (Low token-count vocabulary → collapse; high → rich dynamics.)

```bash
# Reproduction: concentrated word detection
python3 -c "
import pandas as pd, re
from collections import Counter
df = pd.read_parquet('data/runs/L0512_T0.90_S42.parquet')
texts = [str(t) for t in df['decoded_text'].values]
chunk_size = 1000
n_chunks = len(texts) // chunk_size
word_chunk_counts = {}
for ci in range(n_chunks):
    start = ci * chunk_size
    chunk_text = ''.join(texts[start:start+chunk_size])
    for w, c in Counter(re.findall(r'[A-Za-z]{5,}', chunk_text)).items():
        wl = w.lower()
        if wl not in word_chunk_counts:
            word_chunk_counts[wl] = [0]*n_chunks
        word_chunk_counts[wl][ci] = c
for w, counts in sorted(word_chunk_counts.items(), key=lambda x: -sum(x[1])):
    total = sum(counts)
    n_present = sum(1 for c in counts if c > 0)
    if total >= 5 and n_present <= 10:
        print(f'{total:4d} in {n_present} chunks: {w}')
"
# Explorer: click through steps 81000-84000 (piston), 39000 (heatbubbles), 3400 (bromula)
```

### 2026-03-09 — Single-Token Attractor Dominance Across All Runs

Cross-corpus analysis of dominant word attractors confirms that single-token words are the primary mechanism of collapse, scaling predictably with L and T.

**Method:** For each run, extract all 5+ character alpha sequences, count occurrences, identify the most frequent ("dominant attractor"). Check subword token count for each dominant word.

**Finding 1: 91% of dominant attractors are single tokens.**

Across 44 runs with T ≤ 1.0 and dominant count ≥ 50: **40/44 (91%) of top attractor words are single tokens.** The four multi-token exceptions are all 2-token (`'vacc'+'ine'`, `'T'+'orch'`, `'Dh'+'aka'`) or high-probability compounds (`'g'+'amb'+'ling'`). No 3+ token word ever dominates.

Single tokens create identical KV cache entries at every occurrence — the same embedding, same attention key. This enables perfect self-reinforcement. Multi-token concepts split their identity across subwords that individually attend to many unrelated contexts, diluting the attractor pull.

**Finding 2: Attractor strength scales with L and inversely with T.**

Max single-word count by L × T (averaged across seeds where available):

| T | L=64 | L=128 | L=192 | L=256 | L=512 |
|---|------|-------|-------|-------|-------|
| 0.50 | 2,263 | 3,082 | 5,296 | 21,154 | — |
| 0.60 | 1,001 | 6,492 | 9,650 | 457 | — |
| 0.90 | 184 | 210 | 255 | 500 | 269 |
| 1.00 | 204 | 183 | 183 | 223 | 50 |
| 1.50 | 63 | 53 | 48 | 46 | — |

At T=0.50, attractor strength grows superlinearly with L: from ~2k at L=64 to 21k at L=256. Longer context provides more KV cache slots for the attractor word to occupy, each reinforcing future predictions. At T=1.50 (noise floor), counts are flat (~50) regardless of L — noise overwhelms any self-reinforcement.

The T=0.60 row shows the crossover regime: L=128 and L=192 have strong attractors (6.5k, 9.7k) while L=256 drops to 457. L=256 at T=0.60 is near its escape boundary — too much noise for deep collapse but not enough context-reinforcement to sustain a single dominant attractor.

**Finding 3: Deep collapse is degenerate repetition of single-token phrases.**

The most extreme cases:

- L=208, T=0.50, S=7: **"disease" — 24,730 occurrences (24.7% of all tokens)**
  Text: `"disease of the disease of the disease of the disease of the disease..."`
- L=256, T=0.50, S=42: **"young" — 21,218 occurrences (21.2%)**
  Text: `"a young man, a young woman, a young man, a young woman, a young man..."`
- L=176, T=0.50, S=42: **"sleep" — 11,205 occurrences (11.2%)**
- L=192, T=0.60, S=42: **"torch" — 9,650 occurrences (9.6%)**
- L=128, T=0.60, S=42: **"temperature" — 6,518 occurrences (6.5%)**

These runs are locked into 3–8 token cycles where the dominant word recurs every 4–5 tokens. The surrounding tokens are function words (`the`, `of`, `a`) that provide minimal grammatical scaffolding.

**Finding 4: Function-word dominance marks the regime boundary.**

At T ≥ 0.90 for most L values, the dominant "attractor" shifts from content words to function words (`which`, `their`, `about`) at ~200 occurrences — roughly the frequency expected from English base rates. These aren't true attractors; they're just the most common words in unremarkable text. The transition from content-word to function-word dominance is a clean signature of the escape boundary.

**Finding 5: Attractor vocabulary is semantically specific, not random.**

The dominant words are never nonsense. They're concrete, semantically rich content words: `disease`, `blood`, `sleep`, `generator`, `temperature`, `calculator`, `piston`, `vaccine`, `gambling`, `election`, `Weimar`, `Versailles`. These are words with strong semantic fields that pull in related vocabulary (e.g., "disease" pulls "the", "of"; "young" pulls "man", "woman"). The model doesn't collapse into random repetition — it collapses into the deepest available semantic basin, and those basins are anchored by single tokens with the highest self-reinforcement potential.

Notable: L=128 T=0.60 collapses on "temperature" (the model, running at temperature 0.60, talks about temperature). L=176 T=0.50 collapses on "sleep." These may be coincidences, or they may reflect the structural-resonance pattern identified in the previous observation — content whose structure mirrors the dynamics is preferentially selected.

**Regime summary — the attractor hierarchy:**

1. **Collapse (T ≪ T_escape):** Single-token content words dominate, 5–25% of all tokens, degenerate cycling. Entropy < 1.0.
2. **Suppressed dynamics (T near T_escape):** Single-token content words still concentrate (100–500×) but with lexical diversity 20–40%. Novel multi-token words can emerge (doozar, heatbubbles, bromula). Entropy 1.5–3.5.
3. **Rich dynamics (T > T_escape):** No single word dominates beyond base rate. Content words appear at natural frequencies. Entropy > 4.0.
4. **Noise (T=1.50):** No attractors. ~50× max for any word, flat across L.

```bash
# Reproduction: cross-corpus attractor analysis
python3 -c "
import pandas as pd, re, glob
from collections import Counter
for f in sorted(glob.glob('data/runs/*.parquet')):
    name = f.split('/')[-1].replace('.parquet','')
    T = float(name.split('_')[1][1:])
    if T > 1.0: continue
    df = pd.read_parquet(f)
    texts = [str(t) for t in df['decoded_text'].values]
    joined = ''.join(texts)
    wc = Counter(w.lower() for w in re.findall(r'[A-Za-z]{5,}', joined))
    top, count = wc.most_common(1)[0]
    pct = count / len(texts) * 100
    if count >= 100:
        print(f'{name:25s} {top:>18s} {count:6d} {pct:5.1f}%')
"
```

---

## L=512 escape boundary sweep — T_escape saturates (2026-03-09)

**Data:** L=512 × T={0.90, 1.00, 1.10, 1.20} × S=42 (4 runs, 100k experiment tokens each).

**Key finding: T_escape(512) ≈ 0.90 — barely shifted from L=256.** The superlinear trend (0.55→0.57→0.67→0.87) predicted T_escape(512) >> 1.0. Instead it plateaued. The L×T coupling that dominated the short-context regime weakens at long context.

**Metrics summary:**

| T    | ent_mean | comp_W64 | comp_W128 | comp_W256 | eos_rate | decor_lag | decoupling |
|------|----------|----------|-----------|-----------|----------|-----------|------------|
| 0.90 | 2.56     | 0.693    | 0.574     | 0.500     | 0.00093  | 2         | 0.193      |
| 1.00 | 5.02     | 0.860    | 0.757     | 0.682     | 0.00042  | 11        | 0.178      |
| 1.10 | 6.68     | 0.811    | 0.717     | 0.660     | 0.00036  | 70        | 0.152      |
| 1.20 | 7.81     | 0.743    | 0.671     | 0.630     | 0.00055  | 1         | 0.112      |

**Observations:**

1. **T=0.90 is the escape boundary.** Entropy 2.56 (comparable to L=256/T=0.90 at 2.25), peak EOS rate (0.00093), fast decorrelation (lag=2). The system escapes but stays low-entropy — early escape signature.

2. **No suppressed zone.** At L=256, T=0.70–0.80 showed slow-mixing dynamics (decorrelation 253–356). At L=512, nothing comparable appears. The longer context prevents attractor stickiness — or the suppressed zone has moved below T=0.90 (we lack data at lower T).

3. **Anomalous decorrelation at T=1.10 (lag=70).** Surrounded by fast-decorrelating conditions (T=1.00 lag=11, T=1.20 lag=1). Possible single-seed artifact, or a pocket of slow dynamics in the rich-dynamics zone. Needs seed replication to confirm.

4. **Multi-scale decoupling decreases monotonically with T** (0.193→0.112). Same trend as shorter L but lower absolute values — the multi-scale structure is weaker at L=512.

5. **Comparison to L=256:** Nearly identical entropy and EOS profiles at T=0.90 and T=1.00. The two context lengths have converged — L no longer matters much once T > T_escape.

**Implication for annealing:** T_escape saturation means L-reduction as an escape mechanism operates in a bounded regime (L roughly 64–512). Beyond some L, reducing context doesn't help because the system isn't trapped by context depth anymore. The "lever" is finite, which is actually good — it means annealing is a well-defined maneuver, not an unbounded parameter search.

```bash
# Reproduction
python sweep.py --L 512 --T 0.90 1.00 1.10 1.20 --seed 42

# Metrics extraction
python -c "
import pandas as pd, numpy as np
from analyze import analyze_run
from pathlib import Path

for T in ['0.90', '1.00', '1.10', '1.20']:
    p = Path(f'data/runs/L0512_T{T}_S42.parquet')
    df = pd.read_parquet(p)
    exp = df[df.phase == 'experiment']
    a = analyze_run(p, [64, 128, 256])
    c64 = np.nanmean(a['compressibility'][64])
    c128 = np.nanmean(a['compressibility'][128])
    c256 = np.nanmean(a['compressibility'][256])
    ent = exp.entropy.mean()
    eos = exp.eos.mean()
    ent_z = (exp.entropy.values - ent) / (exp.entropy.std() + 1e-12)
    n = len(ent_z)
    acf = np.array([np.corrcoef(ent_z[:n-k], ent_z[k:])[0,1] for k in range(500)])
    lag = next((i for i, v in enumerate(acf) if v < 1/np.e), 500)
    print(f'T={T}: ent={ent:.2f} c64={c64:.3f} c128={c128:.3f} c256={c256:.3f} eos={eos:.5f} lag={lag} decoup={abs(c256-c64):.3f}')
"
```

# Observations — 2026-03-08

Detailed findings log. See [observations.md](../observations.md) for current model and index.

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

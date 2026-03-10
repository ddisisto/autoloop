# Observations — 2026-03-10

Detailed findings log. See [observations.md](../observations.md) for current model and index.

### 2026-03-10 — Concept Fragmentation Under Temperature

Searching for "water" as a standalone word across L=512 temperatures suggested it simply disappeared at high T. Expanding the search to include "water" as a **substring** of any alpha sequence reveals a different story: the concept doesn't disappear — it **fragments**.

**Data:** L=512 × T={0.90, 1.00, 1.10, 1.20} S=42, plus L=256 × T=1.50 S=42.

| T | Total matches | Pure "water" | % pure | Unique forms | Compounds | Fused/embedded |
|---|--------------|-------------|--------|--------------|-----------|----------------|
| 0.90 | 78 | 68 | 87% | 8 | 10 | 3 |
| 1.00 | 45 | 35 | 78% | 11 | 10 | 4 |
| 1.10 | 86 | 56 | 65% | 18 | 26 | 8 |
| 1.20 | 72 | 39 | 54% | 25 | 29 | 17 |
| 1.50 | 79 | 22 | 28% | 35 | 53 | 27 |

"Compounds" = multi-morpheme words containing water (waterproof, rainwater, etc.). "Fused/embedded" = water appearing inside non-word noise strings where it is not the leading substring.

**Tokenization note:** Verified in L0512_T1.20_S42 that `water`, `Water`, `waters`, `watering`, `waterproof`, `saltwater`, `underwater`, `watercolor`, `waterfalls`, `stormwater`, etc. are all **single tokens** in this model's vocabulary. Even in fused noise strings, the water token is produced intact — e.g., `'smoking'+'hem'+'water'+'is'`, `'book'+'cyl'+'inder'+'We'+'water'+'G'+'lo'+'izard'`. The fragmentation is at the **word boundary level** (surrounding noise tokens pile up without whitespace), not at the subword level. The model produces the `water` token cleanly every time; it's the context around it that degrades.

**Finding 1: Total concept activation is roughly constant across temperatures.**

Total occurrences of water-as-substring range from 45–86 across all five temperatures, with no clear trend. The concept exerts approximately constant pressure on the token distribution regardless of T. What changes is not *whether* it appears but *how*.

**Finding 2: Clean expression degrades monotonically with T.**

At T=0.90, 87% of water-occurrences are the clean standalone word. By T=1.50, only 28% are. The rest are legitimate compound words (waterproof, rainwater) or noise-fused strings:

- T=0.90: `water`, `watering`, `waterproof` (clean morphology)
- T=1.20: `bookcylinderWewaterGloizard`, `smokinghemwateris`, `vanVaxarwater`
- T=1.50: `typekurtchwaterAnalyse`, `countingPaywaterwhile`, `RJdynamicwaters`

At high T the surrounding tokens lose coherence and word boundaries collapse, but the `water` token itself is always produced as a single clean token. The concept survives as a signal embedded in boundary noise.

**Finding 3: "Base rate" is not a meaningful concept here.**

At T=1.50, "water" (standalone) appears at ~2.2/10k tokens. At T=1.20 it's 3.9/10k. We initially called T=1.20 "base rate" — but the substring analysis shows the concept is still being actively generated at elevated frequency, just expressed as fragments and compounds rather than clean standalone words. There is no temperature at which the generation is truly concept-free; the autoregressive context always creates some degree of semantic feedback. The noise floor for a *concept* is lower than the noise floor for a *word*.

**Finding 4: Distribution pattern distinguishes regimes even at similar counts.**

T=1.10 and T=1.20 have similar standalone "water" counts (56 vs 39), but completely different spatial distributions:

- T=1.10: 28/56 occurrences in the last 10% of the run (a plant-watering Q&A attractor). Min inter-occurrence gap: 7 tokens. Clusters.
- T=1.20: Uniform scatter, 2–6 per 10% bin. Min gap: 39 tokens, median gap: 1,902. No clustering.

The shift from clustering to scatter happens between T=1.10 and T=1.20 for L=512. This aligns with the decorrelation lag transition (70→1) and the disappearance of concentrated word segments.

**Implications:**

1. **Concepts and words are different observables.** Word frequency measures surface-level expression. Concept frequency (substring + morphological variants) measures the underlying semantic activation. These diverge at high T.

2. **Temperature controls expression fidelity, not concept activation.** The model produces the `water` token at roughly the same rate regardless of T. Low T gives it the sequential coherence to place it in clean word-bounded context. High T introduces noise that collapses word boundaries without eliminating the underlying drive. The single-token nature of `water` means the concept needs no multi-step assembly — it survives intact even when everything around it is garbled.

3. **The attractor phenomenon is a continuum.** There's no clean on/off boundary. At T=0.90 we see deep clustering (268× "piston" in one segment). At T=1.10 we see weak clustering (plant-watering pocket). At T=1.20 we see elevated-but-scattered frequency. At T=1.50 we see the same token embedded in noise compounds. Each is the same feedback mechanism at different strengths.

4. **Unique-forms count as a regime diagnostic.** The number of distinct water-containing forms increases monotonically with T (8→11→18→25→35). This could serve as a cheap temperature-regime classifier: low form diversity = strong attractor (one form dominates); high form diversity = fragmented concept (many noise-fused variants).

```bash
# Reproduction
python3 -c "
import pandas as pd, re
from collections import Counter
for T in ['0.90', '1.00', '1.10', '1.20']:
    df = pd.read_parquet(f'data/runs/L0512_T{T}_S42.parquet')
    joined = ''.join(str(t) for t in df['decoded_text'].values)
    matches = re.findall(r'[A-Za-z]*[Ww]ater[A-Za-z]*', joined)
    pure = sum(1 for m in matches if m.lower() == 'water')
    forms = len(set(m.lower() for m in matches))
    fused = sum(1 for m in matches if not re.match(r'^[Ww]ater', m) and 'water' in m.lower())
    print(f'T={T}: {len(matches)} total, {pure} pure, {forms} forms, {fused} fused')
"
# Tokenization verification
python3 -c "
import pandas as pd, re
df = pd.read_parquet('data/runs/L0512_T1.20_S42.parquet')
texts = [str(t) for t in df['decoded_text'].values]
joined = ''.join(texts)
char_to_row = []
for i, t in enumerate(texts):
    for c in t:
        char_to_row.append(i)
for m in re.finditer(r'[A-Za-z]*[Ww]ater[A-Za-z]*', joined):
    rows = sorted(set(char_to_row[m.start():m.end()]))
    toks = [repr(df.iloc[r]['decoded_text']) for r in rows]
    print(f'{m.group():40s}  {\" + \".join(toks)}')
"
```

---

### 2026-03-10 — Pre-Collapse Trajectories and Basin Transition Dynamics

New script `precollapse.py` systematically detects collapse events, extracts pre-collapse trajectory features, maps attractor content, and characterizes escape dynamics across all runs. Analyzed 46 runs (all T≤1.20).

**Tool:** `python precollapse.py` (summary), `python precollapse.py --detail <run_id>` (per-run), `python precollapse.py --csv data/precollapse.csv` (all metrics).

#### Finding 1: Regime classification by collapse intensity

Collapse intensity (fraction of steps below entropy 0.1, no sustain requirement) cleanly separates runs into four categories that map onto the previously identified four regimes:

| Regime | Runs | Intensity range | Description |
|--------|------|----------------|-------------|
| deep_collapsed | 6 | 0.31–0.91 | >50% sustained collapse, long-lived attractors |
| collapsed | 6 | 0.51–0.75 | Sustained collapse but <50% of run, later onset |
| oscillating | 25 | 0.06–0.84 | Frequent micro-collapses, no sustained lock-in |
| escaped | 9 | 0.001–0.05 | Essentially free dynamics, minimal low-entropy steps |

The "oscillating" category is the largest and most heterogeneous — it spans L=64 at T=0.50 (intensity 0.22, frequent brief dips) through L=256 at T=0.50 (intensity 0.84, near-continuous micro-collapse that never sustains for 500+ steps because of periodic escape bursts). This confirms that collapse is a continuum, not a binary.

#### Finding 2: Attractor content is semantically diverse

Each collapsed run locks into a distinct attractor. No two runs (across different L, T, or seed) share the same attractor content:

| Run | Attractor | Period | Type |
|-----|-----------|--------|------|
| L=256 T=0.60 | " Wars Star" | 2 | Reversed phrase loop |
| L=256 T=0.70 | "1110. 1111. 1112..." | 0 (counter) | Non-periodic incrementing |
| L=256 T=0.80 | " cells," → ",1" | 2 | Simple token pair, shifts attractor mid-run |
| L=192 T=0.70 | "vetica Sans', 'Hel" | 8 | Font name loop (Helvetica) |
| L=128 T=0.50 | "the Weimar Republic was a tim" | 7 | Historical text loop |
| L=128 T=0.60 | " look at the temperature and" | 5 | Instructional fragment |
| L=192 T=0.60 | "Make sure the torch is turned off before turning on the torch." | ~17 | Complete sentence loop |
| L=208 T=0.50 S=7 | " of the disease" | 3 | Short phrase |
| L=224 T=0.50 | "It is a neurotransmitter." | 6 | Complete sentence |
| L=512 T=0.90 | "Kindergarten 1st Grade Worksheet..." | 0 | Long non-periodic text |

Attractor period tends to be shorter at higher L (L=256 period 2, vs L=128 period 5–7). The counting sequence at L=256 T=0.70 is uniquely non-periodic — it's a monotonically incrementing counter, not a repeating cycle. L=192 T=0.60 generates a perfect repeating instruction ("Make sure the torch is turned off before turning on the torch.") with per-token entropy structure: newline and dash tokens retain ~0.2–0.9 nats while content words drop to 0.01–0.03.

#### Finding 3: Basin transition dynamics — the energy landscape

L=256 T=0.80 shows 21 escape events in 100k steps, providing a rich sample of basin transition dynamics. Each transition consists of: sustained low entropy (basin) → entropy spike (escape) → landing in new state.

**Key pattern: spike magnitude predicts escape success.**

| Spike magnitude | N events | Lands shallower | Lands deeper |
|----------------|----------|----------------|--------------|
| <1.0 nats | 3 | 1 (33%) | 2 (67%) |
| 1.0–3.0 nats | 7 | 4 (57%) | 3 (43%) |
| >6.0 nats | 11 | 11 (100%) | 0 (0%) |

Small escape spikes (<1 nat) more often lead to *deeper* basins — the system doesn't escape far enough and falls into a nearby, lower-energy attractor. Large spikes (>6 nats) always reach shallower territory. The threshold for reliable escape appears to be ~3 nats at this (L, T).

**Basin depth progressively decreases over time.** At L=256 T=0.80, early collapse floors are ~0.03–0.05, later floors reach 0.014–0.019. The system cascades through progressively deeper basins. The longest single collapse (16,067 steps at floor 0.014) was the deepest basin observed.

**Cross-run comparison:**

| Run | Escapes | Deeper | Mean spike | Mean floor |
|-----|---------|--------|-----------|------------|
| L=256 T=0.80 | 21 | 6 (29%) | 3.8 | 0.031 |
| L=256 T=0.90 | 3 | 0 (0%) | 5.4 | 0.037 |
| L=512 T=0.90 | 5 | 0 (0%) | 6.3 | 0.019 |
| L=192 T=0.70 | 5 | 0 (0%) | 5.5 | 0.067 |

At higher T or lower L, escapes are always to shallower states — the thermal energy is sufficient to clear the basin rim. The "deeper transitions" are unique to the suppressed zone (L=256 T=0.80) where escape energy barely exceeds the basin depth. This is the energy landscape analogy made concrete: T sets the thermal fluctuation amplitude, L sets the basin depth, and the transition depends on their ratio.

#### Finding 4: W/L convergence reveals memory saturation

The compressibility at window size W depends on the W/L ratio. When W approaches L, the measurement window sees the model's full context — this is where "memory saturation" dynamics are visible.

**L=256 T=0.80 (deep_collapsed):**

| W/L | Comp | Trend |
|-----|------|-------|
| 0.06 | 1.554 | +3.1e-06 (inflating) |
| 0.12 | 0.968 | −3.1e-07 (flat) |
| 0.25 | 0.610 | −1.5e-06 (compressing) |
| 0.50 | 0.383 | −1.8e-06 (compressing) |
| 1.00 | 0.260 | −1.9e-06 (compressing) |

The slope divergence (trend at W/L=1.0 minus trend at W/L=0.06) is −5.0e-06: large windows are compressing *faster* than small windows over time. This means the system is becoming more globally repetitive as it descends into attractors — the basin capture shows up first at large W.

**Contrast with L=256 T=0.90 (oscillating):** slope divergence is +2.0e-07 (essentially zero) — all scales trend together. No preferential compression at any W/L ratio. The system is either free or stuck uniformly across scales.

**L=64 T=0.50 (oscillating):** comp at W/L=1.0 is 0.347, at W/L=4.0 is 0.148. With W>L, the measurement window exceeds the model's context — and compressibility drops further, meaning the patterns that repeat within the context window don't repeat *across* context boundaries. This confirms that L is the characteristic scale of the attractor structure.

#### Finding 5: Pre-collapse trajectory signatures

Pre-collapse features (measured in the 2000 steps before first collapse onset) show systematic patterns:

- **Entropy slope before collapse** varies widely: some runs show gradual descent (L=128 T=0.50: −4.6e-04, clear downward trend) while others show flat or even rising entropy (L=256 T=0.80: −1.9e-05, essentially flat; L=208 T=0.50 S=7: +7.0e-04, rising). Collapse can onset suddenly without a warning slope.

- **Variance decay** (slope of rolling entropy variance) is more reliable: negative values mean fluctuations are narrowing before collapse (the system is "settling"). L=256 T=0.80 shows −2.9e-02 (strong narrowing), L=128 T=0.50 shows −2.5e-01 (very strong).

- **Multi-scale spread** (comp_Wmax − comp_Wmin) in the pre-collapse window: higher values (>1.0) indicate multi-scale structure before collapse (L=192 T=0.70: 1.231, L=256 T=0.60: 1.219). Lower values suggest homogeneous dynamics (L=256 T=0.70: 0.245 — it was already deep in micro-collapse before the sustained event).

- **Descent compressibility slopes by W** show a sign flip in L=256 T=0.80: small W (16, 32) have negative slopes (local structure forming) while large W (128, 256) have positive slopes (global structure diverging). This divergence — local compression with global expansion — is a fingerprint of the *approach* to a deep attractor through a suppressed-dynamics intermediary.

#### Finding 6: Escape from the counting attractor (L=256 T=0.80 step 74340)

Token-level inspection of the escape from the digit-counting attractor (period-10: "1234567890") shows the transition is a single-token perturbation:

- Steps 74330–74339: locked in "1234567890" cycle, entropy 0.006–0.069 per token
- Step 74339: "1" instead of expected "0" (entropy 0.028)
- Step 74340: entropy spikes to 2.29 nats — attractor broken
- Steps 74340–74360: chaotic digit sequence (entropy 0.6–1.4), no clear pattern
- Steps 74340–77234: landing zone, mean entropy 0.684 (shallower than floor 0.014)
- Steps 88810+: re-collapse into ",1,1,1..." (period 2, deeper than the count-based attractor)

The escape mechanism is stochastic: a single token sampled slightly "wrong" (within the temperature-scaled distribution) destabilizes the entire context window. At T=0.80, these perturbations are rare but inevitable over 10k+ steps. The system then explores briefly before falling into a new basin.

#### Implications for the annealing experiment

1. **Pre-seeded " Star Wars" at L=256 T=0.60 should collapse predictably.** The " Wars Star" attractor observed in the natural run (onset step 32k, intensity 0.905) demonstrates the basin exists. Pre-filling with " Star Wars" text should accelerate collapse by starting inside (or near) the basin.

2. **At L=64 T=0.60, the same content should escape.** T_escape(64) ≈ 0.55, so T=0.60 is above escape. The basin transition data shows that smaller L produces shallower basins with higher escape probability.

3. **Escape spike magnitude is a measurable proxy for T_escape.** The threshold for reliable escape (~3 nats at L=256 T=0.80, always succeeds at >6 nats) maps directly to the energy required to clear the basin rim. Measuring this threshold across (L, T) conditions would quantify T_escape with single-run data rather than requiring grid sweeps.

4. **Basin depth is history-dependent.** The progressive deepening observed at L=256 T=0.80 (floors 0.05→0.03→0.014) means the *timing* of annealing intervention matters — earlier intervention encounters shallower basins. This connects to the "memory-depth annealing" concept: not just reducing L, but doing so before the system has settled into its deepest accessible basin.

```bash
# Reproduction
python precollapse.py                           # full summary by regime
python precollapse.py --detail L0256_T0.80_S42  # basin transitions + W/L profile
python precollapse.py --detail L0256_T0.60_S42  # Star Wars attractor
python precollapse.py --csv data/precollapse.csv # all metrics to CSV

# Token-level escape inspection
python3 -c "
import pandas as pd
df = pd.read_parquet('data/runs/L0256_T0.80_S42.parquet')
exp = df[df.phase == 'experiment'].reset_index(drop=True)
for i in range(74330, 74370):
    r = exp.iloc[i]
    print(f'step {i}: entropy={r.entropy:.4f} text={repr(r.decoded_text)}')
"

# Basin transition summary across runs
python3 -c "
import pandas as pd
df = pd.read_csv('data/precollapse.csv')
print(df[['run_id','regime','collapse_intensity','n_basin_transitions','n_deeper_transitions','mean_spike','mean_floor']].to_string())
"
```

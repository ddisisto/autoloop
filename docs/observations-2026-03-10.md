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

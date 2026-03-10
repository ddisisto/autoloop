# Observations — 2026-03-10f: Semantic Theme Mapping & Controller Text Analysis

## L=256 controller run

Launched `ctrl_S42_256_0.70` — the steepest escape boundary test.

```bash
python controller.py --seed 42 --total-steps 10000 --segment-steps 1000 --start-L 256 --start-T 0.70
```

Trajectory:
- T ramps steadily 0.70→1.00 over 7 segments (β stuck at 0.73–0.78 the whole time, below dead zone)
- T=1.00 hits the zone — β=0.94, holds one segment
- β spikes to 1.32, controller drops T to 0.95
- Settles at T=0.95, β=0.95 for final 2 segments
- Zero rollbacks — no collapse events, just a long slow climb through the suppressed zone

Balance T=0.95 for L=256, consistent with T_escape(L=256)≈0.87. Controller had no escape mechanism except T↑ — it never needed L adjustment or rollback.

## Balance-point text is L-dependent

Inspected text from three controller balance points:

| Run | Balance | Character |
|-----|---------|-----------|
| ctrl_S42_8_0.60 (β=0.85) | L=8, T=0.70 | Topic soup — jumps every few sentences (colony demographics → vitamin D → bacterial membranes). No n-gram repeats above count 3. |
| ctrl_S42_128_0.70 (β=0.86) | L=128, T=0.95 | Thematic orbits — education policy → tomato growing → azalea watering loops → acid-base chemistry cycle. Top 4-gram: "the equation for the reaction" (×39). |
| ctrl_S42_256_0.70 (β=0.95) | L=256, T=0.95 | Thematic orbits — education → Mars rover missions on repeat ("the eleventh mission will be a radar rover") → water infrastructure. Top trigram: "will be a" (×38). |

**Key finding: β≈0.90 manifests differently at different L.** Short context (L=8) produces diversity through forgetting — each sentence starts fresh. Long context (L=128, L=256) produces diversity through exploration within a semantic basin — the model orbits related sub-topics. Both are valid β≈0.90; the texture differs. This is not a sensor failure — it's how context length shapes the generative strategy.

```bash
# Reproduction: n-gram analysis of controller runs
python -c "
import pandas as pd
from collections import Counter
for run in ['ctrl_S42_8_0.60', 'ctrl_S42_128_0.70', 'ctrl_S42_256_0.70']:
    df = pd.read_parquet(f'data/runs/{run}.parquet')
    exp = df[df.phase == 'experiment'].reset_index(drop=True)
    words = ''.join(exp.decoded_text.tolist()).lower().split()
    trigrams = Counter(' '.join(words[i:i+3]) for i in range(len(words)-2))
    print(f'{run}: {len(words)} words, top trigrams:')
    for g, c in trigrams.most_common(5): print(f'  {c:4d}  {g}')
"
```

## Semantic theme mapping (new tooling)

Added `--clouds` and `--themes` modes to `semantic.py`:

```bash
python semantic.py --clouds                              # auto-discover themes + basins + co-occurrence
python semantic.py --themes water book food pressure      # multi-theme compact density report
python semantic.py --theme temperature                    # legacy single-theme full analysis
```

### Theme discovery across 59 runs

Auto-discovered 60 content themes (>100 hits, present in 3+ runs). Key structural finding: **every T=0.50 run is dominated by 1-3 themes at 10-50% word density**. These are the attractor basins. Higher-T runs converge to a shared low-density background (health, water, study, research at 0.001-0.003 density).

### Attractor basin catalog (by semantic fingerprint)

Each collapsed run locks into a distinct semantic basin. The `--clouds` mode maps these automatically:

| Run | Basin (top themes by density) |
|-----|------|
| L0208_T0.50_S7 | **disease** (0.748) — 75% of all content words |
| L0256_T0.50_S42 | **young + woman** (0.577 + 0.289) |
| L0256_T0.60_S42 | **star + wars** (0.453 + 0.453) |
| L0224_T0.50_S42 | **book + jewish + history** (0.375 + 0.166 + 0.166) |
| L0192_T0.60_S42 | **torch + sure + turned + turning** (0.288 + 0.171 + 0.144 + 0.144) |
| L0176_T0.50_S42 | **sleep + enough + getting** (0.262 + 0.141 + 0.141) |
| L0160_T0.50_S123 | **city + resource** (0.256 + 0.228) |
| L0192_T0.50_S7 | **gambling + arguments + experiencing + friends** (0.213 + 0.212 + 0.117 + 0.117) |
| L0128_T0.50_S42 | **republic + weimar** (0.200 + 0.200) |
| L0160_T0.50_S7 | **treaty + versailles + signed + allies** (0.198 × 4) — WWI basin |
| L0176_T0.50_S123 | **technology + term + mission** (0.168 + 0.167 + 0.158) |
| L0224_T0.50_S7 | **diabetes + diabetic + retinopathy + complication** (0.192 + 0.170 × 3) |
| L0208_T0.50_S123 | **woman + research + helicopter** (0.211 + 0.177 + 0.174) |
| L0208_T0.50_S42 | **children + describe** (0.338 + 0.099) |
| L0176_T0.50_S7 | **vaccine + report + misuse + entitled** (0.149 + 0.144 × 3) |

### Co-occurrence structure

Theme pairs with Jaccard=1.0 (only appear together in one run) reveal tight basins:
- star + wars → L0256_T0.60_S42
- republic + weimar → L0128_T0.50_S42
- treaty + versailles + signed + allies → L0160_T0.50_S7
- diabetes + diabetic + retinopathy + complication → L0224_T0.50_S7
- gambling + arguments + experiencing + friends + caught → L0192_T0.50_S7
- sleep + enough → L0176_T0.50_S42

### Seed dependence

Same (L,T) with different seeds → different basins:
- L=64 T=0.50: S42 → generator+study+culture, S123 → book+heart+disease, S7 → describe+house
- L=208 T=0.50: S42 → children+describe, S123 → woman+helicopter, S7 → disease (0.748!)
- L=224 T=0.50: S42 → book+jewish+history, S123 → pressure+water+mental, S7 → diabetes+retinopathy

Collapse is deterministic (all seeds collapse at T=0.50) but content is seed-dependent — confirming the earlier finding from 10c, now with full semantic fingerprinting.

### Theme class structure

Three classes of themes emerged:

1. **Attractor words** — spike in single runs at extreme density (0.10-0.75). E.g. disease, star, wars, weimar, versailles, retinopathy. These define specific basins. Spikiness >50.
2. **Widespread content words** — present in 40+ runs at moderate density (0.001-0.01). E.g. health, water, study, research, children, development. These form the shared background vocabulary.
3. **Structural/institutional words** — present in 40+ runs, low density, topically neutral. E.g. govern, control, self. Their neighbor profiles shift by regime: "control" neighbors "diabetes mellitus" at T=0.50, neighbors Python code (`def`, `self.`, `__`) at T=0.60, neighbors abstract governance at T=0.90.

### "self" as a regime marker

"self" (8,760 hits, 45 runs) has a revealing neighbor profile:
- T=0.50: **self-esteem, low, lack, sleep** (psychological attractor content)
- T=0.50 also: **`__`, class, `.__`, def** (Python code generation in some collapsed runs)
- T=0.60-0.90: increasingly code-like (`_`, `=`, `(`, `):`), then grammatical
- T=1.00+: **himself, itself, yourself** (morphological variants dominate)

The model's relationship to "self" literally changes with temperature — from psychological concept, to programming construct, to grammatical form.

### Controller runs match their L-peers

Controller balance points land in the same semantic neighborhoods as fixed-parameter runs at similar (L,T):
- ctrl_S42_128_0.70 → water(0.019), same as L0128_T0.70_S42
- ctrl_S42_256_0.70 → mission(0.019), related to L0176_T0.50_S123's technology+mission basin

```bash
# Full reproduction
python semantic.py --clouds
python semantic.py --clouds --csv data/theme_basins.csv
python semantic.py --themes water book temperature author food pressure disease
python semantic.py --themes self robot auto control capture govern
```

## Open questions updated

Answered:
- ~~What does the generated text look like at the balance point?~~ → L-dependent texture: short L = topic soup, long L = thematic orbits. Both achieve β≈0.90.

New:
- Can semantic basin fingerprints predict which attractor a run will find? (Does initial context bias the basin?)
- Is there a topology to the basin space? (Which basins are "adjacent" — reachable by small perturbations?)
- The "self" profile suggests the model's conceptual repertoire reorganizes with temperature. How deep does this go?
- Substring matching in theme search catches compound forms (wastewater, waterproof) — is this a feature or a bug for basin fingerprinting? Whole-word mode may miss morphological variants that are part of the same attractor.

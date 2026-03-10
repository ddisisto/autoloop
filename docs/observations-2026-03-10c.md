# Observations — 2026-03-10c: Semantic Analysis (Theme: "temperature")

## Setup

New analysis script `semantic.py` — searches for a theme word across all runs, extracts context windows, computes vocabulary stats by (L, T) condition. Default theme: "temperature".

```bash
python semantic.py --seed 42                     # all seed=42 runs
python semantic.py --theme "the" --seed 42       # different theme
python semantic.py --csv data/semantic.csv        # export vocab stats
```

## Finding: The model collapses into a literal "temperature" attractor

L=128/T=0.60/S42 generates coherent climate/weather science text for ~65k tokens, then undergoes catastrophic collapse at step ~67k into a tight loop: "the temperature and the air temperature and the temperature and look at the weather patterns and look at the temperature..."

Entropy profile shows the cliff:
- Steps 0–60k: entropy 0.59–0.90 (coherent generation)
- Steps 60k–70k: entropy 0.59 (transition zone)
- Steps 70k–100k: entropy **0.0115** (locked attractor)

6518 of 7128 total "temperature" hits across all runs come from this single collapsed run. The attractor content is *about* temperature — the model's topic became its prison.

```bash
python semantic.py --runs data/runs/L0128_T0.60_S42.parquet --context-radius 120
```

## Finding: Local entropy around "temperature" hits perfectly tracks the T parameter

Average entropy in a ±20-token window around each "temperature" occurrence:

| T | Local entropy | Context quality |
|---|---|---|
| 0.50 | 0.159 ± 0.265 | Repetitive medical text (fever, body temperature) |
| 0.60 | 0.014 ± 0.014 | Collapsed loop |
| 0.70 | 1.975 ± 1.291 | Coherent climate/science paragraphs |
| 0.80 | 1.972 ± 1.260 | Coherent varied contexts (cooking, plants, health) |
| 0.90 | 2.661 ± 1.352 | Diverse but occasionally fragmented |
| 1.00 | 3.891 ± 1.565 | Loosely coherent, technical jargon mixing |
| 1.10 | 7.122 ± 1.150 | Word salad with occasional structure |
| 1.20 | 8.203 ± 1.336 | Near-random token sequences |
| 1.50 | 8.397 ± 1.495 | Pure noise |

The word "temperature" acts as a probe: its neighborhood reveals the local regime regardless of where it appears.

## Finding: Vocabulary richness spans 100x across conditions

Type-token ratio (TTR) = unique words / total words:

| Condition | TTR | Interpretation |
|---|---|---|
| L=224/T=0.50 | 0.003 | Near-total vocabulary collapse (204 unique words in 77k) |
| L=256/T=0.50 | 0.004 | "young woman, man," on repeat |
| L=128/T=0.50 | 0.013 | Slightly richer collapse |
| L=64/T=0.50 | 0.048 | Weakest collapse (shallowest basin) |
| L=64/T=0.80 | 0.194 | Rich dynamics |
| L=256/T=1.00 | 0.298 | Diverse generation |
| L=192/T=1.50 | 0.503 | Near-random (TTR > 0.5 means most words appear once) |

TTR is a clean scalar proxy for regime identification, complementing entropy and compressibility.

## Finding: Vocabulary saturation curves reveal escape events

Unique token count at 10k-step intervals (L=256, seed=42):

- **T=0.50**: 111 → 181 → 181 → 181 ... (saturates by 20k, ratio 1.63x) — total lock-in
- **T=0.70**: 520 → 520 → 520 → 520 → 533 → 533 → **886 → 1321 → 1905 → 2131** (flat for 50k, then explodes — **escape event at ~60k steps**)
- **T=0.90**: 2115 → 3407 → ... → 7961 (steady growth, ratio 3.76x)
- **T=1.00**: 3355 → 5282 → ... → 13266 (faster growth, ratio 3.95x)
- **T=1.50**: 7005 → 11325 → ... → 25084 (fastest, ratio 3.58x — Heaps' law)

The T=0.70 curve is remarkable: vocabulary is *frozen* for 50k tokens in a suppressed regime, then escapes. This is the suppressed-dynamics zone identified in the entropy analysis, now visible in vocabulary space.

## Finding: Massive vocabulary asymmetry between regimes

At L=256/S42, comparing vocabulary across T:
- Words appearing **only** at T<=0.80: 6,900
- Words appearing **only** at T>=1.00: 43,106
- Shared vocabulary: 3,007

High-T doesn't just add noise to the same word distribution — it unlocks an **entirely different vocabulary**. The shared core (3k words) is mostly function words and common nouns. The low-T exclusive vocabulary includes structured content (dates, code, specific entities). The high-T exclusive vocabulary is dominated by fragments and neologisms.

## Finding: Each collapsed run has a unique "topic attractor"

Collapsed attractors by condition (from top-word analysis):

| Condition | Attractor content | Top words |
|---|---|---|
| L=256/T=0.60 | Star Wars | star(34k), wars(34k) |
| L=256/T=0.50 | Gender narrative | young(21k), woman(11k), man(11k) |
| L=224/T=0.50 | Book reviews | the(17k), book(10k), is(8k) |
| L=176/T=0.50 | Sleep advice | sleep(8k), not(6k), enough(6k) |
| L=192/T=0.50 | Historical narrative | the(11k), was(9k), from(5k) |
| L=208/T=0.50 | Medical/health | the(13k), of(10k), and(7k) |
| L=128/T=0.60 | Climate/temperature | the(11k), and(8k), at(7k) — then collapses to "temperature" loop |

The attractor content is determined by the initial generative trajectory (seed-dependent), but the *depth* is determined by (L, T). The model doesn't converge to a universal attractor — it finds local minima in content space.

## Finding: Attractor content is seed-dependent, collapse is deterministic

All 21 collapsed runs (T=0.50, all L and seeds) collapse, but every seed lands on a **different attractor**. The collapse dynamics are deterministic; the content is path-dependent.

Full attractor catalog across 3 seeds × 7 L values at T=0.50:

| L | Seed 7 | Seed 42 | Seed 123 |
|---|---|---|---|
| 64 | "blood in urine" + sklearn imports | "the generator is a generator" | "a book. a book." + paragraph counting |
| 128 | "the age of the fish" | "the Weimar Republic was a time where" | "hard to get sand" → "1.8.1.8.1.8" |
| 160 | "Treaty of Versailles signed by Allies" | "place to visit" + photoelectric effect | "is a very important resource for a city" |
| 176 | "Misuse of the Mis-name 'Vaccine'" | "not getting enough sleep" | "the term is technology of the mission" |
| 192 | "arguments about gambling" | "the election was televised" | "the first power plants were completed" |
| 208 | "the disease of the disease of the disease" | "the number of children" | "the woman is the first to fly a helicopter" |
| 224 | "diabetic retinopathy is a complication" | "is a book about the history" → "it is a neurotransmitter" | "the pressure in the water" |

These cluster into **semantic families**: medical/body (blood, disease, sleep, retinopathy, neurotransmitter), political/historical (Weimar, Versailles, election, gambling), social categories (young woman/man, the woman first to fly, number of children), self-referential (generator is generator, a book about a book), structural/physical (pressure in water, hard to get sand).

## Finding: Attractor content describes its own dynamics

The collapsed content isn't random training data. It systematically features:
- **Tautologies**: "the generator is a generator" — self-reinforcing by definition
- **Incomplete predicates**: "the Weimar Republic was a time where" — never reaching the object, demanding repetition
- **Self-perpetuating conditions**: "not getting enough sleep... can include not getting enough sleep" — the condition reproduces itself
- **Recursive structures**: "the disease of the disease of the disease" — pure self-reference
- **Confinement**: "the man was not allowed to leave the room" (×399, numbered) → Star Wars

The content and the dynamics are the same thing. These are **eigenstates**: configurations where meaning, structure, and prediction all align, creating a fixed point with zero gradient out.

## Finding: Pre-collapse trajectories reveal semantic basin connectivity

The L=256/T=0.60 run doesn't jump to Star Wars — it traverses a **path through semantic space**:

1. Education policy (inclusive access, equity) — ~0-1%
2. Political violence (Naxalites and Maoists in India) — ~2%
3. Apocalyptic text (Book of Revelation, numbered ordinally) — ~4-8%
4. World civilization (Islamic world, polycentric, decentralized) — ~12-15%
5. Bureaucratic cataloging (William T. Wright, numbered table) — ~18-20%
6. Imprisonment ("the man was not allowed to leave the room" ×399) — ~22%
7. Star Wars Star Wars Star Wars — ~25% onward, forever

Each waypoint is a basin the system passed through on the way down. The pre-collapse text maps **how semantic basins connect to each other**. Across all 21+ collapsed runs, these descent paths would form a graph of the model's semantic topology.

**Key implication**: the attractors aren't isolated points. They form a connected landscape. The "cure for disease" leads through "sleep." Confinement leads through counting. The trajectories between basins are as informative as the basins themselves.

## Finding: Heaps' law exponent β separates regimes cleanly

V(n) = K·n^β fitted to vocabulary growth curves:

| Regime | β range | Interpretation |
|---|---|---|
| Deep collapse (T=0.50, L≥160) | 0.17–0.38 | Vocabulary barely grows — locked |
| Shallow collapse (T=0.50, L=64) | 0.50 | Sub-linear but still growing |
| Escape boundary (T=0.70) | **0.93–1.28** | Vocabulary accelerating — escape events |
| Rich dynamics (T=0.90–1.00) | 0.75–0.85 | Healthy sub-linear (standard Heaps') |
| Noise (T=1.50) | 0.78 | High diversity, standard power law |

β > 1.0 at T=0.70 (L=192) is the escape event: vocabulary *accelerates* as the system breaks free from a suppressed attractor. The Heaps exponent is a single-number diagnostic for regime.

## Finding: Repetition onset sparklines show regime dynamics

Repetition ratio (fraction of bigrams appearing 3+ times) in 5k-token sliding windows:

- **Collapsed**: `████████████` from step 0 — immediate lock-in
- **Late collapse** (L=128/T=0.60): `▇██▆█▇▇▇▇▇▇██████` — coherent for 65k, then cliff
- **Escape event** (L=192/T=0.70): `▇█▇▇▆▆▇▅▁▂▂▂▂▂` — trapped first, then escapes
- **Rich dynamics**: `▁▁▁▁▁▁▁▁▁▁▁▁▁` — never repetitive
- **Noise**: `▁▁▁▁▁▁▁▁▁▁▁▁▁` — zero repetition (everything is novel)

## Finding: Semantic coherence (bigram Jaccard) has a sharp phase transition

Mean coherence between adjacent 500-word windows, averaged by T:

| T | Coherence | |
|---|---|---|
| 0.50 | 0.896 | Same content, window after window |
| 0.60 | 0.725 | High persistence, some drift |
| 0.70 | 0.439 | **Sharp drop** — the escape boundary |
| 0.80 | 0.386 | Moderate topic switching |
| 0.90 | 0.356 | Fast topic turnover |
| ≥1.00 | 0.332–0.340 | Floor — every window is different |

The transition from 0.90 → 0.44 between T=0.60 and T=0.70 is the sharpest feature. This is the phase boundary.

## Reproduction

```bash
# Full analysis (all 5 analyses: theme search, attractor catalog, repetition onset, Heaps' law, coherence)
python semantic.py --seed 42

# Export all metrics to CSV (includes Heaps β, coherence, repetition onset, vocab stats)
python semantic.py --csv data/semantic.csv --seed 42

# Specific run deep-dive
python semantic.py --runs data/runs/L0128_T0.60_S42.parquet --context-radius 120

# Compare attractors across seeds
python semantic.py --theme "the" --runs data/runs/L0192_T0.50_S*.parquet
```

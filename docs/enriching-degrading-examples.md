# Enriching vs Degrading Dynamics: Concrete Examples

**Date:** 2026-03-20
**Source data:** SmolLM-135M, L=128, seeds {42}, 100k tokens/run

All examples from sweep runs at L=128 with seed 42. Entropy H and surprisal S in nats. Gap = H + log_prob = H - S. Enriching tokens have gap < 0 (the sampled token was more surprising than the distribution's entropy); degrading tokens have gap > 0 (more predictable than average).

## The spectrum is not binary

Five regimes emerge along the temperature axis, each with a distinct relationship between entropy, surprisal, and the gap:

| Regime | T | H | Surprisal | Gap | % Enriching | Character |
|---|---|---|---|---|---|---|
| Deep basin | 0.60 (late) | 0.01 | 0.000 | +0.01 | 0% | Verbatim token loop |
| Suppressed | 0.70 | 1.37 | 0.66 | +0.71 | 9% | Repetitive but grammatical |
| Rich dynamics | 0.80 | 1.57 | 1.07 | +0.50 | 16% | Coherent, topically shifting |
| High entropy | 1.00 | 4.40 | 4.18 | +0.22 | 27% | Loosely coherent, drifting |
| Noise | 1.50 | 8.26 | 9.64 | -1.37 | 83% | Unstructured word fragments |

At the extremes, the gap sign flips. Deep basins have gap ≈ 0+ (surprisal matches near-zero entropy — the model is certain and correct). Noise has gap < 0 on average (surprisal exceeds entropy — the model is uncertain and still wrong). Neither extreme produces structured output.

The functionally interesting regime — where the system generates coherent, non-repetitive text — sits in the middle (T ≈ 0.70–1.00 at this L). Here, most tokens are degrading (reinforcing local structure, maintaining grammaticality) but a minority are enriching (introducing new topics, breaking patterns). The balance between these determines whether the system drifts toward collapse or sustains diverse output.

## Example 1: Collapse in real time

**Run:** `L0128_T0.60_S42`, steps 67200–67712

Four consecutive context windows (128 tokens each) spanning the transition from structured output into a permanent attractor basin:

**Window 1** (steps 67200–67328) — H=1.743, surprisal=0.784, 12% enriching:
> than it was in the southern hemisphere. "That's because the ocean is more active in the tropics, so it's more likely to be affected by tropical cyclones," McHugh said. "This is a very strong storm, an

**Window 2** (steps 67328–67456) — H=1.615, surprisal=0.643, 8% enriching:
> tell that is by looking at the weather patterns," McHugh said. "We're going to have to look at the weather patterns and look at the climate." "We're going to have to look at the climate," McHugh said.

**Window 3** (steps 67456–67584) — H=0.196, surprisal=0.004, 0% enriching:
> look at the temperature and look at the temperature and look at the temperature and look at the temperature and look at the temperature and look at the temperature and look at the temperature and look

**Window 4** (steps 67584–67712) — H=0.012, surprisal=0.000, 0% enriching:
> temperature and look at the temperature and look at the temperature and look at the temperature and look at the temperature and look at the temperature and look at the temperature and look at the temp

The content narrows progressively: "weather patterns" → "climate" → "temperature" → verbatim 5-token loop. Surprisal drops four orders of magnitude across two context rotations. The system was already marginal — "look at the weather patterns and look at the climate" contains the seed of the loop — and the narrowing vocabulary removes the remaining enriching tokens that could have broken the pattern.

Once locked, every token in the basin has surprisal < 0.00001:

| Token | Surprisal | Entropy |
|---|---|---|
| " and" | 0.00001 | 0.033 |
| " look" | 0.00000 | 0.012 |
| " at" | 0.00000 | 0.003 |
| " the" | 0.00000 | 0.004 |
| " temperature" | 0.00000 | 0.006 |

The model assigns ~100% probability to the next token. The loop is a fixed point: content, structure, and prediction are perfectly aligned.

This basin persists unchanged from step ~67500 to the end of the run at step 100000 — over 32000 tokens, 250+ context rotations.

## Example 2: Enriching vs degrading tokens

**Run:** `L0128_T0.80_S42`, steps 94500–95000 — H=3.10, surprisal=2.31, 23% enriching

> 4/10/09/education/classroom/09physics.html.&lt;|endoftext|&gt;- About Us - Contact Us - Apply for a Job - Get Involved Biographical Data for Bruce Atkins Bruce Atkins was born in Yarmouth, Massachusetts on the 10th of August 1898. The elder Bruce was born in the 2nd of December 1883, which, according to the Surname Guide, was in the town of Yarmouth...

Topics shift across sentence boundaries — a URL fragment, a navigation menu, a biographical entry. The text is grammatical and locally coherent but globally wandering. Individual tokens split cleanly into two populations:

Sample enriching tokens (gap < 0):

| Token | Entropy | Surprisal | Gap |
|---|---|---|---|
| "physics" | 6.95 | 8.15 | -1.20 |
| "education" | 2.53 | 5.72 | -3.19 |
| "class" | 2.50 | 6.09 | -3.59 |
| "9" | 1.07 | 5.45 | -4.37 |
| `<endoftext>` | 0.64 | 2.20 | -1.56 |

These tokens were more surprising than the distribution's entropy — the model did not expect them. They introduce new content ("physics", "education") or mark structural breaks (`<endoftext>`) that redirect the generation.

Sample degrading tokens (gap > 0):

| Token | Entropy | Surprisal | Gap |
|---|---|---|---|
| " Us" | 3.10 | 0.23 | +2.86 |
| "-" | 6.29 | 3.52 | +2.77 |
| " About" | 6.37 | 4.17 | +2.20 |
| "-" | 1.14 | 0.02 | +1.12 |
| "." | 1.15 | 0.09 | +1.06 |

These are predictable continuations — punctuation, function words, expected phrases — that reinforce the current pattern. The model anticipated them and was right.

## Example 3: Noise is not enrichment

**Run:** `L0128_T1.50_S42`, steps 50000–50300 — H=8.26, surprisal=9.64, 83% enriching

> corresponded significant sess flora far theory undesirable Migration lupus land thus ongoing volume somewhat matplotlib ministry PatchCivil deepen pennant suidae domain Termkhah command commandivity reason Reason Reserar gave Taiwan separated proniamospheric roll leaving vision singleschedule Sri Vishnu southern reception syngest bi Former uses logger favorite gut vent heroin resonant Rosa government raptlearn square TCP era Rocky associate shortage suicidal monetary meteorological sea moved in

At T=1.50 the gap is negative on average: surprisal (9.64) exceeds entropy (8.26). The model is uncertain (high H) and the sampled token is even less likely than the average possibility (high S). By the gap metric, 83% of tokens are "enriching."

But the output is unstructured noise — no grammar, no topic, no coherence. The enriching tokens are not introducing meaningful new structure; they are random draws from a near-uniform distribution. High surprisal at every token is the signature of noise, not of productive novelty.

## Summary of signals

| Signal | Deep basin | Suppressed | Rich | Noise |
|---|---|---|---|---|
| Surprisal | ≈ 0 | Low (< 1) | Moderate (1–2) | High (> 5) |
| Entropy | ≈ 0 | Low | Moderate | High |
| Gap | ≈ 0+ | Positive | Mixed | Negative |
| % enriching | 0% | 2–9% | 16–27% | > 50% |
| Content | Verbatim loop | Repetitive lists | Coherent prose | Word salad |
| Vocabulary growth | Zero | Saturating | Sublinear | Linear (random) |

Degrading dynamics taken to their extreme produce attractor basins. Enriching dynamics taken to their extreme produce noise. Structured, diverse output — the regime where autoregressive generation is functional — requires both, in tension.

## Reproduction

```bash
# The collapse transition
loop grep "look at the temperature" --type sweep --L 128 --T 0.60 --seed 42 -C 100

# Rich dynamics text
loop grep "physics" --type sweep --L 128 --T 0.80 --seed 42 -C 50
```

All data from runs under `data/runs/sweep/`. Parquet columns used: `entropy`, `log_prob`, `decoded_text`. Gap = `entropy + log_prob`.

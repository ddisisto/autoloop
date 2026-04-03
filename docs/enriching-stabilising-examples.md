# Enriching vs Stabilising Dynamics: Concrete Examples

**Date:** 2026-03-20 (terminology updated 2026-03-30)
**Source data:** SmolLM-135M, L in {128, 192}, seed 42, 100k tokens/run

All examples from sweep runs. Entropy H and surprisal S in nats. Gap = H + log_prob = H - S. Enriching tokens have gap < 0 (the sampled token was more surprising than the distribution's entropy); stabilising tokens have gap > 0 (more predictable than average). The spectrum table uses L=128 throughout for consistency.

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

The functionally interesting regime — where the system generates coherent, non-repetitive text — sits in the middle (T ≈ 0.70–1.00 at this L). Here, most tokens are stabilising (maintaining local structure, continuing grammatical trajectories) while a minority are enriching (introducing new topics, breaking patterns). This is normal and necessary — stabilising tokens are the scaffolding that makes coherent output possible. The balance between enriching and stabilising determines whether the system drifts toward degeneration or sustains diverse output.

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

## Example 2: Enriching vs stabilising tokens in prose

In the following excerpts, **bold** tokens are enriching (gap < 0: the model found them more surprising than its own uncertainty). Normal tokens are stabilising (gap > 0: more predictable than average). The pattern is consistent: content-bearing words that introduce new information are enriching; grammatical glue, punctuation, and expected continuations are stabilising.

**Run:** `L0192_T0.70_S42`, steps 55000–55128 — 20% enriching

> Now, how does **our** **magical** metal come into play? **Dental** implants are surgically **inserted** into your jawbone through small incisions. Once placed, they fuse with your **bone**, creating a **stable** **base** for your new tooth **roots**. Over time, your **natural** **bone** **returns**,** giving you a brand new set of **roots** that look and feel like your **actual** teeth. **Ne**at, huh? But wait, there's more! Did you know that dental implants **actually** **hide** **their** roots** in** the** soft** tissue inside your mouth?

The enriching tokens are the specific nouns and modifiers that carry semantic content: "magical", "Dental", "inserted", "bone", "stable", "roots". These are the tokens the model did not fully anticipate — each one narrows the topic in a way that wasn't determined by what came before. The stabilising tokens are the structural scaffolding: "into your", "creating a", "that look and feel like" — phrases the model predicted with high confidence because grammar and topic together constrain them.

**Run:** `L0128_T0.80_S42`, steps 17000–17128 — 16% enriching

> 5,000 was initially borrowed **due** to the collateral. **Sometimes**, such situations happen when **losses** occur. **Should** **a** borrower fail to meet their obligations, the lender could step in and take action. But what happens **now**? Does the lender get to decide who pays back the loan? Or does the borrower simply continue to repay the balance? This brings us **back** to our **concept** of **deb**entures. When **using** a debenture, the borrower agrees to repay the loan **after** **the** **business** **settles** **the** **balance**. **Over** time, **though**, the business might **think** that the loan **has** been **taken**, causing the borrower to default.

**Run:** `L0128_T0.80_S42`, steps 15500–15628 — 17% enriching

> small town** called** Harmonyville, lived** four** best friends - Timmy the Turtle, Sally the **Sal**amander, Benny the **Beaver**, and Daisy the Deer. They** all** **had** different jobs, but they worked together to make sure **they** were taken care of and happy. One day, while playing near the river, **Sally** noticed something strange. **"**Timmy**,** why **does** your shell look **like** **that**?" she asked **curiously**. Timmy replied, "**Oh**, I **am** a tree, Sally! I **lay** **there** for **days**,** making** my shell.

In all three passages, roughly 15–20% of tokens are enriching. Remove them and the text becomes a generic template; remove the stabilising tokens and the text becomes an incoherent keyword list. Functional generation requires both: the stabilising majority maintains structure while the enriching minority injects the content that keeps the system from degenerating into a loop.

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

Stabilising dynamics taken to their degenerate extreme produce attractor basins — the enrichment fraction drops to zero and the system perfectly self-predicts. Enriching dynamics taken to their extreme produce noise — every token is surprising but none are meaningful. Structured, diverse output — the regime where autoregressive generation is functional — requires both, in tension.

## Reproduction

```bash
# The collapse transition
loop grep "look at the temperature" --type sweep --L 128 --T 0.60 --seed 42 -C 100

# Rich dynamics text
loop grep "physics" --type sweep --L 128 --T 0.80 --seed 42 -C 50
```

All data from runs under `data/runs/sweep/`. Parquet columns used: `entropy`, `log_prob`, `decoded_text`. Gap = `entropy + log_prob`.

# Observations — 2026-03-10d: Annealing Semantics & Escape by Mutation

## Session summary

Integrated annealing experiment data into semantic analysis pipeline. Fixed `anneal.py --check` to not accidentally run experiments for non-probe phases. Fixed `semantic.py` to load runs with non-standard filenames (anneal_*, sched_*) by falling back to parquet column values for L/T/seed.

Ran full semantic analysis across all 40 seed=42 runs (standard grid + 4 annealing runs). The annealing runs reveal a new escape mechanism and confirm the suppressed-dynamics regime at an unexpected scale.

## Finding 1: Escape by semantic mutation (period-doubling route to chaos)

**The L=16 pre-seeded Star Wars run escapes not by jumping out of the attractor but by mutating it from within.**

The escape sequence (steps 9260-9600):
1. Steps 0-9262: Pure " Star Wars" cycle (H alternating 0.034/0.607)
2. Step 9263: First mutation — " Star Wars **2000**" (year inserted, H=2.096)
3. Steps 9263-9310: Cycling " Star Wars 2000" and " Star Wars 2001" — incrementing within the attractor
4. Step ~9380: "Star Wars: The Old Republic" — expanded to a longer franchise name
5. Step ~9392: "The Seventh Planet" — drifting out of Star Wars namespace
6. Step ~9400: Fabricated YouTube URL
7. Step ~9433: Code block, then biochemistry, then free generation

**The model tunnels out by increasing the attractor's period.** "Star Wars" (period 2) → "Star Wars 2000" (period 6) → "Star Wars: The Old Republic" (period ~8) → escape. Each mutation adds tokens to the cycle, diluting the mutual-prediction lock between " Star" and " Wars" until the pattern can no longer self-reinforce.

This is analogous to period-doubling as a route to chaos in dynamical systems. The attractor doesn't break — it complexifies until it can't sustain itself.

**Only one Star Wars block in the entire 100k run.** Once escaped, the system never returns. At L=16, the 8-copy lock-in is right at the threshold — one successful mutation is enough to break the cycle permanently.

```bash
# Reproduce
python generate.py --context-length 16 --temperature 0.60 --seed 42 \
  --num-tokens 100000 --prefill-text " Star Wars" \
  --model-dir data/model/SmolLM-135M --output-dir data/runs \
  --device cuda --run-name anneal_L016_stuck_S42
```

## Finding 2: L=16 pre-seeded has β=2.926 — highest in entire dataset

Heaps' law exponent β=2.926 (R²=0.840) for the L=16 Star Wars run. For comparison:
- Previous maximum: β=1.277 (L=192 T=0.70, a boundary run that escapes mid-way)
- Deep collapse: β=0.17-0.38
- Rich dynamics: β=0.75-0.85
- Escape events: β>1.0

β approaching 3 means vocabulary is *explosively* accelerating, but the R²=0.840 indicates this isn't a clean power law — the run has two distinct phases (stuck then escaped) that fight the single-β fit. The low R² is itself diagnostic: it fingerprints a run with a structural phase transition.

The repetition onset profile confirms: `█▇▃▃▂▃▄▃▃▄▃▅` — starts maximally repetitive, drops, then oscillates. Not monotonic in either direction.

## Finding 3: L=16 pre-seeded is dynamically equivalent to L=256 suppressed zone

The L=16 pre-seeded Star Wars run (T=0.60) has:
- Coherence: mean 0.451, std 0.234
- Vocabulary: TTR 0.1748, 10087 unique words

Compare L=256 T=0.70 (natural suppressed zone):
- Coherence: mean 0.448, std 0.220
- Vocabulary: TTR 0.1330, 5949 unique words

Nearly identical coherence dynamics despite 16x difference in context length and different temperatures. The pre-seeded attractor at L=16 creates the same phenomenology as the natural suppressed zone at L=256 — structure present but slow mixing, intermittent escape attempts, high coherence variance.

This suggests the suppressed dynamics regime is defined by the *ratio* of basin depth to thermal energy, not by absolute L or T values. A shallow basin (L=16, 8 copies) at low temperature (T=0.60) behaves like a deep basin (L=256, natural collapse) at moderate temperature (T=0.70).

## Finding 4: L=128 T=0.60 "temperature" eigenstate

L=128 at T=0.60 contributes 6518 of 7148 total "temperature" hits across the full grid. The run is trapped in: "look at the temperature and the air temperature and the temperature and look at the temperature..."

Neighbor token frequencies confirm total lock-in: 'temperature'(32594), 'the'(26106), 'and'(26071), 'at'(26049), 'look'(26045).

This is an eigenstate where the *word* "temperature" is part of the attractor content — the model found a fixed point that literally contains the concept of temperature measurement. The attractor is about looking at temperature, and what it does is look at temperature.

## Finding 5: Annealing runs confirm the full attractor landscape

With 15 collapsed runs now visible in a single analysis:

| Run | Attractor content | Period |
|-----|------------------|--------|
| L=64 T=0.50 | "the generator is a generator" | ~5 |
| L=64 T=0.60 | `print(".") print(".")` / "it's a plate" | ~3-4 |
| L=128 T=0.50 | "the Weimar Republic was a time where" | ~7 |
| L=128 T=0.60 | "look at the temperature and..." | ~6 |
| L=160 T=0.50 | "place to visit" / "Doctor of Dental Surgery" | ~5 |
| L=176 T=0.50 | "not getting enough sleep... can include not getting enough sleep" | ~8 |
| L=192 T=0.50 | "the election was televised from the White House" | ~10 |
| L=192 T=0.60 | "is turned off before turning on the torch" | ~8 |
| L=208 T=0.50 | "the number of children and the number of..." | ~6 |
| L=224 T=0.50 | "is a book about the history of" | ~7 |
| L=256 T=0.50 | "a young woman, a young man, a young..." | ~4 |
| L=256 T=0.60 | "Star Wars" / "the man was not allowed to leave" | 2 / ~8 |
| L=256 T=0.70 | "29xz. 29yz. 29zz." / "find the population mean" | 3 / ~5 |
| L=256 T=0.80 | "# # #" / "cells, cells, cells," | 1-3 |
| anneal L=256 control | "Star Wars" | 2 |

Every collapsed run has unique content. Period generally shortens with L (L=192: period ~10; L=256: period 2-4), consistent with prior observations.

## Methodological fixes

### anneal.py --check now safe for all phases
Previously, `--check` only worked for `probes` phase. For other phases (tier1, tier2, tier5), it fell through to `run_batch()` and started running experiments. Fixed: `--check` now shows status table (same as `--dry-run`) for non-probe phases.

### semantic.py loads non-standard run names
`load_run()` previously required filenames matching `L{L}_T{T}_S{seed}` pattern. Runs with names like `anneal_L016_stuck_S42` were silently dropped. Fixed: falls back to parquet column values (`context_length`, `temperature`) when filename doesn't match, extracts seed from `S{seed}` substring.

Default run discovery now includes `anneal_*.parquet` and `sched_*.parquet` in addition to `L*.parquet`.

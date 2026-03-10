# Observations — 2026-03-10b: Pre-seeded basin escape probes

## Context

First test of the annealing hypothesis using pre-seeded context (new `--prefill-text` in generate.py). Instead of forking checkpoints, we fill the context with repeated attractor tokens and observe whether the system escapes at different L values.

## Finding 1: Pre-seeded basins are much deeper than BOS-measured T_escape predicts

**Hysteresis is real.** T_escape from BOS (building up from noise) does NOT predict the temperature needed to escape a pre-existing attractor.

| Probe | L | T | Prefill | Ent (last 1k) | Comp (last 1k) | Status |
|---|---|---|---|---|---|---|
| probe_L256 | 256 | 0.60 | " Star Wars" | 0.013 | 0.012 | STUCK |
| probe_L064_T060 | 64 | 0.60 | " Star Wars" | 0.043 | 0.012 | STUCK |
| probe_L064_T080 | 64 | 0.80 | " Star Wars" | 0.043 | 0.012 | STUCK |
| probe_L064_T100 | 64 | 1.00 | " Star Wars" | 3.734 | 0.503 | ESCAPE |
| probe_L032_T060 | 32 | 0.60 | " Star Wars" | 0.103 | 0.012 | STUCK |
| probe_young_L064 | 64 | 0.60 | " young" | 1.301 | 0.118 | ESCAPE |

From BOS, T_escape(64) ≈ 0.55. But pre-seeded " Star Wars" at L=64 survives T=0.60 and T=0.80 — only breaking at T=1.00. The basin is at least 0.4T units deeper when pre-loaded vs when building from noise.

Meanwhile " young" (single-token repeat) escapes trivially at L=64/T=0.60. The 2-token mutual-prediction cycle is a qualitatively different kind of attractor.

**Reproduction:**
```bash
python generate.py --context-length 64 --temperature 0.60 --seed 42 \
  --num-tokens 5000 --prefill-text " Star Wars" --run-name probe_L064_T060 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

python generate.py --context-length 64 --temperature 0.60 --seed 42 \
  --num-tokens 5000 --prefill-text " young" --run-name probe_young_L064 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda
```

## Finding 2: Basin depth is a smooth function of L, with a sharp transition

Probing L=2,4,8,16,32,64 at T=0.60 with " Star Wars" prefill reveals a clean gradient from escaped to locked:

| L | Copies of cycle | Ent (last 1k) | Comp (last 1k) | Top tokens | Status |
|---|---|---|---|---|---|
| 2 | 1 | 5.01 | 0.497 | '0':111 ' the':66 '\n':62 | ESCAPE |
| 4 | 2 | 4.40 | 0.521 | ' the':64 '\n':49 ' of':48 | ESCAPE |
| 8 | 4 | 3.55 | 0.530 | ',':51 '\n':50 '.':44 | ESCAPE |
| 16 | 8 | 0.32 | 0.012 | ' Star':500 ' Wars':500 | STUCK |
| 32 | 16 | 0.10 | 0.012 | ' Star':500 ' Wars':500 | STUCK |
| 64 | 32 | 0.04 | 0.012 | ' Star':500 ' Wars':500 | STUCK |

**The transition is between L=8 and L=16** — between 4 and 8 copies of the 2-token cycle. Below L=8, the model's training priors overwhelm the in-context pattern. Above L=16, the pattern has enough reinforcement to lock in completely.

Key observations:
- **Entropy decreases smoothly** from L=2 (5.01) through L=8 (3.55), then drops abruptly to L=16 (0.32). Not a gradual fade — a cliff.
- **Compressibility is bimodal**: ~0.5 for escaped runs (normal language), 0.012 for stuck runs (perfect 2-token cycle). No intermediate regime.
- **Escape destinations are generic**: function words ("the", "of", commas, newlines). The model returns to its natural prior, no competing attractor emerges.
- **At L=2** (exactly one copy of the cycle), the model is essentially memoryless — entropy 5.01 is close to unconstrained generation.
- **Entropy still decreases from L=2→8** even though all three escape: more copies = more in-context evidence for the pattern, even if insufficient to lock in. The model is "trying" to repeat but can't fully commit.

**Reproduction:**
```bash
for L in 2 4 8 16 32 64; do
  python generate.py --context-length $L --temperature 0.60 --seed 42 \
    --num-tokens 5000 --prefill-text " Star Wars" \
    --run-name "probe_L$(printf '%03d' $L)_T060" \
    --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda
done
```

## Finding 3: Single-token vs multi-token attractor depth

" young" (1-token repeat) escapes at L=64/T=0.60 where " Star Wars" (2-token cycle) does not.

This is counterintuitive — a tighter, simpler cycle is *less* sticky. The explanation lies in the reinforcement mechanism:

- **2-token cycle (" Star Wars")**: Each position uniquely predicts the next. " Star" → " Wars" and " Wars" → " Star" are both near-deterministic given the context. The two tokens form a closed mutual-prediction loop.
- **1-token repeat (" young")**: " young" must predict " young", but in training data " young" is followed by many different things ("man", "woman", "people", "er", etc.). The in-context evidence for " young"→" young" competes against a broad distribution of learned continuations.

The key variable is **mutual information between cycle positions**, not cycle length per se. A cycle where each token has only one plausible successor (given the context) is deeper than one where each token has many natural continuations.

This predicts that:
- Named entities (" Star Wars", " New York") should form deep basins (low-entropy bigrams in training)
- Common words (" young", " disease") should form shallow basins (high-entropy continuations in training)
- Rare/invented words (" bromula") might be intermediate (no strong priors either way)

## Implications for annealing experiment design

1. **L-reduction to cycle length is insufficient** for " Star Wars" — even L=16 (8 copies) is stuck. Need L≤8 (≤4 copies) for escape at T=0.60.
2. **T-escalation needed for deep multi-token attractors** — T=1.00 escapes at L=64 where T=0.80 does not.
3. **The interesting experiment is the L-titration curve itself.** The shape of basin depth as a function of L/cycle_length is potentially universal — does the transition always occur at ~4-8 copies? Does this ratio hold for longer cycles? Across models?
4. **Hysteresis magnitude is measurable.** T_escape(from BOS) vs T_escape(from attractor) at each L quantifies how much deeper the basin is once you're in it.
5. **The gradual entropy decrease from L=2→8** suggests the basin has walls, not a cliff. The model gradually comes under the attractor's influence as copies increase, even in the escaped regime.

## Open questions

- Where exactly is the L=8→16 transition? L=10, 12, 14 would pinpoint it, but the shape matters more than the exact boundary.
- Does the transition ratio (~4-8 copies) hold for longer cycles (3-token, 4-token)?
- Does it hold across models? If the ratio is model-independent, it reveals something about in-context learning generally.
- For the full annealing cycle: use L=4-8 for escape phase, not L=64. The schedule should be calibrated to the actual escape boundary.
- Can a hill-climb controller find the escape L automatically? Start at L=256, observe stuck, binary-search downward until escape detected, then ramp back up.

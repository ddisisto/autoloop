# Annealing Experiment Design

## Core Question

Can L-reduction escape a stuck attractor mid-run, and what determines where the system goes next?

## Background

The regime map shows T_escape(L) — the temperature above which a BOS-started run avoids collapse:

| L | T_escape (from BOS) |
|---|---|
| 64 | ~0.55 |
| 128 | ~0.57 |
| 192 | ~0.67 |
| 256 | ~0.87 |
| 512 | ~0.90 |

The key insight: reducing L is equivalent to shallowing the attractor basin. A system stuck at L=256/T=0.60 (deep below T_escape=0.87) could escape if L is reduced to 64 (where T=0.60 is *above* T_escape=0.55).

**Pre-seeding replaces checkpoint forking.** A context filled with 128x " Star Wars" is functionally identical to the stuck attractor at step 31984+ of L=256/T=0.60/S=42 — the attractor is self-reinforcing, so the context *is* the state.

## Known Attractors (prefill candidates)

From the sweep data, these are documented collapse patterns:

| Attractor | Condition | Tokens | Dominance | Cycle |
|---|---|---|---|---|
| " Star Wars" | L=256, T=0.60 | 2 | — | 2-token cycle |
| "young" | L=256, T=0.50 | 1 | 21.2% | "a young man, a young woman" (8-token) |
| "disease" | L=208, T=0.50 | 1 | 24.7% | "disease of the disease of the" (3-5 token) |
| "piston" | L=512, T=0.90 | 1 | 268 occ | "blow a little farther away from the bottom of the piston" x267 |
| "torch" | L=192, T=0.60 | 1 | 9.6% | 3-8 token cycles |

Single-token attractors dominate (91% of collapse). Multi-token attractors like " Star Wars" (2 tokens) are rarer — the subword boundary slightly dilutes self-reinforcement.

## Probe Results (Phase 0 — complete)

Probes revealed that the original Tier 1-2 design was miscalibrated. Key findings:

1. **Hysteresis is massive.** " Star Wars" at L=64/T=0.60 stays STUCK (ent=0.04). Even T=0.80 at L=64 stays stuck. Only T=1.00 escapes. BOS-measured T_escape is useless for pre-seeded basins.
2. **L-titration works, but the threshold is much lower than expected.** L=8 escapes, L=16 does not. The lock-in requires ~4-8 copies of the cycle in context.
3. **Single-token repeats (" young") are shallow** — escape trivially at L=64/T=0.60.
4. **Escape destinations are generic** (function words), not competing attractors.

See [observations-2026-03-10b.md](observations-2026-03-10b.md) for full data.

## Experiment Tiers (recalibrated)

### Tier 1 — Escape Threshold (fixed L, prefill " Star Wars")

Now calibrated to the actual escape boundary: L=8 escapes, L=16 doesn't. Probes confirmed L=64 at T=0.60 and T=0.80 are stuck.

| Run name | L | T | Prefill | Expected | Rationale |
|---|---|---|---|---|---|
| `anneal_L256_control` | 256 | 0.60 | " Star Wars" | Stuck | Probe confirmed |
| `anneal_L016_stuck` | 16 | 0.60 | " Star Wars" | Stuck | Probe confirmed (8 copies locks in) |
| `anneal_L008_escape` | 8 | 0.60 | " Star Wars" | Escape | Probe confirmed (4 copies escapes) |
| `anneal_L004_escape` | 4 | 0.60 | " Star Wars" | Escape | Probe confirmed (2 copies escapes) |

Boundary probes (pinpoint the transition):

| Run name | L | T | Prefill | Expected | Rationale |
|---|---|---|---|---|---|
| `anneal_L010_probe` | 10 | 0.60 | " Star Wars" | Unknown | 5 copies — between escape and lock-in |
| `anneal_L012_probe` | 12 | 0.60 | " Star Wars" | Unknown | 6 copies |
| `anneal_L014_probe` | 14 | 0.60 | " Star Wars" | Unknown | 7 copies |

N=100,000 for confirmed conditions. Seeds: 42, 123, 7 (3 seeds each).
N=10,000 for boundary probes. Seed 42 only.

### Tier 2 — Return Dynamics (the full annealing cycle)

**Recalibrated: escape phase uses L=8, not L=64.** L=64 stays stuck — the escape phase must drop below the lock-in threshold.

After escaping at L=8, what happens when L is restored to 256? Three possibilities:
- **Re-collapse to " Star Wars"**: the attractor still exists and recaptures
- **Find a different attractor**: path-dependence confirmed
- **Sustain rich dynamics**: escape is durable

| Run name | Schedule | Prefill | Key question |
|---|---|---|---|
| `anneal_cycle_short` | 20k:L256:T0.60, 10k:L8:T0.60, 70k:L256:T0.60 | " Star Wars" | Does it re-collapse after escape? |
| `anneal_cycle_long` | 20k:L256:T0.60, 30k:L8:T0.60, 50k:L256:T0.60 | " Star Wars" | Does longer escape phase change the return? |
| `anneal_cycle_brief` | 20k:L256:T0.60, 3k:L8:T0.60, 77k:L256:T0.60 | " Star Wars" | Minimum escape duration? |
| `anneal_no_return` | 20k:L256:T0.60, 80k:L8:T0.60 | " Star Wars" | What does sustained L=8/T=0.60 look like? |

Seeds: 42, 123, 7 (12 runs per seed set).

**This is the flagship question.** If the system returns to L=256 and finds a *different* attractor (or stays dynamic), that's direct evidence that trajectory through parameter space matters — the foundation for schedule-based control.

### Tier 3 — Attractor Generality

Probes already showed single-token (" young") escapes at L=64 while multi-token (" Star Wars") does not. Expand to map the lock-in threshold for different attractor types.

| Run name | L | T | Prefill | Expected |
|---|---|---|---|---|
| `anneal_young_L256` | 256 | 0.60 | " young" | Stuck (control) |
| `anneal_young_L016` | 16 | 0.60 | " young" | Escape? (1-token, shallower basin) |
| `anneal_young_L032` | 32 | 0.60 | " young" | Borderline? |
| `anneal_disease_L064` | 64 | 0.60 | " disease" | Escape? |
| `anneal_disease_L256` | 256 | 0.60 | " disease" | Stuck (control) |
| `anneal_newyork_L064` | 64 | 0.60 | " New York" | Stuck? (2-token named entity, deep like Star Wars?) |
| `anneal_newyork_L008` | 8 | 0.60 | " New York" | Escape? |

Seeds: 42 only. Tests the hypothesis that basin depth depends on mutual information between cycle positions, not cycle length.

### Tier 4 — Path Dependence (gradual vs sudden)

Does the *shape* of L-reduction matter, or only the minimum L reached? Recalibrated: ramp must pass through L=8 to escape.

| Run name | Schedule | Prefill |
|---|---|---|
| `anneal_sudden` | 20k:L256:T0.60, 10k:L8:T0.60, 70k:L256:T0.60 | " Star Wars" |
| `anneal_gradual` | 20k:L256:T0.60, 5k:L64:T0.60, 5k:L16:T0.60, 5k:L8:T0.60, 5k:L16:T0.60, 5k:L64:T0.60, 55k:L256:T0.60 | " Star Wars" |
| `anneal_ramp_down` | 20k:L256:T0.60, 5k:L64:T0.60, 5k:L16:T0.60, 70k:L8:T0.60 | " Star Wars" |

Seeds: 42 only (3 runs). Exploratory.

**What this reveals:** Whether the gradual ramp through L=64→16→8 produces different escape dynamics than a sudden drop to L=8. We know L=16 is stuck and L=8 escapes — does the system show any pre-escape loosening at L=16 during the ramp?

### Tier 5 — T-annealing comparison

Does raising T achieve the same escape as reducing L? Compares the two actuators directly.

| Run name | Schedule | Prefill |
|---|---|---|
| `anneal_T_escape` | 20k:L256:T0.60, 10k:L256:T1.00, 70k:L256:T0.60 | " Star Wars" |
| `anneal_L_escape` | 20k:L256:T0.60, 10k:L8:T0.60, 70k:L256:T0.60 | " Star Wars" |
| `anneal_both` | 20k:L256:T0.60, 10k:L8:T1.00, 70k:L256:T0.60 | " Star Wars" |

Seeds: 42 only (3 runs).

**What this reveals:** T-annealing vs L-annealing — do they escape to the same region, or do the two actuators reach different parts of phase space? If L-reduction escapes to richer dynamics than T-increase, that validates L as the structural actuator (changes attractor landscape) vs T as noise injection (random perturbation).

## Execution Plan

### Phase 0 — Probe (minutes, not hours)

Quick feasibility check before committing to full runs. 5k tokens each, seed 42 only.

| Probe | L | T | Prefill | Question |
|---|---|---|---|---|
| `probe_L256` | 256 | 0.60 | " Star Wars" | Control: confirm attractor holds |
| `probe_L064_T060` | 64 | 0.60 | " Star Wars" | Does L=64 escape at T=0.60? |
| `probe_L064_T080` | 64 | 0.80 | " Star Wars" | Escape with more thermal noise? |
| `probe_L064_T100` | 64 | 1.00 | " Star Wars" | Escape with strong thermal noise? |
| `probe_L032_T060` | 32 | 0.60 | " Star Wars" | Even shallower basin? |
| `probe_young_L064` | 64 | 0.60 | " young" | Single-token attractor stickier? |

**Decision gate:** Check entropy/compressibility of last 1k tokens in each probe.
- If probe escapes: proceed to Tier 1 full runs at that condition.
- If nothing escapes at T=0.60: the pre-seeded basin is deeper than BOS-measured T_escape. Escalate: try T=0.80, T=1.00, L=32. This itself is a finding — hysteresis between basin entry and exit.
- If only high-T escapes: recalibrate Tier 1-5 temperatures upward.

### Phase A — Tier 1 (calibrated by probes)

Full 100k runs at conditions the probes identified as interesting. 3 seeds each. Focus on the escape boundary — the narrowest parameter change that flips stuck→escape.

### Phase B — Tier 2 (return dynamics)

The flagship. Run after Phase A confirms escape conditions. Schedule runs with the calibrated escape parameters.

### Phase C — Tiers 3-5

Contingent on Phase A/B results being interesting. Expand seed counts on whatever shows the strongest signal.

### Time budget

- Phase 0: ~6 probes x 5k tokens ≈ 10-15 min GPU
- Phase A: ~12 runs x 100k tokens ≈ 2 hours
- Phase B: ~12 runs x 100k tokens ≈ 2 hours
- Phase C: ~14 runs contingent ≈ 2-3 hours

## Analysis Plan

For each run, extract:
- **Escape step**: first step where trailing-64 compressibility drops below 0.9 (or whatever threshold separates stuck from dynamic)
- **Escape destination**: what regime does it enter? (entropy, compressibility, decorrelation lag in final 20k tokens)
- **Re-collapse**: does the system return to the original attractor after L is restored?
- **Attractor identity**: what tokens dominate the final segment? Same attractor, different one, or no dominant pattern?

Key comparisons:
- T_escape(L) from prefill vs from BOS — hysteresis test
- Return-segment statistics vs fixed-parameter baselines at same (L, T) — path-dependence test
- Cross-attractor escape thresholds — basin depth test

## Commands

```bash
# --- Phase 0: Probes (5k tokens each) ---

# Control: confirm attractor holds at L=256
python generate.py --context-length 256 --temperature 0.60 --seed 42 \
  --num-tokens 5000 --prefill-text " Star Wars" --run-name probe_L256 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

# Does L=64 escape at T=0.60?
python generate.py --context-length 64 --temperature 0.60 --seed 42 \
  --num-tokens 5000 --prefill-text " Star Wars" --run-name probe_L064_T060 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

# Escape with more thermal noise?
python generate.py --context-length 64 --temperature 0.80 --seed 42 \
  --num-tokens 5000 --prefill-text " Star Wars" --run-name probe_L064_T080 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

# Escape with strong thermal noise?
python generate.py --context-length 64 --temperature 1.00 --seed 42 \
  --num-tokens 5000 --prefill-text " Star Wars" --run-name probe_L064_T100 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

# Even shallower basin?
python generate.py --context-length 32 --temperature 0.60 --seed 42 \
  --num-tokens 5000 --prefill-text " Star Wars" --run-name probe_L032_T060 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

# Single-token attractor stickier?
python generate.py --context-length 64 --temperature 0.60 --seed 42 \
  --num-tokens 5000 --prefill-text " young" --run-name probe_young_L064 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

# --- Phase A: Tier 1 full runs (recalibrated from probes) ---

# Use anneal.py for batch execution:
python anneal.py tier1              # 15 runs: 3 seeds x 4 L-values + 3 boundary probes
python anneal.py tier1 --dry-run    # preview

# Example individual runs:
python generate.py --context-length 8 --temperature 0.60 --seed 42 \
  --num-tokens 100000 --prefill-text " Star Wars" --run-name anneal_L008_escape_S42 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

# --- Phase B: Tier 2 schedule runs (escape at L=8, not L=64) ---

python anneal.py tier2              # 12 runs: 3 seeds x 4 schedule variants
python anneal.py tier2 --dry-run    # preview

# Example:
python generate.py --schedule "20000:L256:T0.60,10000:L8:T0.60,70000:L256:T0.60" \
  --seed 42 --prefill-text " Star Wars" --run-name anneal_cycle_short_S42 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

# --- Phase B: Tier 5 T vs L comparison ---

python anneal.py tier5              # 3 runs: T-escape, L-escape, both
python anneal.py tier5 --dry-run    # preview
```

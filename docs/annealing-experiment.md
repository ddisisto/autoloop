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

## Experiment Tiers

### Tier 1 — Escape Threshold (fixed L, prefill " Star Wars")

Does L-reduction escape work, and does T_escape measured from BOS match T_escape from a pre-existing attractor?

| Run name | L | T | Prefill | Expected | Rationale |
|---|---|---|---|---|---|
| `anneal_L256_control` | 256 | 0.60 | " Star Wars" | Stuck | T=0.60 << T_escape(256)=0.87 |
| `anneal_L192_stuck` | 192 | 0.60 | " Star Wars" | Stuck | T=0.60 < T_escape(192)=0.67 |
| `anneal_L128_border` | 128 | 0.60 | " Star Wars" | Borderline | T=0.60 ≈ T_escape(128)=0.57 |
| `anneal_L064_escape` | 64 | 0.60 | " Star Wars" | Escape | T=0.60 > T_escape(64)=0.55 |

**What this reveals:** Whether the escape boundary from a pre-existing attractor matches the collapse boundary from BOS start. If they differ, that's evidence for hysteresis — the system needs more energy to *leave* a basin than to *avoid* it.

N=100,000 for all. Seeds: 42, 123, 7 (3 seeds x 4 conditions = 12 runs).

### Tier 2 — Return Dynamics (the full annealing cycle)

After escaping at L=64, what happens when L is restored to 256? Three possibilities:
- **Re-collapse to " Star Wars"**: the attractor still exists and recaptures
- **Find a different attractor**: path-dependence confirmed
- **Sustain rich dynamics**: escape is durable

| Run name | Schedule | Prefill | Key question |
|---|---|---|---|
| `anneal_cycle_short` | 20k:L256:T0.60, 10k:L64:T0.60, 70k:L256:T0.60 | " Star Wars" | Does it re-collapse after escape? |
| `anneal_cycle_long` | 20k:L256:T0.60, 30k:L64:T0.60, 50k:L256:T0.60 | " Star Wars" | Does longer escape phase change the return? |
| `anneal_cycle_brief` | 20k:L256:T0.60, 3k:L64:T0.60, 77k:L256:T0.60 | " Star Wars" | Minimum escape duration? |
| `anneal_no_return` | 20k:L256:T0.60, 80k:L64:T0.60 | " Star Wars" | What does sustained L=64/T=0.60 look like? |

Seeds: 42, 123, 7 (12 runs per seed set).

**This is the flagship question.** If the system returns to L=256 and finds a *different* attractor (or stays dynamic), that's direct evidence that trajectory through parameter space matters — the foundation for schedule-based control.

### Tier 3 — Attractor Generality

Is escape a property of L-reduction generally, or specific to " Star Wars"? Single-token attractors may be stickier (perfect KV cache self-reinforcement) than multi-token ones.

| Run name | L | T | Prefill | Expected |
|---|---|---|---|---|
| `anneal_young_L064` | 64 | 0.60 | " young" | Escape (single-token, may be stickier) |
| `anneal_young_L256` | 256 | 0.60 | " young" | Stuck (control) |
| `anneal_disease_L064` | 64 | 0.60 | " disease" | Escape |
| `anneal_disease_L256` | 256 | 0.60 | " disease" | Stuck (control) |

Seeds: 42 only (8 runs). Expand if results are interesting.

**What this reveals:** Whether different attractors have different escape thresholds. If " young" (single-token, from deeper collapse at T=0.50) resists L=64 escape while " Star Wars" (2-token) doesn't, that tells us about basin depth vs token structure.

### Tier 4 — Path Dependence (gradual vs sudden)

Does the *shape* of L-reduction matter, or only the minimum L reached?

| Run name | Schedule | Prefill |
|---|---|---|
| `anneal_sudden` | 20k:L256:T0.60, 10k:L64:T0.60, 70k:L256:T0.60 | " Star Wars" |
| `anneal_gradual` | 20k:L256:T0.60, 5k:L192:T0.60, 5k:L128:T0.60, 5k:L64:T0.60, 5k:L128:T0.60, 5k:L192:T0.60, 55k:L256:T0.60 | " Star Wars" |
| `anneal_ramp_down` | 20k:L256:T0.60, 5k:L192:T0.60, 5k:L128:T0.60, 70k:L64:T0.60 | " Star Wars" |

Seeds: 42 only (3 runs). This is exploratory.

**What this reveals:** Whether gradual L-reduction through the escape boundary produces different dynamics than a sudden drop. If the system escapes at L=128 during the ramp (where T=0.60 ≈ T_escape) but not at L=192, we see the boundary in real time.

### Tier 5 — T-annealing comparison

For completeness: does raising T achieve the same escape as reducing L? This compares the two actuators directly.

| Run name | Schedule | Prefill |
|---|---|---|
| `anneal_T_escape` | 20k:L256:T0.60, 10k:L256:T1.00, 70k:L256:T0.60 | " Star Wars" |
| `anneal_L_escape` | 20k:L256:T0.60, 10k:L64:T0.60, 70k:L256:T0.60 | " Star Wars" |
| `anneal_both` | 20k:L256:T0.60, 10k:L64:T1.00, 70k:L256:T0.60 | " Star Wars" |

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

# --- Phase A: Tier 1 full runs (calibrate L/T from probe results) ---

python generate.py --context-length 256 --temperature 0.60 --seed 42 \
  --num-tokens 100000 --prefill-text " Star Wars" --run-name anneal_L256_control_S42 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

python generate.py --context-length 64 --temperature 0.60 --seed 42 \
  --num-tokens 100000 --prefill-text " Star Wars" --run-name anneal_L064_escape_S42 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

# --- Phase B: Tier 2 schedule runs ---

python generate.py --schedule "20000:L256:T0.60,10000:L64:T0.60,70000:L256:T0.60" \
  --seed 42 --prefill-text " Star Wars" --run-name anneal_cycle_short_S42 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda

# --- Phase B: Tier 5 T vs L comparison ---

python generate.py --schedule "20000:L256:T0.60,10000:L256:T1.00,70000:L256:T0.60" \
  --seed 42 --prefill-text " Star Wars" --run-name anneal_T_escape_S42 \
  --model-dir data/model/SmolLM-135M --output-dir data/runs --device cuda
```

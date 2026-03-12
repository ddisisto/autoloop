# Basin Mapping — Phase 1: L=8 Pilot

Tracking doc for the L=8 pilot survey. Goal: tune capture/escape detection, run to saturation, seed the basin catalogue.

## Status

**Current:** Capture detection needs tuning before production runs.

## Detection Tuning

### Capture detection (COOLING → CAPTURED)

**Implementation:** Two independent gates, either sufficient.

```
CAPTURE_BETA_THRESHOLD = 0.40   # β < 0.40 = vocabulary death (d=3.4, zero false positives)
CAPTURE_COMP_THRESHOLD = 0.45   # comp_W64 < 0.45 = highly repetitive output
MIN_COOLING_SEGMENTS = 5        # minimum segments before capture can fire
```

Gate 1 (β): Clean collapse wall from regime analysis (50 runs). Only evaluated when n_words >= 50 (β=0.0 means insufficient data, not collapse).

Gate 2 (compressibility): Catches short-cycle basins where β is unmeasurable. At L=8 T=0.10, the "1.1.1." basin has entropy=4.53 (model's output distribution is broad) and β=0.0 (no words >1 char), but comp=0.391 (perfectly repetitive). Entropy cannot detect this basin; compressibility can.

**Previous approaches (superseded):**
1. Entropy-delta stability (N consecutive low-delta readings) — fired on normal dynamics (ent=3.83, β=0.80)
2. β + entropy<1.0 gate — missed short-cycle basins where entropy stays high (~4.5) despite total lock-in

### Escape detection (HEATING → TRANSIT)

**Current implementation:** Entropy rises > ESCAPE_ENTROPY_RISE above basin floor, OR T reaches T_max ceiling.

```
ESCAPE_ENTROPY_RISE = 1.0       # absolute rise above basin floor
```

**Status:** Works for deep basins. The T_max ceiling fallback prevents deadlock for shallow captures where the rise threshold is unreachable. The real fix is better capture detection — if we only capture real basins, the escape threshold becomes reachable.

### Deeper basin detection (HEATING → CAPTURED)

**Implementation:** Same dual gates as capture (β or comp), plus entropy < 80% of current basin floor. During heating, if the system falls into a deeper basin than the one it escaped, re-capture it.

**Status:** Untested in practice. Shares the same validated gates as primary capture.

### Re-capture avoidance

**Problem:** After escaping a basin and flushing context, cooling may land in the same basin again. No mechanism to avoid this.

**Tracked in:** `SurveyState.consecutive_known` — counts consecutive captures matching existing types. Currently tracked but not acted on.

**Options:**
- Raise T_min for the next cooling attempt after consecutive re-captures
- Skip capture (don't record) if same type as previous, continue cooling deeper
- After N consecutive known, advance to next L (treat as saturation signal)

## Run Log

*No production runs yet. Tuning in progress.*

| Run | Steps | Captures | Novel | Known | Notes |
|-----|-------|----------|-------|-------|-------|
| smoke test (2k steps) | 2,000 | 1 | 1 | 0 | Premature capture at ent=3.83; proved cycle mechanics work |
| smoke test (5k steps) | 5,000 | 1 | 1 | 0 | Same premature capture issue |

## Basin Taxonomy

*Will be populated as captures accumulate.*

| type_id | W* | entropy_floor | β | representative_text (truncated) | hit_count | notes |
|---------|-----|---------------|------|--------------------------------|-----------|-------|

## Observations

### Smoke test (L=8, seed=42, T_min=0.10, T_max=0.70)

- T ramps down correctly from 0.70: 0.66 → 0.63 → 0.60 → 0.57 → 0.54 → 0.51 → 0.49
- First capture at step 458, T=0.489, ent=3.83, β=0.80 — premature (not a real basin)
- HEATING ramps up correctly, but entropy stays in 3.7–3.9 range (no escape signal)
- T_max ceiling fix triggers escape when T reaches 0.70
- Full cycle completes but capture quality is poor

### Key finding: basins can have high entropy

At L=8 T=0.10, the "1.1.1." basin (token 30 "." and token 33 "1" alternating) has entropy=4.53 — the model's output distribution is broad across many tokens, but at T=0.10 the slight probability edge for "." and "1" always wins. Compressibility (0.391) detects this; entropy (4.53) and β (0.0, unmeasurable) do not. This motivated the dual-gate approach: β catches word-level basins, comp catches token-level cycles.

## Commands

```bash
# Smoke test (quick cycle check)
loop survey --seed 42 -L 8 --total-steps 2000

# Pilot run (once thresholds are tuned)
loop survey --seed 42 -L 8 --total-steps 100000

# Check captures
loop index build
loop index query --type survey

# Inspect text at a specific step range
loop grep "pattern" --type survey -C 30
```

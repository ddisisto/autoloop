# Basin Mapping — Phase 1: L=8 Pilot

Tracking doc for the L=8 pilot survey. Goal: tune capture/escape detection, run to saturation, seed the basin catalogue.

## Status

**Current:** Capture detection needs tuning before production runs.

## Detection Tuning

### Capture detection (COOLING → CAPTURED)

**Implementation:** β + entropy gate, derived from regime analysis (50 sweep runs, F-stat + Cohen's d).

```
CAPTURE_BETA_THRESHOLD = 0.40   # β < 0.40 is a clean collapse wall (d=3.4, zero false positives)
CAPTURE_ENTROPY_THRESHOLD = 1.0 # entropy < 1.0 separates real basins from suppressed dynamics
MIN_COOLING_SEGMENTS = 5        # minimum segments before capture can fire
```

Both conditions must hold. β is the primary discriminator — it's the single cleanest regime gate in the dataset. Entropy confirms we're at a real basin floor, not just slow dynamics.

**Previous approach (superseded):** entropy-delta stability (N consecutive low-delta readings). This fired on normal dynamics — captured at ent=3.83, β=0.80 in the smoke test, which is rich dynamics not a basin.

### Escape detection (HEATING → TRANSIT)

**Current implementation:** Entropy rises > ESCAPE_ENTROPY_RISE above basin floor, OR T reaches T_max ceiling.

```
ESCAPE_ENTROPY_RISE = 1.0       # absolute rise above basin floor
```

**Status:** Works for deep basins. The T_max ceiling fallback prevents deadlock for shallow captures where the rise threshold is unreachable. The real fix is better capture detection — if we only capture real basins, the escape threshold becomes reachable.

### Deeper basin detection (HEATING → CAPTURED)

**Implementation:** Same β + entropy gates as capture, plus entropy < 80% of current basin floor. During heating, if the system falls into a deeper basin than the one it escaped, re-capture it.

**Status:** Untested in practice. Now shares the same validated gates as primary capture.

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

### Key question (resolved)

What does a real basin look like at L=8? From sweep data, L=8 at T=0.50 should collapse to ent < 1.0. The smoke test's ent=3.83 capture was nowhere near collapse — stability detection triggered on normal fluctuations. **Fixed:** replaced with β < 0.40 + entropy < 1.0 gates from regime analysis (50 runs, zero false positives).

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

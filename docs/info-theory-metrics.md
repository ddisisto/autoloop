# Information-theoretic metric augmentation

**Date:** 2026-03-19
**Status:** In progress

## Motivation

The sister project (`../framework`) grounds autoregressive self-play dynamics in information theory — rate-distortion, statistical complexity, compressive novelty. This project's existing metrics (entropy, compressibility, Heaps' beta) are already information-theoretic in substance but linguistic in framing. Adding metrics that align with Framework's terminology and fill known discriminative gaps.

Key gap: compressibility detects collapse but doesn't discriminate within rich dynamics. Framework predicts this — it's an algorithmic-level metric. Organisational/semantic-level measures are needed.

## Approach

**Augment, don't migrate.** Existing metrics have calibrated baselines across ~70 runs. New metrics are added alongside, computed over existing parquet data, and compared before changing any detection logic.

### New metrics (in implementation order)

1. **Surprisal (step-level derived metric)**
   - Definition: `-log_prob` — model's surprise at the token it sampled
   - Already in parquet as `log_prob`; register as a proper step-level derived metric
   - Enables the entropy-surprisal gap (see below)

2. **Entropy-surprisal gap (step-level derived metric)**
   - Definition: `entropy - (-log_prob)` = `entropy + log_prob`
   - Positive: sampled token more predictable than distribution average (reinforcing)
   - Negative: sampled token more surprising than average (potentially enriching)
   - Maps directly to Framework's "compressive novelty" — the governing variable at all timescales
   - Zero additional compute; derived from existing parquet columns

3. **Lempel-Ziv complexity (window-level metric)**
   - Definition: count of distinct phrases in LZ76 parsing of token sequence
   - More principled than gzip ratio: directly measures process complexity
   - Same sliding-window architecture as compressibility
   - Operates on token IDs (integer sequence), not byte strings — avoids encoding artifacts

4. **Block entropy scaling (window-level metric, future)**
   - Entropy rate as function of block length k=1,2,4,...,K
   - Excess entropy (intercept extrapolation) approximates statistical complexity
   - Deferred until metrics 1-3 are evaluated

### Sensor integration

- Add `surprisal_gap` to `SensorReading` — mean of `entropy + log_prob` over trailing window
- Available to controllers as a direct "compressive novelty" signal

### Run-level metrics

- `surprisal_gap_mean`, `surprisal_gap_std` — scalar summaries
- `lz_complexity` at each window size — scalar summaries matching compressibility pattern

### Impact on existing work

- **No existing observations invalidated.** Regime boundaries, basin taxonomy, escape dynamics are empirical phenomena independent of metric choice.
- **Basin detection unchanged initially.** New metrics computed alongside; evaluate discriminative power before modifying detection pipeline.
- **Cache version bump required** when adding new window metrics to analysis cache.
- **Observations entry needed** after computing new metrics over existing runs: compare old vs new metric behaviour across known regimes.

## Cross-reference

- `../framework/` — information-theoretic framework for autoregressive dynamics
- Framework Chapter 4 (autoregressive loop) maps directly to this project's experimental setup
- Framework's "compressive novelty" = this project's entropy-surprisal gap
- Framework's "statistical complexity" ≈ this project's future block entropy scaling metric

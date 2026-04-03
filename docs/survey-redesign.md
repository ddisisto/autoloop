# Survey Redesign: Surprisal-First Basin Detection

**Date:** 2026-03-20 (terminology updated 2026-03-30)
**Status:** Proposed — replaces current multi-gate approach

## Motivation

The survey protocol has gone through several gate iterations: gzip compressibility → LZ complexity → LZ + beta. Each change improved discrimination but added parameters and windowing decisions that don't scale cleanly across L values. Meanwhile, the data consistently shows that one signal dominates: **surprisal** (= -log_prob).

Surprisal is the model's own assessment of how predictable each token was. A basin *is* the system predicting itself — surprisal measures that directly. It requires:

- No windowing parameter (per-token quantity)
- No L-dependent threshold (the model's certainty is L-independent)
- No additional computation (log_prob is already computed in the forward pass)
- No normalization or baseline correction

Empirically, at L=8 and L=12 capture points, mean segment surprisal is < 0.05 (median) while non-capture segments sit at ~0.53. The separation is orders of magnitude, not percentages.

## Connection to Framework

The sister project (`../framework`) identifies the enrichment fraction — the proportion of tokens doing novel compressive work — as the central continuous variable governing autoregressive dynamics. A token is **enriching** when it surfaces genuinely new structure; **stabilising** when it maintains trajectories, continues grammar, or restates what the context already implies. The entropy-surprisal gap (= entropy + log_prob) is the direct measure: positive gap means the sampled token was more predictable than the distribution average (stabilising); negative means more surprising (enriching).

Stabilising tokens are not pathological — they are essential scaffolding. Coherent generation requires a majority of stabilising tokens providing structure while a minority of enriching tokens inject novelty. The pathological endpoint is **degeneration**: the enrichment fraction drops to zero, surprisal approaches zero, and the model produces exactly what it expects, adding length without adding any structure. This is Framework's "accumulation" — and it is exactly what a basin is.

The capture gate, then, is not detecting "repetitive output" or "a shift to stabilising dynamics." Stabilising dynamics are normal and necessary. The gate detects **degeneration** — the extreme where the system has lost all enriching tokens and entered complete self-prediction. This is a sharper definition than regime-boundary detection, and it is what we should build around.

## Proposed Design

### Gate

Single gate: **mean surprisal over trailing segment < threshold**.

- Segment size: 2×L tokens (unchanged — gives 10+ context rotations at MIN_COOLING_SEGMENTS=5)
- Threshold: calibrate from existing L=8 and L=12 capture data. Preliminary analysis suggests ~0.50 nats captures 93% of current captures. Tighter thresholds (0.10-0.20) would catch only deep basins.
- Sensitivity parameter: the threshold controls the depth at which basins are captured. Lower threshold = only deep basins. Higher = shallower template attractors too. This is a single, interpretable knob.

### What's dropped from gate logic

- LZ complexity gate (CAPTURE_LZ_THRESHOLD)
- Beta gate (CAPTURE_BETA_THRESHOLD)
- comp_W64 in SensorReading (for gating purposes)
- lz_W64 in SensorReading (for gating purposes)

### What's kept

- **Embeddings** for clustering (orthogonal to gating — the model's representation of basin content)
- **COOLING→HEATING→TRANSIT state machine** (the temperature sweep protocol works)
- **LZ spectrum, gzip comp spectrum, entropy, beta, gap** as *descriptors* stored per capture. These characterize the basin but don't control the gate
- **Enrichment fraction** as a per-capture descriptor — what proportion of tokens in the capture segment were enriching? Deep basins will have enrichment ≈ 0; shallower template attractors may retain a small enriching minority
- **Escape detection** via entropy rise (unchanged)

### SensorReading simplification

The sensor reading still computes all metrics for logging and characterization, but the gate decision uses only:

```
captured = mean_surprisal_segment < CAPTURE_SURPRISAL_THRESHOLD
```

This is a degeneration detector: it fires when the model's own uncertainty about its next token drops below the threshold across a sustained segment. The lower the threshold, the deeper the degeneration required to trigger capture.

### Capture record

Each capture stores:
- Embedding (576-dim) — for clustering
- Context text (L tokens) — for inspection
- Attractor text (W* tokens) — for broader context
- Surprisal mean/std over capture segment — the gate signal
- Entropy, beta, gap, LZ spectrum, comp spectrum — descriptors
- T, L, step, seed — conditions

## Validation Plan

Before implementing, confirm with existing data:

1. **Coverage check:** For all 825 existing captures (L=8 + L=12), compute mean segment surprisal. What threshold captures 95%? 99%? Are the missed captures genuine basins or borderline?

2. **False positive check:** Sample non-capture segments from survey runs. At the proposed threshold, how often would the gate fire incorrectly?

3. **L-independence:** Compare the surprisal distribution at capture for L=8 vs L=12. If the threshold needs to differ, surprisal isn't truly L-independent and the design needs revision.

4. **Edge cases:** The 7% of L=12 captures with surprisal > 0.50 — examine their content. Are they genuine basins caught by beta/LZ that surprisal would miss? Or are they borderline captures that shouldn't have been caught?

## Scaling Considerations

### Performance

Surprisal is free — no additional computation. The current LZ gate requires O(W²) computation per segment; removing it makes the survey loop faster, especially at larger L where W scales.

### Clustering pipeline

No change needed. Clustering uses embeddings, not gate metrics. The persistent clustering roadmap (recluster command, saved labels) proceeds independently.

### Cross-L analysis

Surprisal-gated captures should produce more comparable basins across L values, since the gate criterion is the same regardless of L. This simplifies cross-L correspondence.

### Future: block entropy scaling

Framework predicts that statistical complexity correlates with capability thresholds. Block entropy scaling (entropy rate as function of block length) approximates statistical complexity from the token stream. This could be computed per basin as a characterization metric — not a gate, but a way to rank basins by structural richness. Combined with the enrichment fraction, this would give a two-dimensional characterization: how degenerate is the basin (enrichment fraction), and how structurally complex was the content it settled on (block entropy scaling). Deferred until the surprisal gate is validated and the persistent catalogue is in place.

## Migration

1. Validate surprisal gate on existing L=8 + L=12 data (no new runs)
2. Implement in survey.py: replace multi-gate with surprisal-only
3. Re-run L=12 with surprisal gate as confirmation
4. Proceed up L-ladder (L=16, 24, ...) with the new gate
5. Existing L=8 data (captured with gzip/LZ gates) remains valid — the captures are real basins regardless of which gate found them. No need to re-collect.

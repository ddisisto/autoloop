# autoloop — Project Brief

**Working paper title:** *Attractor Dynamics in Closed-Loop Autoregressive Generation*

**Status:** Pre-registration draft
**Date:** March 2026

---

## 1. Motivation

When an autoregressive language model generates tokens indefinitely — with old tokens rolling out of a fixed-length context window as new tokens enter — the system becomes a discrete stochastic dynamical system operating on its own output. After the original prompt exits the window, the model is conditioning entirely on self-generated text. The resulting dynamics are a property of the model itself: its learned weights, the context length, and the sampling temperature.

Despite the simplicity of this setup, the resulting system has received almost no formal study. Basic questions remain open: What attractors does the system converge to? Is there a crossover between repetitive collapse and incoherent noise? How sharp is that crossover, and what structural modes persist near it?

This project systematically characterizes the dynamical landscape of autoregressive self-play — first at fixed temperature, then under controlled temperature ramps — to map the phase structure and identify attractor dynamics intrinsic to the model.

## 2. Research Questions

**RQ1 — Phase structure.** Is there a crossover between an ordered regime (repetitive collapse) and a disordered regime (incoherent generation)? How sharp is this crossover, and where does it occur in temperature space?

**RQ2 — Attractor characterization.** What structural modes does the system visit at and near the crossover? What are their dwell time distributions and transition dynamics?

**RQ3 — Path dependence.** Does the system's behavior at a given temperature depend on how that temperature was reached? Do temperature ramps in opposite directions reveal hysteresis, indicating multistability or genuine attractor structure?

## 3. Experimental Design

### 3.1 System Architecture

The core system consists of:

- A frozen pretrained language model (inference only, no weight updates)
- A sliding context window of configurable length $L$
- Per-step measurement of softmax entropy and sampled token log-probability
- A comprehensive logging pipeline capturing tokens, decoded text, and per-step measurements

All windowed and derived measurements (output compressibility, smoothed entropy, spectral analysis) are computed post-hoc from the logged data, not during generation. This keeps the collection pipeline minimal and parameter-free, and allows exploration of different analysis choices without rerunning generation.

### 3.2 Model

**SmolLM-135M (HuggingFace).** Small enough for rapid iteration and dense sweeps, while being recent and well-trained for its size class.

### 3.3 Initialization: Self-Consistent Pre-Fill

The context window is pre-filled before the experiment begins, avoiding a non-stationary growth phase where the effective context length would be changing.

**Pre-fill procedure:**
1. Begin with BOS token
2. Generate $L$ tokens from the model at the experimental temperature $T$
3. The experimental phase begins at step $L + 1$, once the context window is full

This produces a self-consistent initial state: the context consists entirely of model-generated output at the same temperature used for the experiment, eliminating both the prompt-washout transient and any adjustment transient from a mismatched pre-fill temperature. The pre-fill is fully determined by model, temperature, and PRNG seed.

### 3.4 Measurements

**Logged per step during generation:**

- **Token ID** — the sampled token
- **Decoded text** — the token decoded to its text representation (cached to avoid needing the model/tokenizer at analysis time)
- **Softmax entropy** — Shannon entropy of the model's output probability distribution: $H_t = -\sum_i p_i \log p_i$. A property of the model's internal state (how uncertain it is about the next token), computationally free since logits are already available, deterministic (independent of which token is sampled)
- **Log-probability of sampled token** — how likely the actually-chosen token was under the distribution. The gap between distribution entropy and negative log-probability indicates how "typical" each sample was
- **Temperature** — current $T_t$ (fixed per run in Phase 1, varying in Phase 2)
- **EOS flag** — whether the sampled token was the end-of-sequence token

**Derived in post-hoc analysis (not computed during generation):**

- Output compressibility over sliding windows (gzip compression ratio of decoded text). Primary window $W = L$ (one context-length of output — the natural memory horizon of the system). Secondary diagnostic window $W = L/4$ for detecting local degenerate collapse
- Smoothed entropy (EMA or other filters at various timescales)
- Autocorrelation and spectral density of entropy and compressibility time series
- 2D phase portraits: softmax entropy vs. output compressibility
- Dwell time distributions in identified regimes
- EOS rate statistics: mean inter-EOS interval, variability, trends
- Transfer functions: $T \to C$ and $T \to H$ curves
- Hysteresis plots: system state vs. instantaneous $T$ for opposing ramp directions (Phase 2)

### 3.5 Generation Loop

```
# Pre-fill
initialize context with BOS token
for t = 1 to L:
    forward pass: context → logits
    sample token from softmax(logits / T)
    append token to context
    log: {t, token_id, decoded_text, H_t, log_p_t, T, eos_flag, phase="prefill"}

# Experimental self-play
for t = L+1 to L+N:
    forward pass: context → logits
    compute softmax entropy H_t from logits
    sample token from softmax(logits / T_t)
    compute log-probability of sampled token
    append token to context, truncate to length L
    log: {t, token_id, decoded_text, H_t, log_p_t, T_t, eos_flag, phase="experiment"}
```

No KV cache across steps — the sliding window drops the oldest token each step, invalidating the entire positional structure of any cached keys/values. A full forward pass over the L-token context is required each step. At 135M parameters this is fast even at L=1024.

EOS tokens are retained in the generation stream.

Sampling uses pure temperature scaling only — no top-k or nucleus (top-p) filtering — to keep a single, interpretable control parameter.

### 3.6 Independent Variables

**Phase 1 (fixed temperature):**

| Variable | Values | Notes |
|---|---|---|
| Context length $L$ | 64, 256, 1024 | Pilot grid; may be extended |
| Temperature $T$ | 0.5, 1.0, 1.5 | Pilot grid; may be extended |
| PRNG seed | 42, 43, 44 | Three replicates per condition |

**Phase 2 (temperature ramps):** Design determined by Phase 1 results. Anticipated variables are ramp direction (increasing / decreasing), ramp rate, and temperature range (spanning the crossover region identified in Phase 1).

### 3.7 Run Parameters

- **Tokens per run:** 100,000 post-pre-fill tokens (absolute, not scaled by $L$)
- **Run length rationale:** Absolute token count gives approximately linear cost scaling with $L$. At $L = 1024$, this provides ~98 full context turnovers; at $L = 64$, ~1,562. If L=1024 runs show non-stationarity at 100k tokens, run length may be extended for that condition.
- **Stationarity assessment:** Each run is split into 5 non-overlapping blocks of 20k tokens. Per-block means and variances of entropy and compressibility are compared. Classification: *stationary* (no trend in block statistics), *transient* (monotonic drift — system still equilibrating), or *structured non-stationarity* (block statistics fluctuate without trend — suggests mode-switching on timescales comparable to run length). Visual inspection plus simple linear regression for drift in Phase 0; may be formalized in Phase 1.
- **Replicates:** 3 seeds per condition in the pilot grid (27 total conditions)
- **Storage estimate:** ~25 MB total for the pilot grid (100k decoded tokens per run ≈ few hundred KB)

## 4. Implementation Plan

### Phase 0 — Infrastructure and Pilot

Build core generation loop. Run the pilot grid.

**Deliverables:**
- Working generation loop with pre-fill procedure
- Per-step logging: token ID, decoded text, softmax entropy, log-probability, temperature, EOS flag
- Logging pipeline (Parquet)
- End-to-end validation on SmolLM-135M
- Complete pilot grid: 3 temperatures × 3 context lengths × 3 seeds = 27 runs
- Post-hoc analysis tooling: compressibility computation, phase portrait plotting
- Per-run timing data
- Assessment: which axes (T, L) show the most interesting variation, and where to focus

### Phase 1 — Fixed-Temperature Characterization

Expand the pilot grid based on Phase 0 findings. The exact grid is determined by Phase 0 results, but the anticipated direction is denser temperature spacing through any identified crossover region, and potentially additional $L$ values if context length effects are substantial.

**Deliverables:**
- Transfer functions: $T \to C$ and $T \to H$ curves at each $L$
- 2D phase portraits: softmax entropy vs. compressibility at each condition
- Identification and characterization of crossover region(s)
- Stationarity assessment across conditions
- Complete fixed-temperature phase map

### Phase 2 — Temperature Ramp Experiments

Controlled temperature ramps through the crossover region identified in Phase 1.

**Design (informed by Phase 1 results):**
- Linear ramps from high $T$ to low $T$, and low $T$ to high $T$, through the crossover region
- Multiple ramp rates
- Multiple seeds per condition

**Deliverables:**
- Hysteresis plots: system state (compressibility, softmax entropy) vs. instantaneous $T$ for opposing ramp directions
- Evidence for or against path dependence and multistability
- Characterization of transition dynamics: does the system snap between modes, or drift smoothly?
- Comparison to fixed-temperature baselines: do ramp experiments visit states not seen in fixed-$T$ runs?

### Phase 3 — Targeted Extensions

Limited extensions guided by findings from Phases 1–2. Possible directions include extended runs at conditions showing slow-timescale dynamics, finer parameter spacing in regions of particular interest, or pre-fill temperature sensitivity checks.

### Phase 4 — Writing

Paper draft, figures, supplementary materials, code and data release.

The paper presents two main results: the fixed-temperature phase map (Phase 1) and the ramp / hysteresis experiments (Phase 2), with Phase 3 extensions as supplementary material if applicable.

## 5. Expected Contributions

1. **A systematic characterization** of attractor behavior, repetition dynamics, and phase structure in autoregressive self-play as a function of sampling temperature and context length.
2. **Hysteresis and path-dependence analysis** revealing multistability and attractor structure through temperature ramp experiments.
3. **Dual-signal methodology** demonstrating the complementary value of output compressibility (information content of the decoded token stream) and softmax entropy (model uncertainty) for characterizing self-play dynamics.
4. **A reusable experimental framework** (`autoloop`) for studying closed-loop autoregressive dynamics.

## 6. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Degenerate collapse dominates at most temperatures | Low-Medium | Dense sweep will locate any non-trivial region if it exists; documenting the collapse boundary is itself a result |
| No clear crossover — gradual, featureless transition | Low | Crossover sharpness and its characterization are themselves a finding; methodology contribution stands regardless |
| Compressibility too noisy at small $W$ | Low-Medium | W is an analysis parameter, not baked into collection; can explore multiple window sizes post-hoc |
| Pilot grid misses interesting region | Low | Pilot spans a wide range; Phase 1 fills in based on findings |

## 7. Open Questions for Resolution During Phase 0

- Per-run timing across the pilot grid → confirms 100k tokens is feasible at all conditions
- Compressor choice: gzip vs. zlib (quick empirical comparison)
- Whether additional $L$ values or $T$ values are needed beyond the pilot grid
- Appropriate smoothing timescales for entropy (EMA alpha or alternative filters)

## 8. Future Directions

Natural extensions beyond this study include:

- **Model scaling:** Replication at 1B, 8B, and beyond
- **Architecture comparison:** Models with different training regimes, tokenizers, or architectural choices
- **Extended context lengths:** Larger $L$ values, potentially up to native context window limits
- **Entropy-regulated control:** A closed-loop controller that adjusts temperature to maintain a target output compressibility, enabling precise navigation of the dynamical landscape
- **Alternative measurements:** Embedding-space analysis, n-gram statistics, alternative compressors
- **Initial condition sensitivity:** Systematic variation of pre-fill temperature, alternative pre-fill strategies
- **Alternative sampling strategies:** Top-k, top-p (nucleus) sampling as additional control variables (this study uses pure temperature scaling only)

## 9. Tools and Dependencies

- **Model:** SmolLM-135M (HuggingFace), pre-downloaded
- **Model loading / tokenization:** HuggingFace Transformers
- **Inference / generation loop:** PyTorch (custom)
- **Logging:** Parquet per run
- **Analysis:** NumPy, SciPy, matplotlib, Python gzip module
- **Compute:** Local GPU (single consumer GPU sufficient for 135M model)
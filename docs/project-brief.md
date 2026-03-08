# autoloop — Project Brief

**Working paper title:** *Multi-Scale Complexity Control in Closed-Loop Autoregressive Generation*

**Status:** Active — Phase 0 pilot in progress
**Date:** March 2026

---

## 1. Motivation

When an autoregressive language model generates tokens indefinitely — with old tokens rolling out of a fixed-length context window as new tokens enter — the system becomes a discrete stochastic dynamical system operating on its own output. After the original prompt exits the window, the model is conditioning entirely on self-generated text. The resulting dynamics are a property of the model itself: its learned weights, the context length, and the sampling temperature.

Despite the simplicity of this setup, the resulting system has received almost no formal study. Basic questions remain open: What attractors does the system converge to? Is there a crossover between repetitive collapse and incoherent noise? How sharp is that crossover, and what structural modes persist near it?

**Central insight (emerging from Phase 0 pilot):** Output compressibility measured at different window sizes probes structure at different scales. Short-range compression (W << L) detects local repetitive collapse. Long-range compression (W ≥ L) detects coherence across the model's full memory horizon. The ratio between these signals defines a multi-dimensional characterization of the output regime — and offers a natural sensor array for closed-loop control.

**Second insight (emerging from cross-L comparison):** Temperature and context length act as orthogonal actuators on fundamentally different axes. Temperature controls the *noise floor* — randomness of each individual sample. Context length controls the *memory horizon* — how much self-generated history the model conditions on, and therefore how deep and sticky attractor basins are. At T=0.50, L=64 shows persistent escape episodes from collapse attractors while L=256 locks in permanently. At T=1.00, L=256 shifts the system to a different operating point (higher entropy, lower compressibility) without causing collapse. This orthogonality suggests a two-actuator controller: T for fast corrections and L for structural regime selection — including using L-reduction as an escape mechanism from stuck attractors, analogous to simulated annealing but for memory depth.

This project systematically characterizes the dynamical landscape of autoregressive self-play, develops multi-scale compression as a diagnostic framework, and builds toward closed-loop complexity control with joint T+L actuation.

## 2. Research Questions

**RQ1 — Phase structure.** Is there a crossover between an ordered regime (repetitive collapse) and a disordered regime (incoherent generation)? How sharp is this crossover, and where does it occur in temperature space?

**RQ2 — Attractor characterization.** What structural modes does the system visit at and near the crossover? What are their dwell time distributions and transition dynamics?

**RQ3 — Multi-scale structure.** How do compressibility signals at different window sizes (W = L/4, L, 2L, 4L) relate to each other? Where do they decouple — i.e., where does local structure exist without global repetition? This decoupling zone is the regime of maximal complexity.

**RQ4 — Context length as control parameter.** How does context length L modulate attractor basin depth, crossover location, and multi-scale structure? Is there a critical L above which collapse attractors become inescapable at a given T? (Emerging findings: L dramatically deepens collapse attractors; the escape-to-lock transition occurs between L=64 and L=256 at T=0.50; at T=1.00, L shifts the operating point without causing collapse.)

**RQ5 — Path dependence.** Does the system's behavior at a given temperature depend on how that temperature was reached? Do temperature ramps in opposite directions reveal hysteresis, indicating multistability or genuine attractor structure?

**RQ6 — Closed-loop complexity control.** Can temperature and context length be dynamically adjusted using multi-scale compression as a feedback signal to maintain the system in a target complexity regime? T for fast corrections (raise when short-range compression drops, lower when entropy spikes), L for structural regime selection (shorten to escape stuck attractors, lengthen to deepen coherence). Can this sustain coherent generation that neither collapses nor becomes noise?

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
- **Temperature** — current $T_t$ (fixed per run in Phase 1, varying in Phase 2+)
- **EOS flag** — whether the sampled token was the end-of-sequence token

**Derived in post-hoc analysis (not computed during generation):**

- Output compressibility over sliding windows at multiple scales:
  - $W = L/4$ — local collapse diagnostic
  - $W = L$ — primary signal (one context-length, the model's memory horizon)
  - $W = 2L$, $W = 4L$ — emergent structure beyond the context window (Phase 1+)
- Multi-scale compression ratio: relationship between compressibility at different W values
- Smoothed entropy (EMA or other filters at various timescales)
- Autocorrelation and spectral density of entropy and compressibility time series
- 2D phase portraits: softmax entropy vs. output compressibility
- Distribution evolution: per-time-block violin plots of entropy and compressibility
- Dwell time distributions in identified regimes
- EOS rate statistics: trailing EOS rate over sliding window, mean inter-EOS interval, variability, trends. EOS is a model-internal coherence signal — the model's assessment of sequence completeness — distinct from entropy (local uncertainty) and compressibility (observer-assessed structure). Emerging finding: EOS rate peaks at T=1.0 (the rich-dynamics regime), not at collapse or noise; L dramatically suppresses EOS in the collapse regime.
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

Checkpoints saved every 1k steps (context tensor, RNG state, accumulated records). Runs can be interrupted and resumed, or extended to longer N by re-running with higher --num-tokens.

### 3.6 Independent Variables

**Phase 0–1 (fixed temperature):**

| Variable | Values | Notes |
|---|---|---|
| Context length $L$ | 64, 128, 192, 256 | Revised: densify 64–256 range where attractor depth transition occurs; 1024 deprioritized |
| Temperature $T$ | 0.5, 1.0, 1.5 | Pilot grid; Phase 1 densifies crossover region (~0.6–0.9) |
| PRNG seed | 42, 123, 7 | Three replicates per condition |

**Phase 2 (temperature ramps):** Design determined by Phase 1 results. Anticipated variables are ramp direction (increasing / decreasing), ramp rate, and temperature range (spanning the crossover region identified in Phase 1).

**Phase 3 (closed-loop control):** Temperature adjusted dynamically using multi-scale compression feedback. Design determined by Phase 1–2 results.

### 3.7 Run Parameters

- **Tokens per run:** 100,000 post-pre-fill tokens (absolute, not scaled by $L$)
- **Run length rationale:** Absolute token count gives approximately linear cost scaling with $L$. At $L = 1024$, this provides ~98 full context turnovers; at $L = 64$, ~1,562. If L=1024 runs show non-stationarity at 100k tokens, run length may be extended for that condition.
- **Stationarity assessment:** Each run is split into 5 non-overlapping blocks of 20k tokens. Per-block means and variances of entropy and compressibility are compared. Classification: *stationary* (no trend in block statistics), *transient* (monotonic drift — system still equilibrating), or *structured non-stationarity* (block statistics fluctuate without trend — suggests mode-switching on timescales comparable to run length). Visual inspection plus simple linear regression for drift in Phase 0; may be formalized in Phase 1.
- **Replicates:** 3 seeds per condition in the pilot grid (27 total conditions)
- **Storage estimate:** ~25 MB total for the pilot grid (100k decoded tokens per run ≈ few hundred KB)

## 4. Implementation Plan

### Phase 0 — Infrastructure and Pilot ← CURRENT

Build core generation loop. Run the pilot grid. Develop analysis and visualization tooling.

**Deliverables:**
- Working generation loop with pre-fill procedure and checkpoint/resume ✓
- Per-step logging to Parquet with JSON metadata sidecar ✓
- End-to-end validation on SmolLM-135M ✓
- Post-hoc analysis: compressibility (W=L, W=L/4), stationarity assessment ✓
- Visualization: entropy/compressibility time series, phase portraits, temporal phase portraits, violin distribution plots ✓
- Plot reproduction script ✓
- Pilot grid runs: 3 temperatures × 3 context lengths × 3 seeds = 27 runs (in progress)
- Assessment: which axes (T, L) show the most interesting variation, and where to focus

**Emerging findings (see observations.md):**
- Three distinct regimes at L=64: collapse (T=0.5), rich dynamics (T=1.0), noise (T=1.5)
- Context length L dramatically deepens collapse attractor (L=256 locks in, L=64 keeps escaping at T=0.5)
- At T=1.0, L=256 shifts operating point (higher entropy, lower compressibility) without collapse — L and T are orthogonal
- Crossover region likely T=0.6–0.9
- W=L/4 and W=L compressibility decouple in the interesting regime
- L-transition between 64–256 is where attractor escape/lock behavior changes — densify here, not at 1024

### Phase 1 — Fixed-Temperature Characterization + Multi-Scale Analysis

Expand the pilot grid based on Phase 0 findings. Dense temperature spacing through crossover region. Introduce compressibility at W=2L and W=4L to probe emergent structure beyond the context window.

**Deliverables:**
- Transfer functions: $T \to C$ and $T \to H$ curves at each $L$
- Multi-scale compression profiles: how compressibility at W=L/4, L, 2L, 4L varies with T
- Identification of the "decoupling zone" where local structure exists without global repetition
- Complete fixed-temperature phase map
- Correlation length analysis: at what scale does structure disappear, and how does this depend on T and L?

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

### Phase 3 — Closed-Loop Complexity Control

Use multi-scale compression as a feedback signal to dynamically control temperature and context length, maintaining the system in a target complexity regime.

**Design (informed by Phase 1–2 results):**
- Controller that monitors compressibility at multiple scales (W=L/4, L, and potentially longer) plus trailing EOS rate as a third sensor (model-internal coherence signal)
- **T actuator (fast):** Raises T when short-range compression drops too low (approaching loop collapse); lowers T when long-range compression gets too high (approaching noise)
- **L actuator (structural):** Shortens L to escape stuck attractors (reducing memory depth makes attractor basins shallower); lengthens L to deepen coherence when the system is in a productive regime. L-reduction is an escape mechanism analogous to simulated annealing for memory depth
- Target: sustain the system in the "decoupling zone" — local structure without global repetition
- Explore different target points in (short-compression, long-compression) space
- Explore the edge-of-chaos overlap zones visible in phase portraits where collapse and rich-dynamics regions share (entropy, compressibility) space

**Deliverables:**
- Working closed-loop controller with joint T+L actuation
- Demonstration that dynamic T+L control can maintain the system at criticality
- Characterization of achievable operating points in multi-scale compression space
- Comparison to fixed-T/fixed-L baselines: does controlled generation produce qualitatively different output?
- Analysis of whether structure can be maintained at scales beyond L (emergent long-range order from limited-memory system)
- Characterization of L-escape dynamics: how quickly does shortening L allow escape from collapse attractors?

### Phase 4 — Targeted Extensions

Limited extensions guided by findings from Phases 1–3. Possible directions include:
- Extended runs at conditions showing slow-timescale dynamics
- Finer parameter spacing in regions of particular interest
- Pre-fill temperature sensitivity checks
- Model scaling experiments (1B, 8B)

### Phase 5 — Writing

Paper draft, figures, supplementary materials, code and data release.

## 5. Expected Contributions

1. **A systematic characterization** of attractor behavior, repetition dynamics, and phase structure in autoregressive self-play as a function of sampling temperature and context length.
2. **Multi-scale compression as a diagnostic framework** for characterizing the complexity of autoregressive output — probing structure at local (W << L), context-scale (W = L), and emergent (W >> L) scales.
3. **Hysteresis and path-dependence analysis** revealing multistability and attractor structure through temperature ramp experiments.
4. **Closed-loop complexity control** demonstrating that joint temperature and context-length modulation ("memory-depth annealing"), guided by multi-scale compression and EOS-rate feedback, can maintain a language model at a target operating point in complexity space.
5. **A reusable experimental framework** (`autoloop`) for studying closed-loop autoregressive dynamics.

## 6. Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Degenerate collapse dominates at most temperatures | Low-Medium | Dense sweep will locate any non-trivial region if it exists; documenting the collapse boundary is itself a result |
| No clear crossover — gradual, featureless transition | Low | Crossover sharpness and its characterization are themselves a finding; methodology contribution stands regardless |
| Compressibility too noisy at small $W$ | Low-Medium | W is an analysis parameter, not baked into collection; can explore multiple window sizes post-hoc |
| Pilot grid misses interesting region | Low | Pilot spans a wide range; Phase 1 fills in based on findings |
| Multi-scale compression signals are too correlated to decouple | Medium | Even high correlation with scale-dependent offsets is informative; the existence or absence of a decoupling zone is itself a finding |
| Closed-loop controller is unstable or oscillates | Medium | Phase 1–2 provide the static characterization needed to design a stable controller; can start with conservative gain and simple proportional control |

## 7. Open Questions for Resolution During Phase 0

- ~~Per-run timing across the pilot grid~~ → confirmed: ~24–38 tok/s at L=64, scales with L
- Compressor choice: gzip vs. zlib (quick empirical comparison)
- ~~Whether additional $L$ values or $T$ values are needed~~ → resolved: densify L=64–256, deprioritize L=1024; T crossover sweep in Phase 1
- Appropriate smoothing timescales for entropy (EMA alpha or alternative filters)
- How does EOS rate behave at intermediate L (128, 192)? Does the L-suppression curve match the attractor depth curve?

## 8. Future Directions

Natural extensions beyond this study include:

- **Model scaling:** Replication at 1B, 8B, and beyond — do phase boundaries and multi-scale structure persist?
- **Architecture comparison:** Models with different training regimes, tokenizers, or architectural choices
- **Extended context lengths:** Larger $L$ values, potentially up to native context window limits
- **Alternative measurements:** Embedding-space analysis, n-gram statistics, alternative compressors
- **Initial condition sensitivity:** Systematic variation of pre-fill temperature, alternative pre-fill strategies
- **Alternative sampling strategies:** Top-k, top-p (nucleus) sampling as additional control variables (this study uses pure temperature scaling only)
- **Multi-dimensional control:** Using multiple actuators (T, L, top-k, top-p) simultaneously, guided by multi-scale compression feedback, to navigate a higher-dimensional control space. Joint T+L control is the primary focus of Phase 3; additional actuators are future extensions
- **Practical applications:** Compression-guided decoding strategies for production language model inference

## 9. Tools and Dependencies

- **Model:** SmolLM-135M (HuggingFace), pre-downloaded to `data/model/SmolLM-135M/`
- **Model loading / tokenization:** HuggingFace Transformers
- **Inference / generation loop:** PyTorch (custom), CUDA 12.6
- **Logging:** Parquet per run, JSON metadata sidecar
- **Analysis:** NumPy, SciPy, scikit-learn, matplotlib, Python gzip module
- **Package management:** uv
- **Compute:** Local GPU (GTX 1070, single consumer GPU sufficient for 135M model)
- **Throughput:** ~24–38 tok/s at L=64, ~10.5 tok/s at L=1024

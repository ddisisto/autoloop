# Memory-Depth Annealing: Escaping Collapse Attractors in Autoregressive Self-Play

**Draft post — target: r/MachineLearning [R] or blog**

---

## Hook

When a language model generates tokens into a fixed-size sliding window — conditioning only on its own output — it becomes a dynamical system with attractors. Low temperature leads to repetitive collapse. High temperature leads to noise. The interesting regime is in between.

We've been mapping this landscape and found something we didn't expect: **context length and temperature are orthogonal control axes** with fundamentally different roles. Temperature controls the noise floor (per-token randomness). Context length controls *memory depth* — how much self-generated history the model conditions on, and therefore how deep and sticky attractor basins are.

This means you can escape a collapse attractor not by raising temperature (which adds noise everywhere), but by *shortening the context window* (which makes the attractor basin shallower). Then extend the window again. We call this **memory-depth annealing** — the dual of simulated annealing, applied to the model's memory horizon instead of its sampling noise.

## The setup

- SmolLM-135M generating into a sliding context window of length L
- No prompt — after pre-fill, the model conditions entirely on its own output
- Pure temperature sampling (no top-k/top-p) — one clean control parameter
- 100k tokens per run, measuring softmax entropy and output compressibility (gzip ratio over sliding windows)

## What we see

**Three regimes** at L=64 (phase portrait: entropy vs compressibility):

[L0064_Tmulti_S42_phase.png]

- T=0.50: Collapse attractor — but with escape episodes (the scatter upward)
- T=1.00: Rich dynamics — broad cloud, the model explores a continuum of states
- T=1.50: Stationary noise — tight cluster at high entropy

**Context length changes everything.** Same temperature (T=0.50), different L:

[Lmulti_T0.50_S42_temporal.png]

- L=64: Wanders across phase space for 100k tokens. Keeps escaping and re-entering the collapse basin. Colors (time) are mixed — no temporal ordering.
- L=256: Collapses to bottom-left corner within 15k tokens and *never escapes*. The attractor is permanent.

More context = more self-reinforcing signal = deeper basin. The model gets stuck in its own echoes.

**But at T=1.00, L doesn't cause collapse** — it shifts the operating point:

[Lmulti_T1.00_S42_temporal.png]

L=256 moves the cloud down-left (lower compressibility, slightly different entropy distribution) compared to L=64. L is pulling toward order, but the system resists collapse. It finds a different equilibrium rather than locking in.

## The insight

Temperature and context length are orthogonal actuators:

| | Temperature (T) | Context length (L) |
|---|---|---|
| **What it controls** | Per-token sampling noise | Memory horizon / attractor depth |
| **Timescale** | Instantaneous | Structural |
| **Collapse escape** | Add noise everywhere | Make the basin shallower |
| **Coherence** | Lower noise (risks re-collapse) | Deepen basin (sustain structure) |

This suggests a **dual-actuator controller** for maintaining generation at the edge of chaos:
- **T** for fast corrections — raise when compressibility drops (approaching collapse), lower when entropy spikes
- **L** for structural regime selection — shorten to escape, lengthen to sustain

Reducing L mid-run is an escape mechanism. Simulated annealing temporarily *increases* a parameter (temperature) to escape local minima. Memory-depth annealing temporarily *decreases* a parameter (context length) to make attractor basins shallower. Same principle, dual mechanism.

## Why this matters

1. **Repetition is the #1 failure mode in open-ended generation.** Current mitigations (repetition penalties, frequency penalties) are heuristic patches. Understanding the attractor dynamics could lead to principled solutions.

2. **The multi-scale compression signal is a natural sensor.** Compressibility at different window sizes (W=L/4 vs W=L) probes structure at different scales. Where these signals decouple — local structure without global repetition — is the regime of maximal complexity. This is a measurable, continuous signal, not a binary "is it repeating?"

3. **Dynamic L is cheap.** Unlike temperature, which requires resampling, changing context length just means truncating the window differently. Zero computational overhead.

4. **This reframes autoregressive generation as a control problem.** The model is a plant. T and L are actuators. Multi-scale compressibility is the sensor array. The target is a region in complexity space. Standard control theory applies.

## What's next

- Densifying the L grid (L=64, 128, 192, 256) to map the attractor depth curve — where exactly does the escape-to-lock transition happen?
- Temperature sweeps through the crossover region (T=0.6–0.9)
- Temperature ramp experiments to test for hysteresis (path dependence)
- Building the actual closed-loop controller

Code and data at [repo link]. Early-stage work — feedback welcome.

---

## Notes to self

- Figures need to be self-contained with clear captions
- Could trim to just the two key plots (L=64 phase portrait + cross-L temporal at T=0.50) for a tighter post
- r/ML audience likes: novel framing, real data, clean visuals, connection to practical problems (repetition)
- Might also work for: LessWrong, ML Twitter/Bluesky thread, complexity science forums
- Consider whether to share code/repo at this stage or wait for more data

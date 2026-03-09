# Interaction Topology

Speculative framing that emerged from Phase 0 observations. Not a proposal — a direction.

## The current paradigm

AI interaction is transactional:
- User provides input (prompt)
- Model processes it within fixed parameters (temperature, max tokens, context window)
- Model produces output bounded by EOS/BOS, matching implicit or explicit spec
- Control surface: prompt engineering (input shaping), maybe temperature (usually hidden)
- Reproducibility is a goal; dynamics are invisible

Everything interesting about the generative process is flattened into a black box between input and output boundaries.

## What the instruments show

The autoloop experiments reveal a continuous dynamical system with observable internal state, not a function mapping inputs to outputs. Key findings:

**Multiple interacting control dimensions:**
- T (temperature): per-step noise floor, fast actuator
- L (context length): memory horizon / attractor depth, structural actuator
- W (measurement window): observer resolution — not a generation parameter, but changes what patterns are visible and therefore what feedback is possible

**These dimensions interact non-linearly.** They form a topology:
- T and L are not orthogonal: longer L shifts the collapse boundary upward in T. T_escape(L) is superlinear — L=64 escapes at T≈0.55, L=256 requires T≈0.87
- Four regimes emerge from the T×L interaction: collapse, suppressed dynamics, rich dynamics, noise
- The suppressed zone (e.g. L=256 at T=0.70–0.80) has structure but slow mixing — decorrelation lags of 250+ steps, multi-scale decoupling up to 0.35
- W reveals or hides structure depending on scale — the "instrument" is part of the system

**EOS is an interior signal, not a boundary marker.** At T=1.00 (richest dynamics), EOS fires from the dense center of phase space — the model tries to end when it's most engaged, not when it's stuck or lost. At T=0.50 (collapse), EOS fires during escape attempts from attractors. The meaning of EOS depends on the regime.

## The reframe

What if interaction with a generative model isn't "submit query, receive answer" but navigating a dynamical system together?

- The user isn't providing input to be processed — they're perturbing the system's trajectory through phase space
- The model isn't producing output to a spec — it's evolving along attractors that the control surface shapes
- EOS isn't "I'm done" — it's a coherence signal from the interior of the dynamics
- The "quality" of output isn't about matching a spec — it's about the region of phase space the system occupies

## Control surface

| Actuator | Timescale | What it controls | Status |
|----------|-----------|-----------------|--------|
| T (temperature) | Per-step | Noise floor, escape probability | Characterized (static); schedule mode designed |
| L (context length) | Structural | Memory depth, attractor stickiness | Characterized (static); schedule mode designed |
| W (measurement window) | Observer | What scales of structure are visible/steerable | Analysis tool; not yet a control input |
| Schedule (T,L trajectory) | Multi-step | Path through parameter space | Designed (schedule.py); open question: does path matter? |
| Feedback controller | Closed-loop | Maintain target regime via metric feedback | Designed (Controller protocol); not yet built |
| Embedding-space steering | Directional | Semantic trajectory | Not yet explored |
| User input | Perturbation | Phase-space displacement | Not yet explored |

Each operates on a different timescale. A controller that navigates all of them simultaneously is qualitatively different from "set temperature to 0.7 and hope for the best."

The central open experiment: does the *trajectory* through T×L space matter, or only the endpoint? If a hot-start (T=1.0) followed by cooling produces different attractor basins than a cold-start at the same final T — path-dependence is real, and schedule design becomes a first-class concern.

## What this implies

1. **The phase space is the interface.** Not the text. The text is a projection of a trajectory through a high-dimensional dynamical system. The interesting thing to observe, steer, and interact with is the trajectory itself.

2. **Reproducibility is the wrong goal.** Two runs through the same phase-space region produce different text but similar dynamics. The dynamics are what matter for interaction quality, not token-level reproducibility.

3. **Boundaries are regime-dependent.** Where to "stop" isn't a fixed criterion — it depends on what regime the system is in. An EOS in the rich-dynamics regime means something different from an EOS in the collapse regime.

4. **The observer's resolution matters.** W isn't just an analysis parameter — it determines what feedback is available for control. A controller with W=16 sees different structure than one with W=256. Multi-scale observation isn't optional, it's fundamental.

## Open questions

- **Path-dependence:** Does annealing (hot start → cool) land in different attractor basins than cold start at the same final (T, L)? This is testable now with schedule mode.
- **Human-in-the-loop:** Can a human interact meaningfully with phase-space information, or does this require automated control? The explorer is the testbed.
- **Embedding-space steering:** Is it another actuator, or does it change the topology itself?
- **Generality:** Is this specific to autoregressive text generation, or does it generalize to other generative modalities?
- **Model-dependence:** The staircase of basins at T=0.50, the superlinear T_escape(L) — are these properties of SmolLM-135M specifically, or of autoregressive generation generally? L=512 sweep is probing this edge.

## Connection to autoloop experiments

This framing emerged directly from observing:
- Four-regime structure (collapse / suppressed / rich / noise) and the transitions between them
- T×L interaction: T_escape(L) is superlinear, actuators are coupled not orthogonal
- Non-monotonic L effects (the L=192 anomaly at T=0.50)
- EOS as interior signal at T=1.00 vs boundary signal at T=0.50
- W as measurement resolution that reveals/hides structure at different scales
- The "memory-depth annealing" concept: L-reduction as escape mechanism

All of this was invisible until we built instruments to see it.

## From instruments to controls

The explorer trajectory (see explorer.md) makes the speculative concrete:

| Concept | Instrument (read-only) | Control (read-write) |
|---------|----------------------|---------------------|
| Phase space | Chart axes (entropy × compressibility) | Live trajectory you steer |
| Context window | Text panel showing L tokens | Slider that reshapes the attractor landscape |
| Temperature | Color-coded regime indicator | Per-step noise floor you adjust in real time |
| EOS | Marker on chart, navigation target | Stop condition in an events system |
| Memory depth | L parameter in run metadata | Annealing lever — drag to escape/deepen attractors |
| Measurement scale | W parameter for compressibility | Observer resolution, part of the feedback loop |

The key transition: "jump in the seat" at any point in a recorded run, reconstruct full model state from the parquet (one forward pass of L tokens → KV cache), adjust parameters, generate forward. The parquet is a complete record; the model state is a pure function of the last L tokens.

This means every recorded run is not just data to analyze — it's a waypoint you can return to and branch from. The phase space becomes navigable, not just observable.

## The schedule system as bridge

The schedule system (designed, not yet built) provides the concrete bridge from instruments to controls:

**Open-loop (schedule.py):** Predefined T(t), L(t) trajectories. Test whether path through parameter space matters. First experiment: three paths to the same (T=0.70, L=192) — cold start, hot start with cooldown, annealing with L ramp. If final-segment statistics differ, path-dependence is confirmed.

**Closed-loop (Controller protocol):** Feedback controllers that observe live metrics (entropy, compressibility, EOS rate, decorrelation lag) and adjust T and L to maintain a target regime. The simplest: proportional control on entropy — if entropy drops below threshold, raise T; if it rises above, lower T. More sophisticated: multi-objective control targeting the rich-dynamics region of phase space.

**The key insight from recent data:** The suppressed-dynamics zone (L=256, T=0.70–0.80) suggests that simply raising T isn't sufficient for large L — the system can have structure without dynamics. A controller needs to observe *mixing rate* (decorrelation lag), not just entropy level. This is why the four-regime model matters for control design: you need to distinguish "structured and dynamic" from "structured and stuck."

## Future directions

Embedding-space projections (UMAP/t-SNE of hidden states) as the "true" phase portrait. Trajectories through embedding space, with metric-derived views as interpretable projections. Steering in embedding space = directional perturbation of generation. The topology of the space becomes the interface itself.

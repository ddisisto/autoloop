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

**These dimensions aren't independent.** They form a topology:
- Phase boundaries exist (T~0.70-0.80 is a crossover zone)
- Attractor basins have depth that scales non-monotonically with L (the L=192 anomaly)
- The same system shows qualitatively different dynamics depending on where it sits in T×L space
- W reveals or hides structure depending on scale — the "instrument" is part of the system

**EOS is an interior signal, not a boundary marker.** At T=1.00 (richest dynamics), EOS fires from the dense center of phase space — the model tries to end when it's most engaged, not when it's stuck or lost. At T=0.50 (collapse), EOS fires during escape attempts from attractors. The meaning of EOS depends on the regime.

## The reframe

What if interaction with a generative model isn't "submit query, receive answer" but navigating a dynamical system together?

- The user isn't providing input to be processed — they're perturbing the system's trajectory through phase space
- The model isn't producing output to a spec — it's evolving along attractors that the control surface shapes
- EOS isn't "I'm done" — it's a coherence signal from the interior of the dynamics
- The "quality" of output isn't about matching a spec — it's about the region of phase space the system occupies

## Control surface (speculative)

| Actuator | Timescale | What it controls |
|----------|-----------|-----------------|
| T (temperature) | Per-step | Noise floor, escape probability |
| L (context length) | Structural | Memory depth, attractor stickiness |
| W (measurement window) | Observer | What scales of structure are visible/steerable |
| Embedding-space steering | Directional | Semantic trajectory (not yet explored) |
| User input | Perturbation | Phase-space displacement |

Each operates on a different timescale. A controller that navigates all of them simultaneously is qualitatively different from "set temperature to 0.7 and hope for the best."

## What this implies

1. **The phase space is the interface.** Not the text. The text is a projection of a trajectory through a high-dimensional dynamical system. The interesting thing to observe, steer, and interact with is the trajectory itself.

2. **Reproducibility is the wrong goal.** Two runs through the same phase-space region produce different text but similar dynamics. The dynamics are what matter for interaction quality, not token-level reproducibility.

3. **Boundaries are regime-dependent.** Where to "stop" isn't a fixed criterion — it depends on what regime the system is in. An EOS in the rich-dynamics regime means something different from an EOS in the collapse regime.

4. **The observer's resolution matters.** W isn't just an analysis parameter — it determines what feedback is available for control. A controller with W=16 sees different structure than one with W=256. Multi-scale observation isn't optional, it's fundamental.

## Open questions

- Can a human interact meaningfully with phase-space information, or does this require automated control?
- What does embedding-space steering look like in this framework? Is it another actuator, or does it change the topology itself?
- Is this specific to autoregressive text generation, or does it generalize to other generative modalities?
- The "staircase of basins" at T=0.50 — are these a property of the model, the tokenizer, or the generation process? Do they exist in larger models?

## Connection to autoloop experiments

This framing emerged directly from observing:
- Three-regime structure (collapse / rich / noise) and the sharp transitions between them
- T and L as orthogonal actuators with different timescales
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

Future: embedding-space projections (UMAP/t-SNE of hidden states) as the "true" phase portrait. Trajectories through embedding space, with metric-derived views as interpretable projections. Steering in embedding space = directional perturbation of generation. The topology of the space becomes the interface itself.

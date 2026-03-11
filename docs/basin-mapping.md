# Basin Taxonomy Survey

## Motivation

The phase diagram maps regimes. The basin taxonomy maps *content*. Each attractor basin is a mode the model "knows how to do" — a register, topic, format, or genre strong enough to capture the system at a given (T, L) operating point. The set of all recoverable basins is an empirical map of the model's behavioral repertoire, extracted from output dynamics without inspecting weights.

Previous work treated basins as obstacles to avoid. This reframes them as the primary object of study. The controller problem (navigate *between* basins) depends on first solving the cartography problem (what basins exist, where are they, what connects to what).

## System

SmolLM-135M, autoregressive free generation. No external input after initial seed. PRNG checkpointed at every step.

**Parameter ranges:**
- T (temperature): 0.50–1.50
- L (context length): 8–1024
- W (measurement window): {16, 32, 64, 128, 256}

## Core Protocol: Cooling/Heating Survey

### Phase 1 — Basin Capture (cooling)

1. Initialise context with seed text (see Seeding Strategy below).
2. Set T to survey temperature T_survey (chosen to be within the suppressed or collapse regime for the current L — i.e., T < T_escape(L)).
3. Generate until the sensor suite indicates basin capture:
   - Entropy stabilises (rolling variance drops below threshold)
   - Compressibility at W=64 stabilises
   - Decorrelation lag converges
4. Continue generating for ≥2× decorrelation lag after stabilisation to characterise the basin floor.
5. Checkpoint PRNG state at basin floor. This is the **basin record point**.

### Phase 2 — Basin Characterisation (at basin record point)

At the basin record point, record:

**Sensor profile (quantitative):**
- Entropy: mean, std, floor value
- Compressibility: comp_W at all W ∈ {16, 32, 64, 128, 256}
- Decoupling index: comp_W64 − comp_W256
- Decorrelation lag (ACF threshold 1/e)
- EOS rate (trailing window)
- Surprisal: mean, kurtosis, gap (H − (−log p))

**Attractor type (categorical):**
- **Repetition**: verbatim token loop. Period measurable in tokens. Fragile to perturbation.
- **Template**: structural scaffolding (header/definition, Q&A, proof/theorem, list). Cycle length measurable in structural units. Moderate perturbation robustness.
- **Self-referential**: meta-level/object-level feedback loop. Contains backward references ("the above," "this article," "as mentioned"). High perturbation robustness.
- **Paraphrase loop**: semantic repetition without token-level repetition. Same propositions restated with surface variation. Intermediate between template and repetition.
- **Hybrid / unclassified**: exhibits features of multiple types, or novel structure.

Note: these categories emerged from L=512 T=0.90 observation and will likely need refinement as taxonomy grows. Types may grade into each other; record the dominant type and any secondary features.

**Content tag (qualitative):**
- Semantic domain (astronomy, mathematics, biology, thermodynamics, SEO/content-mill, forum post, encyclopedia, fiction, ...)
- Register (formal academic, conversational, instructional, promotional, ...)
- Notable features (functional self-reference, training-data metadata leakage, cross-basin residue, ...)

**Depth profile (via checkpoint forking):**
From the basin record point, fork N branches (N ≥ 5) with perturbed parameters:
- T_perturb ∈ {T_survey + 0.05, T_survey + 0.10, T_survey + 0.20}
- L_perturb ∈ {L/2, L, 2L} (where hardware allows)
- Measure: steps of coherent elaboration before collapse or paraphrase onset
- This yields a depth score: mean elaboration steps across forks

### Phase 3 — Escape and Transit (heating)

1. From basin record point, raise T above T_escape(L) (use phase diagram estimates).
2. Generate through the escape transient. Record:
   - Steps to escape (entropy rises above basin floor + threshold)
   - Escape trajectory in (entropy, compressibility) phase space
   - Semantic residue: which tokens/topics from the old basin persist in the new context
3. Allow free generation in the heated regime for a transit period (≥L tokens, to allow context turnover).
4. Cool again (return T to T_survey). System falls into the next basin.
5. Repeat from Phase 1.

### Cycle Structure

One **survey cycle** = capture → characterise → escape → transit → recapture. Each cycle yields one basin record. A **survey run** is a sequence of cycles from a single initial seed. Multiple survey runs from different seeds and different initial contexts sample different regions of the basin landscape.

## Seeding Strategy

The initial context determines which basin neighborhood the system enters first. Systematic variation of seeds maps different regions:

**Seed types:**
- **Domain seeds**: short text fragments from known domains (a LaTeX equation, a code snippet, a news headline, a conversational opener, a recipe, a poem fragment). Biases first capture toward that domain's basins.
- **Null seeds**: minimal or empty context. Lets the model's unconditional distribution select the first basin. Reveals the "default" or highest-probability basins.
- **Cross-domain seeds**: juxtapose fragments from different domains. Tests whether hybrid basins exist or whether one domain dominates.
- **Adversarial seeds**: context designed to be far from any expected basin (random Unicode, contradictory framings). Tests basin capture robustness.

## L-Dependent Survey Design

The basin landscape changes with L. Short context (L=8–32) has shallow basins, fast turnover, many escapes — surveys yield many basins per run but characterisation is noisy. Long context (L=512–1024) has deep basins, slow turnover — fewer basins per run but richer characterisation.

**Recommended survey configurations:**

| L range | T_survey | T_escape (est.) | Expected regime | Cycles per 100k tokens |
|---------|----------|-----------------|-----------------|----------------------|
| 8–32 | 0.50 | ~0.50 | Rapid cycling, shallow basins | 50–200 |
| 64–128 | 0.50 | 0.55–0.60 | Moderate basins, periodic escape | 10–50 |
| 192–256 | 0.60 | 0.65–0.90 | Deep basins, slow cycling | 5–15 |
| 512 | 0.80 | >0.90 | Very deep basins, multi-register | 2–8 |
| 1024 | 0.90–1.00 | >1.00 (est.) | Potentially permanent capture | 1–3 |

The short-L regime (8–32) is underexplored in current data and may reveal a qualitatively different basin landscape — basins too shallow for sustained characterisation but cycling fast enough to yield statistical distributions of basin types. This is the "rapid survey" mode.

The long-L regime (512–1024) is the "deep characterisation" mode. Fewer basins per run, but each basin can be depth-probed extensively from checkpoints.

## Taxonomy Data Structure

Each basin record is a row in the taxonomy database:

```
basin_id:         unique identifier (run_id + step_range)
run_id:           survey run identifier
seed_text:        initial context (or hash)
seed_type:        domain / null / cross-domain / adversarial
L:                context length at capture
T_survey:         temperature at capture
capture_step:     step at which basin capture was detected
record_step:      step of basin record point (checkpoint)

# Sensor profile
entropy_mean:     float
entropy_std:      float
entropy_floor:    float
comp_W16..W256:   float × 5
decoupling:       float (comp_W64 − comp_W256)
decorrelation:    int (lag at ACF < 1/e)
eos_rate:         float
surprisal_mean:   float
surprisal_kurt:   float
surprisal_gap:    float

# Classification
attractor_type:   repetition | template | self_referential | paraphrase | hybrid
content_domain:   string (free tag)
register:         string (free tag)
features:         list[string] (notable features)

# Depth
depth_score:      float (mean elaboration steps under perturbation)
depth_T_profile:  dict[T_perturb → elaboration_steps]
depth_L_profile:  dict[L_perturb → elaboration_steps]

# Transition
escape_T:         temperature used for escape
escape_steps:     steps from heating to escape detection
escape_residue:   list[string] (semantic residue tags)
prev_basin_id:    preceding basin in survey run
next_basin_id:    following basin in survey run
```

## Analysis Targets

Once the taxonomy has sufficient entries (target: ≥100 basins across the full L×T range):

**1. Basin census.** How many distinct basins exist at each L? Does the count scale with L (more context = more expressible modes) or saturate? At what L does the model's repertoire exhaust itself?

**2. Basin type distribution.** What proportion are repetition / template / self-referential / paraphrase at each (T, L)? Prediction: repetition dominates at low T and high L; self-referential appears only above some minimum L (enough context for the reference frame to establish).

**3. Transition graph.** Which basins connect to which? Is the graph sparse or dense? Are there hub basins (high in-degree, easy to reach from many starting points) and peripheral basins (reachable only from specific seeds)? The graph structure is a topology of the model's knowledge.

**4. Depth vs. type correlation.** Do self-referential basins consistently have greater depth under perturbation than template basins? Is depth a function of attractor type, or of content domain, or both?

**5. Cross-L basin correspondence.** Does the "pseudo-math proof" basin exist at L=32? At L=1024? How does its sensor profile change with L? Some basins may only be expressible above a minimum L (e.g., the self-referential frame needs enough context for backward references to work).

**6. Semantic residue network.** When basin A transitions to basin B, which content features persist? This yields a directed, weighted graph of semantic influence between basins — the "bleed" structure observed qualitatively (astronomy → math → taxonomy) made quantitative.

**7. Basin stability under intra-run T/L modification.** Using the depth-probing forks: what is the minimum ΔT or ΔL needed to escape each basin type? This is the per-basin escape energy, directly useful for controller design.

## Relationship to Controller Design

The taxonomy is prerequisite to the controller, not a replacement for it. Once the basin map exists:

- **Navigation** becomes pathfinding on the transition graph: from current basin, what sequence of T/L modifications reaches the target basin (or target basin *type*)?
- **Avoidance** is informed: some basins are dead ends (high in-degree, low out-degree), some are hubs. The controller can bias away from dead ends and toward hubs when exploring.
- **Task framing** becomes seed selection: choose a seed that places the system in the neighborhood of basins relevant to the desired output.
- **Quality control** becomes basin classification: is the current basin a repetition attractor (abort), a template attractor (possibly useful), or a self-referential attractor (evaluate depth)?

The controller doesn't avoid all basins. It *navigates* them.
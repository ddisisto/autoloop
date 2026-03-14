# Working Theory

*What we believe we've shown, what we haven't, and what would change our minds.*

This document states claims and their boundaries. It is not a paper, a README, or a lab notebook. Every sentence is either a claim, evidence for a claim, a caveat on a claim, or an anti-claim. For the evidence trail with reproduction commands, see `observations.md`. For the broader research design, see `docs/project-brief.md`.

**System:** SmolLM-135M generating tokens into a fixed-length sliding window of L tokens, conditioned entirely on its own output. No external input after the initial seed exits the window. Pure temperature scaling (no top-k/top-p). This is a discrete stochastic dynamical system whose behavior is determined by the model's learned weights, the context length L, and the sampling temperature T.

---

## Established findings

These are claims we would expect to hold with more seeds, more steps, and modest parameter changes. The qualitative patterns are robust across the conditions tested (3 seeds, 12 L values, 12 T values, ~70 runs).

### 1. Four regimes exist in the (T, L) parameter space

The system occupies one of four qualitatively distinct regimes at any operating point:

- **Collapse:** the system locks into a repeating attractor. Entropy drops to near-zero, vocabulary growth halts. Heaps' beta < 0.40.
- **Suppressed dynamics:** structure is present but mixing is slow. Entropy is low but nonzero, decorrelation lags are long (hundreds of steps). The system is near an attractor but not fully locked.
- **Rich dynamics:** fast mixing, high entropy, steady vocabulary growth. Heaps' beta 0.75-0.85.
- **Noise:** entropy saturates, linguistic structure dissolves. The model's per-token confidence decouples from word-boundary position.

The boundaries between regimes are sharp in the collapse-to-suppressed transition and smooth in the rich-to-noise transition. Data-driven classification using Heaps' beta and entropy separates the regimes with large effect sizes (Cohen's d > 3 for the primary discriminators).

*Evidence: observations.md current model, observations-2026-03-12.md (metric separability analysis across 50 runs, 18 metrics).*

*What would change our minds: a fifth regime appearing at untested (L, T) combinations, or the regime boundaries shifting qualitatively with a different model architecture.*

### 2. The collapse boundary is L-dependent and saturating

T_escape(L) — the minimum temperature to avoid collapse from a cold start — increases with context length and then saturates. The trend across tested L values rises steeply from L=64 to L=256, then flattens by L=512. This suggests a characteristic scale: below some L, context amplifies collapse; above it, temperature alone determines the regime.

We state the trend, not precise threshold values. The exact T_escape at each L depends on how "collapse" is operationalized (which sensor, what threshold, how many steps), and our estimates come from grid sweeps at 0.05-0.10 T resolution with 3 seeds. The qualitative shape — steep rise then saturation — is consistent across operationalizations.

*Evidence: sweep runs across L in {64, 128, 192, 256, 512} at multiple T values. Controller runs at L in {8, 16} confirm low T_escape at short context.*

*What would change our minds: T_escape continuing to rise linearly with L beyond L=512, or a non-monotonic T_escape(L) curve.*

### 3. Collapse is a staircase, not a switch

At sub-threshold temperature, the system does not jump to a zero-entropy floor. It descends through a hierarchy of metastable states, each with a distinct entropy level. Context length L sets the speed of descent — longer context reaches deeper floors faster. This is visible as a stepped entropy time series, with each plateau lasting thousands of steps before the next drop.

*Evidence: entropy time series at T=0.50 across L in {64, 128, 256}, 3 seeds. All show stepped structure rather than immediate collapse.*

*What would change our minds: observing direct jumps to the deepest attractor without intermediate plateaus at some L or T we haven't tested.*

### 4. Basin escape is hysteretic

The temperature required to escape a pre-existing attractor is substantially higher than the temperature required to avoid falling into it from a cold start. A multi-token attractor (e.g., a 2-token cycle) pre-seeded into the context survives well above T_escape measured from BOS. The gap is roughly 0.4T in the one attractor tested in detail, though this number should be treated as indicative rather than universal.

Single-token repeats are much shallower — they lack the mutual-prediction reinforcement between cycle positions that stabilizes multi-token attractors.

*Evidence: pre-seeded probe runs at L=64 with " Star Wars" (2-token cycle) and " young" (1-token repeat). Observations-2026-03-10b.md.*

*Caveat: the 0.4T gap is measured for one specific attractor at one L. The gap likely varies with attractor period, L, and cycle content. We have not systematically measured this.*

### 5. Closed-loop control works

A simple rule-based controller (adjust T by +/-0.05 per segment, adjust L when T saturates) can hold Heaps' beta near a target value of ~0.90, maintaining the system in the rich-dynamics regime. The balance temperature tracks T_escape(L) — the controller independently discovers the collapse boundary. Beta ~0.90 appears to be a natural equilibrium for this model and generation setup.

*Evidence: controller runs at L in {8, 16, 128, 256} with different starting temperatures. All converge to beta ~0.90. Observations-2026-03-10e.md, 2026-03-10f.md.*

*What would change our minds: a different model equilibrating at a qualitatively different beta, which would suggest the equilibrium is model-specific rather than a property of the generation setup.*

### 6. Vocabulary growth rate separates regimes cleanly

Heaps' beta (the exponent of vocabulary growth as a function of tokens generated) is the single best collapse discriminator. It separates collapse (beta < 0.40) from non-collapse with Cohen's d = 3.4 in the tested population. Entropy mean is the best rich/suppressed discriminator (d = 3.4). The combination of these two metrics gives clean three-way regime classification.

*Evidence: metric separability analysis across 50 runs, 18 metrics, 4 regime labels. Observations-2026-03-12.md.*

### 7. Compressibility detects collapse but does not discriminate rich dynamics

Multi-window compressibility (gzip compression ratio at window sizes W) is sensitive to the presence of repetitive structure but does not distinguish between different qualities of rich-dynamics output. At T=1.00, the compressibility signal at W > L is only ~0.03 below the noise floor. The in-context contribution at supra-L scales is minimal. Entropy and Heaps' beta are better control signals for navigating the escape boundary.

*Evidence: W > L analysis across 64 runs. Observations-2026-03-10e.md.*

---

## Provisional findings

These are patterns that are consistent and suggestive but rest on limited instances, small sample sizes, or single-condition observations. We believe them but would not stake a strong claim without more data.

### 8. Attractor content is systematically self-describing

Across 21 collapsed runs (3 seeds x 7 L values at T=0.50), attractor content features tautologies, incomplete predicates, self-perpetuating conditions, recursive structures, and confinement motifs. The content does not just repeat — it describes states of repetition, incompleteness, or entrapment. This pattern appears across seeds and L values, though each seed finds different specific content.

We describe these as **behavioral fixed points**: configurations where the content the model produces is also the content most likely to perpetuate itself under autoregressive generation. The content and the dynamics are aligned — the text describes a condition (confinement, recursion, incompleteness) that is structurally self-reinforcing in context.

*Anti-claim: we do not claim these are eigenstates in any formal mathematical sense. There is no linear operator whose spectrum we have computed. "Eigenstate" is a metaphor we have used in informal descriptions, but it implies a formal structure we have not established. The correct description is: these are fixed points of a stochastic dynamical system whose content happens to be semantically aligned with their dynamical role. Whether this alignment is a deep property of how language models organize their probability landscapes or a surface coincidence at this model scale is an open question.*

*Anti-claim: we have not computed actual gradients at these fixed points. "Zero-gradient" is inferred from behavioral stability (the system stays there indefinitely), not from inspecting the loss landscape. A zero-gradient claim would require computing the Jacobian of the next-token distribution with respect to the context at the fixed point, which we have not done.*

*Evidence: text extraction from all T=0.50 collapsed runs. Observations-2026-03-10c.md.*

*What would change our minds: finding attractor content at a different model scale that is purely random repetition with no self-describing quality, which would suggest this is a SmolLM-135M artifact.*

### 9. Two distinct escape mechanisms exist

We observe two structurally different ways the system escapes from attractor basins:

**Path A: Progressive mutation (continuous).** At threshold L (where the context holds just enough copies of the cycle to maintain lock-in), the model escapes by incrementally mutating the attractor content. The cycle period lengthens, the content drifts, and eventually the mutual-prediction lock between cycle positions weakens enough that the system breaks free. Observed in detail for one attractor (" Star Wars" at L=16): " Star Wars" -> " Star Wars 2000" -> " Star Wars: The Old Republic" -> free generation.

**Path B: EOS-mediated escape (discrete).** An EOS token fires from within the attractor — often at a natural sentence boundary, with low entropy at the EOS position itself. The token *after* EOS then has near-maximal entropy (~6.3 nats), because the model treats EOS as a sequence boundary and effectively restarts generation while attractor residue remains in the context window. The post-EOS content is typically unrelated to the attractor. Whether the escape succeeds depends on context length: at shorter L, the attractor content exits the window faster, giving the new content time to establish itself before recapture.

Across 81 EOS events from deep attractors (pre-entropy < 0.5) in T=0.50 sweep runs, 38% led to sustained escape and 62% were recaptured. The escape rate is strongly L-dependent: ~58% at L=64, declining to ~12% at L=208 and 0% at L=256 (n=1). Every EOS event produces a transient entropy spike regardless of outcome — mean post-EOS entropy is 1.83 vs pre-EOS 0.20. The key difference between escape and recapture appears to be whether the attractor content rotates out of the window before the mutual-prediction lock can re-establish.

The two mechanisms are complementary: progressive mutation is gradual and works by diluting cycle coherence from within; EOS escape is instantaneous and works by injecting a sequence-boundary signal that resets the generative context. Both are consistent with the same underlying principle — the attractor is stabilized by mutual prediction between cycle positions, and anything that disrupts this mutual prediction enables escape.

*Anti-claim: we do not claim progressive mutation is period-doubling in the Feigenbaum sense. We have not measured period ratios, checked for convergence to the Feigenbaum constant, or established a formal connection to the period-doubling route to chaos. "Progressive cycle lengthening preceding escape" is the honest description.*

*Caveat on Path A: observed for one attractor at one L value. Caveat on Path B: the 81 EOS events are from T=0.50 only, and the L-dependent escape rates have small per-L sample sizes (8-15 events each). The trend is consistent but the exact rates should be treated as approximate. We have not yet established whether EOS-mediated escape and progressive mutation co-occur, alternate, or are triggered by different basin structures.*

*Evidence: Path A — text analysis of L=16 T=0.60 pre-seeded run, observations-2026-03-10d.md. Path B — post-EOS entropy analysis across all T=0.50 sweep runs (81 events from deep attractors, 145 total EOS events).*

### 10. Suppressed dynamics may be scale-invariant

A shallow basin at low temperature (L=16 pre-seeded at T=0.60) produces coherence and vocabulary statistics nearly identical to a deep basin at moderate temperature (L=256 at T=0.70). This suggests the suppressed-dynamics regime is defined by the ratio of basin depth to thermal energy, not by absolute parameter values.

*Caveat: this is a comparison between two conditions. Scale invariance is a strong claim that would require systematic variation across many (depth, T) combinations. What we have is a suggestive coincidence between two data points.*

*Evidence: observations-2026-03-10d.md.*

### 11. Within-basin deepening is consistent and directional

In the L=8 survey data, 25 of 29 consecutive same-type recaptures show lower entropy than the previous capture of the same type. Small perturbations (5% temperature increase during heating) consistently tighten the attractor rather than loosening it. This suggests basins have a deepening dynamic — once captured, the system trends toward tighter lock-in absent a sufficiently large perturbation.

*Caveat: L=8 only. The survey protocol's heating rate may be too gentle to escape before the deepening effect dominates. Needs replication at other L values.*

*Evidence: basin capture data from 3 survey runs, 201 captures. Docs/basin-mapping.md.*

### 12. Basin discovery has a long tail at L=8

Across three seeds at L=8 (100k steps each), 17 basin types were identified. New types continued appearing late — the last novel type was discovered at 98% through the third seed's run. Four types are universal (found in all 3 seeds); eight are unique to a single seed. The distribution has rare types that depend on initial conditions.

*Caveat: type assignment originally used cosine threshold (0.3) on 576-dim embeddings, now replaced by HDBSCAN clustering on 14-dim feature space (22 clusters). The qualitative finding — continued discovery without saturation — is robust to re-clustering.*

*Evidence: survey runs at L=8, seeds 42/123/7. Docs/basin-mapping.md.*

---

## Open questions

Each is tied to a specific finding above and states what we'd need to see.

### Does the self-describing quality of attractors hold at scale?

(Tied to finding 8.) If a larger model (1B+) in the same setup produces attractors with purely random content, the self-describing quality is a small-model artifact — perhaps the 135M parameter model's limited capacity forces it into regions where content and structure are tightly coupled. If the quality persists or intensifies, it says something about how autoregressive models organize their probability landscapes generally.

### How do the two escape mechanisms interact?

(Tied to finding 9.) We have identified two paths — progressive mutation and EOS-mediated escape — but do not know their relative frequency in natural (non-pre-seeded) runs, whether they co-occur in the same escape event, or whether certain basin structures favor one mechanism over the other. The EOS data comes from T=0.50 sweeps where progressive mutation was not the primary dynamic (those runs are mostly deep collapse); the progressive mutation observation comes from a pre-seeded probe at threshold L. A systematic study would track both mechanisms simultaneously across a range of (L, T) near the escape boundary to establish whether they are independent, complementary, or one dominates.

### What is the basin count at higher L?

(Tied to finding 12.) Does the number of recoverable basin types grow with L, saturate, or peak? L=8 has at least 17 with no sign of saturation. If L=64 has hundreds, the model's behavioral repertoire is richer than expected. If L=64 has fewer (because longer context stabilizes fewer, deeper basins), that tells us something about how context length shapes the attractor landscape.

### Is beta ~0.90 model-specific or setup-specific?

(Tied to finding 5.) The controller equilibrium at beta ~0.90 could be a property of SmolLM-135M's weight distribution, a property of the sliding-window generation setup, or a more general property of autoregressive generation. Testing with a different model in the same setup would distinguish the first from the latter two.

### Does any of this survive external input?

The entire experimental setup strips away external signal to expose raw dynamics. Real systems — chain-of-thought, agentic loops, multi-turn dialogue — have prompts, loss signals, and structured inputs that may suppress, reshape, or eliminate the phase structure observed here. We do not claim these findings generalize to prompted generation. The dynamics described are properties of unprompted autoregressive self-consumption. Whether they persist in degraded or modified form when external signal is introduced is an empirical question we have not tested.

*This is the largest gap between what we've shown and what the work might imply. Bridging it would require experiments where minimal external signal is injected (e.g., a fixed system prompt, periodic token injection) and the phase structure is re-measured.*

### Is basin topology structured or random?

(Tied to finding 12.) Pre-collapse trajectories trace paths through connected semantic basins, but we haven't established whether the transition graph has structure (hubs, dead ends, preferred paths) or is effectively random (any basin can follow any other, determined by stochastic fluctuation). The distinction matters: structured topology would mean the model's learned representations impose constraints on which modes can follow which; random topology would mean basins are isolated peaks with no inter-basin organization.

### Do the compression spectrum clusters match semantic clusters?

The basin-mapping design proposes compression spectrum shape as mechanistic identity and embedding distance as semantic identity. When these agree, basin identity is confident. When they disagree (same topic, different structure, or different topic, same structure), the disagreement reveals something about how the model organizes its modes. We don't yet have enough data to know how often they agree.

---

## Vocabulary notes

Terms we use, what we mean by them, and where the language risks overselling the evidence.

**Basin / attractor:** a repeating or near-repeating pattern the system locks into at low temperature. This is standard dynamical systems terminology and is used appropriately — the system does exhibit fixed points and limit cycles.

**Basin depth:** informally, how hard a basin is to escape. We have not defined an energy function, so "depth" is a metaphor for escape difficulty (measured by the temperature required to exit, or the number of steps at elevated temperature before escape). It is not a measurement of a potential well. When we say "deeper," we mean "harder to escape from," not "lower energy" in any formal sense.

**Behavioral fixed point:** a configuration where the system, left to run, stays indefinitely. The content in the context window reproduces itself under autoregressive generation. This is what we mean when we have previously used the term "eigenstate" — but without the formal connotation of a linear operator's spectrum.

**Escape boundary / T_escape:** the temperature above which the system avoids collapse from a cold start at a given L. Operationally defined by sensor thresholds (primarily Heaps' beta). Not a phase transition in the thermodynamic sense — the system is not at equilibrium, and we have not established an order parameter.

**Period-doubling / progressive mutation:** the observed increase in attractor cycle length before escape. We use "progressive mutation" as the primary description. The connection to Feigenbaum's period-doubling route to chaos is noted as a suggestive analogy, not an established correspondence.

**Scale invariance (of suppressed dynamics):** the observation that two different (L, T) combinations produce similar statistics when the basin-depth-to-thermal-energy ratio is similar. This is a single comparison, not a demonstrated scaling law.

**Compression spectrum:** the vector of gzip compression ratios at multiple window sizes W. This is a measurement, not a metaphor. Gzip's dictionary literally encodes the repeating byte sequences in the attractor — the dictionary content *is* the attractor's constituent structure at each scale.

# Observations Log

Append-only record of findings. Each entry includes reproduction commands.

## Current Model

*Rewritten as understanding evolves. Entries below are the evidence trail.*

**System:** SmolLM-135M generating into its own context. No external input. Pure autoregressive dynamics.

**Two coupled actuators:**
- T (temperature): per-step noise floor. Controls escape probability from attractors.
- L (context length): memory horizon. Controls attractor basin depth, stickiness, and collapse boundary.

**Four regimes** at fixed L: collapse (T≤T_escape), suppressed dynamics (structure but slow mixing), rich dynamics (T well above T_escape), noise (T≥1.50). The collapse boundary T_escape is L-dependent, sharp, and hysteretic. The noise boundary is smooth — no phase transition, just diminishing returns above T≈2.0. Data-driven classification: β<0.40 → collapse; entropy>3.5 → high-entropy zone (comp_W256>0.65 → noise, else → rich); remainder → suppressed. Heaps' β is the best collapse/suppressed discriminator (Cohen's d=3.4); entropy_mean is the best rich/suppressed discriminator (d=3.4).

**Collapse boundary T_escape(L) increases then saturates.** Estimated escape temperatures: L=64/128 ≈ 0.55–0.60, L=192 ≈ 0.65–0.70, L=256 ≈ 0.85–0.90, L=512 ≈ 0.90. The steep rise from L=128→256 (~+0.3 in T) flattens out by L=512 (~+0.03). This suggests a characteristic scale: below L≈256, context amplifies collapse; above it, context is "sufficient" and temperature alone determines the regime.

**A fourth regime: suppressed dynamics.** L=256 at T=0.70–0.80 is neither collapsed (entropy 0.4–0.6, not <0.1) nor escaped (entropy well below the L≤192 values of 1.1–1.6). Decorrelation lags of 253–356 steps reveal slow-mixing attractor dynamics. Surprisal kurtosis is intermediate (29–60, vs 500+ in collapse and <10 in rich dynamics). The system has local structure (comp_W64 ~0.6) but no large-scale repetition (comp_W256 ~0.25). This suppressed zone is where the multi-scale decoupling is strongest.

**Attractor structure at T=0.50 is a staircase, not a binary.** Each L value has a distinct entropy floor — L=64 sits on a high meta-stable basin (~0.2–0.4 nats), L=128 on a lower false floor (~0.1–0.2 nats), L=256 hits the true zero-entropy floor by ~15k steps. Collapse is a timescale phenomenon: every T=0.50 run may collapse eventually; L sets how fast you descend the staircase.

**Basin transitions reveal an energy landscape.** Escape from attractors requires an entropy spike exceeding a threshold that depends on (L, T). At L=256 T=0.80: spikes >6 nats always reach shallower basins; spikes <1 nat lead to deeper basins 67% of the time. Basin depth progressively decreases over a run (floors 0.05→0.03→0.014), creating an energy cascade. The "escape velocity" — minimum spike for reliable escape — is a measurable proxy for T_escape at any (L, T). Higher T or lower L eliminates deeper transitions entirely.

**L-densification at T=0.50: jagged, not smooth.** L-profile from 64→256 is non-monotonic with local dips at L=208 and L=128, local peaks at L=176 and L=192. Seed variance (std 0.04–0.10) is comparable to inter-L differences (Δmean ~0.03–0.08) at n=3 seeds. The overall downward trend is robust; the fine structure may be noise. No clean phase transition or bifurcation point was found — the "critical L" hypothesis was falsified.

**Slope-flip in the T dimension.** Compressibility (W=64) *decreases* with L at T≤0.60 (deeper collapse = less local structure) but *increases* with L at T=1.00 (longer context = more structure). The sign flip occurs around T=0.70–0.80. This is the phase boundary in the L dimension.

**EOS peak tracks the escape boundary.** Peak EOS rate: L=64 at T=0.90 (0.00092), L=128 at T=0.90 (0.00104), L=192 at T=0.70 (0.00083), L=256 at T=0.90 (0.00087). The peak fires at or just above T_escape — where the system is actively transitioning between collapse and rich dynamics. L=192's peak at T=0.70 reflects its lower escape boundary.

**T=1.50 is not a hard noise floor — the noise boundary is smooth and relational.** From T=1.5 to T=10, entropy gains only +0.4 (L=64) or +0.3 (L=256), saturating at ~55% of the theoretical max (8.4-8.7 vs log2(49152)=15.58). Even at T=10, the model retains strong opinions at ~45% of positions — the distribution is bimodal (55% near-uniform, 45% confident). L matters more than T above T≈2.0. The noise boundary is structurally defined by the space-prefix gradient: at T=1.0, the model's per-token confidence correlates with word-boundary position (gradient +0.57); at T=10, this correlation vanishes (gradient ~0) and token-type frequencies match the vocab baseline. The collapse boundary is intrinsic (the system locks itself); the noise boundary is relational (the model's dynamics decouple from linguistic structure that an observer can parse).

**W (measurement window) is a third dimension.** Compressibility depends strongly on W. Standard grid: W ∈ {16, 32, 64, 128, 256}. At T=0.50, L-curves separate dramatically across W. At T≥0.90, L barely matters at any W. Gzip has fixed overhead (~20B header + Huffman table); at W=16 this inflates ratios above 1.0. Useful range: W≥64 for quantitative work, W≥32 qualitatively. Correction possible by normalizing against incompressible baseline at matched byte length.

**Three-sensor framework:** entropy (model uncertainty), compressibility (observer-assessed structure), EOS rate (model-assessed coherence). Each probes a different aspect. All three needed for regime identification.

**Transfer functions confirm the phase structure.** T→compressibility curves for different L cross at a single pivot point (T≈0.70). Below: longer L = less structure (deeper collapse). Above: longer L = more structure (richer dynamics). T→entropy curves change shape with L: L=64 is roughly linear, L=192 has a sharp elbow at T=0.70 (stuck below, released above). EOS peak is sharper and shifted to lower T for longer L.

**Entropy autocorrelation reveals temporal structure.** Decorrelation lag (first lag where ACF < 1/e) cleanly separates regimes. Collapsed runs: lag 2–8 (locked, no fluctuations to decorrelate). Suppressed zone (L=256, T=0.70–0.80): lag 253–356 (slow-mixing attractor dynamics — the longest in the grid). Rich dynamics (all L at T≥0.90): lag 1–10 (fast mixing regardless of L). Noise (T=1.50): lag 1 (memoryless). Key insight: decorrelation is NOT proportional to L in the rich regime — once escaped, mixing is fast. The interesting temporal structure lives in the suppressed zone near T_escape.

**Attractor content is semantically diverse and L-dependent.** Each collapsed run locks into a distinct attractor: " Wars Star" (L=256 T=0.60), counting sequences (L=256 T=0.70), "Weimar Republic" (L=128 T=0.50), "Helvetica Sans" (L=192 T=0.70), instructional loops (L=192 T=0.60). Attractor period shortens with L (L=128: period 5–7; L=256: period 2). No two runs share the same attractor.

**W/L convergence fingerprints the approach to collapse.** When W approaches L, compressibility descent slopes show a sign flip in the suppressed zone: small W slopes negative (local structure forming), large W slopes positive (global structure diverging). This divergence — local compression with global expansion — is a fingerprint of attractor approach. Slope divergence is near-zero in escaped runs (all scales trend together).

**Basin escape hysteresis.** T_escape measured from BOS (avoiding collapse) does NOT predict the temperature needed to escape a pre-existing attractor. " Star Wars" (2-token cycle) pre-seeded at L=64 survives T=0.60 and T=0.80, only escaping at T=1.00 — despite T_escape(L=64) ≈ 0.55 from BOS. The basin is ~0.4T deeper once occupied. Meanwhile " young" (1-token repeat) escapes trivially at L=64/T=0.60 — single-token repeats lack the mutual-prediction lock of multi-token cycles.

**Basin depth is a smooth function of L with a sharp transition.** Pre-seeding " Star Wars" across L=2,4,8,16,32,64 at T=0.60: entropy drops smoothly from 5.01 (L=2) to 3.55 (L=8), then collapses to 0.32 (L=16). The lock-in threshold is 4-8 copies of the cycle in context. Below this, model priors dominate; above, the pattern self-reinforces. Compressibility is bimodal — 0.5 (escaped) or 0.012 (locked) — with no intermediate regime. Escape destinations are generic (function words), with no competing attractors.

**Vocabulary richness is a clean regime diagnostic.** Type-token ratio (TTR) spans 100x: from 0.003 (L=224/T=0.50, 204 unique words in 77k) to 0.503 (T=1.50, most words appear once). Vocabulary saturation curves reveal dynamics invisible in entropy: L=256/T=0.70 holds steady at 520 unique tokens for 50k steps, then escapes to 2131 — the suppressed-dynamics zone is visible as a vocabulary plateau followed by a cliff. High-T (>=1.00) doesn't just add noise; it unlocks 43k words absent from low-T vocabulary entirely, with only 3k shared across regimes. Heaps' law exponent β cleanly separates regimes: 0.17–0.38 (deep collapse), 0.75–0.85 (rich dynamics), >1.0 (escape events).

**Attractor content is not random — it describes its own dynamics.** Across 21 collapsed runs (3 seeds × 7 L values at T=0.50), every seed finds a unique attractor, but the content systematically features tautologies ("the generator is a generator"), incomplete predicates ("was a time where"), self-perpetuating conditions ("not getting enough sleep... can include not getting enough sleep"), recursive structures ("the disease of the disease"), and confinement ("the man was not allowed to leave"). These are eigenstates: configurations where content, structure, and prediction align into zero-gradient fixed points. The collapsed content clusters into semantic families — medical/body, political/historical, social categories, self-referential — exposing the sharpest peaks in the model's probability landscape.

**Escape by semantic mutation (period-doubling route to chaos).** At L=16 (threshold lock-in), the model doesn't jump out of the Star Wars attractor — it tunnels out by mutating it. "Star Wars" → "Star Wars 2000" → "Star Wars 2001" → "Star Wars: The Old Republic" → "The Seventh Planet" → freedom. Each mutation increases the cycle's period, diluting the mutual-prediction lock until it can't self-reinforce. Analogous to period-doubling as a route to chaos. Once escaped, the system never returns (only 1 Star Wars block in 100k tokens). The post-escape phase has β=2.926 (highest in dataset) — explosive vocabulary growth from a standing start.

**Suppressed dynamics is scale-invariant.** L=16 pre-seeded at T=0.60 has coherence (0.451±0.234) and TTR (0.175) nearly identical to L=256 at T=0.70 (coherence 0.448±0.220, TTR 0.133). The regime is defined by the ratio of basin depth to thermal energy, not absolute L or T. A shallow basin at low temperature behaves like a deep basin at moderate temperature.

**Pre-collapse trajectories map semantic basin connectivity.** Runs don't jump to attractors — they traverse paths through semantic space. L=256/T=0.60 walks: education → political violence → apocalyptic text → civilization → bureaucratic cataloging → imprisonment ("the man was not allowed to leave" ×399) → Star Wars (forever). Each waypoint is a basin the system passed through. Across all collapsed runs, these descent paths form a graph of the model's semantic topology — the basins are connected, and the paths between them reveal how concepts relate in the learned representation.

**Closed-loop control is possible.** A simple controller (adjust T ±0.05 per segment, adjust L ±16 when T saturates) can hold Heaps' β near a target. Balance points exist: L=8/T=0.70, L=16/T=0.75, L=128/T=0.90-0.95 — the balance T tracks T_escape(L). At small L the balance basin is wide (β stays 0.85-1.07 for any T in 0.70-1.00); at L=128 it oscillates (β swings 0.60-1.27), revealing a sharper escape boundary. β ≈ 0.90 appears to be a natural equilibrium for SmolLM-135M, regardless of L or starting T.

**Compressibility is a collapse detector, not a rich-dynamics discriminator.** W>L analysis across 64 runs: at T=1.00, comp_W256 at L=64 (W>L) is only 0.03 below the noise floor (T=1.50). The in-context contribution at W>L scales is minimal. The controller's real operating range is at W≤L scales. Entropy and Heaps' β are the right sensors for navigating the escape boundary.

**Balance-point texture is L-dependent.** Controller runs at β≈0.90 produce qualitatively different text depending on L. L=8: topic soup (jumps every few sentences, no n-gram repeats >3). L=128: thematic orbits (azalea watering → acid-base chemistry, top 4-gram ×39). L=256: thematic orbits (Mars rover missions on repeat, top trigram ×38). Short context = diversity through forgetting; long context = diversity through exploration within a semantic basin. Both achieve β≈0.90 legitimately — the metric measures vocabulary growth rate, and both strategies produce it.

**L=256 controller finds T=0.95.** `ctrl_S42_256_0.70` ramped T steadily from 0.70→1.00 over 7 segments (β stuck at 0.73-0.78 below the dead zone), then settled at T=0.95, β=0.95. Zero rollbacks. Balance T tracks T_escape(L) as predicted: T=0.70 (L=8), T=0.75 (L=16), T=0.90-0.95 (L=128, L=256).

**Semantic theme mapping reveals basin fingerprints.** Auto-discovery across 59 runs finds 60 content themes. Every T=0.50 run is dominated by 1-3 themes at 10-50% word density (the attractor basin). Higher-T runs share a low-density background (health, water, study, research at 0.001-0.003). Seeds determine which basin: L=64/T=0.50 → S42 gets generator+culture, S123 gets book+heart, S7 gets describe+house. Three theme classes: attractor words (spike in one run, spikiness >50), widespread content words (40+ runs, moderate density), structural words (govern, control — topically neutral, neighbor profile shifts by regime).

**Open questions:**
- Does the lock-in ratio (~4-8 copies) hold for longer cycles (3-token, 4-token) and across models? If universal, it reveals a property of in-context learning generally.
- What drives the suppressed-dynamics regime? L=256 at T=0.70–0.80 has high comp_W64 (~0.6) but low entropy (~0.4–0.6). Is this a single deep attractor or switching between multiple shallow ones? Basin transition analysis suggests the latter: 21 escape events at L=256 T=0.80, with progressive deepening.
- Is the L=512 T=1.10 decorrelation anomaly (lag=70) a single-seed artifact or a real pocket of slow dynamics in the rich-dynamics zone?
- Do the four regimes and annealing mechanism generalize beyond SmolLM-135M?
- Can escape spike magnitude serve as a single-run estimator of T_escape, replacing grid sweeps?
- Is the escape-by-mutation mechanism (period expansion → chaos) a general route, or specific to short cycles at threshold L?
- Is β ≈ 0.90 model-specific or a property of the generation setup? Would a different model equilibrate at the same point?
- Can proportional T control stabilize the L=128 oscillation, or does the escape boundary have intrinsic bistability?
- Can semantic basin fingerprints predict which attractor a run will find? Is there a topology to basin space?
- The "self" neighbor profile reorganizes with temperature (psychological → code → grammatical). How deep does this conceptual reorganization go?
- Does the space-prefix gradient generalize to other structural features (punctuation position, digit context, etc.)? Is there a family of "structural coupling" metrics?
- At what T does the space-prefix gradient cross zero? The data shows +0.41 at T=2.0 and ~0 at T=10 — the crossover is somewhere in between.

---

## Evidence Log

Detailed entries archived by date. Each file contains full reproduction commands.

| Date | File | Key findings |
|------|------|-------------|
| 2026-03-06 | [observations-2026-03-06.md](docs/observations-2026-03-06.md) | Three regimes at L=64, context length deepens attractors, violin temporal structure |
| 2026-03-07 | [observations-2026-03-07.md](docs/observations-2026-03-07.md) | L=256 sweep, L as second control parameter, memory-depth annealing concept, EOS signal analysis |
| 2026-03-08 | [observations-2026-03-08.md](docs/observations-2026-03-08.md) | Attractor staircase at T=0.50, multi-window analysis (W dimension), seed replication confirms L=192 anomaly, pre-registered L-densification predictions |
| 2026-03-09 | [observations-2026-03-09.md](docs/observations-2026-03-09.md) | L-densification results (jagged profile), collapse boundary is L-dependent, transfer functions, L=256 crossover (suppressed zone discovered), L=512 escape boundary saturates, single-token attractor dominance, structural resonance |
| 2026-03-10 | [observations-2026-03-10.md](docs/observations-2026-03-10.md) | Concept fragmentation under temperature; pre-collapse trajectories and basin transition dynamics (energy landscape, escape spike thresholds, W/L convergence, attractor diversity) |
| 2026-03-10b | [observations-2026-03-10b.md](docs/observations-2026-03-10b.md) | Pre-seeded basin escape probes: hysteresis (basin exit >> basin avoidance), L-titration of basin depth (lock-in at 4-8 copies), single-token vs multi-token attractor depth, mutual prediction as basin depth mechanism |
| 2026-03-10c | [observations-2026-03-10c.md](docs/observations-2026-03-10c.md) | Semantic analysis: "temperature" attractor at L=128/T=0.60, vocabulary richness 100x range, Heaps' law β separates regimes, attractor content describes own dynamics (eigenstates), pre-collapse trajectories map semantic basin connectivity, seed-dependent content but deterministic collapse |
| 2026-03-10d | [observations-2026-03-10d.md](docs/observations-2026-03-10d.md) | Escape by semantic mutation (period-doubling route to chaos), L=16 β=2.926 anomaly, suppressed zone equivalence across scales, L=128/T=0.60 "temperature" eigenstate, full 15-attractor catalog |
| 2026-03-10e | [observations-2026-03-10e.md](docs/observations-2026-03-10e.md) | Controller v1 with balance points (β≈0.90 equilibrium), comp_stats interface fix, W>L analysis (compressibility as collapse detector) |
| 2026-03-10f | [observations-2026-03-10f.md](docs/observations-2026-03-10f.md) | Semantic theme mapping (--clouds/--themes), L=256 controller (T=0.95 balance), balance-point text is L-dependent, basin fingerprint catalog, three theme classes, "self" profile shifts with T |
| 2026-03-12 | [observations-2026-03-12.md](docs/observations-2026-03-12.md) | Metric separability analysis (F-stat + Cohen's d across 50 runs, 18 metrics). β<0.40 is clean collapse wall, entropy_mean is top separator (F=72), surprisal_kurtosis is extreme-event detector, decorrelation_lag is bimodal. Data-driven regime classifier. Capture detection gates: β first, entropy second. |
| 2026-03-12b | [observations-2026-03-12b.md](docs/observations-2026-03-12b.md) | Extreme-T probes (T=2-10, L=64/256). Entropy saturates at ~55% of theoretical max. β trajectory is T-independent (same asymptote, different rate). Space-prefix gradient defines the noise boundary structurally: collapse boundary is intrinsic, noise boundary is relational (model-observer coupling). |
| 2026-03-14 | [observations-2026-03-14.md](docs/observations-2026-03-14.md) | EOS-mediated basin escape: discrete escape mechanism distinct from progressive mutation. 81 EOS events from deep attractors, 38% escape rate, strongly L-dependent. Thread across EOS boundary is register/format, not topic. Hypothesis: register-defined basins resist EOS escape. |

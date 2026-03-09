# Observations Log

Append-only record of findings. Each entry includes reproduction commands.

## Current Model

*Rewritten as understanding evolves. Entries below are the evidence trail.*

**System:** SmolLM-135M generating into its own context. No external input. Pure autoregressive dynamics.

**Two coupled actuators:**
- T (temperature): per-step noise floor. Controls escape probability from attractors.
- L (context length): memory horizon. Controls attractor basin depth, stickiness, and collapse boundary.

**Four regimes** at fixed L: collapse (T≤T_escape), suppressed dynamics (structure but slow mixing), rich dynamics (T well above T_escape), noise (T≥1.50). The collapse boundary T_escape is L-dependent, but the coupling saturates at large L.

**Collapse boundary T_escape(L) increases then saturates.** Estimated escape temperatures: L=64/128 ≈ 0.55–0.60, L=192 ≈ 0.65–0.70, L=256 ≈ 0.85–0.90, L=512 ≈ 0.90. The steep rise from L=128→256 (~+0.3 in T) flattens out by L=512 (~+0.03). This suggests a characteristic scale: below L≈256, context amplifies collapse; above it, context is "sufficient" and temperature alone determines the regime.

**A fourth regime: suppressed dynamics.** L=256 at T=0.70–0.80 is neither collapsed (entropy 0.4–0.6, not <0.1) nor escaped (entropy well below the L≤192 values of 1.1–1.6). Decorrelation lags of 253–356 steps reveal slow-mixing attractor dynamics. Surprisal kurtosis is intermediate (29–60, vs 500+ in collapse and <10 in rich dynamics). The system has local structure (comp_W64 ~0.6) but no large-scale repetition (comp_W256 ~0.25). This suppressed zone is where the multi-scale decoupling is strongest.

**Attractor structure at T=0.50 is a staircase, not a binary.** Each L value has a distinct entropy floor — L=64 sits on a high meta-stable basin (~0.2–0.4 nats), L=128 on a lower false floor (~0.1–0.2 nats), L=256 hits the true zero-entropy floor by ~15k steps. Collapse is a timescale phenomenon: every T=0.50 run may collapse eventually; L sets how fast you descend the staircase.

**L-densification at T=0.50: jagged, not smooth.** L-profile from 64→256 is non-monotonic with local dips at L=208 and L=128, local peaks at L=176 and L=192. Seed variance (std 0.04–0.10) is comparable to inter-L differences (Δmean ~0.03–0.08) at n=3 seeds. The overall downward trend is robust; the fine structure may be noise. No clean phase transition or bifurcation point was found — the "critical L" hypothesis was falsified.

**Slope-flip in the T dimension.** Compressibility (W=64) *decreases* with L at T≤0.60 (deeper collapse = less local structure) but *increases* with L at T=1.00 (longer context = more structure). The sign flip occurs around T=0.70–0.80. This is the phase boundary in the L dimension.

**EOS peak tracks the escape boundary.** Peak EOS rate: L=64 at T=0.90 (0.00092), L=128 at T=0.90 (0.00104), L=192 at T=0.70 (0.00083), L=256 at T=0.90 (0.00087). The peak fires at or just above T_escape — where the system is actively transitioning between collapse and rich dynamics. L=192's peak at T=0.70 reflects its lower escape boundary.

**T=1.50 is a universal noise floor.** Compressibility ~0.705 and entropy ~8.0–8.3 regardless of L. At high T, context length is irrelevant — thermal noise dominates.

**W (measurement window) is a third dimension.** Compressibility depends strongly on W. Standard grid: W ∈ {16, 32, 64, 128, 256}. At T=0.50, L-curves separate dramatically across W. At T≥0.90, L barely matters at any W. Gzip has fixed overhead (~20B header + Huffman table); at W=16 this inflates ratios above 1.0. Useful range: W≥64 for quantitative work, W≥32 qualitatively. Correction possible by normalizing against incompressible baseline at matched byte length.

**Three-sensor framework:** entropy (model uncertainty), compressibility (observer-assessed structure), EOS rate (model-assessed coherence). Each probes a different aspect. All three needed for regime identification.

**Transfer functions confirm the phase structure.** T→compressibility curves for different L cross at a single pivot point (T≈0.70). Below: longer L = less structure (deeper collapse). Above: longer L = more structure (richer dynamics). T→entropy curves change shape with L: L=64 is roughly linear, L=192 has a sharp elbow at T=0.70 (stuck below, released above). EOS peak is sharper and shifted to lower T for longer L.

**Entropy autocorrelation reveals temporal structure.** Decorrelation lag (first lag where ACF < 1/e) cleanly separates regimes. Collapsed runs: lag 2–8 (locked, no fluctuations to decorrelate). Suppressed zone (L=256, T=0.70–0.80): lag 253–356 (slow-mixing attractor dynamics — the longest in the grid). Rich dynamics (all L at T≥0.90): lag 1–10 (fast mixing regardless of L). Noise (T=1.50): lag 1 (memoryless). Key insight: decorrelation is NOT proportional to L in the rich regime — once escaped, mixing is fast. The interesting temporal structure lives in the suppressed zone near T_escape.

**Open questions:**
- Can L-reduction escape a stuck attractor mid-run? (The annealing experiment — fork a collapsed run, reduce L temporarily, observe whether it finds a different basin.)
- What drives the suppressed-dynamics regime? L=256 at T=0.70–0.80 has high comp_W64 (~0.6) but low entropy (~0.4–0.6). Is this a single deep attractor or switching between multiple shallow ones?
- Is the L=512 T=1.10 decorrelation anomaly (lag=70) a single-seed artifact or a real pocket of slow dynamics in the rich-dynamics zone?
- Do the four regimes and annealing mechanism generalize beyond SmolLM-135M?

---

## Evidence Log

Detailed entries archived by date. Each file contains full reproduction commands.

| Date | File | Key findings |
|------|------|-------------|
| 2026-03-06 | [observations-2026-03-06.md](docs/observations-2026-03-06.md) | Three regimes at L=64, context length deepens attractors, violin temporal structure |
| 2026-03-07 | [observations-2026-03-07.md](docs/observations-2026-03-07.md) | L=256 sweep, L as second control parameter, memory-depth annealing concept, EOS signal analysis |
| 2026-03-08 | [observations-2026-03-08.md](docs/observations-2026-03-08.md) | Attractor staircase at T=0.50, multi-window analysis (W dimension), seed replication confirms L=192 anomaly, pre-registered L-densification predictions |
| 2026-03-09 | [observations-2026-03-09.md](docs/observations-2026-03-09.md) | L-densification results (jagged profile), collapse boundary is L-dependent, transfer functions, L=256 crossover (suppressed zone discovered), L=512 escape boundary saturates, single-token attractor dominance, structural resonance |
| 2026-03-10 | [observations-2026-03-10.md](docs/observations-2026-03-10.md) | Concept fragmentation under temperature — concepts persist but expression degrades with T |

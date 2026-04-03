# Observations — 2026-03-19b

## Information-theoretic metric augmentation

### Context

Sister project (`../framework`) grounds autoregressive self-play dynamics in information theory — rate-distortion, statistical complexity, compressive novelty. This prompted an evaluation of whether information-theoretic metrics could fill known gaps in regime discrimination, particularly within the rich-dynamics regime where gzip compressibility is uninformative.

Added three new metrics to the registry (no new runs required — computed from existing parquet data):
- **Surprisal gap** (step-level): `entropy + log_prob` = entropy minus surprisal. Positive means the sampled token was more predictable than the distribution average (reinforcing); negative means more surprising (enriching). Maps to Framework's "compressive novelty."
- **LZ complexity** (window-level): Lempel-Ziv 76 phrase count over sliding windows of token IDs. Counts distinct phrases in a greedy parse — directly measures process complexity without gzip's byte-encoding artifacts.
- **Surprisal gap mean/std** (run-level): scalar summaries.

Recomputed all 58 sweep runs with full regime separability analysis.

### Finding: LZ complexity is uniformly better than gzip compressibility

F-statistic ranking across four regimes (collapse/suppressed/rich/noise):

| Window | Gzip F-stat | LZ F-stat | LZ advantage |
|--------|------------|-----------|--------------|
| W=32   | 11.67      | 40.05     | 3.4x         |
| W=64   | 29.91      | 58.56     | 2.0x         |
| W=128  | 44.77      | 73.61     | 1.6x         |
| W=256  | 53.91      | 80.38     | 1.5x         |

LZ beats gzip at every window size. The advantage is largest at small windows (W=32: 3.4x), where gzip's fixed overhead (~20B header + Huffman table) inflates ratios above 1.0. LZ has no such overhead — it counts phrases in token-ID space, not byte space.

LZ also has tighter standard deviations within regimes, meaning cleaner separation. At W=256: LZ collapse std=13.0 vs gzip std=0.018 (different scales, but within-regime coefficient of variation is lower for LZ).

**Implication:** LZ complexity should replace gzip compressibility as the primary window-level complexity metric. This includes the W* approach in basin detection (compression spectrum), sensor readings, and the three-sensor framework. Gzip can be retained for backward compatibility but LZ is strictly superior.

### Finding: Surprisal gap tracks enriching/stabilising regime boundary

The gap's sign cleanly separates regime types:

| Regime     | Gap mean | Gap std | Interpretation |
|------------|----------|---------|----------------|
| Suppressed | **+0.42** | 0.20   | Most reinforcing — tokens consistently more predictable than distribution expects |
| Collapse   | +0.19    | 0.08   | Reinforcing but distribution is already peaked, less room for gap |
| Noise      | -0.32    | 0.45   | Slightly surprising, near random |
| Rich       | **-1.41** | 0.89   | Strongly enriching — tokens genuinely more surprising than the distribution predicts |

The gap's overall F-statistic is 48 (8th of 35 metrics). But its value is in specific pairwise discrimination:

- **Rich vs suppressed:** Cohen's d = 2.84. This was the hardest discrimination for existing metrics (entropy separates them at d=4.48, but compressibility doesn't help). The gap adds an independent signal grounded in token-level dynamics rather than aggregate statistics.
- **Rich vs noise:** Cohen's d = 1.54. The gap distinguishes these because rich dynamics produces tokens that are *more* surprising than the model's own distribution (negative gap), while noise produces tokens that are merely *as* surprising as a near-uniform distribution (gap near zero).
- **Collapse vs suppressed:** d = 1.48. Weaker than `heaps_beta` (d=3.42) for this pair, which remains the best collapse detector.

**Key insight:** Suppressed dynamics has a *higher* positive gap than collapse. This is counterintuitive until you think about what the gap measures. In collapse, the distribution is already concentrated on the attractor tokens — the gap between expected and sampled surprise is small because both are low. In suppressed dynamics, the distribution is still relatively broad (entropy 0.4-2.9 nats) but sampling consistently lands on the predictable side. The system is *actively* reinforcing, not merely locked.

This maps directly to Framework's prediction: the stabilising extreme — degeneration — is characterized by compressive novelty approaching zero, where each token does less representational work relative to what's already established. Suppressed dynamics is the clearest example: surface statistics (entropy, fluency) look moderate while the gap reveals that the system is consistently choosing the predictable option. Note that stabilising tokens are not inherently pathological — they are the necessary scaffolding of coherent generation. The pathology is when the enrichment fraction drops to zero and only stabilising tokens remain.

### Finding: metric hierarchy aligns with Framework's compression levels

The F-statistic ranking naturally clusters by Framework's compression hierarchy:

**Algorithmic level** (surface statistics):
- `entropy_mean` F=129, `surprisal_mean` F=117 — strong gross separators, weak within-regime discriminators

**Organisational level** (structural complexity):
- `lz_W256_mean` F=80, `lz_W128_mean` F=74 — process complexity, measures phrase structure in token sequences
- `comp_W256_mean` F=54 — same level but noisier (byte-encoding artifacts)
- `heaps_beta` F=39 — vocabulary growth rate, another organisational measure

**Between algorithmic and semantic** (token-distribution coupling):
- `surprisal_gap_mean` F=48 — measures how the sampling process interacts with the model's own distribution

This ordering — algorithmic metrics separate grossly, organisational metrics separate within low-entropy regimes, the gap separates within high-entropy regimes — is exactly what Framework predicts about hierarchical compression levels having different discriminative domains.

### Reproduction

```bash
# Compute all metrics across sweep runs
python scripts/regime_analysis.py

# Report only (from saved CSV)
python scripts/regime_analysis.py --report

# Quick comparison on individual runs
python -c "
from autoloop.analyze import analyze_run, default_window_sizes, load_experiment_df
from autoloop.analyze.scalars import run_scalars
from pathlib import Path
import numpy as np

for name in ['L0064_T0.50_S42', 'L0064_T1.00_S42']:
    p = Path(f'data/runs/sweep/{name}.parquet')
    exp = load_experiment_df(p)
    L = int(p.stem.split('_')[0][1:])
    cache = analyze_run(p, default_window_sizes(L), exp=exp)
    s = run_scalars(exp, cache)
    gap = s['surprisal_gap_mean']
    lz = np.nanmean(cache['lz_complexity'][256])
    print(f'{name}: gap={gap:+.4f} lz_W256={lz:.1f}')
"
```

### Implications for basin detection

The current basin detection pipeline uses:
1. `heaps_beta` < threshold as collapse gate (keep)
2. `entropy` as secondary gate (keep)
3. W* compressibility spectrum for basin fingerprinting (replace with LZ)
4. 576-dim embeddings for clustering (keep)

LZ complexity should replace gzip compressibility in the compression spectrum (`comp_spectrum` in engine.py) and the basin capture sidecar. The W* approach (finding the most compressible window) translates directly: find the window size with the *lowest* normalized LZ complexity (phrases/window_size ratio). This is a cleaner signal because it's not contaminated by gzip header overhead at small windows.

The surprisal gap could be added as a basin fingerprint feature — it would help distinguish captures in the transition zone between suppressed and rich dynamics. But this is additive, not a replacement.

These changes are deferred to the next session. The metrics are implemented and validated; the detection pipeline update is a separate piece of work.

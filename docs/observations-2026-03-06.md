# Observations — 2026-03-06

Detailed findings log. See [observations.md](../observations.md) for current model and index.

### 2026-03-06 — Initial L=64 temperature sweep (seed=42)

**Data:** `L0064_T{0.50,1.00,1.50}_S42.parquet` (100k tokens each)

**Throughput:** ~24-38 tok/s on GTX 1070 at L=64.

**Three distinct regimes observed:**

- **T=0.50**: Degenerate loop attractor. Entropy mean 0.68 nats. Text collapses to verbatim repetition ("Ethnomusicology is the study of music." repeating). Compressibility 0.35. Classified as transient — block-level entropy fluctuates (1.03 → 0.29 → 0.84), suggesting mode-switching between tighter and looser loops.
- **T=1.00**: Rich dynamics. Entropy mean 3.72 nats. Coherent-ish text with degraded grammar. Compressibility 0.74. Also transient. Wide spread in phase portrait — continuum of states rather than discrete modes.
- **T=1.50**: Near-random. Entropy mean 7.99 nats, very stationary (trend -0.009/block). Word salad. Compressibility 0.71 — similar to T=1.00 despite very different entropy, because gzip picks up character-level patterns in English tokens even when token-order is random.

**Phase portrait** shows three clearly separated clusters with a gap in entropy space (~2-3 nats) between T=0.50 and T=1.00. Crossover region is below T=1.0.

```bash
# Summary stats
python -c "
import pandas as pd
for t in ['0.50', '1.00', '1.50']:
    df = pd.read_parquet(f'data/runs/L0064_T{t}_S42.parquet')
    exp = df[df.phase == 'experiment']
    print(f'T={t}: entropy mean={exp.entropy.mean():.3f} std={exp.entropy.std():.3f}, EOS={exp.eos.sum()}')
"

# Plots
python plot.py --runs data/runs/L0064_T*_S42.parquet
```

---

### 2026-03-06 — Context length effect at T=0.50 (L=64 vs L=256, partial)

**Data:** `L0064_T0.50_S42.parquet` (100k), `L0256_T0.50_S42.parquet` (49k, partial)

**Context length dramatically deepens the loop attractor.** L=256 collapses to near-zero entropy (~0.1 nats) by ~15k steps and stays there. L=64 keeps having escape episodes throughout the full 100k run (entropy excursions up to ~3 nats).

**Phase portrait:** L=256 is a tight cluster at (entropy ~0.1, compressibility ~0.1). L=64 scatters widely up to entropy ~3, compressibility ~0.75. Longer context = more self-reinforcing repetitive signal = deeper attractor basin.

**Implication:** Context length is a significant experimental variable, not just a nuisance parameter. Attractor basin depth scales with L.

```bash
python plot.py --runs data/runs/L0064_T0.50_S42.parquet data/runs/L0256_T0.50_S42.parquet
```

---

### 2026-03-06 — Violin plots reveal temporal collapse structure

**Data:** Same as above.

**Violin plot** (entropy + compressibility distributions per 10k-token time block) shows the collapse dynamics much more clearly than time series alone:

- **T=0.50 L=64**: Entropy distribution progressively narrows and shifts left, but non-monotonically — blocks 50-70k show re-broadening (escape episodes). Compressibility mirrors this.
- **T=0.50 L=256**: Entropy collapses to a spike by 20-30k and stays. Compressibility reaches ~0.1-0.2 (vs L=64's 0.3-0.5). Longer context = more self-reinforcing repetitive signal = deeper attractor.
- **T=1.00**: Broad, stable distributions throughout. Some compressibility bimodality in individual blocks.
- **T=1.50**: Narrow, stable. Stationary noise.

W=L/4 (lighter overlay) is consistently tighter than W=L, confirming local-scale repetition structure.

```bash
python plot.py --runs data/runs/L0064_T*_S42.parquet --plots violin
python plot.py --runs data/runs/L0064_T0.50_S42.parquet data/runs/L0256_T0.50_S42.parquet --plots violin
```

# Observations — 2026-03-12

## Metric separability analysis: which signals best separate regimes?

Built `scripts/regime_analysis.py` — computes all 18 registered metrics across all 50 sweep runs, classifies regimes, and performs F-statistic + Cohen's d analysis.

### Regime classifier (data-driven, three gates)

```
β < 0.40             → collapse
entropy_mean > 3.5   → (high-entropy zone)
  comp_W256 > 0.65   → noise
  else               → rich
else                 → suppressed
```

Distribution: 11 collapse, 28 suppressed, 9 rich, 2 noise.

### Top metric separators (F-statistic ranking)

| Metric | F-stat | What it separates |
|--------|--------|-------------------|
| entropy_mean | 71.8 | Everything (strongest single signal) |
| comp_W256_mean | 32.5 | Collapse from rich/noise (d=31-35) |
| heaps_beta | 32.1 | Collapse from suppressed (d=3.4); noise from rich (d=5.5) |
| entropy_std | 23.5 | Collapse (0.44) from rich (1.85) |
| comp_W64_mean | 20.6 | All regimes with moderate separation |
| surprisal_kurtosis | 13.6 | Collapse (956) from everything else (<2) |

### Pairwise discrimination (best single metric, Cohen's d)

| Pair | Best metric | d |
|------|-------------|---|
| collapse vs noise | comp_W256_mean | 34.5 |
| collapse vs rich | comp_W256_mean | 31.2 |
| **collapse vs suppressed** | **heaps_beta** | **3.4** |
| noise vs rich | heaps_beta | 5.5 |
| noise vs suppressed | surprisal_var | 4.7 |
| **rich vs suppressed** | **entropy_mean** | **3.4** |

The hard distinctions are collapse↔suppressed and rich↔suppressed. Both achieve d≈3.4, which is good but not overwhelming — these are the interesting boundaries.

### Key findings

**1. β < 0.40 is a clean collapse wall.** All 11 collapse runs have β ∈ [0.09, 0.40]. Only 1 of 28 suppressed runs has β < 0.50 (L=192 T=0.50 S=7 at β=0.403). This is the single best capture detection gate — cleaner than any entropy threshold.

**2. Decorrelation lag is bimodal within collapse.** F-stat ranks it last (F=2.0), but this misses the structure. Collapse runs split into:
- Locked cycles: lag 2-11 (e.g., L=256 T=0.50: lag=2, L=176 T=0.50: lag=3)
- Slow descent: lag 173-2000 (e.g., L=160 T=0.50 S=7: lag=2000, L=208 T=0.50 S=7: lag=725)
These correspond to different positions on the attractor staircase — locked cycles are at the floor, slow descent is still descending. A useful secondary signal but not a separator by itself.

**3. Surprisal kurtosis is an extreme-event detector.** Collapse: 956±511. Suppressed: 244±375. Rich: 1.5±1.9. Noise: -0.4±0.2. The distribution shape (heavy-tailed vs normal) cleanly separates regimes, but the high variance in collapse/suppressed limits its use as a threshold.

**4. Suppressed dynamics is the widest regime (n=28, 56% of runs).** It spans L=64-512, T=0.50-0.90, with β from 0.40 to 1.45. This is the "everything that isn't extreme" category — the regime where the system has structure but isn't locked. Substructure within this regime (e.g., near-collapse vs near-escape) is a ripe target for further analysis.

**5. The T=0.50 boundary is seed-dependent at L=128-224.** At T=0.50, three seeds at L=160 produce collapse (β=0.12, 0.23, 0.40) — the β=0.40 case is right at the boundary. At L=176, two collapse + one suppressed (β=0.51). At L=192, one collapse + two suppressed. The collapse/suppressed boundary at fixed T is stochastic and L-dependent.

**6. comp_W256 > 0.65 distinguishes noise from rich.** Noise runs (near-uniform sampling) have comp_W256 ≈ 0.66-0.68; rich dynamics have 0.58-0.63. The gap is small (Δ≈0.05) but consistent. Only tested at L=512 T=1.00-1.20 (n=4 runs) — thin.

### Implications for capture detection

For the survey controller, the data supports:
1. **Primary gate: β < 0.40** — a hard wall for "this is collapsed." No false positives in 50 runs.
2. **Secondary gate: entropy_mean < 1.0** — separates collapse+deep-suppressed from everything else.
3. **Tertiary: surprisal_kurtosis > 100** — confirms extreme-event regime (captures vs plateau).
4. **Don't use decorrelation lag as a gate** — too bimodal within regimes. Better as a post-hoc descriptor.
5. **Don't use compressibility for capture detection** — it separates regimes well at the extremes (collapse vs rich: d=31) but poorly at the boundary (collapse vs suppressed: d=1.8 for comp_W256). Confirmed: collapse detector, not rich-dynamics discriminator.

### Reproduction

```bash
python scripts/regime_analysis.py              # full compute + report
python scripts/regime_analysis.py --report     # report from saved CSV
cat data/regime_analysis.csv                   # raw data (50 runs x 30 metrics)
```

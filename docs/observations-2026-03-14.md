# Observations — 2026-03-14

## EOS-mediated basin escape

### Context

While drafting THEORY.md, identified EOS as a structurally distinct escape mechanism from the progressive mutation path previously documented (observations-2026-03-10d.md). Analyzed existing T=0.50 sweep data — no new runs required.

### Finding: EOS is a discrete escape mechanism

Every T=0.50 sweep run stores per-step `eos` boolean and `token_id` in parquet. Extracted all EOS events and classified post-EOS trajectories.

**Population:** 145 EOS events across 22 T=0.50 sweep runs (L=64 to L=256, seeds 42/123/7). 81 events fired from deep attractors (pre-EOS entropy < 0.5 over preceding L steps).

**Post-EOS entropy signature:** Every EOS event from a deep attractor produces a transient entropy spike. The EOS token itself often has low entropy (the model naturally reaches a sentence boundary within the attractor). The spike occurs at step+1: entropy jumps to ~6.3 nats because the model treats EOS as a sequence boundary and effectively restarts generation while attractor content remains in the context window.

- Mean pre-EOS entropy: 0.20
- Mean post-EOS entropy (first L steps): 1.83
- EOS-token entropy: variable (0.04–1.75), often low

**Escape rate is L-dependent:**

| L | EOS from deep | Escaped | Rate |
|---|--------------|---------|------|
| 64 | 12 | 7 | 58% |
| 128 | 14 | 5 | 36% |
| 160 | 12 | 5 | 42% |
| 176 | 10 | 5 | 50% |
| 192 | 15 | 5 | 33% |
| 208 | 8 | 1 | 12% |
| 224 | 9 | 3 | 33% |
| 256 | 1 | 0 | 0% |

Overall: 31/81 (38%) led to sustained escape. The trend — higher escape rate at shorter L — is consistent with the mechanism: shorter context means attractor content exits the window faster, giving post-EOS content time to establish before recapture.

### Finding: Thread across EOS boundary is register, not topic

Inspecting decoded text around EOS events reveals that content after EOS is topically unrelated to the attractor but often preserves its register/format:

**Escaped (L=64):**
```
PRE:  ... A good essay is one that is well-organized and well-written.
POST: <|endoftext|>The 2016 election was a historic event in American history.
```
"Describe a good essay" → starts writing an essay. Topic breaks, instructional frame persists.

```
PRE:  ... is elected by the House. The Speaker is elected by the House.
POST: <|endoftext|>The two most important elements of a physical education program...
```
Governance → physical education. Topic breaks completely, encyclopedic register persists.

**Recaptured (L=192):**
```
PRE:  ...types of jobs in the world. There are different types of jobs in the world.
POST: <|endoftext|>The 19th century was a time of great change and progress in the world.
```
Even the phrase "in the world" carries across. Textbook register intact.

**Recaptured (L=64, code register):**
```
PRE:  ...def get_user_name_3006(self): return self.user_name_hash_3006
POST: <|endoftext|>import numpy as np\nimport matplotlib.pyplot as plt...
```
Repetitive Python method definitions → Python imports. Topic shifts (method defs → data science setup), code register survives completely. The format signal in the window is strong enough that EOS merely pivots within the same register-basin.

### Hypothesis: register-defined basins resist EOS escape

The EOS mechanism breaks the *topic lock* but not necessarily the *format lock*. If a basin is defined primarily by register/format (code, lists, Q&A), the format signal persists in the post-EOS window and can recapture the system even after a topical pivot. Basins defined primarily by topic but with generic format (encyclopedic prose about a specific subject) should be more susceptible to EOS escape.

**Prediction:** Among recaptured EOS events, the post-EOS text shares format but not topic with the pre-EOS attractor. Among escaped events, neither format nor topic is preserved.

**Prediction:** Attractors with strong format signatures (code, numbered lists, Q&A) should have lower EOS escape rates than attractors with strong topic signatures but generic prose format, controlling for L.

Not tested. Requires either systematic classification of attractor format types across existing data or targeted pre-seeded probes with format-defined vs topic-defined attractors.

### Relation to progressive mutation

Two escape paths now identified:

1. **Progressive mutation (continuous):** attractor period lengthens, content drifts, mutual-prediction lock weakens gradually. Observed at threshold L for pre-seeded attractors (observations-2026-03-10d.md).
2. **EOS-mediated escape (discrete):** single EOS token breaks autoregressive chain, model restarts generation conditioned on attractor residue in window. Whether escape succeeds depends on L (window turnover speed) and potentially on whether the basin is format-defined or topic-defined.

Both reduce to the same principle: attractor stability comes from mutual prediction between cycle positions, and anything that disrupts this enables escape. The two mechanisms differ in timescale (gradual vs instantaneous) and in what they disrupt (cycle content vs sequence framing).

Open: do the two mechanisms co-occur? Does progressive mutation precede EOS (weakening the lock until EOS can finish the job)? Or are they independent paths triggered by different basin structures?

### Reproduction

```python
# Extract all EOS events from T=0.50 runs and classify outcomes
import pandas as pd, numpy as np, glob

results = []
for f in sorted(glob.glob('data/runs/sweep/L*_T0.50_*.parquet')):
    L = int(f.split('/')[-1].split('_')[0][1:])
    df = pd.read_parquet(f, columns=['step','eos','entropy'])
    for eos_idx in df.index[df['eos']]:
        pre_ent = df.iloc[max(0,eos_idx-L):eos_idx]['entropy'].mean()
        post = df.iloc[eos_idx+1:min(len(df),eos_idx+L+1)]['entropy'].values
        if len(post) >= L//2 and pre_ent < 0.5:
            escaped = post[-1] > max(pre_ent * 2, 1.0)
            print(f'L={L} step={int(df.iloc[eos_idx]["step"])} pre={pre_ent:.3f} post_final={post[-1]:.3f} {"ESCAPED" if escaped else "recaptured"}')
```

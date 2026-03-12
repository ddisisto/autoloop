# Handover — 2026-03-10d: Annealing Semantics & Window Sizing

## What happened this session

1. **Fixed anneal.py `--check`** — was accidentally running experiments for non-probe phases. Now shows status table (like `--dry-run`) for all phases.

2. **Fixed semantic.py to load non-standard runs** — `anneal_*` and `sched_*` filenames now parsed by falling back to parquet column values for L/T/seed. Default run discovery includes all three patterns.

3. **Ran semantic analysis across full grid + annealing runs** (40 runs, seed=42). Key findings:
   - L=16 pre-seeded Star Wars escapes via **semantic mutation** — period expansion ("Star Wars" → "Star Wars 2000" → "The Old Republic" → freedom). Analogous to period-doubling route to chaos.
   - L=16 has β=2.926 (highest in dataset by 2x), indicating explosive vocabulary growth post-escape.
   - L=16/T=0.60 pre-seeded is dynamically equivalent to L=256/T=0.70 natural suppressed zone (coherence 0.451 vs 0.448). **Suppressed dynamics is scale-invariant** — defined by basin-depth/thermal-energy ratio.
   - L=128/T=0.60 is the "temperature" eigenstate — 6518 hits for the word "temperature" out of 7148 total, locked in a loop literally about measuring temperature.
   - 15 collapsed runs catalogued with unique attractor content.

4. **Updated `default_window_sizes()` to `[32,64,128,256]`** — floor at 32 (gzip overhead dominates below), and always includes W>L. This enables detecting training-prior structure beyond the context window.

5. **Verified W>L analysis works**: at L=4, comp_W256=0.602 — significant long-range structure despite 4-token context. This is pure training distribution, not in-context pattern formation.

## What to do next

### Immediate: W>L analysis across full grid
The highest-value next step. Run default_window_sizes (now [32,64,128,256]) on ALL runs and compare:
- How does comp_W256 vary with L at fixed T? At L=4/8/16 it reveals training prior; at L=256 it reveals in-context patterns. Where's the crossover?
- Does comp_W>L separate regimes differently than comp_W≤L?
- The multi-scale decoupling metric (|comp_W256−comp_W64|) should be recalculated with the new windows.

```bash
# Recompute analysis at new standard windows for all runs
python -c "
from analyze import analyze_run, default_window_sizes
from pathlib import Path
ws = default_window_sizes(0)  # [32,64,128,256]
for p in sorted(Path('data/runs').glob('*.parquet')):
    analyze_run(p, ws)
    print(f'  done: {p.stem}')
"
```

### Resume tier1 annealing
4/15 tier1 runs done (seed 42 only):
- `anneal_L004_escape_S42`: 73k/100k tokens (partial, has checkpoint)
- `anneal_L008_escape_S42`: complete (100k)
- `anneal_L016_stuck_S42`: complete (100k)
- `anneal_L256_control_S42`: 820/100k (barely started, has checkpoint)

Remaining: seeds 123 and 7 for all four L values, plus boundary probes L=10/12/14. Run with `python anneal.py tier1`.

**Note**: `is_done()` just checks parquet existence — the two partial runs show as DONE but aren't complete. Either delete their parquets to force rerun, or rely on checkpoint resume (generate.py should handle this).

### Semantic topology mapping
Still the main research priority from 10c. Sliding window topic detection → transition graph → semantic basin network.

### Cosine embedding distances
Embed token windows with the model's own embeddings, measure cosine distances between consecutive windows (semantic velocity) and distance to eventual attractor.

## Recent architectural change
`analyze.py` was restructured into `analyze/` package (commit `93e4526`). Imports are the same (`from analyze import analyze_run, default_window_sizes`). `metrics.py` was folded in. Check the package structure if anything breaks.

## Files changed this session
- `anneal.py` — `--check` fix for non-probe phases
- `semantic.py` — non-standard filename loading, default discovery patterns
- `analyze/summary.py` — `default_window_sizes()` now returns `[32,64,128,256]`
- `observations.md` — new current-model entries, evidence log row for 10d
- `docs/observations-2026-03-10d.md` — full findings writeup
- `CLAUDE.md` — updated analyze description, new findings
- `MEMORY.md` — updated architecture, insights, priorities

## Commits this session
- `23aaa99` — Prior session work: semantic.py, CLAUDE.md, anneal.py, observations-10c
- `77ca762` — Escape by semantic mutation, anneal/semantic fixes, observations-10d
- (pending) — default_window_sizes update, CLAUDE.md, handover

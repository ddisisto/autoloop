# Handover — 2026-03-10c: Semantic Analysis Session

## What happened this session

Built `semantic.py` — a semantic analysis tool that runs 5 analyses across all parquet runs:
1. **Theme search**: find a word, extract context windows, local entropy, neighbor tokens
2. **Attractor catalog**: top n-grams per run, repetition ratios, collapsed vs escaped classification
3. **Repetition onset**: detect when n-gram repetition begins (sparkline profiles)
4. **Heaps' law**: fit V(n) = K·n^β to vocabulary growth, report β per condition
5. **Semantic coherence**: bigram Jaccard overlap between adjacent 500-word sliding windows

## The big discovery

**Attractor content is not random.** Across 21 collapsed runs (3 seeds × 7 L values at T=0.50), every seed finds a unique attractor, but the content systematically describes its own dynamics:

- "The generator is a generator" — tautology as fixed point
- "The Weimar Republic was a time where" — incomplete predicate, never reaching the object
- "Not getting enough sleep... can include not getting enough sleep" — self-perpetuating condition
- "The disease of the disease of the disease" — pure recursive self-reference
- "The man was not allowed to leave the room" (×399) → Star Wars — confinement then surrender

These are **eigenstates** — configurations where content, structure, and prediction all align.

**Pre-collapse trajectories are not random jumps.** L=256/T=0.60 walks: education → political violence → apocalypse → civilization → cataloging → imprisonment → Star Wars. Each waypoint is a basin the system passed through. The runs map the **topology of the model's semantic space**.

The user's key insight: the attractors form a **connected network**. Disease connects to sleep. Confinement connects to counting. The paths between basins are as informative as the basins themselves. This is the model's energy landscape with labels on the minima.

## Files created/modified

- **Created**: `semantic.py` — new analysis script (5 analyses, CLI, CSV export)
- **Created**: `docs/observations-2026-03-10c.md` — full findings with reproduction commands
- **Created**: `data/semantic.csv` — exported metrics for all seed=42 runs
- **Modified**: `observations.md` — added current model entries + evidence log row
- **Modified**: `CLAUDE.md` — added semantic.py to layout and CLI reference
- **Modified**: `MEMORY.md` — updated insights and next priorities

## What to do next

### Immediate: Semantic topology mapping
The highest-priority follow-up. Extract topic sequences from pre-collapse trajectories and build a graph of how semantic basins connect.

Approach:
1. For each collapsed run, slide a window (e.g. 500 words, stride 250) across the pre-collapse text
2. Assign each window a "topic" — could be top-3 bigrams, or cluster embeddings
3. Build a transition graph: nodes = topics, edges = sequential transitions
4. Overlay all 21+ runs onto the same graph
5. The result is a map of the model's semantic topology — which basins connect to which

### Immediate: Cosine embedding distances
User mentioned this as the next analysis after semantic.py. Use the model's own embeddings to:
- Embed sliding windows of generated text
- Measure cosine distances between consecutive windows (semantic velocity)
- Measure distance from each window to the eventual attractor (approach trajectory)
- This gives a proper geometry to the semantic landscape, not just bigram overlap

### Ongoing: Annealing experiments
`anneal.py tier1` may still be collecting data. Check with `anneal.py tier1 --check`. Tier 2 (return dynamics: L=256→L=8→L=256) is next after tier1 completes.

## Key commands

```bash
# Full semantic analysis
python semantic.py --seed 42

# Export all metrics
python semantic.py --csv data/semantic.csv --seed 42

# Check annealing status
python anneal.py tier1 --check

# Sweep status
python sweep.py --status
```

## Conceptual frame

The session started as "let's search for the word temperature" and ended at "these attractors are eigenstates of the model's semantic space, and the pre-collapse trajectories map the topology of how meaning is organized in the learned representation." The quantitative tools (Heaps' β, coherence Jaccard, repetition onset) are solid, but the qualitative reading of the attractor content is where the real insight lives. The next session should focus on making the topology mapping rigorous — either via embedding geometry or via the sliding-window topic graph.

# Interactive Explorer — Design

## Goals

Web-based interactive explorer for autoloop run data. Replace static plot pipeline for exploratory analysis; eventually evolve into real-time control dashboard.

**Core principles:**
- Adaptable: metrics, chart types, and UI elements change rapidly — nothing hardcoded in the frontend
- Shareable: URL hash encodes full view state — paste into observations, share with collaborators
- Simple: one Python file for backend, one HTML file for frontend, minimal JS framework overhead

## Architecture

```
explorer.py          # FastAPI backend: data loading, API endpoints, serves static files
static/
  index.html         # Single-page app: Plotly.js + vanilla JS (or Alpine.js if needed)
```

Two files. Backend is the source of truth for what exists (runs, metrics, chart types). Frontend discovers everything from the API and builds UI dynamically.

### Key boundary: the backend describes, the frontend renders

The frontend never hardcodes metric names, run lists, or chart configurations. It asks the backend "what do you have?" and builds controls from the response. Adding a new metric = backend change only. Adding a new chart type = small contract extension, but frontend handles layout generically.

## Data Model

### Runs

Each run is a parquet file + optional analysis cache. Backend scans `data/runs/` on startup, extracts metadata from filenames and JSON sidecars.

```
GET /api/runs
→ [
    {
      "id": "L0064_T0.70_S42",
      "L": 64, "T": 0.70, "seed": 42,
      "tokens": 100000,
      "path": "L0064_T0.70_S42.parquet"
    },
    ...
  ]
```

### Metrics

Two kinds:
- **Step-level**: one value per generation step (entropy, logprob, eos flag). Source: parquet columns.
- **Block-level**: one value per analysis block (compressibility at window W, EOS rate). Source: analysis cache or computed on load.

Backend publishes a metric registry — frontend uses this to populate axis selectors, know what's plottable against what, etc.

```
GET /api/metrics
→ [
    {
      "id": "entropy",
      "name": "Softmax Entropy",
      "resolution": "step",     // "step" or "block"
      "source": "parquet",      // "parquet" or "analysis"
      "column": "entropy"       // how to find it in the data
    },
    {
      "id": "compressibility_W_L",
      "name": "Compressibility (W=L)",
      "resolution": "block",
      "source": "analysis",
      "description": "Gzip compressibility over L-token windows"
    },
    ...
  ]
```

### Data endpoint

Frontend requests data for specific runs and metrics. Backend handles downsampling, block alignment, and missing data.

```
GET /api/data?runs=L0064_T0.70_S42,L0128_T0.70_S42&metrics=entropy,compressibility_W_L&downsample=100
→ {
    "L0064_T0.70_S42": {
      "entropy": { "x": [...], "y": [...] },
      "compressibility_W_L": { "x": [...], "y": [...] }
    },
    ...
  }
```

- Step-level metrics: x = step number, y = value. Downsampled by striding or averaging.
- Block-level metrics: x = block midpoint step, y = value. No downsampling (already sparse).

## Chart Types

Start with two, designed to be extended:

### Time Series
- X = step (shared across all traces)
- Y = one or more metrics
- One trace per run, colored by run identity
- Secondary Y axis for metrics with different scales

### Phase Portrait
- X = metric A, Y = metric B
- One trace per run (scatter or connected line)
- Color by run, or by time (cividis colormap) for single-run view

### Future chart types (not MVP)
- Distribution: violin/histogram of metric values across runs or time blocks
- Heatmap: metric value in (L, T) grid space
- Transfer function: metric summary (mean, std) vs T or L

## URL State

URL hash encodes the full view configuration. Changing any control updates the hash; loading a URL restores the view exactly.

```
#runs=L0064_T0.70_S42,L0128_T0.70_S42&chart=timeseries&y=entropy,compressibility_W_L&downsample=100
#runs=L0064_T*_S42&chart=phase&x=entropy&y=compressibility_W_L
```

Run selectors support glob patterns (resolved server-side) so URLs stay readable even for multi-run views.

**State fields:**
- `runs` — selected run IDs or glob pattern
- `chart` — chart type
- `x`, `y` — metric(s) on each axis
- `downsample` — downsampling factor
- `xrange`, `yrange` — zoom state (optional, Plotly handles this natively)

### Favorites

Save button stores current URL hash + optional label to localStorage. Favorites panel shows saved views. Export as markdown links for pasting into observations.md.

## Backend Design (explorer.py)

```python
# Responsibilities:
# 1. Scan data/runs/ on startup, build run index
# 2. Lazy-load run data (parquet + analysis) on first request, cache in memory
# 3. Serve metric registry (derived from what's actually available in the data)
# 4. Serve data slices with downsampling
# 5. Serve static files (index.html, JS)
# 6. Resolve glob patterns in run selectors

# Key classes/functions:
# - RunIndex: scans data dir, holds metadata, resolves globs
# - MetricRegistry: defines available metrics, how to extract them
# - load_run_data(run_id, metrics): loads and caches, returns dict
# - downsample(series, factor): stride or average

# NOT responsible for:
# - Analysis computation (delegates to analyze.py)
# - Plot rendering (that's the frontend)
# - Any HTML generation
```

### Analysis integration

Backend imports `analyze_run()` and `default_window_sizes()` from `analyze.py`. Analysis is triggered lazily on first data request and cached. The explorer passes its already-loaded experiment DataFrame to `analyze_run(path, ws, exp=exp)` to avoid double-loading parquets. Analysis results are cached in a single `.analysis.pkl` per run, with incremental window-size support (requesting new W values only computes the missing ones).

### Metric registry is data-driven

On startup, backend checks what columns exist in the parquets and what's in the analysis caches. Only metrics that actually exist appear in the registry. If we add a new metric to analyze.py, it shows up in the explorer automatically once the cache is regenerated.

## Frontend Design (static/index.html)

Single HTML file. Dependencies loaded from CDN:
- Plotly.js (charting, pan/zoom/toggle built-in)
- Possibly Alpine.js or similar for reactive UI (evaluate whether vanilla JS suffices)

```
┌─────────────────────────────────────────────┐
│ Run Selector          │ Chart Controls      │
│ ☐ Group by L/T/seed   │ Chart type: [v]     │
│ ☑ L=64  T=0.70 S=42  │ X axis: [v]         │
│ ☑ L=128 T=0.70 S=42  │ Y axis: [v]         │
│ ☐ L=192 T=0.70 S=42  │ Downsample: [___]   │
│ ...                   │ [Share] [★ Save]    │
├─────────────────────────────────────────────┤
│                                             │
│              Plotly Chart                   │
│                                             │
│                                             │
└─────────────────────────────────────────────┘
```

### Run selector features
- Group by L, T, or seed — toggle entire groups
- Quick patterns: "all T=0.70", "all L=64", "crossover only"
- Visual indicator of what's selected vs available

### Reactivity
- Any control change → update URL hash → fetch data → rerender
- Debounce rapid changes (e.g. toggling multiple runs)
- Loading indicator for data fetches

## Interaction model

The explorer evolves along a single trajectory from read-only instrument to interactive control surface. Each phase adds capability without replacing what came before.

```
read-only                                              read-write
   |                                                       |
   popover → drawer → scroll → flags → jump-in → generate → steer
   |___________ explorer ____________|______ control ______|
```

### Click → popover → drawer flow

One interaction, not three features:

1. **Click any point on any chart** → small popover appears near the click. Shows ~20 tokens centered on that step, entropy-colored. Metadata line (run, step, metric values). Two actions: **[×] dismiss** or **[→ Open]**.
2. **Open** → right drawer slides in, showing W tokens either side of the clicked step. Chart X-axis indicates the visible range. W defaults to L but is adjustable.
3. **Now in continuous document mode.** Scroll text → chart position follows. Zoom chart → text viewport adjusts. Bidirectional sync.

### Adjustable display window (W_display)

The display window is decoupled from the model's context length L:
- Default: W_display = L (shows what the model could see)
- User can increase to see more surrounding context, or decrease for a tighter view
- The L-token context window is visually indicated within the text (shaded band = "what the model saw at this step")
- Slider or numeric input in drawer header

### Flags (annotations)

User-created markers on (run, step) pairs:
- Click a flag icon on the popover or drawer to mark the current position
- Each flag stores: run ID, step, label (optional), metric values at that point
- Flags appear as markers on all charts (vertical lines or diamonds)
- Listed in a sidebar section, clickable to navigate
- Persisted in localStorage (like favorites)
- Export as markdown table for observations.md

### Events system

A unified abstraction for navigation (now) and stop conditions (future):

```
Event = predicate on token stream state

Built-in predicates:
  eos                           # token is EOS
  entropy_above(threshold)      # entropy > X
  entropy_below(threshold)      # entropy < X
  compressibility_crosses(val)  # comp crosses threshold (either direction)
  flag                          # user-flagged step

Navigation:  [< prev event] [> next event]  with event type selector
Control:     "run forward until [event]"     same predicates, different backend
```

In read-only mode, events are evaluated against stored parquet data. In control mode, the same predicates become real-time stop conditions.

### Jump in seat

Reconstruct full model state at any step, cheaply:

1. Read `token_id[N-L+1 ... N]` from the parquet (the L tokens ending at step N)
2. One forward pass through the model → KV cache is rebuilt
3. User is "in the seat" at step N — sees the context, the phase-space position, all metrics

The parquet is the complete state record. The KV cache is a pure function of the last L tokens. No mid-run checkpoints needed — reconstruction is one forward pass (~milliseconds on GPU).

Changing L at jump-in is free: just take fewer or more trailing tokens. Memory-depth annealing becomes a slider you drag while watching the phase portrait respond.

### Live generation

From the jumped-in state:
- User adjusts parameters (T, L, W for measurement)
- Hits "generate" → tokens stream to the chart in real time via websocket
- Events system provides stop conditions
- New tokens append to a branch parquet (forked from the original run at step N)
- Phase portrait updates live — watch the trajectory evolve

### Embedding-space projections (future)

The phase portraits currently use derived metrics (entropy, compressibility) as axes. A natural extension:
- Project token embeddings (or hidden states) into 2D/3D via UMAP/t-SNE/PCA
- Plot trajectories through embedding space — the "true" phase portrait
- Overlay metric-derived phase portraits for comparison
- Steering in embedding space = directional perturbation of the generation process

This connects back to interaction-topology.md: the phase space IS the interface.

## Implementation Plan

### Phase 1: MVP (done)
- [x] `explorer.py`: run index, metric registry, data endpoint, static file serving
- [x] `static/index.html`: run selector, metric dropdowns, time series chart, URL hash sync
- [x] Phase portrait mode
- [x] Shareable URL hash state
- [x] Context inspection (click-to-read, scrubber, EOS navigation)
- [x] Favorites (localStorage + markdown export)
- [x] Run grouping (by L, T, seed)
- [x] Glob patterns in run selector

### Phase 2: Context redesign
- [ ] DOM restructure: `.workspace` wrapping `.chart-container` + `.context-drawer`
- [ ] Popover on chart click (lightweight, positioned near click point)
- [ ] Pin-to-drawer flow (popover → right drawer opens)
- [ ] Adjustable W_display in drawer header (independent of L)
- [ ] L-window highlight in continuous text (shaded band)
- [ ] Bidirectional scroll/zoom sync (text ↔ chart X axis)
- [ ] Resize handle between chart and drawer
- [ ] Flags: create, display on charts, list in sidebar, persist, export

### Phase 3: Events & extended views
- [ ] Event predicates (eos, entropy threshold, compressibility threshold, flag)
- [ ] Event navigation: [< prev] [> next] with type selector
- [ ] Multi-chart stacking (2 charts sharing context drawer)
- [ ] Secondary Y axis for dual-metric overlay
- [ ] Heatmap: summary metric across (L, T) grid
- [ ] Distribution views (violin/histogram per run or time block)

### Phase 4: Jump in seat
- [ ] `/api/reconstruct` endpoint: load model, feed L tokens, return ready state
- [ ] UI: "Jump in" button in drawer, shows model state indicator
- [ ] Parameter adjustment panel (T slider, L slider)
- [ ] Generate forward N steps (batch mode, results appended to chart)
- [ ] Events as stop conditions for generation

### Phase 5: Live control
- [ ] WebSocket for streaming generation (token-by-token metrics to chart)
- [ ] Real-time phase portrait updates
- [ ] Branch management: fork runs at arbitrary points, compare branches
- [ ] Multi-scale measurement during live generation (W as observer parameter)
- [ ] Embedding-space projection views (UMAP/t-SNE of hidden states)

## Open Questions

- **Popover positioning**: anchor to click point (Plotly coordinates → screen coordinates) or to a fixed position near the chart edge? Click-anchored is more natural but needs Plotly coordinate mapping.
- **Continuous scroll performance**: 10k+ tokens in DOM may need virtualization. Measure first — modern browsers handle large DOM well with `content-visibility: auto`.
- **Branch divergence visualization**: when a user generates forward from step N, how to show the branch point and diverging trajectories on the same chart? Plotly trace grouping, or separate subplot?
- **Embedding projections**: compute server-side (hidden states are large) or ship a reduced representation? PCA components could be pre-computed and cached like compressibility.
- **Events UI**: dropdown selector for predicate type, or a mini query language? Start with dropdown, consider DSL if combinatorial predicates are needed (e.g., "entropy > 3 AND compressibility < 0.3").

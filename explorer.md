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

## Implementation Plan

### Phase 1: MVP
- [ ] Add fastapi + uvicorn deps
- [ ] `explorer.py`: run index, metric registry, data endpoint, static file serving
- [ ] `static/index.html`: run selector, metric dropdowns, time series chart, URL hash sync
- [ ] Basic phase portrait mode
- [ ] Verify shareable URLs work end-to-end

### Phase 2: Polish
- [ ] Favorites (localStorage + export as markdown)
- [ ] Run grouping (by L, T, seed)
- [ ] Glob patterns in run selector
- [ ] Secondary Y axis for dual-metric views
- [ ] Color schemes: by run identity, by time, by parameter value

### Phase 3: Extended charts
- [ ] Heatmap: summary metric across (L, T) grid
- [ ] Distribution views
- [ ] Transfer function curves

### Phase 4: Live control (future)
- [ ] WebSocket for streaming data from active generation
- [ ] Control panel: T and L sliders that send commands to running generation
- [ ] Real-time chart updates

## Open Questions

- **Alpine.js vs vanilla JS**: How complex does the UI state get before we need a reactive framework? Start vanilla, add Alpine if wiring gets painful.
- **Downsampling strategy**: Stride (fast, may miss spikes) vs. LTTB (better visual fidelity, more complex). Start with stride.
- **Multi-Y-axis**: Plotly supports it but it gets messy with many metrics. May want a "overlay" vs "subplot" toggle.
- **Data size**: 24 runs × 100k steps × few columns = manageable in browser memory. At 100+ runs, may need server-side aggregation.

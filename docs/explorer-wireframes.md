# Explorer Layout Wireframes

## Problem

The MVP explorer has a left sidebar (run selector + chart controls + favorites) and a main chart area. Context inspection (click a point, see the L tokens at that step) was added as a bottom pane inside `.chart-area`. This is broken:

- Chart autoscaling fights the panel — Plotly resize events are unreliable when the container shrinks
- The spatial relationship between chart point and context text is weak — you click at x=45000 on the chart, then look *down* at a text pane with no spatial link
- The bottom pane eats vertical chart space, which is the most valuable axis for time series
- No room for multi-chart (future)
- Context panel navigation (scrubber, EOS jumps) takes up vertical space before you even see text

## Current layout (for reference)

```
+------+------------------------------------------+
|header|  autoloop explorer           2/24 runs    |
+------+------------------------------------------+
|      |                                           |
|side  |                                           |
|bar   |              Chart (Plotly)               |
|      |                                           |
|runs  |                                           |
|      |                                           |
|ctrls +-------------------------------------------+
|      | Context: L=64 T=0.50 S=42          [x]   |
|favs  | Step [45000] [<EOS] [<] [>] [>EOS]        |
|      | =====[======||=========]======= (scrubber)|
|      | the quick brown fox jumped over the lazy  |
|      | dog and then the |cat| sat on the mat ... |
+------+-------------------------------------------+
```

Problems: chart loses ~40% of vertical space when context opens. Scrubber + nav + header = ~80px overhead before any text. No visual link between the clicked chart point and the text below.

## Design constraints

- Desktop, large monitor (1920x1080 or larger)
- Chart is king — maximum area, always
- Context inspection is a secondary view that should feel connected to the chart, not bolted on
- Must not break when context is closed (most of the time it's closed)
- Multi-chart (future): two metrics stacked or side-by-side
- Keyboard-navigable (future, but don't preclude it)
- Research tool — information density over polish

## Workflows

| ID | Workflow | Key requirement |
|----|----------|-----------------|
| A | Exploration | Chart dominates, context is quick peek |
| B | Comparison | Multiple runs overlaid, inspect both at same step |
| C | Documentation | Save view, copy link — sidebar-level |
| D | Deep inspection | Context dominates, scrub through time, chart is reference |

---

## Option 1: Right drawer

Context panel slides in from the right as a vertical drawer, splitting the chart area horizontally.

### Closed

```
+------+--------------------------------------------------+
|      |                                                   |
|side  |                                                   |
|bar   |                  Chart (full width)                |
|280px |                                                   |
|      |                                                   |
|      |                                                   |
+------+--------------------------------------------------+
```

### Open (default ~400px, resizable)

```
+------+-------------------------------+------------------+
|      |                               | Context          |
|side  |                               | L=64 T=0.50 S=42|
|bar   |        Chart (shrinks         |                  |
|      |         horizontally)         | Step [45000]     |
|      |                               | [<EOS][<][>][>EOS|
|      |                               |                  |
|      |         * <-- click here      | ...the quick     |
|      |         |                     | brown fox jumped  |
|      |         | (vert. line at step)| over the |lazy|  |
|      |                               | dog and then ... |
|      |                               |                  |
|      |                               | ---scrubber---   |
+------+-------------------------------+------------------+
       |<---- resizable drag handle -->|
```

### Good at

- **Chart keeps full vertical height** — time series Y axis is never compromised
- **Vertical step indicator on chart and text are side by side** — look left at the metric spike, look right at the text. Spatial relationship is strong.
- **Deep inspection (D)**: text gets full height, good for reading long context windows
- **Multi-chart (future)**: stack two charts vertically in the left portion; context panel serves both
- **Natural keyboard flow**: Tab between chart and context, arrow keys in context

### Bad at

- **Horizontal chart space reduced** — for time series, you lose some X-axis resolution when context is open. On a wide monitor this is fine. On 1920px, sidebar(280) + context(400) = 680px, leaving 1240px for chart. Acceptable.
- **Comparison (B)**: only shows one run's context at a time (but this is true of all layouts)
- **Phase portraits**: the vertical step indicator doesn't apply; would need a highlighted point instead

### Multi-chart variant

```
+------+-------------------------------+------------------+
|      |        Chart 1 (entropy)      | Context          |
|side  |                               |                  |
|bar   +-------------------------------+ (shared across   |
|      |        Chart 2 (compress.)    |  both charts)    |
|      |                               |                  |
+------+-------------------------------+------------------+
```

---

## Option 2: Bottom pane (fixed, minimal)

Keep the bottom position but redesign to be much thinner — a single-line scrubber strip that expands only on demand.

### Closed

```
+------+--------------------------------------------------+
|      |                                                   |
|side  |                  Chart (full height)               |
|bar   |                                                   |
|      |                                                   |
+------+--------------------------------------------------+
```

### Collapsed (after clicking a point — just the scrubber/indicator bar)

```
+------+--------------------------------------------------+
|      |                                                   |
|side  |                  Chart                            |
|bar   |                    * <-- clicked                  |
|      |                    |                              |
|      |                    v                              |
+------+--------------------------------------------------+
| Step 45000 [<EOS][<][>][>EOS]  =====[===||====]======== |
+------+--------------------------------------------------+
```

~36px strip. Shows step position, scrubber, nav. No text yet.

### Expanded (click the strip or press Enter)

```
+------+--------------------------------------------------+
|      |                                                   |
|side  |                  Chart (loses ~30% height)        |
|bar   |                                                   |
+------+--------------------------------------------------+
| Step 45000 [<EOS][<][>][>EOS]  =====[===||====]======== |
|                                                          |
| ...the quick brown fox jumped over the |lazy| dog and   |
| then the cat sat on the mat and the bird flew away...   |
|                                                          |
+----------------------------------------------------------+
```

### Good at

- **Minimal intrusion when just scrubbing** — the collapsed bar is nearly free
- **Exploration (A)**: chart dominates, scrubber is a thin tool
- **Horizontal text layout** — long context wraps naturally in a wide pane

### Bad at

- **Deep inspection (D)**: expanded mode steals vertical space from the chart, same as current MVP
- **Spatial disconnect** — chart point is above, text is below, separated by the scrubber bar. The connection is only through the vertical line indicator.
- **Multi-chart (future)**: stacking charts AND a bottom context pane means three vertical sections fighting for space
- **Resize is awkward** — how tall should the expanded text area be? If user resizes, chart needs to re-layout. Current bug lives here.

---

## Option 3: Overlay / floating panel

Context is a floating panel (like a devtools inspector) that can be positioned anywhere, including overlaid on the chart.

### Closed

```
+------+--------------------------------------------------+
|      |                                                   |
|side  |                  Chart (full)                     |
|bar   |                                                   |
|      |                                                   |
+------+--------------------------------------------------+
```

### Open (floating, draggable)

```
+------+--------------------------------------------------+
|      |                                                   |
|side  |           *  <-- clicked point                    |
|bar   |          /                                        |
|      |    +----/---------------------------+             |
|      |    | Context  L=64 T=0.50 S=42 [x]  |             |
|      |    | Step [45000] [<EOS][<][>][>EOS] |             |
|      |    | ...the quick brown fox jumped   |             |
|      |    | over the |lazy| dog and then    |             |
|      |    | the cat sat on the mat...       |             |
|      |    +--------------------------------+             |
|      |                                                   |
+------+--------------------------------------------------+
```

### Good at

- **Zero layout disruption** — chart never resizes, no Plotly relayout bugs
- **Flexible positioning** — user drags it where convenient
- **Quick peek** — open, read, close. No layout shift.

### Bad at

- **Occludes the chart** — the thing you're inspecting is hidden behind the inspector. Defeats the purpose.
- **Dragging state** — position needs to be saved/restored, fragile
- **Multi-chart**: floating panels over stacked charts is confusing
- **Deep inspection (D)**: can't make the panel large enough to read comfortably without covering the chart entirely
- **Comparison (B)**: two floating panels? No.
- **Mobile/resize**: floating panel positions break on window resize

---

## Recommendation: Option 1 (Right drawer)

**Rationale:**

1. **Spatial coherence.** The vertical step indicator on the chart and the context text are side by side. When you scrub through time, your eyes move horizontally between metric values (left) and text (right). This is the natural reading direction for "what happened at this step?"

2. **Chart keeps its vertical axis.** Time series are the primary chart type. Vertical space is entropy/compressibility range — compressing that is worse than compressing the time axis. A 1240px-wide chart on a 1920px monitor is plenty of X resolution, especially with pan/zoom.

3. **Multi-chart stacking works.** Two vertically-stacked charts in the left portion with a shared context panel on the right is the natural extension. The context panel becomes a "detail view" for whichever chart you clicked.

4. **Clean state transitions.** Closed = chart takes full width. Open = chart animates narrower, context slides in. No vertical reflow, no Plotly height bugs. A horizontal resize is much more reliable than vertical for Plotly — it just reflows the X axis.

5. **Deep inspection.** Full-height text area means ~800px of vertical space for context tokens. At 14px font + 1.7 line-height, that is ~33 lines visible — more than enough for a 256-token context window.

6. **Scrubber fits naturally.** Put the scrubber at the top of the context panel (horizontal, full-width of the panel). Nav buttons below it. Text fills the rest. No overhead eating into chart space.

### Implementation sketch

```
Layout structure:

.main {
  display: flex;         /* horizontal */
}

.sidebar { width: 280px; }

.workspace {
  flex: 1;
  display: flex;         /* horizontal */
}

.chart-container {
  flex: 1;               /* takes remaining space */
  display: flex;
  flex-direction: column; /* for future multi-chart stacking */
}

.context-drawer {
  width: 0;              /* closed */
  transition: width 0.2s;
  overflow: hidden;
}

.context-drawer.open {
  width: 400px;          /* default, resizable via drag handle */
  min-width: 300px;
  max-width: 600px;
}
```

### Context drawer internal layout

```
+------------------+
| L=64 T=0.50  [x] |  <- header (fixed, 32px)
+------------------+
| =[====||=====]=  |  <- scrubber with EOS ticks (fixed, 36px)
+------------------+
| Step [45000]     |  <- nav row (fixed, 32px)
| [<EOS][<][>][>EOS|
+------------------+
|                  |  <- token text (flex: 1, scrollable)
| the quick brown  |
| fox jumped over  |
| the |lazy| dog   |
| and then the cat |
| sat on the mat   |
| and the bird     |
| flew away over   |
| the hills and    |
| far away...      |
|                  |
+------------------+
```

### Resizable drag handle

A 4px vertical strip between `.chart-container` and `.context-drawer`. Cursor: `col-resize`. Drag to resize. Store width in localStorage.

### Chart-context visual link

When context is open, the chart shows a vertical dashed line at the inspected step (already implemented). Enhancement: the line color matches the inspected run's trace color, not just accent.

### Keyboard shortcuts (future-ready)

| Key | Action |
|-----|--------|
| `Escape` | Close context panel |
| `[` / `]` | Previous / next step |
| `{` / `}` | Previous / next EOS |
| `G` + number | Jump to step |
| `C` | Toggle context panel |

These don't need to be implemented now, but the layout doesn't preclude them.

### Multi-chart stacking (future)

```
+------+-------------------------------+------------------+
|      | Chart 1                       |                  |
|side  | (entropy time series)    [=]  | Context          |
|bar   +-------------------------------+                  |
|      | Chart 2                       | (shared, shows   |
|      | (compressibility)        [=]  |  step from       |
|      |                               |  whichever chart |
+------+-------------------------------+  was clicked)    |
                                       +------------------+
```

Each chart gets an equal vertical slice of `.chart-container`. A small `[=]` button on each chart allows expanding it to full height. Both charts share the vertical step indicator line.

### Heatmap/grid view (future)

When chart type is "heatmap", the context panel could repurpose as a "cell detail" panel — click a cell in the (L, T) grid, see summary stats and a mini time series for that condition.

### URL state additions

Add `ctx=RUNID:STEP` to the hash when context is open. Example:
```
#runs=L0064_T*_S42&chart=timeseries&y=entropy&ctx=L0064_T0.50_S42:45000
```

This makes context inspection shareable — paste the URL, collaborator sees the same text at the same step.

### Migration from current layout

1. Move `.context-panel` from inside `.chart-area` to a new `.context-drawer` sibling
2. Change `.chart-area` flex-direction from column to... keep it column (for future chart stacking), but wrap both in a horizontal `.workspace` flex container
3. Remove `max-height: 50vh` and `min-height: 180px` from context panel — it now takes full height
4. Add resize handle between chart and context
5. Transition: `width` instead of `display: none/flex`
6. Keep all existing context JS (fetch, render, nav) — only the DOM structure and CSS changes

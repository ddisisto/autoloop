// ---------------------------------------------------------------------------
// Panel system: composable chart strips with shared X axis
// ---------------------------------------------------------------------------
import { state, fetchData, getRunColor, getRunDash, getRunLabel, nextPanelId } from './state.js';
import { updateStatus } from './app.js';
import { ctxState, openContextPanel, updateStepIndicator, onChartZoom } from './context.js';

const _selfZoom = {}; // panelId -> bool, prevents zoom echo loops

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------
function panelChartId(panelId) { return 'panel-chart-' + panelId; }

function metricName(id) {
  const m = state.metrics.find(m => m.id === id);
  return m ? m.name : id;
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------
function darkLayout(panel) {
  const layout = {
    template: 'plotly_dark',
    paper_bgcolor: '#1a1a2e',
    plot_bgcolor: '#16213e',
    font: { family: 'Consolas, Monaco, Courier New, monospace', color: '#e0e0e0' },
    margin: { t: 8, r: panel.metrics.y2 ? 60 : 30, b: 36, l: 60 },
    legend: {
      bgcolor: 'rgba(22, 33, 62, 0.8)',
      bordercolor: '#333', borderwidth: 1,
      font: { size: 10 },
      orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom',
    },
    xaxis: {
      gridcolor: '#333', zeroline: false,
      title: panel.type === 'phase' ? metricName(panel.metrics.x) : null,
    },
    yaxis: {
      title: metricName(panel.metrics.y1),
      gridcolor: '#333', zeroline: false,
    },
  };

  if (panel.metrics.y2) {
    layout.yaxis2 = {
      title: metricName(panel.metrics.y2),
      overlaying: 'y', side: 'right',
      gridcolor: 'rgba(51,51,51,0.3)', zeroline: false,
    };
  }

  // Apply shared X range for timeseries
  if (panel.type === 'timeseries' && state.xRange) {
    layout.xaxis.range = [state.xRange.min, state.xRange.max];
  }

  return layout;
}

// ---------------------------------------------------------------------------
// Trace building
// ---------------------------------------------------------------------------
function buildTraces(panel, data, runIds) {
  const traces = [];

  if (panel.type === 'timeseries') {
    for (const metricKey of ['y1', 'y2']) {
      const metricId = panel.metrics[metricKey];
      if (!metricId) continue;
      const yaxis = metricKey === 'y2' ? 'y2' : undefined;
      const opacity = metricKey === 'y2' ? 0.6 : 1;

      for (const runId of runIds) {
        const runData = data[runId];
        if (!runData || !runData[metricId]) continue;
        const run = state.runs.find(r => r.id === runId);
        if (!run) continue;

        const label = getRunLabel(run);
        const suffix = panel.metrics.y2 ? ` (${metricName(metricId)})` : '';
        traces.push({
          x: runData[metricId].x,
          y: runData[metricId].y,
          type: 'scattergl',
          mode: 'lines',
          name: label + suffix,
          yaxis,
          line: { color: getRunColor(run), dash: getRunDash(run), width: 1.5 },
          opacity,
          hovertemplate: `${label}<br>step=%{x}<br>${metricName(metricId)}=%{y:.4f}<extra></extra>`,
          _runId: runId,
        });
      }
    }
  } else if (panel.type === 'phase') {
    const xId = panel.metrics.x;
    const yId = panel.metrics.y1;
    if (!xId || !yId) return traces;

    for (const runId of runIds) {
      const runData = data[runId];
      if (!runData || !runData[xId] || !runData[yId]) continue;
      const run = state.runs.find(r => r.id === runId);
      if (!run) continue;

      const xd = runData[xId], yd = runData[yId];
      const len = Math.min(xd.y.length, yd.y.length);
      traces.push({
        x: xd.y.slice(0, len),
        y: yd.y.slice(0, len),
        type: 'scattergl',
        mode: 'lines+markers',
        name: getRunLabel(run),
        line: { color: getRunColor(run), dash: getRunDash(run), width: 1 },
        marker: { size: 3, color: getRunColor(run), opacity: 0.6 },
        hovertemplate: `${getRunLabel(run)}<br>${metricName(xId)}=%{x:.4f}<br>${metricName(yId)}=%{y:.4f}<extra></extra>`,
        _runId: runId,
      });
    }
  }

  return traces;
}

// ---------------------------------------------------------------------------
// Panel DOM creation
// ---------------------------------------------------------------------------
function createMetricPill(panelId, axis, metricId) {
  const pill = document.createElement('span');
  pill.className = 'metric-pill';
  pill.dataset.axis = axis;

  const label = document.createElement('span');
  label.textContent = metricName(metricId);
  pill.appendChild(label);

  const remove = document.createElement('button');
  remove.className = 'pill-remove';
  remove.textContent = '\u00d7';
  remove.title = 'Remove metric';
  remove.addEventListener('click', (e) => {
    e.stopPropagation();
    const panel = state.panels.find(p => p.id === panelId);
    if (!panel) return;
    panel.metrics[axis] = null;
    renderPanel(panel);
    updatePanelDOM(panel);
  });

  pill.appendChild(remove);
  return pill;
}

export function createPanelDOM(panel) {
  const strip = document.createElement('div');
  strip.className = 'panel-strip';
  strip.dataset.panelId = panel.id;

  // Header
  const header = document.createElement('div');
  header.className = 'panel-header';

  const metricBar = document.createElement('div');
  metricBar.className = 'panel-metric-bar';
  metricBar.dataset.panelId = panel.id;
  buildMetricBar(metricBar, panel);

  const actions = document.createElement('div');
  actions.className = 'panel-actions';

  const typeSelect = document.createElement('select');
  typeSelect.className = 'panel-type-select';
  for (const [val, label] of [['timeseries', 'TS'], ['phase', 'Phase']]) {
    const opt = document.createElement('option');
    opt.value = val; opt.textContent = label;
    typeSelect.appendChild(opt);
  }
  typeSelect.value = panel.type;
  typeSelect.addEventListener('change', () => {
    panel.type = typeSelect.value;
    if (panel.type === 'phase' && !panel.metrics.x) {
      const comp = state.metrics.find(m => m.id.startsWith('compressibility'));
      panel.metrics.x = comp ? comp.id : (state.metrics[1] ? state.metrics[1].id : state.metrics[0]?.id);
    }
    updatePanelDOM(panel);
    renderPanel(panel);
  });

  const closeBtn = document.createElement('button');
  closeBtn.className = 'panel-close';
  closeBtn.textContent = '\u00d7';
  closeBtn.title = 'Close panel';
  closeBtn.addEventListener('click', () => removePanel(panel.id));

  actions.appendChild(typeSelect);
  actions.appendChild(closeBtn);
  header.appendChild(metricBar);
  header.appendChild(actions);
  strip.appendChild(header);

  // Chart div
  const chartDiv = document.createElement('div');
  chartDiv.className = 'panel-chart';
  chartDiv.id = panelChartId(panel.id);
  chartDiv.style.height = panel.height + 'px';
  strip.appendChild(chartDiv);

  // Resize handle
  const resizeHandle = document.createElement('div');
  resizeHandle.className = 'panel-resize-handle';
  wireResizeHandle(resizeHandle, panel, chartDiv);
  strip.appendChild(resizeHandle);

  return strip;
}

function buildMetricBar(bar, panel) {
  bar.innerHTML = '';

  // For phase: show x metric pill
  if (panel.type === 'phase' && panel.metrics.x) {
    const xPill = createMetricPill(panel.id, 'x', panel.metrics.x);
    xPill.classList.add('axis-x');
    const sep = document.createElement('span');
    sep.className = 'axis-separator';
    sep.textContent = 'vs';
    bar.appendChild(xPill);
    bar.appendChild(sep);
  }

  if (panel.metrics.y1) {
    bar.appendChild(createMetricPill(panel.id, 'y1', panel.metrics.y1));
  }
  if (panel.metrics.y2) {
    bar.appendChild(createMetricPill(panel.id, 'y2', panel.metrics.y2));
  }

  // Add metric button
  const addBtn = document.createElement('button');
  addBtn.className = 'add-metric-btn';
  addBtn.textContent = '+';
  addBtn.title = 'Add metric';
  addBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    showMetricDropdown(panel.id, addBtn);
  });
  bar.appendChild(addBtn);
}

function updatePanelDOM(panel) {
  const strip = document.querySelector(`.panel-strip[data-panel-id="${panel.id}"]`);
  if (!strip) return;
  const bar = strip.querySelector('.panel-metric-bar');
  if (bar) buildMetricBar(bar, panel);
  const typeSelect = strip.querySelector('.panel-type-select');
  if (typeSelect) typeSelect.value = panel.type;
}

// ---------------------------------------------------------------------------
// Metric dropdown picker
// ---------------------------------------------------------------------------
let activeDropdown = null;

function closeDropdown() {
  if (activeDropdown) {
    activeDropdown.remove();
    activeDropdown = null;
  }
  document.removeEventListener('click', onDropdownOutsideClick);
}

function onDropdownOutsideClick(e) {
  if (activeDropdown && !activeDropdown.contains(e.target)) closeDropdown();
}

function showMetricDropdown(panelId, anchor) {
  closeDropdown();
  const panel = state.panels.find(p => p.id === panelId);
  if (!panel) return;

  const dd = document.createElement('div');
  dd.className = 'metric-dropdown';

  const search = document.createElement('input');
  search.className = 'metric-dropdown-search';
  search.placeholder = 'Search metrics...';
  dd.appendChild(search);

  const list = document.createElement('div');
  dd.appendChild(list);

  function renderItems(filter) {
    list.innerHTML = '';
    const f = (filter || '').toLowerCase();
    for (const m of state.metrics) {
      if (f && !m.name.toLowerCase().includes(f) && !m.id.toLowerCase().includes(f)) continue;
      const item = document.createElement('div');
      item.className = 'metric-dropdown-item';
      item.textContent = `${m.name} [${m.resolution}]`;
      item.addEventListener('click', () => {
        // Determine which axis slot to fill
        if (panel.type === 'phase' && !panel.metrics.x) {
          panel.metrics.x = m.id;
        } else if (!panel.metrics.y1) {
          panel.metrics.y1 = m.id;
        } else if (!panel.metrics.y2) {
          panel.metrics.y2 = m.id;
        } else {
          // Replace y1
          panel.metrics.y1 = m.id;
        }
        closeDropdown();
        updatePanelDOM(panel);
        renderPanel(panel);
      });
      list.appendChild(item);
    }
  }

  search.addEventListener('input', () => renderItems(search.value));
  renderItems('');

  // Position relative to anchor
  const rect = anchor.getBoundingClientRect();
  dd.style.position = 'fixed';
  dd.style.left = rect.left + 'px';
  dd.style.top = rect.bottom + 2 + 'px';

  document.body.appendChild(dd);
  activeDropdown = dd;
  search.focus();

  setTimeout(() => document.addEventListener('click', onDropdownOutsideClick), 0);
}

// ---------------------------------------------------------------------------
// Panel resize handle
// ---------------------------------------------------------------------------
function wireResizeHandle(handle, panel, chartDiv) {
  let startY = 0, startH = 0;

  handle.addEventListener('mousedown', (e) => {
    e.preventDefault();
    startY = e.clientY;
    startH = chartDiv.offsetHeight;
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';

    const onMove = (e) => {
      const delta = e.clientY - startY;
      const newH = Math.max(120, Math.min(600, startH + delta));
      panel.height = newH;
      chartDiv.style.height = newH + 'px';
      Plotly.Plots.resize(chartDiv.id);
    };
    const onUp = () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
}

// ---------------------------------------------------------------------------
// Render a single panel (fetch + plot)
// ---------------------------------------------------------------------------
async function renderPanel(panel) {
  const selectedIds = Array.from(state.selectedRuns);
  const chartId = panelChartId(panel.id);
  const chartEl = document.getElementById(chartId);
  if (!chartEl) return;

  if (selectedIds.length === 0) {
    Plotly.react(chartId, [], {
      ...darkLayout(panel),
      annotations: [{ text: 'Select runs', xref: 'paper', yref: 'paper',
        x: 0.5, y: 0.5, showarrow: false, font: { size: 14, color: '#888' } }],
    }, { responsive: true });
    return;
  }

  // Collect needed metrics
  const needed = new Set();
  if (panel.metrics.y1) needed.add(panel.metrics.y1);
  if (panel.metrics.y2) needed.add(panel.metrics.y2);
  if (panel.type === 'phase' && panel.metrics.x) needed.add(panel.metrics.x);
  if (needed.size === 0) return;

  try {
    const data = await fetchData(selectedIds, [...needed], state.downsample);
    const traces = buildTraces(panel, data, selectedIds);

    await Plotly.react(chartId, traces, darkLayout(panel), { responsive: true });
    wirePanelEvents(panel);

    // Restore step indicator if context panel is open
    if (ctxState.open && panel.type === 'timeseries') {
      addStepIndicatorToPanel(panel, ctxState.step);
    }
  } catch (err) {
    console.error(`Panel ${panel.id} render failed:`, err);
  }
}

// ---------------------------------------------------------------------------
// Wire click + zoom events per panel
// ---------------------------------------------------------------------------
function wirePanelEvents(panel) {
  const chartId = panelChartId(panel.id);
  const chartEl = document.getElementById(chartId);
  if (!chartEl) return;

  chartEl.removeAllListeners && chartEl.removeAllListeners();

  // Zoom sync
  chartEl.on('plotly_relayout', (data) => {
    if (_selfZoom[panel.id]) {
      _selfZoom[panel.id] = false;
      return;
    }

    if (panel.type !== 'timeseries') return;

    if (data['xaxis.autorange']) {
      state.xRange = null;
    } else if (data['xaxis.range[0]'] != null) {
      state.xRange = { min: Math.round(data['xaxis.range[0]']), max: Math.round(data['xaxis.range[1]']) };
    } else if (data['xaxis.range']) {
      state.xRange = { min: Math.round(data['xaxis.range'][0]), max: Math.round(data['xaxis.range'][1]) };
    } else {
      return;
    }

    // Sync other timeseries panels
    for (const other of state.panels) {
      if (other.id === panel.id || other.type !== 'timeseries') continue;
      const otherId = panelChartId(other.id);
      _selfZoom[other.id] = true;
      if (state.xRange) {
        Plotly.relayout(otherId, { 'xaxis.range': [state.xRange.min, state.xRange.max] });
      } else {
        Plotly.relayout(otherId, { 'xaxis.autorange': true });
      }
    }

    // Sync context drawer
    onChartZoom(data);
  });

  // Click to open context
  chartEl.on('plotly_click', (eventData) => {
    if (!eventData || !eventData.points || eventData.points.length === 0) return;
    const point = eventData.points[0];
    const trace = chartEl.data[point.curveNumber];
    const runId = trace?._runId;
    if (!runId) return;

    let step;
    if (panel.type === 'timeseries') {
      step = Math.round(point.x);
    } else {
      step = point.pointIndex * state.downsample;
    }
    openContextPanel(runId, step);
  });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
export async function renderAllPanels() {
  const promises = state.panels.map(p => renderPanel(p));
  await Promise.all(promises);
  updateStatus();
}

export function addPanel(config) {
  const panel = {
    id: nextPanelId(),
    type: config?.type || 'timeseries',
    metrics: {
      y1: config?.y1 || (state.metrics[0] ? state.metrics[0].id : ''),
      y2: config?.y2 || null,
      x: config?.x || null,
    },
    height: config?.height || 250,
  };
  state.panels.push(panel);

  const container = document.getElementById('panelContainer');
  container.appendChild(createPanelDOM(panel));
  renderPanel(panel);
  return panel;
}

export function removePanel(panelId) {
  state.panels = state.panels.filter(p => p.id !== panelId);
  const strip = document.querySelector(`.panel-strip[data-panel-id="${panelId}"]`);
  if (strip) strip.remove();
  // Don't allow empty — add default if last panel removed
  if (state.panels.length === 0) {
    addPanel({ y1: 'entropy' });
  }
}

export function rebuildAllPanelDOM() {
  const container = document.getElementById('panelContainer');
  container.innerHTML = '';
  for (const panel of state.panels) {
    container.appendChild(createPanelDOM(panel));
  }
}

export function resizeAllPanels() {
  for (const panel of state.panels) {
    const el = document.getElementById(panelChartId(panel.id));
    if (el) Plotly.Plots.resize(el);
  }
}

// ---------------------------------------------------------------------------
// Step indicator (called from context.js)
// ---------------------------------------------------------------------------
function addStepIndicatorToPanel(panel, step) {
  const chartId = panelChartId(panel.id);
  const chartEl = document.getElementById(chartId);
  if (!chartEl || !chartEl.layout) return;

  const shapes = (chartEl.layout.shapes || []).filter(s => !s._autoloop_step_indicator);
  shapes.push({
    type: 'line', x0: step, x1: step, y0: 0, y1: 1, yref: 'paper',
    line: { color: 'rgba(83, 194, 201, 0.6)', width: 1.5, dash: 'dot' },
    _autoloop_step_indicator: true,
  });
  Plotly.relayout(chartId, { shapes });
}

export function updateStepIndicatorAll(step) {
  for (const panel of state.panels) {
    if (panel.type === 'timeseries') {
      addStepIndicatorToPanel(panel, step);
    }
  }
}

export function removeStepIndicatorAll() {
  for (const panel of state.panels) {
    const chartId = panelChartId(panel.id);
    const chartEl = document.getElementById(chartId);
    if (!chartEl || !chartEl.layout) continue;
    const shapes = (chartEl.layout.shapes || []).filter(s => !s._autoloop_step_indicator);
    Plotly.relayout(chartId, { shapes });
  }
}

// Zoom from context overview bar → all panels
export function syncXRangeFromContext(min, max) {
  state.xRange = { min, max };
  for (const panel of state.panels) {
    if (panel.type !== 'timeseries') continue;
    const chartId = panelChartId(panel.id);
    _selfZoom[panel.id] = true;
    Plotly.relayout(chartId, { 'xaxis.range': [min, max] });
  }
}

export function resetXRange() {
  state.xRange = null;
  for (const panel of state.panels) {
    if (panel.type !== 'timeseries') continue;
    const chartId = panelChartId(panel.id);
    _selfZoom[panel.id] = true;
    Plotly.relayout(chartId, { 'xaxis.autorange': true });
  }
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  runs: [],            // from /api/runs
  metrics: [],         // from /api/metrics
  selectedRuns: new Set(),
  groupBy: 'L',
  chartType: 'timeseries',
  xMetric: '',
  yMetric: '',
  downsample: 500,
};

let debounceTimer = null;
let favoritesOpen = true;

// ---------------------------------------------------------------------------
// Color scheme
// ---------------------------------------------------------------------------
// Color by T value (sequential), differentiate L by dash style
const T_COLORS = {
  '0.50': '#636EFA',
  '0.60': '#5B8FF9',
  '0.70': '#00CC96',
  '0.80': '#19D3F3',
  '0.90': '#FECB52',
  '1.00': '#EF553B',
  '1.10': '#FF6692',
  '1.20': '#B6E880',
  '1.50': '#AB63FA',
};

const L_DASHES = {
  '64': 'solid',
  '128': 'dash',
  '192': 'dot',
  '256': 'dashdot',
  '512': 'longdash',
  '1024': 'longdashdot',
};

// Fallback palette for unknown T values
const PALETTE = [
  '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
  '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
];

function getRunColor(run) {
  const tKey = parseFloat(run.T).toFixed(2);
  return T_COLORS[tKey] || PALETTE[state.runs.indexOf(run) % PALETTE.length];
}

function getRunDash(run) {
  return L_DASHES[String(run.L)] || 'solid';
}

function getRunLabel(run) {
  return `L=${run.L} T=${parseFloat(run.T).toFixed(2)} S=${run.seed}`;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
async function apiFetch(path) {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`API error: ${resp.status} ${resp.statusText}`);
  return resp.json();
}

async function fetchRuns() {
  state.runs = await apiFetch('/api/runs');
}

async function fetchMetrics() {
  state.metrics = await apiFetch('/api/metrics');
}

async function fetchData(runIds, metricIds, downsample) {
  const params = new URLSearchParams();
  params.set('runs', runIds.join(','));
  params.set('metrics', metricIds.join(','));
  params.set('downsample', String(downsample));
  return apiFetch(`/api/data?${params}`);
}

// ---------------------------------------------------------------------------
// UI: Run list
// ---------------------------------------------------------------------------
function renderRunList() {
  const container = document.getElementById('runList');
  container.innerHTML = '';

  const groupKey = state.groupBy;
  const groups = {};
  for (const run of state.runs) {
    const key = String(run[groupKey]);
    if (!groups[key]) groups[key] = [];
    groups[key].push(run);
  }

  // Sort group keys numerically
  const sortedKeys = Object.keys(groups).sort((a, b) => parseFloat(a) - parseFloat(b));

  for (const key of sortedKeys) {
    const groupRuns = groups[key];
    const groupDiv = document.createElement('div');
    groupDiv.className = 'run-group';

    // Group header with checkbox
    const allSelected = groupRuns.every(r => state.selectedRuns.has(r.id));
    const someSelected = groupRuns.some(r => state.selectedRuns.has(r.id));

    const header = document.createElement('div');
    header.className = 'run-group-header';

    const groupCb = document.createElement('input');
    groupCb.type = 'checkbox';
    groupCb.checked = allSelected;
    groupCb.indeterminate = someSelected && !allSelected;
    groupCb.addEventListener('change', () => {
      for (const r of groupRuns) {
        if (groupCb.checked) state.selectedRuns.add(r.id);
        else state.selectedRuns.delete(r.id);
      }
      renderRunList();
      scheduleUpdate();
    });

    const label = document.createElement('span');
    label.textContent = `${groupKey}=${key}`;
    label.addEventListener('click', (e) => {
      e.stopPropagation();
      groupCb.checked = !groupCb.checked;
      groupCb.dispatchEvent(new Event('change'));
    });

    header.appendChild(groupCb);
    header.appendChild(label);
    groupDiv.appendChild(header);

    // Individual runs
    for (const run of groupRuns) {
      const item = document.createElement('div');
      item.className = 'run-item';

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = state.selectedRuns.has(run.id);
      cb.addEventListener('change', () => {
        if (cb.checked) state.selectedRuns.add(run.id);
        else state.selectedRuns.delete(run.id);
        renderRunList();
        scheduleUpdate();
      });

      const swatch = document.createElement('span');
      swatch.className = 'color-swatch';
      swatch.style.background = getRunColor(run);

      const text = document.createElement('span');
      text.textContent = getRunLabel(run);

      item.appendChild(cb);
      item.appendChild(swatch);
      item.appendChild(text);
      groupDiv.appendChild(item);
    }

    container.appendChild(groupDiv);
  }
}

// ---------------------------------------------------------------------------
// UI: Metric dropdowns
// ---------------------------------------------------------------------------
function populateMetricDropdowns() {
  const xSel = document.getElementById('xMetric');
  const ySel = document.getElementById('yMetric');
  xSel.innerHTML = '';
  ySel.innerHTML = '';

  for (const m of state.metrics) {
    const opt1 = document.createElement('option');
    opt1.value = m.id;
    opt1.textContent = `${m.name} [${m.resolution}]`;
    xSel.appendChild(opt1);

    const opt2 = opt1.cloneNode(true);
    ySel.appendChild(opt2);
  }

  // Restore from state or set defaults
  if (state.yMetric && ySel.querySelector(`option[value="${state.yMetric}"]`)) {
    ySel.value = state.yMetric;
  } else if (state.metrics.length > 0) {
    // Default: first metric that looks like entropy
    const entropyMetric = state.metrics.find(m => m.id === 'entropy');
    ySel.value = entropyMetric ? 'entropy' : state.metrics[0].id;
    state.yMetric = ySel.value;
  }

  if (state.xMetric && xSel.querySelector(`option[value="${state.xMetric}"]`)) {
    xSel.value = state.xMetric;
  } else if (state.metrics.length > 1) {
    // Default: second metric or compressibility
    const compMetric = state.metrics.find(m => m.id.startsWith('compressibility'));
    xSel.value = compMetric ? compMetric.id : state.metrics[1].id;
    state.xMetric = xSel.value;
  } else if (state.metrics.length > 0) {
    xSel.value = state.metrics[0].id;
    state.xMetric = xSel.value;
  }
}

function updateChartControlsVisibility() {
  const xRow = document.getElementById('xMetricRow');
  const yLabel = document.querySelector('#yMetric').parentElement.querySelector('label');
  if (state.chartType === 'phase') {
    xRow.style.display = 'flex';
    yLabel.textContent = 'Y axis';
  } else {
    xRow.style.display = 'none';
    yLabel.textContent = 'Y axis';
  }
}

// ---------------------------------------------------------------------------
// UI: Favorites
// ---------------------------------------------------------------------------
function getFavorites() {
  try {
    return JSON.parse(localStorage.getItem('autoloop_favorites') || '[]');
  } catch { return []; }
}

function saveFavorites(favs) {
  localStorage.setItem('autoloop_favorites', JSON.stringify(favs));
}

function renderFavorites() {
  const list = document.getElementById('favoritesList');
  const favs = getFavorites();
  list.innerHTML = '';

  if (favs.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'fav-empty';
    empty.textContent = 'No saved views yet.';
    list.appendChild(empty);
    return;
  }

  for (let i = 0; i < favs.length; i++) {
    const fav = favs[i];
    const item = document.createElement('div');
    item.className = 'fav-item';

    const link = document.createElement('a');
    link.href = '#' + fav.hash;
    link.textContent = fav.label || fav.hash.substring(0, 40);
    link.title = fav.label || fav.hash;
    link.addEventListener('click', (e) => {
      e.preventDefault();
      window.location.hash = fav.hash;
      loadFromHash();
    });

    const del = document.createElement('span');
    del.className = 'fav-delete';
    del.textContent = '\u00d7';
    del.title = 'Delete';
    del.addEventListener('click', () => {
      const updated = getFavorites();
      updated.splice(i, 1);
      saveFavorites(updated);
      renderFavorites();
    });

    item.appendChild(link);
    item.appendChild(del);
    list.appendChild(item);
  }
}

function toggleFavorites() {
  favoritesOpen = !favoritesOpen;
  const list = document.getElementById('favoritesList');
  const actions = document.querySelector('.fav-actions');
  const arrow = document.querySelector('.favorites-header .arrow');
  list.style.display = favoritesOpen ? '' : 'none';
  actions.style.display = favoritesOpen ? '' : 'none';
  arrow.classList.toggle('open', favoritesOpen);
}

// ---------------------------------------------------------------------------
// URL hash state
// ---------------------------------------------------------------------------
function encodeHash() {
  const parts = [];
  if (state.selectedRuns.size > 0) {
    parts.push('runs=' + Array.from(state.selectedRuns).join(','));
  }
  parts.push('chart=' + state.chartType);
  if (state.chartType === 'phase' && state.xMetric) {
    parts.push('x=' + state.xMetric);
  }
  if (state.yMetric) parts.push('y=' + state.yMetric);
  parts.push('downsample=' + state.downsample);
  parts.push('group=' + state.groupBy);
  return parts.join('&');
}

function updateHash() {
  const hash = encodeHash();
  history.replaceState(null, '', '#' + hash);
}

function decodeHash(hash) {
  const params = new URLSearchParams(hash);
  return {
    runs: params.get('runs') ? params.get('runs').split(',') : null,
    chart: params.get('chart') || 'timeseries',
    x: params.get('x') || '',
    y: params.get('y') || '',
    downsample: params.get('downsample') ? parseInt(params.get('downsample')) : 500,
    group: params.get('group') || 'L',
  };
}

function loadFromHash() {
  const hash = window.location.hash.replace(/^#/, '');
  if (!hash) return false;

  const decoded = decodeHash(hash);

  state.chartType = decoded.chart;
  state.downsample = decoded.downsample;
  state.groupBy = decoded.group;
  if (decoded.x) state.xMetric = decoded.x;
  if (decoded.y) state.yMetric = decoded.y;

  if (decoded.runs) {
    state.selectedRuns.clear();
    for (const id of decoded.runs) {
      // Only add if the run actually exists
      if (state.runs.some(r => r.id === id)) {
        state.selectedRuns.add(id);
      }
    }
  }

  // Update UI controls
  document.getElementById('chartType').value = state.chartType;
  document.getElementById('downsample').value = state.downsample;

  // Update group toggle
  for (const btn of document.querySelectorAll('#groupToggle button')) {
    btn.classList.toggle('active', btn.dataset.group === state.groupBy);
  }

  updateChartControlsVisibility();
  populateMetricDropdowns();
  renderRunList();
  scheduleUpdate(0);
  return true;
}

// ---------------------------------------------------------------------------
// Chart rendering
// ---------------------------------------------------------------------------
function showLoading(on) {
  document.getElementById('loadingOverlay').classList.toggle('active', on);
}

async function updateChart() {
  const selectedIds = Array.from(state.selectedRuns);
  if (selectedIds.length === 0 || state.metrics.length === 0) {
    Plotly.purge('chart');
    Plotly.newPlot('chart', [], {
      ...darkLayout(),
      annotations: [{
        text: selectedIds.length === 0
          ? 'Select one or more runs to begin.'
          : 'No metrics available.',
        xref: 'paper', yref: 'paper',
        x: 0.5, y: 0.5,
        showarrow: false,
        font: { size: 16, color: '#888' },
      }],
    }, { responsive: true });
    updateStatus();
    return;
  }

  // Determine which metrics to fetch
  let metricsToFetch = [];
  if (state.chartType === 'phase') {
    metricsToFetch = [state.xMetric, state.yMetric].filter(Boolean);
  } else {
    metricsToFetch = [state.yMetric].filter(Boolean);
  }

  if (metricsToFetch.length === 0) return;

  showLoading(true);

  try {
    const data = await fetchData(selectedIds, metricsToFetch, state.downsample);
    renderChart(data, selectedIds, metricsToFetch);
    // Restore step indicator if context panel is open
    if (ctxState.open && state.chartType === 'timeseries') {
      updateStepIndicator(ctxState.step);
    }
  } catch (err) {
    console.error('Failed to fetch data:', err);
    showToast('Failed to load data: ' + err.message, 'error');
  } finally {
    showLoading(false);
    updateStatus();
  }
}

function darkLayout() {
  return {
    template: 'plotly_dark',
    paper_bgcolor: '#1a1a2e',
    plot_bgcolor: '#16213e',
    font: { family: 'Consolas, Monaco, Courier New, monospace', color: '#e0e0e0' },
    margin: { t: 30, r: 30, b: 50, l: 60 },
    legend: {
      bgcolor: 'rgba(22, 33, 62, 0.8)',
      bordercolor: '#333',
      borderwidth: 1,
      font: { size: 11 },
    },
  };
}

function renderChart(data, runIds, metricsToFetch) {
  const traces = [];

  if (state.chartType === 'timeseries') {
    const metricId = state.yMetric;
    const metricDef = state.metrics.find(m => m.id === metricId);
    const metricName = metricDef ? metricDef.name : metricId;

    for (const runId of runIds) {
      const runData = data[runId];
      if (!runData || !runData[metricId]) continue;
      const run = state.runs.find(r => r.id === runId);
      if (!run) continue;

      traces.push({
        x: runData[metricId].x,
        y: runData[metricId].y,
        type: 'scattergl',
        mode: 'lines',
        name: getRunLabel(run),
        line: {
          color: getRunColor(run),
          dash: getRunDash(run),
          width: 1.5,
        },
        hovertemplate: `${getRunLabel(run)}<br>step=%{x}<br>${metricName}=%{y:.4f}<extra></extra>`,
      });
    }

    Plotly.react('chart', traces, {
      ...darkLayout(),
      xaxis: { title: 'Step', gridcolor: '#333', zeroline: false },
      yaxis: { title: metricName, gridcolor: '#333', zeroline: false },
    }, { responsive: true }).then(() => wireChartClickHandler());

  } else if (state.chartType === 'phase') {
    const xMetricId = state.xMetric;
    const yMetricId = state.yMetric;
    const xDef = state.metrics.find(m => m.id === xMetricId);
    const yDef = state.metrics.find(m => m.id === yMetricId);
    const xName = xDef ? xDef.name : xMetricId;
    const yName = yDef ? yDef.name : yMetricId;

    for (const runId of runIds) {
      const runData = data[runId];
      if (!runData || !runData[xMetricId] || !runData[yMetricId]) continue;
      const run = state.runs.find(r => r.id === runId);
      if (!run) continue;

      // For phase portrait, we need to align x and y data
      // Use the shorter array length
      const xd = runData[xMetricId];
      const yd = runData[yMetricId];
      const len = Math.min(xd.y.length, yd.y.length);

      traces.push({
        x: xd.y.slice(0, len),
        y: yd.y.slice(0, len),
        type: 'scattergl',
        mode: 'lines+markers',
        name: getRunLabel(run),
        line: {
          color: getRunColor(run),
          dash: getRunDash(run),
          width: 1,
        },
        marker: {
          size: 3,
          color: getRunColor(run),
          opacity: 0.6,
        },
        hovertemplate: `${getRunLabel(run)}<br>${xName}=%{x:.4f}<br>${yName}=%{y:.4f}<extra></extra>`,
      });
    }

    Plotly.react('chart', traces, {
      ...darkLayout(),
      xaxis: { title: xName, gridcolor: '#333', zeroline: false },
      yaxis: { title: yName, gridcolor: '#333', zeroline: false },
    }, { responsive: true }).then(() => wireChartClickHandler());
  }
}

// ---------------------------------------------------------------------------
// Debounced update
// ---------------------------------------------------------------------------
function scheduleUpdate(delay = 300) {
  clearTimeout(debounceTimer);
  updateHash();
  debounceTimer = setTimeout(() => updateChart(), delay);
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------
function updateStatus() {
  const el = document.getElementById('statusText');
  const n = state.selectedRuns.size;
  const total = state.runs.length;
  el.textContent = `${n}/${total} runs selected`;
}

// ---------------------------------------------------------------------------
// Toast notifications
// ---------------------------------------------------------------------------
function showToast(message, type = 'success') {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = 'toast ' + type + ' show';
  setTimeout(() => { toast.classList.remove('show'); }, 2000);
}

// ---------------------------------------------------------------------------
// Event wiring
// ---------------------------------------------------------------------------
function wireEvents() {
  // Group toggle
  for (const btn of document.querySelectorAll('#groupToggle button')) {
    btn.addEventListener('click', () => {
      for (const b of document.querySelectorAll('#groupToggle button')) b.classList.remove('active');
      btn.classList.add('active');
      state.groupBy = btn.dataset.group;
      renderRunList();
      updateHash();
    });
  }

  // Chart type
  document.getElementById('chartType').addEventListener('change', (e) => {
    state.chartType = e.target.value;
    updateChartControlsVisibility();
    scheduleUpdate();
  });

  // X metric
  document.getElementById('xMetric').addEventListener('change', (e) => {
    state.xMetric = e.target.value;
    scheduleUpdate();
  });

  // Y metric
  document.getElementById('yMetric').addEventListener('change', (e) => {
    state.yMetric = e.target.value;
    scheduleUpdate();
  });

  // Downsample
  document.getElementById('downsample').addEventListener('change', (e) => {
    state.downsample = parseInt(e.target.value) || 500;
    scheduleUpdate();
  });

  // Share
  document.getElementById('btnShare').addEventListener('click', () => {
    const url = window.location.href;
    navigator.clipboard.writeText(url).then(() => {
      showToast('Copied!');
    }).catch(() => {
      // Fallback
      const ta = document.createElement('textarea');
      ta.value = url;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      showToast('Copied!');
    });
  });

  // Save favorite
  document.getElementById('btnSave').addEventListener('click', () => {
    const hash = encodeHash();
    const label = prompt('Label for this view:', buildDefaultLabel());
    if (label === null) return; // cancelled
    const favs = getFavorites();
    favs.push({
      hash: hash,
      label: label || buildDefaultLabel(),
      timestamp: new Date().toISOString(),
    });
    saveFavorites(favs);
    renderFavorites();
    showToast('View saved!');
  });

  // Favorites toggle
  document.getElementById('favoritesHeader').addEventListener('click', toggleFavorites);

  // Copy as markdown
  document.getElementById('btnCopyMarkdown').addEventListener('click', () => {
    const favs = getFavorites();
    if (favs.length === 0) {
      showToast('No favorites to copy.', 'error');
      return;
    }
    const base = window.location.origin + window.location.pathname;
    const md = favs.map(f => `- [${f.label}](${base}#${f.hash})`).join('\n');
    navigator.clipboard.writeText(md).then(() => {
      showToast('Markdown copied!');
    });
  });

  // Hash change
  window.addEventListener('hashchange', () => {
    loadFromHash();
  });

  // Responsive chart resize
  window.addEventListener('resize', () => {
    Plotly.Plots.resize('chart');
  });

  // Context panel events
  wireContextEvents();
}

function buildDefaultLabel() {
  const parts = [];
  const selected = Array.from(state.selectedRuns);
  if (selected.length <= 3) {
    parts.push(selected.join(', '));
  } else {
    parts.push(`${selected.length} runs`);
  }
  parts.push(state.chartType);
  if (state.chartType === 'phase') {
    parts.push(`${state.xMetric} vs ${state.yMetric}`);
  } else {
    parts.push(state.yMetric);
  }
  return parts.join(' | ');
}

// ---------------------------------------------------------------------------
// Context Inspection Panel
// ---------------------------------------------------------------------------
const ctxState = {
  open: false,
  runId: null,
  runMeta: null,       // { L, T, seed, ... } from state.runs
  step: 0,
  stepRange: null,     // cached /api/step_range response
  contextData: null,   // latest /api/context response
  loading: false,
  vertLineIndex: null, // Plotly shape index for step indicator
};

const stepRangeCache = {}; // runId -> step_range response
let ctxFetchController = null; // AbortController for in-flight context fetches
let scrubberDebounce = null;

async function fetchStepRange(runId) {
  if (stepRangeCache[runId]) return stepRangeCache[runId];
  const data = await apiFetch(`/api/step_range?run=${encodeURIComponent(runId)}`);
  stepRangeCache[runId] = data;
  return data;
}

async function fetchContext(runId, step, windowSize) {
  // Cancel any in-flight request
  if (ctxFetchController) ctxFetchController.abort();
  ctxFetchController = new AbortController();

  const params = new URLSearchParams({ run: runId, step: String(step), window: String(windowSize) });
  const resp = await fetch(`/api/context?${params}`, { signal: ctxFetchController.signal });
  if (!resp.ok) throw new Error(`API error: ${resp.status}`);
  return resp.json();
}

function openContextPanel(runId, step) {
  const run = state.runs.find(r => r.id === runId);
  if (!run) return;

  ctxState.open = true;
  ctxState.runId = runId;
  ctxState.runMeta = run;
  ctxState.step = step;

  document.getElementById('contextPanel').classList.add('open');
  updateContextTitle();

  // Fetch step_range then context
  ctxSetLoading(true);
  fetchStepRange(runId).then(sr => {
    ctxState.stepRange = sr;
    setupScrubber(sr);
    renderEosTicks(sr);
    return loadContextAtStep(step);
  }).catch(err => {
    if (err.name !== 'AbortError') {
      console.error('Context fetch failed:', err);
      showToast('Failed to load context: ' + err.message, 'error');
      ctxSetLoading(false);
    }
  });
}

function closeContextPanel() {
  ctxState.open = false;
  document.getElementById('contextPanel').classList.remove('open');
  removeStepIndicator();
  // Resize chart to reclaim space
  setTimeout(() => Plotly.Plots.resize('chart'), 50);
}

function updateContextTitle() {
  const run = ctxState.runMeta;
  if (!run) return;
  const label = `Context: L=${run.L} T=${parseFloat(run.T).toFixed(2)} S=${run.seed}`;
  document.getElementById('contextTitle').textContent = label;
}

function ctxSetLoading(on) {
  ctxState.loading = on;
  const el = document.getElementById('contextLoading');
  el.style.display = on ? 'flex' : 'none';
  if (on) {
    // Clear text content but keep loading indicator
    const textEl = document.getElementById('contextText');
    // Remove all children except loading
    Array.from(textEl.children).forEach(c => {
      if (c.id !== 'contextLoading') textEl.removeChild(c);
    });
  }
}

async function loadContextAtStep(step) {
  const sr = ctxState.stepRange;
  if (!sr) return;

  // Clamp step
  step = Math.max(sr.min_step, Math.min(sr.max_step, step));
  ctxState.step = step;

  // Update UI controls
  document.getElementById('contextStepInput').value = step;
  document.getElementById('contextScrubber').value = step;
  updateNavButtonState();
  updateStepIndicator(step);

  ctxSetLoading(true);
  try {
    const L = ctxState.runMeta.L;
    const data = await fetchContext(ctxState.runId, step, L);
    ctxState.contextData = data;
    renderContextTokens(data);
  } catch (err) {
    if (err.name !== 'AbortError') {
      console.error('Context load failed:', err);
      showToast('Failed to load context', 'error');
    }
  } finally {
    ctxSetLoading(false);
  }
}

function setupScrubber(stepRange) {
  const scrubber = document.getElementById('contextScrubber');
  scrubber.min = stepRange.min_step;
  scrubber.max = stepRange.max_step;
  scrubber.value = ctxState.step;

  const stepInput = document.getElementById('contextStepInput');
  stepInput.min = stepRange.min_step;
  stepInput.max = stepRange.max_step;
}

function renderEosTicks(stepRange) {
  const container = document.getElementById('eosTicks');
  container.innerHTML = '';
  if (!stepRange.eos_steps || stepRange.eos_steps.length === 0) return;

  const min = stepRange.min_step;
  const max = stepRange.max_step;
  const range = max - min;
  if (range <= 0) return;

  for (const eosStep of stepRange.eos_steps) {
    const pct = ((eosStep - min) / range) * 100;
    const tick = document.createElement('div');
    tick.className = 'eos-tick';
    tick.style.left = pct + '%';
    container.appendChild(tick);
  }
}

function renderContextTokens(data) {
  const textEl = document.getElementById('contextText');
  // Keep loading indicator, clear everything else
  const loadingEl = document.getElementById('contextLoading');

  // Build token HTML
  const frag = document.createDocumentFragment();

  if (!data.tokens || data.tokens.length === 0) {
    const msg = document.createElement('div');
    msg.style.cssText = 'color: var(--text-dim); font-style: italic; padding: 12px 0;';
    msg.textContent = 'No tokens available at this step.';
    frag.appendChild(msg);
  } else {
    // Determine entropy range for coloring
    const entropies = data.tokens.map(t => t.entropy).filter(e => e != null && !isNaN(e));
    const minE = entropies.length > 0 ? Math.min(...entropies) : 0;
    const maxE = entropies.length > 0 ? Math.max(...entropies) : 1;
    const rangeE = maxE - minE || 1;

    for (const tok of data.tokens) {
      if (tok.eos) {
        const eos = document.createElement('span');
        eos.className = 'ctx-eos';
        eos.textContent = 'EOS';
        eos.title = `Step ${tok.step} | EOS token`;
        frag.appendChild(eos);
        continue;
      }

      const span = document.createElement('span');
      span.className = 'ctx-token';
      if (tok.step === ctxState.step) {
        span.classList.add('current');
      }

      // Entropy-based coloring: subtle background opacity
      if (tok.entropy != null && !isNaN(tok.entropy)) {
        const norm = (tok.entropy - minE) / rangeE; // 0 = low entropy, 1 = high
        // Low entropy: dimmer (less saturated), high entropy: warmer accent
        const alpha = 0.05 + norm * 0.20; // subtle range: 0.05 to 0.25
        span.style.backgroundColor = `rgba(239, 85, 59, ${alpha.toFixed(3)})`;
      }

      span.textContent = tok.text;
      span.title = `Step ${tok.step} | H=${tok.entropy != null ? tok.entropy.toFixed(3) : '?'} | logp=${tok.log_prob != null ? tok.log_prob.toFixed(3) : '?'}`;
      frag.appendChild(span);
    }
  }

  // Replace content
  textEl.innerHTML = '';
  textEl.appendChild(frag);
  textEl.appendChild(loadingEl);

  // Scroll to make the current token visible
  const currentEl = textEl.querySelector('.ctx-token.current');
  if (currentEl) {
    currentEl.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
}

function updateNavButtonState() {
  const sr = ctxState.stepRange;
  if (!sr) return;

  const step = ctxState.step;
  document.getElementById('ctxPrev').disabled = (step <= sr.min_step);
  document.getElementById('ctxNext').disabled = (step >= sr.max_step);

  // EOS navigation
  const eosSteps = sr.eos_steps || [];
  const prevEos = eosSteps.filter(s => s < step);
  const nextEos = eosSteps.filter(s => s > step);
  document.getElementById('ctxPrevEos').disabled = (prevEos.length === 0);
  document.getElementById('ctxNextEos').disabled = (nextEos.length === 0);
}

function navigateStep(delta) {
  loadContextAtStep(ctxState.step + delta);
}

function navigateToEos(direction) {
  const sr = ctxState.stepRange;
  if (!sr || !sr.eos_steps) return;
  const step = ctxState.step;

  if (direction < 0) {
    // Previous EOS
    const prev = sr.eos_steps.filter(s => s < step);
    if (prev.length > 0) loadContextAtStep(prev[prev.length - 1]);
  } else {
    // Next EOS
    const next = sr.eos_steps.filter(s => s > step);
    if (next.length > 0) loadContextAtStep(next[0]);
  }
}

// Vertical step indicator on chart
function updateStepIndicator(step) {
  if (!ctxState.open) return;
  if (state.chartType !== 'timeseries') {
    removeStepIndicator();
    return;
  }

  const chartEl = document.getElementById('chart');
  if (!chartEl || !chartEl.layout) return;

  const shapes = (chartEl.layout.shapes || []).filter(s => s._autoloop_step_indicator !== true);
  shapes.push({
    type: 'line',
    x0: step, x1: step,
    y0: 0, y1: 1,
    yref: 'paper',
    line: { color: 'rgba(83, 194, 201, 0.6)', width: 1.5, dash: 'dot' },
    _autoloop_step_indicator: true,
  });

  Plotly.relayout('chart', { shapes: shapes });
}

function removeStepIndicator() {
  const chartEl = document.getElementById('chart');
  if (!chartEl || !chartEl.layout) return;
  const shapes = (chartEl.layout.shapes || []).filter(s => s._autoloop_step_indicator !== true);
  Plotly.relayout('chart', { shapes: shapes });
}

function wireContextEvents() {
  // Close button
  document.getElementById('contextClose').addEventListener('click', closeContextPanel);

  // Nav buttons
  document.getElementById('ctxPrev').addEventListener('click', () => navigateStep(-1));
  document.getElementById('ctxNext').addEventListener('click', () => navigateStep(1));
  document.getElementById('ctxPrevEos').addEventListener('click', () => navigateToEos(-1));
  document.getElementById('ctxNextEos').addEventListener('click', () => navigateToEos(1));

  // Step input (direct jump)
  const stepInput = document.getElementById('contextStepInput');
  stepInput.addEventListener('change', () => {
    const val = parseInt(stepInput.value);
    if (!isNaN(val)) loadContextAtStep(val);
  });
  stepInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.target.blur();
      const val = parseInt(stepInput.value);
      if (!isNaN(val)) loadContextAtStep(val);
    }
  });

  // Scrubber (debounced)
  const scrubber = document.getElementById('contextScrubber');
  scrubber.addEventListener('input', () => {
    // Update step display immediately for responsiveness
    const val = parseInt(scrubber.value);
    document.getElementById('contextStepInput').value = val;
    ctxState.step = val;
    updateStepIndicator(val);

    // Debounce the actual fetch
    clearTimeout(scrubberDebounce);
    scrubberDebounce = setTimeout(() => {
      loadContextAtStep(val);
    }, 150);
  });

  // Plotly click event — wired after chart exists
  // We use MutationObserver or re-wire after each chart render
}

function wireChartClickHandler() {
  const chartEl = document.getElementById('chart');
  // Remove previous handler if any, then add
  chartEl.removeAllListeners && chartEl.removeAllListeners('plotly_click');
  chartEl.on('plotly_click', (eventData) => {
    if (!eventData || !eventData.points || eventData.points.length === 0) return;

    const point = eventData.points[0];
    const traceIndex = point.curveNumber;

    // Get selected run IDs in order (matches trace order)
    const selectedIds = Array.from(state.selectedRuns);
    if (traceIndex >= selectedIds.length) return;

    const runId = selectedIds[traceIndex];

    // For time series, x is the step. For phase, we need the underlying step.
    let step;
    if (state.chartType === 'timeseries') {
      step = Math.round(point.x);
    } else {
      // Phase portrait: use point index to estimate step
      // The data is downsampled, so pointIndex * downsample gives approximate step
      step = point.pointIndex * state.downsample;
    }

    openContextPanel(runId, step);
  });
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
async function init() {
  try {
    // Fetch runs and metrics in parallel
    await Promise.all([fetchRuns(), fetchMetrics()]);

    // Populate controls
    populateMetricDropdowns();
    updateChartControlsVisibility();
    renderFavorites();
    wireEvents();

    // Try to restore from URL hash
    if (!loadFromHash()) {
      // Default: select all runs with the smallest L value, show time series of entropy
      if (state.runs.length > 0) {
        const minL = Math.min(...state.runs.map(r => r.L));
        for (const run of state.runs) {
          if (run.L === minL) state.selectedRuns.add(run.id);
        }
      }
      renderRunList();
      scheduleUpdate(0);
    }

    updateStatus();
  } catch (err) {
    console.error('Init failed:', err);
    document.getElementById('statusText').textContent = 'failed to connect';
    showToast('Failed to connect to backend: ' + err.message, 'error');
  }
}

init();

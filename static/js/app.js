// ---------------------------------------------------------------------------
// App: init, event wiring, hash state, toast, scheduling
// ---------------------------------------------------------------------------
import { state, fetchRuns, fetchMetrics } from './state.js';
import { renderRunList, populateMetricDropdowns, updateChartControlsVisibility,
         renderFavorites, toggleFavorites, getFavorites, saveFavorites } from './sidebar.js';
import { updateChart } from './chart.js';
import { wireContextEvents } from './context.js';

let debounceTimer = null;

// ---------------------------------------------------------------------------
// Debounced update
// ---------------------------------------------------------------------------
export function scheduleUpdate(delay = 300) {
  clearTimeout(debounceTimer);
  updateHash();
  debounceTimer = setTimeout(() => updateChart(), delay);
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------
export function updateStatus() {
  const el = document.getElementById('statusText');
  const n = state.selectedRuns.size;
  const total = state.runs.length;
  el.textContent = `${n}/${total} runs selected`;
}

// ---------------------------------------------------------------------------
// Toast notifications
// ---------------------------------------------------------------------------
export function showToast(message, type = 'success') {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = 'toast ' + type + ' show';
  setTimeout(() => { toast.classList.remove('show'); }, 2000);
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

export function updateHash() {
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
      if (state.runs.some(r => r.id === id)) {
        state.selectedRuns.add(id);
      }
    }
  }

  document.getElementById('chartType').value = state.chartType;
  document.getElementById('downsample').value = state.downsample;

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
// Event wiring
// ---------------------------------------------------------------------------
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
    if (label === null) return;
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

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
async function init() {
  try {
    await Promise.all([fetchRuns(), fetchMetrics()]);

    populateMetricDropdowns();
    updateChartControlsVisibility();
    renderFavorites();
    wireEvents();

    if (!loadFromHash()) {
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

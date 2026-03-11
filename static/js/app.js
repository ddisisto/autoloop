// ---------------------------------------------------------------------------
// App: init, event wiring, hash state, toast, scheduling
// ---------------------------------------------------------------------------
import { state, fetchRuns, fetchMetrics, rescanRuns, nextPanelId } from './state.js';
import { renderRunList, renderFavorites, toggleFavorites, getFavorites, saveFavorites } from './sidebar.js';
import { renderAllPanels, rebuildAllPanelDOM, addPanel, resizeAllPanels } from './panels.js';
import { wireContextEvents } from './context.js';
import { PRESETS, applyPreset } from './presets.js';

let debounceTimer = null;

// ---------------------------------------------------------------------------
// Debounced update
// ---------------------------------------------------------------------------
export function scheduleUpdate(delay = 300) {
  clearTimeout(debounceTimer);
  updateHash();
  debounceTimer = setTimeout(() => renderAllPanels(), delay);
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------
export function updateStatus() {
  const el = document.getElementById('statusText');
  const n = state.selectedRuns.size;
  const total = state.runs.length;
  el.textContent = `${n}/${total} runs | ${state.panels.length} panels`;
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
function encodePanels() {
  // ts:entropy+eos_ema|phase:entropy,compressibility_W64
  return state.panels.map(p => {
    let s = p.type === 'phase' ? 'phase:' : 'ts:';
    if (p.type === 'phase') {
      s += (p.metrics.x || '') + ',' + (p.metrics.y1 || '');
    } else {
      s += p.metrics.y1 || '';
      if (p.metrics.y2) s += '+' + p.metrics.y2;
    }
    return s;
  }).join('|');
}

function decodePanels(str) {
  if (!str) return null;
  return str.split('|').map(part => {
    const [typeStr, rest] = part.split(':');
    const type = typeStr === 'phase' ? 'phase' : 'timeseries';
    const metrics = { y1: null, y2: null, x: null };

    if (type === 'phase' && rest) {
      const [x, y1] = rest.split(',');
      metrics.x = x || null;
      metrics.y1 = y1 || null;
    } else if (rest) {
      const [y1, y2] = rest.split('+');
      metrics.y1 = y1 || null;
      metrics.y2 = y2 || null;
    }
    return { type, metrics };
  });
}

function encodeHash() {
  const parts = [];
  if (state.selectedRuns.size > 0) {
    parts.push('runs=' + Array.from(state.selectedRuns).join(','));
  }
  parts.push('colorBy=' + state.colorBy);
  parts.push('downsample=' + state.downsample);
  parts.push('group=' + state.groupBy);
  parts.push('panels=' + encodePanels());
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
    colorBy: params.get('colorBy') || 'T',
    downsample: params.get('downsample') ? parseInt(params.get('downsample')) : 500,
    group: params.get('group') || 'L',
    panels: params.get('panels') || null,
  };
}

function loadFromHash() {
  const hash = window.location.hash.replace(/^#/, '');
  if (!hash) return false;

  const decoded = decodeHash(hash);

  state.colorBy = decoded.colorBy;
  state.downsample = decoded.downsample;
  state.groupBy = decoded.group;

  if (decoded.runs) {
    state.selectedRuns.clear();
    for (const id of decoded.runs) {
      if (state.runs.some(r => r.id === id)) {
        state.selectedRuns.add(id);
      }
    }
  }

  // Restore panels
  const panelDefs = decodePanels(decoded.panels);
  if (panelDefs && panelDefs.length > 0) {
    state.panels = panelDefs.map(p => ({
      id: nextPanelId(),
      type: p.type,
      metrics: { ...p.metrics },
      height: 250,
    }));
  } else if (state.panels.length === 0) {
    applyPreset('overview');
  }

  // Update UI controls
  document.getElementById('colorBy').value = state.colorBy;
  document.getElementById('downsample').value = state.downsample;
  document.getElementById('presetSelect').value = '';

  for (const btn of document.querySelectorAll('#groupToggle button')) {
    btn.classList.toggle('active', btn.dataset.group === state.groupBy);
  }

  rebuildAllPanelDOM();
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
  if (selected.length <= 3) parts.push(selected.join(', '));
  else parts.push(`${selected.length} runs`);
  parts.push(state.panels.map(p => p.metrics.y1 || p.type).join('+'));
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

  // Color-by
  document.getElementById('colorBy').addEventListener('change', (e) => {
    state.colorBy = e.target.value;
    renderRunList();
    scheduleUpdate();
  });

  // Preset
  document.getElementById('presetSelect').addEventListener('change', (e) => {
    const name = e.target.value;
    if (!name) return;
    applyPreset(name);
    rebuildAllPanelDOM();
    scheduleUpdate(0);
    e.target.value = ''; // reset to "Custom" after applying
  });

  // Add panel
  document.getElementById('addPanelBtn').addEventListener('click', () => {
    addPanel({ y1: 'entropy' });
  });

  // Downsample
  document.getElementById('downsample').addEventListener('change', (e) => {
    state.downsample = parseInt(e.target.value) || 500;
    scheduleUpdate();
  });

  // Share
  document.getElementById('btnShare').addEventListener('click', () => {
    const url = window.location.href;
    navigator.clipboard.writeText(url).then(() => showToast('Copied!'))
      .catch(() => showToast('Copy failed', 'error'));
  });

  // Save favorite
  document.getElementById('btnSave').addEventListener('click', () => {
    const hash = encodeHash();
    const label = prompt('Label for this view:', buildDefaultLabel());
    if (label === null) return;
    const favs = getFavorites();
    favs.push({ hash, label: label || buildDefaultLabel(), timestamp: new Date().toISOString() });
    saveFavorites(favs);
    renderFavorites();
    showToast('View saved!');
  });

  // Favorites toggle
  document.getElementById('favoritesHeader').addEventListener('click', toggleFavorites);

  // Copy as markdown
  document.getElementById('btnCopyMarkdown').addEventListener('click', () => {
    const favs = getFavorites();
    if (favs.length === 0) { showToast('No favorites to copy.', 'error'); return; }
    const base = window.location.origin + window.location.pathname;
    const md = favs.map(f => `- [${f.label}](${base}#${f.hash})`).join('\n');
    navigator.clipboard.writeText(md).then(() => showToast('Markdown copied!'));
  });

  // Rescan for new runs
  document.getElementById('rescanBtn').addEventListener('click', async () => {
    const btn = document.getElementById('rescanBtn');
    btn.disabled = true;
    btn.textContent = '...';
    try {
      const before = state.runs.length;
      await rescanRuns();
      renderRunList();
      const after = state.runs.length;
      showToast(after > before ? `Found ${after - before} new run(s)` : 'No new runs');
    } catch (e) {
      showToast('Rescan failed', 'error');
    }
    btn.disabled = false;
    btn.textContent = '\u21bb Refresh';
  });

  // Hash change
  window.addEventListener('hashchange', () => loadFromHash());

  // Responsive resize
  window.addEventListener('resize', () => resizeAllPanels());

  // Context panel events
  wireContextEvents();
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
async function init() {
  try {
    await Promise.all([fetchRuns(), fetchMetrics()]);

    // Populate preset dropdown
    const presetSelect = document.getElementById('presetSelect');
    for (const [key, preset] of Object.entries(PRESETS)) {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = preset.label;
      presetSelect.appendChild(opt);
    }

    renderFavorites();
    wireEvents();

    if (!loadFromHash()) {
      // Default: apply overview preset + select all runs with min L
      applyPreset('overview');
      if (state.runs.length > 0) {
        const minL = Math.min(...state.runs.map(r => r.L));
        for (const run of state.runs) {
          if (run.L === minL) state.selectedRuns.add(run.id);
        }
      }
      rebuildAllPanelDOM();
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

// ---------------------------------------------------------------------------
// Sidebar: run list, metric dropdowns, favorites
// ---------------------------------------------------------------------------
import { state, getRunColor, getRunLabel } from './state.js';
import { scheduleUpdate, updateHash } from './app.js';

// ---------------------------------------------------------------------------
// Run list
// ---------------------------------------------------------------------------
export function renderRunList() {
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
// Metric dropdowns
// ---------------------------------------------------------------------------
export function populateMetricDropdowns() {
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
    const entropyMetric = state.metrics.find(m => m.id === 'entropy');
    ySel.value = entropyMetric ? 'entropy' : state.metrics[0].id;
    state.yMetric = ySel.value;
  }

  if (state.xMetric && xSel.querySelector(`option[value="${state.xMetric}"]`)) {
    xSel.value = state.xMetric;
  } else if (state.metrics.length > 1) {
    const compMetric = state.metrics.find(m => m.id.startsWith('compressibility'));
    xSel.value = compMetric ? compMetric.id : state.metrics[1].id;
    state.xMetric = xSel.value;
  } else if (state.metrics.length > 0) {
    xSel.value = state.metrics[0].id;
    state.xMetric = xSel.value;
  }
}

export function updateChartControlsVisibility() {
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
// Favorites
// ---------------------------------------------------------------------------
let favoritesOpen = true;

export function getFavorites() {
  try {
    return JSON.parse(localStorage.getItem('autoloop_favorites') || '[]');
  } catch { return []; }
}

export function saveFavorites(favs) {
  localStorage.setItem('autoloop_favorites', JSON.stringify(favs));
}

export function renderFavorites() {
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
      // loadFromHash is called via hashchange event
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

export function toggleFavorites() {
  favoritesOpen = !favoritesOpen;
  const list = document.getElementById('favoritesList');
  const actions = document.querySelector('.fav-actions');
  const arrow = document.querySelector('.favorites-header .arrow');
  list.style.display = favoritesOpen ? '' : 'none';
  actions.style.display = favoritesOpen ? '' : 'none';
  arrow.classList.toggle('open', favoritesOpen);
}

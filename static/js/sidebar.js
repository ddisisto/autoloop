// ---------------------------------------------------------------------------
// Sidebar: run list, favorites
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

  const sortedKeys = Object.keys(groups).sort((a, b) => parseFloat(a) - parseFloat(b));

  for (const key of sortedKeys) {
    const groupRuns = groups[key];
    const groupDiv = document.createElement('div');
    groupDiv.className = 'run-group';

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
// Favorites
// ---------------------------------------------------------------------------
let favoritesOpen = true;

export function getFavorites() {
  try { return JSON.parse(localStorage.getItem('autoloop_favorites') || '[]'); }
  catch { return []; }
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

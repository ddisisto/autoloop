// ---------------------------------------------------------------------------
// Chart presets: hardcoded starting views + user-saved layouts
// ---------------------------------------------------------------------------
import { state, nextPanelId } from './state.js';

export const PRESETS = {
  overview: {
    label: 'Overview',
    panels: [
      { type: 'timeseries', metrics: { y1: 'entropy', y2: 'eos_ema', x: null } },
      { type: 'timeseries', metrics: { y1: 'compressibility_W64', y2: null, x: null } },
    ],
  },
  regime: {
    label: 'Regime comparison',
    panels: [
      { type: 'timeseries', metrics: { y1: 'entropy', y2: null, x: null } },
      { type: 'timeseries', metrics: { y1: 'compressibility_W64', y2: null, x: null } },
      { type: 'timeseries', metrics: { y1: 'eos_ema', y2: null, x: null } },
    ],
  },
  window: {
    label: 'Window scaling',
    panels: [
      { type: 'timeseries', metrics: { y1: 'compressibility_W16', y2: 'compressibility_W256', x: null } },
      { type: 'timeseries', metrics: { y1: 'compressibility_W64', y2: 'compressibility_W128', x: null } },
    ],
  },
  phase: {
    label: 'Phase portrait',
    panels: [
      { type: 'phase', metrics: { x: 'entropy', y1: 'compressibility_W64', y2: null } },
    ],
  },
  suppressed: {
    label: 'Suppressed zone',
    panels: [
      { type: 'timeseries', metrics: { y1: 'entropy', y2: null, x: null } },
      { type: 'timeseries', metrics: { y1: 'compressibility_W256', y2: 'compressibility_W64', x: null } },
      { type: 'phase', metrics: { x: 'entropy', y1: 'compressibility_W256', y2: null } },
    ],
  },
};

// Validate metric IDs against available metrics, falling back gracefully
function resolveMetricId(id) {
  if (!id) return null;
  if (state.metrics.some(m => m.id === id)) return id;
  // Fallback: if a compressibility metric is missing, try the first available one
  if (id.startsWith('compressibility_')) {
    const fallback = state.metrics.find(m => m.id.startsWith('compressibility_'));
    return fallback ? fallback.id : null;
  }
  return null;
}

export function applyPreset(name) {
  const preset = PRESETS[name];
  if (!preset) return false;

  state.panels = preset.panels.map(p => ({
    id: nextPanelId(),
    type: p.type,
    metrics: {
      y1: resolveMetricId(p.metrics.y1) || (state.metrics[0] ? state.metrics[0].id : ''),
      y2: resolveMetricId(p.metrics.y2),
      x: resolveMetricId(p.metrics.x),
    },
    height: 250,
  }));
  state.xRange = null;
  return true;
}

// User presets in localStorage
const STORAGE_KEY = 'autoloop_user_presets';

export function getUserPresets() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); }
  catch { return {}; }
}

export function saveUserPreset(name) {
  const presets = getUserPresets();
  presets[name] = {
    label: name,
    panels: state.panels.map(p => ({
      type: p.type,
      metrics: { ...p.metrics },
    })),
    timestamp: new Date().toISOString(),
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(presets));
}

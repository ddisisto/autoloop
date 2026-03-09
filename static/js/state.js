// ---------------------------------------------------------------------------
// Application state, color scheme, API helpers
// ---------------------------------------------------------------------------

let _panelCounter = 0;
export function nextPanelId() { return 'p' + (++_panelCounter); }

export const state = {
  runs: [],            // from /api/runs
  metrics: [],         // from /api/metrics
  selectedRuns: new Set(),
  groupBy: 'L',
  colorBy: 'T',       // 'T' | 'L' | 'seed' | 'run'
  colorOverrides: {},  // runId -> hex color
  panels: [],          // [{id, type, metrics: {y1, y2, x}, height}]
  xRange: null,        // shared zoom {min, max} or null
  downsample: 500,
};

// ---------------------------------------------------------------------------
// Color maps
// ---------------------------------------------------------------------------
const T_COLORS = {
  '0.50': '#636EFA', '0.60': '#5B8FF9', '0.70': '#00CC96',
  '0.80': '#19D3F3', '0.90': '#FECB52', '1.00': '#EF553B',
  '1.10': '#FF6692', '1.20': '#B6E880', '1.50': '#AB63FA',
};

const L_COLORS = {
  '64': '#636EFA', '128': '#19D3F3', '160': '#00CC96',
  '176': '#3DDB85', '192': '#FECB52', '208': '#FFA15A',
  '224': '#FF6692', '256': '#EF553B', '512': '#AB63FA',
  '1024': '#FF97FF',
};

const SEED_COLORS = {
  '7': '#00CC96', '42': '#636EFA', '123': '#EF553B',
};

const PALETTE = [
  '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
  '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
];

const DASH_STYLES = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot'];

function colorMapFor(key) {
  if (key === 'T') return T_COLORS;
  if (key === 'L') return L_COLORS;
  if (key === 'seed') return SEED_COLORS;
  return {};
}

export function getRunColor(run) {
  if (state.colorOverrides[run.id]) return state.colorOverrides[run.id];
  if (state.colorBy === 'run') {
    return PALETTE[state.runs.indexOf(run) % PALETTE.length];
  }
  const map = colorMapFor(state.colorBy);
  const key = String(state.colorBy === 'T' ? parseFloat(run.T).toFixed(2) : run[state.colorBy]);
  return map[key] || PALETTE[state.runs.indexOf(run) % PALETTE.length];
}

export function autoDashBy() {
  if (state.colorBy === 'T') return 'L';
  if (state.colorBy === 'L') return 'T';
  if (state.colorBy === 'seed') return 'L';
  return null; // run mode: no dash
}

export function getRunDash(run) {
  return 'solid';
}

export function getRunLabel(run) {
  return `L=${run.L} T=${parseFloat(run.T).toFixed(2)} S=${run.seed}`;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
export async function apiFetch(path) {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error(`API error: ${resp.status} ${resp.statusText}`);
  return resp.json();
}

export async function fetchRuns() {
  state.runs = await apiFetch('/api/runs');
}

export async function fetchMetrics() {
  state.metrics = await apiFetch('/api/metrics');
}

export async function fetchData(runIds, metricIds, downsample) {
  const params = new URLSearchParams();
  params.set('runs', runIds.join(','));
  params.set('metrics', metricIds.join(','));
  params.set('downsample', String(downsample));
  return apiFetch(`/api/data?${params}`);
}

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
// Cool-to-hot gradient stops (blue → red)
const COOL_HOT = [
  [49, 54, 149],   // #313695
  [69, 117, 180],  // #4575B4
  [116, 173, 209], // #74ADD1
  [171, 217, 233], // #ABD9E9
  [254, 224, 144], // #FEE090
  [253, 174, 97],  // #FDAE61
  [244, 109, 67],  // #F46D43
  [215, 48, 39],   // #D73027
  [165, 0, 38],    // #A50026
];

function interpolateGradient(stops, t) {
  const n = stops.length - 1;
  const i = Math.min(Math.floor(t * n), n - 1);
  const f = t * n - i;
  const a = stops[i], b = stops[i + 1];
  const r = Math.round(a[0] + f * (b[0] - a[0]));
  const g = Math.round(a[1] + f * (b[1] - a[1]));
  const bl = Math.round(a[2] + f * (b[2] - a[2]));
  return `rgb(${r},${g},${bl})`;
}

function dynamicTColor(run) {
  const vals = [...new Set(state.runs.map(r => parseFloat(r.T)))].sort((a, b) => a - b);
  const v = parseFloat(run.T);
  if (vals.length <= 1) return interpolateGradient(COOL_HOT, 0.5);
  const t = (v - vals[0]) / (vals[vals.length - 1] - vals[0]);
  return interpolateGradient(COOL_HOT, t);
}

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
  if (key === 'L') return L_COLORS;
  if (key === 'seed') return SEED_COLORS;
  return {};
}

export function getRunColor(run) {
  if (state.colorOverrides[run.id]) return state.colorOverrides[run.id];
  if (state.colorBy === 'run') {
    return PALETTE[state.runs.indexOf(run) % PALETTE.length];
  }
  if (state.colorBy === 'T') return dynamicTColor(run);
  const map = colorMapFor(state.colorBy);
  const key = String(run[state.colorBy]);
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

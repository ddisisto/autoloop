// ---------------------------------------------------------------------------
// Application state and color scheme
// ---------------------------------------------------------------------------
export const state = {
  runs: [],            // from /api/runs
  metrics: [],         // from /api/metrics
  selectedRuns: new Set(),
  groupBy: 'L',
  chartType: 'timeseries',
  xMetric: '',
  yMetric: '',
  downsample: 500,
};

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

export function getRunColor(run) {
  const tKey = parseFloat(run.T).toFixed(2);
  return T_COLORS[tKey] || PALETTE[state.runs.indexOf(run) % PALETTE.length];
}

export function getRunDash(run) {
  return L_DASHES[String(run.L)] || 'solid';
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

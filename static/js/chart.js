// ---------------------------------------------------------------------------
// Chart rendering (Plotly)
// ---------------------------------------------------------------------------
import { state, fetchData, getRunColor, getRunDash, getRunLabel } from './state.js';
import { showToast, updateStatus } from './app.js';
import { ctxState, openContextPanel, updateStepIndicator, onChartZoom } from './context.js';

function showLoading(on) {
  document.getElementById('loadingOverlay').classList.toggle('active', on);
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

export async function updateChart() {
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
    renderChart(data, selectedIds);
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

function renderChart(data, runIds) {
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

export function wireChartClickHandler() {
  const chartEl = document.getElementById('chart');

  // Remove all previous listeners, then re-add both
  chartEl.removeAllListeners && chartEl.removeAllListeners();

  // Zoom sync: chart zoom → context overview
  chartEl.on('plotly_relayout', (data) => onChartZoom(data));

  chartEl.on('plotly_click', (eventData) => {
    if (!eventData || !eventData.points || eventData.points.length === 0) return;

    const point = eventData.points[0];
    const traceIndex = point.curveNumber;

    const selectedIds = Array.from(state.selectedRuns);
    if (traceIndex >= selectedIds.length) return;

    const runId = selectedIds[traceIndex];

    let step;
    if (state.chartType === 'timeseries') {
      step = Math.round(point.x);
    } else {
      step = point.pointIndex * state.downsample;
    }

    openContextPanel(runId, step);
  });
}

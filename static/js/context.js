// ---------------------------------------------------------------------------
// Context inspection panel (right drawer)
// ---------------------------------------------------------------------------
import { state, apiFetch } from './state.js';
import { showToast } from './app.js';

// ---------------------------------------------------------------------------
// Context state
// ---------------------------------------------------------------------------
export const ctxState = {
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

// ---------------------------------------------------------------------------
// Drag handle for drawer resize
// ---------------------------------------------------------------------------
let dragging = false;

function initDragHandle() {
  const handle = document.getElementById('dragHandle');
  const drawer = document.getElementById('contextPanel');

  handle.addEventListener('mousedown', (e) => {
    if (!ctxState.open) return;
    e.preventDefault();
    dragging = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const mainRect = document.querySelector('.workspace').getBoundingClientRect();
    const newWidth = mainRect.right - e.clientX;
    const clamped = Math.max(300, Math.min(600, newWidth));
    drawer.style.width = clamped + 'px';
  });

  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    Plotly.Plots.resize('chart');
  });
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------
async function fetchStepRange(runId) {
  if (stepRangeCache[runId]) return stepRangeCache[runId];
  const data = await apiFetch(`/api/step_range?run=${encodeURIComponent(runId)}`);
  stepRangeCache[runId] = data;
  return data;
}

async function fetchContext(runId, step, windowSize) {
  if (ctxFetchController) ctxFetchController.abort();
  ctxFetchController = new AbortController();

  const params = new URLSearchParams({ run: runId, step: String(step), window: String(windowSize) });
  const resp = await fetch(`/api/context?${params}`, { signal: ctxFetchController.signal });
  if (!resp.ok) throw new Error(`API error: ${resp.status}`);
  return resp.json();
}

// ---------------------------------------------------------------------------
// Panel open/close
// ---------------------------------------------------------------------------
export function openContextPanel(runId, step) {
  const run = state.runs.find(r => r.id === runId);
  if (!run) return;

  ctxState.open = true;
  ctxState.runId = runId;
  ctxState.runMeta = run;
  ctxState.step = step;

  document.getElementById('contextPanel').classList.add('open');
  document.getElementById('dragHandle').classList.add('active');
  updateContextTitle();

  // Resize chart after drawer transition
  setTimeout(() => Plotly.Plots.resize('chart'), 220);

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

export function closeContextPanel() {
  ctxState.open = false;
  document.getElementById('contextPanel').classList.remove('open');
  document.getElementById('dragHandle').classList.remove('active');
  removeStepIndicator();
  setTimeout(() => Plotly.Plots.resize('chart'), 220);
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
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
    const textEl = document.getElementById('contextText');
    Array.from(textEl.children).forEach(c => {
      if (c.id !== 'contextLoading') textEl.removeChild(c);
    });
  }
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
export async function loadContextAtStep(step) {
  const sr = ctxState.stepRange;
  if (!sr) return;

  step = Math.max(sr.min_step, Math.min(sr.max_step, step));
  ctxState.step = step;

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

function navigateStep(delta) {
  loadContextAtStep(ctxState.step + delta);
}

function navigateToEos(direction) {
  const sr = ctxState.stepRange;
  if (!sr || !sr.eos_steps) return;
  const step = ctxState.step;

  if (direction < 0) {
    const prev = sr.eos_steps.filter(s => s < step);
    if (prev.length > 0) loadContextAtStep(prev[prev.length - 1]);
  } else {
    const next = sr.eos_steps.filter(s => s > step);
    if (next.length > 0) loadContextAtStep(next[0]);
  }
}

function updateNavButtonState() {
  const sr = ctxState.stepRange;
  if (!sr) return;

  const step = ctxState.step;
  document.getElementById('ctxPrev').disabled = (step <= sr.min_step);
  document.getElementById('ctxNext').disabled = (step >= sr.max_step);

  const eosSteps = sr.eos_steps || [];
  const prevEos = eosSteps.filter(s => s < step);
  const nextEos = eosSteps.filter(s => s > step);
  document.getElementById('ctxPrevEos').disabled = (prevEos.length === 0);
  document.getElementById('ctxNextEos').disabled = (nextEos.length === 0);
}

// ---------------------------------------------------------------------------
// Scrubber and EOS ticks
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Token rendering
// ---------------------------------------------------------------------------
function renderContextTokens(data) {
  const textEl = document.getElementById('contextText');
  const loadingEl = document.getElementById('contextLoading');

  const frag = document.createDocumentFragment();

  if (!data.tokens || data.tokens.length === 0) {
    const msg = document.createElement('div');
    msg.style.cssText = 'color: var(--text-dim); font-style: italic; padding: 12px 0;';
    msg.textContent = 'No tokens available at this step.';
    frag.appendChild(msg);
  } else {
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

      if (tok.entropy != null && !isNaN(tok.entropy)) {
        const norm = (tok.entropy - minE) / rangeE;
        const alpha = 0.05 + norm * 0.20;
        span.style.backgroundColor = `rgba(239, 85, 59, ${alpha.toFixed(3)})`;
      }

      span.textContent = tok.text;
      span.title = `Step ${tok.step} | H=${tok.entropy != null ? tok.entropy.toFixed(3) : '?'} | logp=${tok.log_prob != null ? tok.log_prob.toFixed(3) : '?'}`;
      frag.appendChild(span);
    }
  }

  textEl.innerHTML = '';
  textEl.appendChild(frag);
  textEl.appendChild(loadingEl);

  const currentEl = textEl.querySelector('.ctx-token.current');
  if (currentEl) {
    currentEl.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
}

// ---------------------------------------------------------------------------
// Step indicator on chart
// ---------------------------------------------------------------------------
export function updateStepIndicator(step) {
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

// ---------------------------------------------------------------------------
// Event wiring (called once from app.js)
// ---------------------------------------------------------------------------
export function wireContextEvents() {
  document.getElementById('contextClose').addEventListener('click', closeContextPanel);

  document.getElementById('ctxPrev').addEventListener('click', () => navigateStep(-1));
  document.getElementById('ctxNext').addEventListener('click', () => navigateStep(1));
  document.getElementById('ctxPrevEos').addEventListener('click', () => navigateToEos(-1));
  document.getElementById('ctxNextEos').addEventListener('click', () => navigateToEos(1));

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

  const scrubber = document.getElementById('contextScrubber');
  scrubber.addEventListener('input', () => {
    const val = parseInt(scrubber.value);
    document.getElementById('contextStepInput').value = val;
    ctxState.step = val;
    updateStepIndicator(val);

    clearTimeout(scrubberDebounce);
    scrubberDebounce = setTimeout(() => {
      loadContextAtStep(val);
    }, 150);
  });

  // Initialize drag-to-resize
  initDragHandle();
}

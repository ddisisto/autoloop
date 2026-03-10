// ---------------------------------------------------------------------------
// Context inspection panel (right drawer) with zoom-synced overview
// ---------------------------------------------------------------------------
import { state, apiFetch } from './state.js';
import { showToast } from './app.js';
import { resizeAllPanels, updateStepIndicatorAll, removeStepIndicatorAll,
         syncXRangeFromContext, resetXRange } from './panels.js';

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
  viewRange: null,     // {min, max} = chart zoom; null = full range
  bufferRange: null,   // {start, end} step indices currently in DOM
  entropyBounds: null, // {min, range} for consistent coloring across extensions
  extending: false,    // lock for concurrent buffer extensions
};

// Buffer/scroll constants
const BUFFER_SIZE = 2000;
const SCROLL_MARGIN_PX = 400;
const EXTEND_SIZE = 500;
const ACTIVE_FRAC = 2 / 3;   // position from top for active token (lower 1/3)
let programmaticScroll = false;

const stepRangeCache = {}; // runId -> step_range response
let ctxFetchController = null;
let scrubberDebounce = null;
let searchDebounce = null;

// Search state
const searchState = {
  query: '',
  matches: [],    // step positions
  currentIdx: -1, // index into matches
  flags: { case: false, word: false, regex: false },
};

// Word cloud state
const wcState = {
  fullNgrams: [],   // top ngrams for entire run
  zoomTerms: null,   // Set of terms present in zoom range (null = not yet loaded)
};
let wcZoomDebounce = null;

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
    const clamped = Math.max(300, Math.min(900, newWidth));
    drawer.style.width = clamped + 'px';
  });

  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
    resizeAllPanels();
  });
}

// ---------------------------------------------------------------------------
// Overview bar: viewport drag, resize, click-to-pan
// ---------------------------------------------------------------------------
let ovDrag = null; // {type: 'pan'|'resize-left'|'resize-right', startX, startRange}

function initOverviewBar() {
  const bar = document.getElementById('overviewBar');
  const vp = document.getElementById('overviewViewport');

  // Click on bar background → center viewport on that position
  bar.addEventListener('mousedown', (e) => {
    if (e.target !== bar) return;
    const sr = ctxState.stepRange;
    if (!sr) return;

    const rect = bar.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    const step = Math.round(sr.min_step + frac * (sr.max_step - sr.min_step));

    const vr = getViewRange();
    const halfSpan = (vr.max - vr.min) / 2;
    const newMin = Math.max(sr.min_step, step - halfSpan);
    const newMax = Math.min(sr.max_step, newMin + (vr.max - vr.min));
    zoomChart(newMin, newMax);
  });

  // Viewport drag: pan or edge resize
  vp.addEventListener('mousedown', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const sr = ctxState.stepRange;
    if (!sr) return;

    const vpRect = vp.getBoundingClientRect();
    const edgeThresh = 6;
    let type = 'pan';
    if (e.clientX - vpRect.left < edgeThresh) type = 'resize-left';
    else if (vpRect.right - e.clientX < edgeThresh) type = 'resize-right';

    ovDrag = { type, startX: e.clientX, startRange: { ...getViewRange() } };
    document.body.style.cursor = type === 'pan' ? 'grabbing' : 'col-resize';
    document.body.style.userSelect = 'none';
  });

  document.addEventListener('mousemove', (e) => {
    if (!ovDrag) return;
    const sr = ctxState.stepRange;
    if (!sr) return;

    const bar = document.getElementById('overviewBar');
    const barRect = bar.getBoundingClientRect();
    const fullRange = sr.max_step - sr.min_step;
    const dx = e.clientX - ovDrag.startX;
    const dStep = Math.round((dx / barRect.width) * fullRange);
    const { startRange } = ovDrag;

    let newMin, newMax;
    const minViewSpan = Math.max(100, fullRange * 0.01); // minimum 1% of range

    if (ovDrag.type === 'pan') {
      const span = startRange.max - startRange.min;
      newMin = Math.max(sr.min_step, Math.min(sr.max_step - span, startRange.min + dStep));
      newMax = newMin + span;
    } else if (ovDrag.type === 'resize-left') {
      newMin = Math.max(sr.min_step, Math.min(startRange.max - minViewSpan, startRange.min + dStep));
      newMax = startRange.max;
    } else {
      newMin = startRange.min;
      newMax = Math.min(sr.max_step, Math.max(startRange.min + minViewSpan, startRange.max + dStep));
    }

    zoomChart(newMin, newMax);
  });

  document.addEventListener('mouseup', () => {
    if (!ovDrag) return;
    ovDrag = null;
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });

  // Double-click overview to reset zoom
  bar.addEventListener('dblclick', () => {
    ctxState.viewRange = null;
    resetXRange();
    syncScrubberToView();
    renderOverviewViewport();
  });
}

// ---------------------------------------------------------------------------
// Zoom helpers
// ---------------------------------------------------------------------------
function getViewRange() {
  const sr = ctxState.stepRange;
  if (!sr) return { min: 0, max: 1 };
  if (ctxState.viewRange) return ctxState.viewRange;
  return { min: sr.min_step, max: sr.max_step };
}

function zoomChart(min, max) {
  ctxState.viewRange = { min, max };
  syncXRangeFromContext(min, max);
  syncScrubberToView();
  renderOverviewViewport();
}

function syncScrubberToView() {
  const vr = getViewRange();
  const scrubber = document.getElementById('contextScrubber');
  scrubber.min = Math.round(vr.min);
  scrubber.max = Math.round(vr.max);

  const stepInput = document.getElementById('contextStepInput');
  stepInput.min = Math.round(vr.min);
  stepInput.max = Math.round(vr.max);
}

function renderOverviewViewport() {
  const sr = ctxState.stepRange;
  if (!sr) return;

  const vp = document.getElementById('overviewViewport');
  const stepEl = document.getElementById('overviewStep');
  const fullRange = sr.max_step - sr.min_step;
  if (fullRange <= 0) return;

  const vr = getViewRange();
  const leftPct = ((vr.min - sr.min_step) / fullRange) * 100;
  const rightPct = ((sr.max_step - vr.max) / fullRange) * 100;
  vp.style.left = leftPct + '%';
  vp.style.right = rightPct + '%';

  // Step position marker (relative to full range)
  const stepPct = ((ctxState.step - sr.min_step) / fullRange) * 100;
  stepEl.style.left = Math.max(0, Math.min(100, stepPct)) + '%';
}

// ---------------------------------------------------------------------------
// Chart zoom event handler (called from chart.js)
// ---------------------------------------------------------------------------
export function onChartZoom(relayoutData) {
  if (!ctxState.open || !ctxState.stepRange) return;

  if (relayoutData['xaxis.autorange']) {
    ctxState.viewRange = null;
  } else if (relayoutData['xaxis.range[0]'] != null) {
    ctxState.viewRange = {
      min: Math.round(relayoutData['xaxis.range[0]']),
      max: Math.round(relayoutData['xaxis.range[1]']),
    };
  } else if (relayoutData['xaxis.range']) {
    ctxState.viewRange = {
      min: Math.round(relayoutData['xaxis.range'][0]),
      max: Math.round(relayoutData['xaxis.range'][1]),
    };
  } else {
    return;
  }

  syncScrubberToView();
  renderOverviewViewport();

  // Debounce word cloud zoom refresh
  clearTimeout(wcZoomDebounce);
  wcZoomDebounce = setTimeout(() => refreshWordCloudZoom(), 400);
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

async function fetchContext(runId, step, { window, before, after, signal } = {}) {
  const params = new URLSearchParams({ run: runId, step: String(step) });
  if (window != null) params.set('window', String(window));
  if (before != null) params.set('before', String(before));
  if (after != null) params.set('after', String(after));
  const opts = signal ? { signal } : {};
  const resp = await fetch(`/api/context?${params}`, opts);
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
  ctxState.viewRange = null; // reset zoom on new panel open
  ctxState.bufferRange = null;
  ctxState.entropyBounds = null;
  ctxState.extending = false;
  searchState.query = '';
  searchState.matches = [];
  searchState.currentIdx = -1;
  searchState.error = null;
  const searchInput = document.getElementById('ctxSearchInput');
  if (searchInput) searchInput.value = '';

  document.getElementById('contextPanel').classList.add('open');
  document.getElementById('dragHandle').classList.add('active');
  updateContextTitle();

  setTimeout(() => resizeAllPanels(), 220);

  ctxSetLoading(true);
  fetchStepRange(runId).then(sr => {
    ctxState.stepRange = sr;
    setupScrubber(sr);
    renderEosTicks(sr);
    renderOverviewViewport();

    // Sync viewport to current shared X range if any
    if (state.xRange) {
      ctxState.viewRange = { ...state.xRange };
      syncScrubberToView();
      renderOverviewViewport();
    }

    loadWordCloud();
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
  document.title = 'autoloop explorer';
  wcState.fullNgrams = [];
  wcState.zoomTerms = null;
  setTimeout(() => resizeAllPanels(), 220);
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
  renderOverviewViewport();

  // If step is within current buffer, just scroll to it
  const br = ctxState.bufferRange;
  if (br && step >= br.start && step <= br.end) {
    updateActiveStep(step);
    scrollToStep(step);
    return;
  }

  // Otherwise load a new buffer centered on this step
  await loadBuffer(step);
}

async function loadBuffer(step) {
  const sr = ctxState.stepRange;
  if (!sr) return;

  const before = Math.min(Math.round(BUFFER_SIZE * ACTIVE_FRAC), step - sr.min_step);
  const after = Math.min(BUFFER_SIZE - before - 1, sr.max_step - step);

  if (ctxFetchController) ctxFetchController.abort();
  ctxFetchController = new AbortController();

  ctxSetLoading(true);
  try {
    const data = await fetchContext(ctxState.runId, step, {
      before, after, signal: ctxFetchController.signal,
    });
    ctxState.contextData = data;
    ctxState.bufferRange = { start: data.window_start, end: data.window_end };

    // Compute and store entropy bounds for consistent coloring
    const entropies = data.tokens.map(t => t.entropy).filter(e => e != null && !isNaN(e));
    const minE = entropies.length ? Math.min(...entropies) : 0;
    const maxE = entropies.length ? Math.max(...entropies) : 1;
    ctxState.entropyBounds = { min: minE, range: (maxE - minE) || 1 };

    renderContextTokens(data);
    updateActiveStep(step);
    scrollToStep(step);
  } catch (err) {
    if (err.name !== 'AbortError') {
      console.error('Buffer load failed:', err);
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
// Scrubber setup
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
function buildTokenSpan(tok, entropyBounds) {
  if (tok.eos) {
    const eos = document.createElement('span');
    eos.className = 'ctx-eos';
    eos.textContent = 'EOS';
    eos.title = `Step ${tok.step} | EOS token`;
    eos.dataset.step = tok.step;
    return eos;
  }

  const span = document.createElement('span');
  span.className = 'ctx-token';
  span.dataset.step = tok.step;

  if (tok.entropy != null && !isNaN(tok.entropy) && entropyBounds) {
    const norm = (tok.entropy - entropyBounds.min) / entropyBounds.range;
    const alpha = 0.05 + norm * 0.20;
    span.style.backgroundColor = `rgba(239, 85, 59, ${alpha.toFixed(3)})`;
  }

  if (searchState.query && tokenMatchesSearch(tok.text)) {
    span.classList.add('search-match');
  }

  span.textContent = tok.text;
  span.title = `Step ${tok.step} | H=${tok.entropy != null ? tok.entropy.toFixed(3) : '?'} | logp=${tok.log_prob != null ? tok.log_prob.toFixed(3) : '?'}`;
  return span;
}

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
    const bounds = ctxState.entropyBounds;
    for (const tok of data.tokens) {
      frag.appendChild(buildTokenSpan(tok, bounds));
    }
  }

  textEl.innerHTML = '';
  textEl.appendChild(frag);
  textEl.appendChild(loadingEl);
}

// ---------------------------------------------------------------------------
// Step indicator on chart (delegated to panels.js)
// ---------------------------------------------------------------------------
export function updateStepIndicator(step) {
  if (!ctxState.open) return;
  updateStepIndicatorAll(step);
}

function removeStepIndicator() {
  removeStepIndicatorAll();
}

// ---------------------------------------------------------------------------
// Scroll positioning and active step management
// ---------------------------------------------------------------------------
function scrollToStep(step) {
  const textEl = document.getElementById('contextText');
  const span = textEl.querySelector(`[data-step="${step}"]`);
  if (!span) return;

  programmaticScroll = true;
  const containerRect = textEl.getBoundingClientRect();
  const spanRect = span.getBoundingClientRect();
  const targetY = containerRect.height * ACTIVE_FRAC;
  textEl.scrollTop += (spanRect.top - containerRect.top) - targetY;
  requestAnimationFrame(() => { programmaticScroll = false; });
}

function getStepAtViewFraction(frac) {
  const textEl = document.getElementById('contextText');
  const rect = textEl.getBoundingClientRect();
  const y = rect.top + rect.height * frac;
  const x = rect.left + rect.width / 2;
  const el = document.elementFromPoint(x, y);
  if (!el) return null;
  // Walk up to find a token with data-step
  let node = el;
  while (node && node !== textEl) {
    if (node.dataset && node.dataset.step != null) return parseInt(node.dataset.step);
    node = node.parentElement;
  }
  return null;
}

function updateActiveStep(step) {
  const textEl = document.getElementById('contextText');
  const L = ctxState.runMeta ? ctxState.runMeta.L : 0;
  const contextStart = step - L + 1;

  // Update current highlight and context window markers
  const prev = textEl.querySelector('.ctx-token.current');
  if (prev) prev.classList.remove('current');

  textEl.querySelectorAll('[data-step]').forEach(el => {
    const s = parseInt(el.dataset.step);
    const inCtx = s >= contextStart && s <= step;
    el.classList.toggle('in-context', inCtx);
    el.classList.toggle('out-of-context', !inCtx);
    if (s === step && el.classList.contains('ctx-token')) {
      el.classList.add('current');
    }
  });
}

// ---------------------------------------------------------------------------
// Infinite scroll: extend buffer at edges
// ---------------------------------------------------------------------------
async function extendBuffer(direction) {
  if (ctxState.extending) return;
  const br = ctxState.bufferRange;
  const sr = ctxState.stepRange;
  if (!br || !sr) return;

  let anchorStep, before, after;
  if (direction === 'up') {
    if (br.start <= sr.min_step) return;
    anchorStep = br.start;
    before = Math.min(EXTEND_SIZE, br.start - sr.min_step);
    after = 0;
  } else {
    if (br.end >= sr.max_step) return;
    anchorStep = br.end;
    before = 0;
    after = Math.min(EXTEND_SIZE, sr.max_step - br.end);
  }
  if (before === 0 && after === 0) return;

  ctxState.extending = true;
  const bufferRef = br; // snapshot to detect if buffer was replaced during fetch
  try {
    const data = await fetchContext(ctxState.runId, anchorStep, { before, after });
    // If buffer was replaced (e.g. user navigated), discard
    if (ctxState.bufferRange !== bufferRef) return;

    const textEl = document.getElementById('contextText');
    const loadingEl = document.getElementById('contextLoading');
    const bounds = ctxState.entropyBounds;
    const frag = document.createDocumentFragment();

    const newTokens = direction === 'up'
      ? data.tokens.filter(t => t.step < br.start)
      : data.tokens.filter(t => t.step > br.end);

    if (newTokens.length === 0) return;

    for (const tok of newTokens) {
      frag.appendChild(buildTokenSpan(tok, bounds));
    }

    if (direction === 'up') {
      const prevHeight = textEl.scrollHeight;
      textEl.insertBefore(frag, textEl.firstChild);
      textEl.scrollTop += textEl.scrollHeight - prevHeight;
      br.start = data.window_start;
    } else {
      textEl.insertBefore(frag, loadingEl);
      br.end = data.window_end;
    }

    // Update context window classes on new tokens
    updateActiveStep(ctxState.step);
  } catch (err) {
    if (err.name !== 'AbortError') console.error('Buffer extend failed:', err);
  } finally {
    ctxState.extending = false;
  }
}

let scrollSyncRAF = null;

function onContextScroll() {
  if (programmaticScroll) return;

  if (scrollSyncRAF) return;
  scrollSyncRAF = requestAnimationFrame(() => {
    scrollSyncRAF = null;

    // Sync step from scroll position
    const step = getStepAtViewFraction(ACTIVE_FRAC);
    if (step != null && step !== ctxState.step) {
      ctxState.step = step;
      document.getElementById('contextStepInput').value = step;
      document.getElementById('contextScrubber').value = step;
      updateActiveStep(step);
      updateStepIndicator(step);
      renderOverviewViewport();
      updateNavButtonState();
    }

    // Check buffer edges for extension
    const textEl = document.getElementById('contextText');
    if (textEl.scrollTop < SCROLL_MARGIN_PX) {
      extendBuffer('up');
    }
    if (textEl.scrollHeight - textEl.clientHeight - textEl.scrollTop < SCROLL_MARGIN_PX) {
      extendBuffer('down');
    }
  });
}

// ---------------------------------------------------------------------------
// Word cloud
// ---------------------------------------------------------------------------
async function loadWordCloud() {
  if (!ctxState.runId) return;

  try {
    const data = await apiFetch(`/api/ngrams?run=${encodeURIComponent(ctxState.runId)}&top=30`);
    wcState.fullNgrams = data.ngrams || [];
    wcState.zoomTerms = null; // full run = all active
    renderWordCloud();
    updateTabTitle();
  } catch (err) {
    console.error('Word cloud fetch failed:', err);
  }
}

async function refreshWordCloudZoom() {
  if (!ctxState.runId || wcState.fullNgrams.length === 0) return;

  const vr = getViewRange();
  const sr = ctxState.stepRange;
  if (!sr) return;

  // If viewing full range, all terms active
  if (!ctxState.viewRange) {
    wcState.zoomTerms = null;
    applyZoomState();
    return;
  }

  try {
    const params = new URLSearchParams({
      run: ctxState.runId,
      top: '100',
      min_step: String(Math.round(vr.min)),
      max_step: String(Math.round(vr.max)),
    });
    const data = await apiFetch(`/api/ngrams?${params}`);
    wcState.zoomTerms = new Set((data.ngrams || []).map(n => n.term));
    applyZoomState();
  } catch (err) {
    console.error('Word cloud zoom refresh failed:', err);
  }
}

function renderWordCloud() {
  const container = document.getElementById('wordcloud');
  container.innerHTML = '';
  if (wcState.fullNgrams.length === 0) return;

  const maxCount = wcState.fullNgrams[0].count;

  for (const ng of wcState.fullNgrams) {
    const el = document.createElement('span');
    el.className = 'wc-term';
    el.textContent = ng.term;

    // Size by rank: top terms bigger
    const ratio = ng.count / maxCount;
    const size = 10 + ratio * 4; // 10px to 14px
    el.style.fontSize = size.toFixed(1) + 'px';
    el.style.color = `rgba(224, 224, 224, ${0.5 + ratio * 0.5})`;

    el.addEventListener('click', () => {
      const searchInput = document.getElementById('ctxSearchInput');
      searchInput.value = ng.term;
      runSearch(ng.term);
      // Highlight active term
      container.querySelectorAll('.wc-term').forEach(t => t.classList.remove('active-search'));
      el.classList.add('active-search');
    });

    container.appendChild(el);
  }
}

function applyZoomState() {
  const container = document.getElementById('wordcloud');
  const terms = container.querySelectorAll('.wc-term');
  terms.forEach(el => {
    if (wcState.zoomTerms === null) {
      el.classList.remove('inactive');
    } else {
      el.classList.toggle('inactive', !wcState.zoomTerms.has(el.textContent));
    }
  });
}

function updateTabTitle() {
  if (wcState.fullNgrams.length === 0) return;
  // Top 3 unigrams for tab title
  const topTerms = wcState.fullNgrams
    .filter(n => n.n === 1)
    .slice(0, 3)
    .map(n => n.term);
  if (topTerms.length > 0) {
    document.title = `${topTerms.join(' / ')} — autoloop`;
  }
}

// ---------------------------------------------------------------------------
// Token search
// ---------------------------------------------------------------------------
function tokenMatchesSearch(text) {
  const q = searchState.query;
  if (!q) return false;
  try {
    let pattern = searchState.flags.regex ? q : q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    if (searchState.flags.word) pattern = `\\b${pattern}\\b`;
    const flags = searchState.flags.case ? '' : 'i';
    return new RegExp(pattern, flags).test(text);
  } catch {
    return false;
  }
}

async function runSearch(query) {
  searchState.query = query;
  searchState.matches = [];
  searchState.currentIdx = -1;

  if (!query || !ctxState.runId) {
    updateSearchUI();
    renderSearchTicks();
    return;
  }

  try {
    const params = new URLSearchParams({ run: ctxState.runId, q: query });
    if (searchState.flags.case) params.set('case', '1');
    if (searchState.flags.word) params.set('word', '1');
    if (searchState.flags.regex) params.set('regex', '1');
    const data = await apiFetch(`/api/search?${params}`);
    if (data.error) {
      searchState.error = data.error;
      updateSearchUI();
      renderSearchTicks();
      return;
    }
    searchState.error = null;
    searchState.matches = data.matches || [];
    // Find nearest match to current step
    if (searchState.matches.length > 0) {
      const step = ctxState.step;
      let bestIdx = 0;
      let bestDist = Math.abs(searchState.matches[0] - step);
      for (let i = 1; i < searchState.matches.length; i++) {
        const dist = Math.abs(searchState.matches[i] - step);
        if (dist < bestDist) { bestDist = dist; bestIdx = i; }
      }
      searchState.currentIdx = bestIdx;
    }
  } catch (err) {
    console.error('Search failed:', err);
  }

  updateSearchUI();
  renderSearchTicks();
}

function searchNavigate(delta) {
  if (searchState.matches.length === 0) return;
  searchState.currentIdx = (searchState.currentIdx + delta + searchState.matches.length) % searchState.matches.length;
  updateSearchUI();
  loadContextAtStep(searchState.matches[searchState.currentIdx]);
}

function updateSearchUI() {
  const status = document.getElementById('ctxSearchStatus');
  const prevBtn = document.getElementById('ctxSearchPrev');
  const nextBtn = document.getElementById('ctxSearchNext');

  if (!searchState.query) {
    status.textContent = '';
    status.title = '';
    prevBtn.disabled = true;
    nextBtn.disabled = true;
  } else if (searchState.error) {
    status.textContent = 'err';
    status.title = searchState.error;
    prevBtn.disabled = true;
    nextBtn.disabled = true;
  } else if (searchState.matches.length === 0) {
    status.textContent = '0/0';
    status.title = '';
    prevBtn.disabled = true;
    nextBtn.disabled = true;
  } else {
    status.textContent = `${searchState.currentIdx + 1}/${searchState.matches.length}`;
    prevBtn.disabled = false;
    nextBtn.disabled = false;
  }
}

function renderSearchTicks() {
  const container = document.getElementById('searchTicks');
  if (!container) return;
  container.innerHTML = '';

  const sr = ctxState.stepRange;
  if (!sr || searchState.matches.length === 0) return;

  const range = sr.max_step - sr.min_step;
  if (range <= 0) return;

  for (const step of searchState.matches) {
    const pct = ((step - sr.min_step) / range) * 100;
    const tick = document.createElement('div');
    tick.className = 'search-tick';
    tick.style.left = pct + '%';
    container.appendChild(tick);
  }
}

// ---------------------------------------------------------------------------
// Event wiring (called once from app.js)
// ---------------------------------------------------------------------------
export function wireContextEvents() {
  document.getElementById('contextClose').addEventListener('click', closeContextPanel);

  document.getElementById('ctxCopy').addEventListener('click', () => {
    const textEl = document.getElementById('contextText');
    const text = textEl.innerText.replace(/Loading context\.\.\.\s*$/, '').trim();
    navigator.clipboard.writeText(text).then(() => {
      showToast('Copied to clipboard', 'success');
    }).catch(() => {
      showToast('Copy failed', 'error');
    });
  });

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
    renderOverviewViewport();

    clearTimeout(scrubberDebounce);
    scrubberDebounce = setTimeout(() => {
      loadContextAtStep(val);
    }, 150);
  });

  // Search
  const searchInput = document.getElementById('ctxSearchInput');
  const triggerSearch = () => {
    clearTimeout(searchDebounce);
    searchDebounce = setTimeout(() => runSearch(searchInput.value.trim()), 300);
  };
  searchInput.addEventListener('input', triggerSearch);
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (e.shiftKey) searchNavigate(-1);
      else searchNavigate(1);
    }
    if (e.key === 'Escape') {
      searchInput.value = '';
      runSearch('');
      searchInput.blur();
    }
  });
  document.getElementById('ctxSearchPrev').addEventListener('click', () => searchNavigate(-1));
  document.getElementById('ctxSearchNext').addEventListener('click', () => searchNavigate(1));

  // Search flag toggles
  function bindSearchFlag(btnId, flagName) {
    const btn = document.getElementById(btnId);
    btn.addEventListener('click', () => {
      searchState.flags[flagName] = !searchState.flags[flagName];
      btn.classList.toggle('active', searchState.flags[flagName]);
      if (searchInput.value.trim()) runSearch(searchInput.value.trim());
    });
  }
  bindSearchFlag('ctxSearchCase', 'case');
  bindSearchFlag('ctxSearchWord', 'word');
  bindSearchFlag('ctxSearchRegex', 'regex');

  // Alt+C/W/R shortcuts when search input focused
  searchInput.addEventListener('keydown', (e) => {
    if (!e.altKey) return;
    const map = { c: 'ctxSearchCase', w: 'ctxSearchWord', r: 'ctxSearchRegex' };
    const btnId = map[e.key.toLowerCase()];
    if (btnId) {
      e.preventDefault();
      document.getElementById(btnId).click();
    }
  });

  // Scroll-synced step tracking
  document.getElementById('contextText').addEventListener('scroll', onContextScroll);

  // Double-click to flag a token as active position
  document.getElementById('contextText').addEventListener('dblclick', (e) => {
    const token = e.target.closest('[data-step]');
    if (!token) return;
    e.preventDefault();
    const step = parseInt(token.dataset.step);
    ctxState.step = step;
    document.getElementById('contextStepInput').value = step;
    document.getElementById('contextScrubber').value = step;
    updateActiveStep(step);
    updateStepIndicator(step);
    renderOverviewViewport();
    updateNavButtonState();
  });

  initDragHandle();
  initOverviewBar();
}

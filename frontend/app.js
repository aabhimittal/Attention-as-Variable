/* Attention as a Variable — direct-manipulation attention editor.
 *
 * State model: per-head interventions sent to the backend on every change.
 *   ops[key]   = {op, value?}          — pattern / zero / scale for head key "l-h"
 *   edits[key] = {"q,k": value, ...}   — painted cell clamps for head key
 * The backend applies the op first, then the cell clamps, then re-runs the
 * forward pass and returns attention + baseline/intervened distributions.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const DEFAULT_REMOTE_API = 'https://abhimittal-attention-as-variable.hf.space';

const RAMP_LIGHT = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
const RAMP_DARK = ['#1a1a19', '#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'];

const state = {
  api: localStorage.getItem('aav.api') || '',
  L: 6, H: 12,
  tokens: [], attention: null, dist: [], kl: 0, topBase: '', topInt: '',
  sel: { l: 0, h: 0 },
  ops: {}, edits: {},
  brush: 0.9, renorm: true,
  minis: [],
};

const key = (l, h) => `${l}-${h}`;
const isDark = () => matchMedia('(prefers-color-scheme: dark)').matches &&
  document.documentElement.dataset.theme !== 'light' ||
  document.documentElement.dataset.theme === 'dark';

function heat(v) {
  const ramp = isDark() ? RAMP_DARK : RAMP_LIGHT;
  const x = Math.pow(Math.min(1, Math.max(0, v)), 0.6) * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(x));
  const t = x - i;
  const a = ramp[i], b = ramp[i + 1];
  const c = (s, o) => parseInt(s.slice(o, o + 2), 16);
  const mix = (o) => Math.round(c(a, o) + (c(b, o) - c(a, o)) * t);
  return `rgb(${mix(1)},${mix(3)},${mix(5)})`;
}

/* ---------------- API ---------------- */

function apiBase() { return state.api.replace(/\/+$/, ''); }

async function resolveApi() {
  const stored = localStorage.getItem('aav.api');
  if (stored) { state.api = stored; return; }
  try {
    const r = await fetch('/api/health', { signal: AbortSignal.timeout(4000) });
    if (r.ok) { state.api = ''; return; }
  } catch { /* static host: fall through to remote default */ }
  state.api = DEFAULT_REMOTE_API;
}

function interventions() {
  const out = [];
  for (const [k, o] of Object.entries(state.ops)) {
    const [l, h] = k.split('-').map(Number);
    out.push({ layer: l, head: h, op: o.op, value: o.value ?? null });
  }
  for (const [k, cells] of Object.entries(state.edits)) {
    const [l, h] = k.split('-').map(Number);
    const cs = Object.entries(cells).map(([qk, value]) => {
      const [q, kk] = qk.split(',').map(Number);
      return { q, k: kk, value };
    });
    if (cs.length) out.push({ layer: l, head: h, op: 'edit', cells: cs, renormalize: state.renorm });
  }
  return out;
}

let timer = null, inflight = false, queued = false;
function schedule(ms = 250) {
  clearTimeout(timer);
  timer = setTimeout(analyze, ms);
}

async function analyze() {
  if (inflight) { queued = true; return; }
  inflight = true;
  setStatus('running…', '');
  try {
    const r = await fetch(apiBase() + '/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: $('text').value, interventions: interventions() }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();
    state.tokens = d.tokens;
    state.attention = d.attention;
    state.dist = d.dist;
    state.kl = d.kl;
    state.topBase = d.top_base;
    state.topInt = d.top_int;
    state.L = d.attention.length;
    state.H = d.attention[0].length;
    setStatus('live', 'ok');
    renderAll();
  } catch (e) {
    setStatus(`error: ${e.message} — check API url (backend may still be waking up)`, 'err');
  } finally {
    inflight = false;
    if (queued) { queued = false; analyze(); }
  }
}

function setStatus(msg, cls) {
  const el = $('status');
  el.textContent = msg;
  el.className = 'status' + (cls ? ' ' + cls : '');
}

/* ---------------- rendering ---------------- */

function renderAll() {
  buildGrid();
  drawMinis();
  drawMain();
  drawDist();
  drawChips();
  $('sel-title').textContent = `Layer ${state.sel.l} · Head ${state.sel.h}`;
}

function buildGrid() {
  const grid = $('grid');
  if (state.minis.length === state.L * state.H) return;
  grid.innerHTML = '';
  state.minis = [];
  for (let l = 0; l < state.L; l++) {
    const row = document.createElement('div');
    row.className = 'grid-row';
    const lbl = document.createElement('span');
    lbl.className = 'lbl';
    lbl.textContent = 'L' + l;
    row.appendChild(lbl);
    for (let h = 0; h < state.H; h++) {
      const c = document.createElement('canvas');
      c.width = c.height = 52;
      c.title = `Layer ${l} · Head ${h}`;
      c.onclick = () => { state.sel = { l, h }; syncControls(); renderAll(); };
      row.appendChild(c);
      state.minis.push(c);
    }
    grid.appendChild(row);
  }
}

function drawMinis() {
  const T = state.tokens.length;
  for (let l = 0; l < state.L; l++) {
    for (let h = 0; h < state.H; h++) {
      const c = state.minis[l * state.H + h];
      if (!c) continue;
      const ctx = c.getContext('2d');
      const s = c.width / T;
      ctx.fillStyle = heat(0);
      ctx.fillRect(0, 0, c.width, c.height);
      const m = state.attention[l][h];
      for (let q = 0; q < T; q++)
        for (let k = 0; k <= q; k++) {
          ctx.fillStyle = heat(m[q][k]);
          ctx.fillRect(k * s, q * s, Math.ceil(s), Math.ceil(s));
        }
      const kk = key(l, h);
      c.classList.toggle('sel', l === state.sel.l && h === state.sel.h);
      c.classList.toggle('edited', !!state.ops[kk] || !!state.edits[kk]);
    }
  }
}

const PAD = 68;
let geom = null; // {cell, T, dpr}

function drawMain() {
  const cv = $('main');
  const T = state.tokens.length;
  const dpr = devicePixelRatio || 1;
  const avail = Math.min(cv.parentElement.clientWidth - 8, 640);
  const cell = Math.max(10, Math.floor((avail - PAD) / T));
  const size = PAD + cell * T;
  cv.width = size * dpr;
  cv.height = size * dpr;
  cv.style.width = size + 'px';
  cv.style.height = size + 'px';
  geom = { cell, T, dpr };
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const css = getComputedStyle(document.documentElement);
  ctx.clearRect(0, 0, size, size);

  const m = state.attention[state.sel.l][state.sel.h];
  for (let q = 0; q < T; q++)
    for (let k = 0; k < T; k++) {
      ctx.fillStyle = k <= q ? heat(m[q][k]) : css.getPropertyValue('--page');
      ctx.fillRect(PAD + k * cell, PAD + q * cell, cell - 1, cell - 1);
    }

  // rings on painted cells (intervened-entity color)
  const cells = state.edits[key(state.sel.l, state.sel.h)] || {};
  ctx.strokeStyle = css.getPropertyValue('--int');
  ctx.lineWidth = 2;
  for (const qk of Object.keys(cells)) {
    const [q, k] = qk.split(',').map(Number);
    if (q < T && k < T) ctx.strokeRect(PAD + k * cell + 1, PAD + q * cell + 1, cell - 3, cell - 3);
  }

  // token labels: rows = query (destination), cols = key (source)
  ctx.fillStyle = css.getPropertyValue('--ink-2');
  ctx.font = `${Math.min(11, cell)}px ui-monospace, monospace`;
  for (let q = 0; q < T; q++) {
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(trim(state.tokens[q]), PAD - 6, PAD + q * cell + cell / 2, PAD - 8);
  }
  for (let k = 0; k < T; k++) {
    ctx.save();
    ctx.translate(PAD + k * cell + cell / 2, PAD - 6);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(trim(state.tokens[k]), 0, 0, PAD + 4);
    ctx.restore();
  }
}

const trim = (s) => (s.length > 8 ? s.slice(0, 8) + '…' : s).replace(/ /g, '␣');

function drawDist() {
  const box = $('dist');
  box.innerHTML = '';
  const max = Math.max(0.0001, ...state.dist.map((d) => Math.max(d.base, d.int)));
  for (const d of state.dist.slice(0, 12)) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML =
      `<span class="tok" title="${esc(d.token)}">${esc(trim(d.token))}</span>` +
      `<span class="bar-pair">` +
      `<span class="bar base" style="width:${(d.base / max) * 78}%"><span>${pct(d.base)}</span></span>` +
      `<span class="bar int" style="width:${(d.int / max) * 78}%"><span>${pct(d.int)}</span></span>` +
      `</span>`;
    box.appendChild(row);
  }
  $('kl').textContent = state.kl.toFixed(4);
  const shifted = state.topBase !== state.topInt;
  $('topshift').innerHTML = shifted
    ? `${esc(trim(state.topBase))} → <b>${esc(trim(state.topInt))}</b>`
    : `${esc(trim(state.topInt))} <small>(unchanged)</small>`;
}

const pct = (p) => (p * 100).toFixed(1) + '%';
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');

function drawChips() {
  const box = $('chips');
  box.innerHTML = '';
  const items = [];
  for (const [k, o] of Object.entries(state.ops))
    items.push([k, o.op === 'scale' ? `scale ${o.value.toFixed(2)}` : o.op, 'ops']);
  for (const [k, cells] of Object.entries(state.edits)) {
    const n = Object.keys(cells).length;
    if (n) items.push([k, `${n} cell${n > 1 ? 's' : ''}`, 'edits']);
  }
  if (!items.length) {
    box.innerHTML = '<span class="hint">none — the model is un-edited</span>';
    return;
  }
  for (const [k, label, kind] of items) {
    const [l, h] = k.split('-');
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.innerHTML = `<b>L${l}H${h}</b> ${label} <span class="x">✕</span>`;
    chip.title = 'click to remove';
    chip.onclick = () => { delete state[kind][k]; syncControls(); schedule(0); };
    box.appendChild(chip);
  }
}

/* ---------------- interaction ---------------- */

function selKey() { return key(state.sel.l, state.sel.h); }

function syncControls() {
  const o = state.ops[selKey()];
  $('scale').value = o && o.op === 'scale' ? o.value : 1;
  $('scale-val').textContent = Number($('scale').value).toFixed(2);
  document.querySelectorAll('.toolbar [data-op]').forEach((b) =>
    b.classList.toggle('on', !!o && o.op === b.dataset.op));
}

document.querySelectorAll('.toolbar [data-op]').forEach((btn) => {
  btn.onclick = () => {
    const k = selKey();
    if (state.ops[k] && state.ops[k].op === btn.dataset.op) delete state.ops[k];
    else state.ops[k] = { op: btn.dataset.op };
    syncControls();
    schedule(0);
  };
});

$('reset-head').onclick = () => {
  delete state.ops[selKey()];
  delete state.edits[selKey()];
  syncControls();
  schedule(0);
};

$('clear-all').onclick = () => {
  state.ops = {};
  state.edits = {};
  syncControls();
  schedule(0);
};

$('scale').oninput = () => {
  const v = Number($('scale').value);
  $('scale-val').textContent = v.toFixed(2);
  const k = selKey();
  if (Math.abs(v - 1) < 1e-9) { if (state.ops[k]?.op === 'scale') delete state.ops[k]; }
  else state.ops[k] = { op: 'scale', value: v };
  schedule(300);
};

$('brush').oninput = () => {
  state.brush = Number($('brush').value);
  $('brush-val').textContent = state.brush.toFixed(2);
};

$('renorm').onchange = () => { state.renorm = $('renorm').checked; schedule(0); };

$('text').addEventListener('input', () => {
  // token positions change with the text — painted cells no longer apply
  state.edits = {};
  schedule(500);
});

$('examples').onchange = () => {
  if ($('examples').value) {
    $('text').value = $('examples').value;
    $('examples').value = '';
    state.edits = {};
    schedule(0);
  }
};

$('api').value = state.api;
$('api').onchange = () => {
  state.api = $('api').value.trim();
  localStorage.setItem('aav.api', state.api);
  schedule(0);
};

/* --- painting on the main canvas --- */

let painting = false;

function cellAt(ev) {
  if (!geom) return null;
  const r = $('main').getBoundingClientRect();
  const x = ev.clientX - r.left - PAD;
  const y = ev.clientY - r.top - PAD;
  if (x < 0 || y < 0) return null;
  const k = Math.floor(x / geom.cell);
  const q = Math.floor(y / geom.cell);
  if (q >= geom.T || k >= geom.T || k > q) return null;
  return { q, k };
}

function paint(ev) {
  const c = cellAt(ev);
  if (!c) return;
  const kk = selKey();
  (state.edits[kk] ??= {})[`${c.q},${c.k}`] = state.brush;
  // immediate local feedback; the server round-trip corrects renormalized rows
  state.attention[state.sel.l][state.sel.h][c.q][c.k] = state.brush;
  drawMain();
  drawChips();
}

const main = $('main');
main.addEventListener('pointerdown', (ev) => {
  painting = true;
  main.setPointerCapture(ev.pointerId);
  paint(ev);
});
main.addEventListener('pointermove', (ev) => {
  if (painting) paint(ev);
  const c = cellAt(ev);
  const tip = $('tooltip');
  if (c && state.attention) {
    const v = state.attention[state.sel.l][state.sel.h][c.q][c.k];
    tip.textContent = `${trim(state.tokens[c.q])} ← ${trim(state.tokens[c.k])} : ${v.toFixed(3)}`;
    tip.style.left = ev.clientX + 12 + 'px';
    tip.style.top = ev.clientY + 12 + 'px';
    tip.hidden = false;
  } else tip.hidden = true;
});
main.addEventListener('pointerup', () => { painting = false; schedule(150); });
main.addEventListener('pointerleave', () => { $('tooltip').hidden = true; });

addEventListener('resize', () => { if (state.attention) drawMain(); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  if (state.attention) renderAll();
});

/* ---------------- boot ---------------- */

(async function boot() {
  await resolveApi();
  $('api').value = state.api;
  try {
    const r = await fetch(apiBase() + '/api/model_info', { signal: AbortSignal.timeout(120000) });
    const info = await r.json();
    $('model-name').textContent = info.model;
    state.L = info.layers;
    state.H = info.heads;
  } catch { /* analyze() will surface the error */ }
  analyze();
})();

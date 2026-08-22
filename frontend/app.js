/* Attention as a Variable — direct-manipulation attention editor.
 *
 * State model: per-head interventions sent to the backend on every change.
 *   ops[key]   = {op, value?}          — pattern / zero / scale / temp / topk
 *   masks[key] = [k, ...]              — key positions the head is blind to
 *   edits[key] = {"q,k": value, ...}   — painted cell clamps
 * The backend applies ops in list order, then the masks, then the cell
 * clamps, re-runs the forward pass, and returns attention, baseline vs
 * intervened distributions, per-head statistics and target metrics.
 */
'use strict';

const $ = (id) => document.getElementById(id);
const DEFAULT_REMOTE_API = 'https://abhimittal-attention-as-variable.hf.space';

const RAMP_LIGHT = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
const RAMP_DARK = ['#1a1a19', '#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'];
const RAMP_IMP_LIGHT = ['#fcfcfb', '#e9f0e4', '#cfe0c2', '#a9cb93', '#7bb45f', '#4f9a35', '#2f7d1c', '#1a5c0d'];
const RAMP_IMP_DARK = ['#1a1a19', '#12300c', '#1a5c0d', '#2f7d1c', '#4f9a35', '#7bb45f', '#a9cb93', '#cfe0c2'];

const state = {
  api: localStorage.getItem('aav.api') || '',
  L: 6, H: 12,
  tokens: [], attention: null, dist: [], kl: 0, topBase: '', topInt: '',
  stats: null, logitDiff: null, ms: null,
  sel: { l: 0, h: 0 },
  ops: {}, masks: {}, edits: {},
  brush: 0.9, renorm: true,
  sweep: null, view: 'att',
  minis: [],
};

const key = (l, h) => `${l}-${h}`;
const isDark = () =>
  (matchMedia('(prefers-color-scheme: dark)').matches &&
    document.documentElement.dataset.theme !== 'light') ||
  document.documentElement.dataset.theme === 'dark';

function ramp(kind) {
  if (kind === 'imp') return isDark() ? RAMP_IMP_DARK : RAMP_IMP_LIGHT;
  return isDark() ? RAMP_DARK : RAMP_LIGHT;
}

function heat(v, kind) {
  const r = ramp(kind);
  const x = Math.pow(Math.min(1, Math.max(0, v)), 0.6) * (r.length - 1);
  const i = Math.min(r.length - 2, Math.floor(x));
  const t = x - i;
  const a = r[i], b = r[i + 1];
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

function targets() {
  const a = $('tgt-a').value, b = $('tgt-b').value;
  return a && b ? [a, b] : a ? [a] : [];
}

function interventions() {
  const out = [];
  for (const [k, o] of Object.entries(state.ops)) {
    const [l, h] = k.split('-').map(Number);
    out.push({ layer: l, head: h, op: o.op, value: o.value ?? null });
  }
  for (const [k, pos] of Object.entries(state.masks)) {
    if (!pos.length) continue;
    const [l, h] = k.split('-').map(Number);
    out.push({ layer: l, head: h, op: 'mask_key', positions: pos });
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

async function post(path, body, timeout = 120000) {
  const r = await fetch(apiBase() + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeout),
  });
  if (!r.ok) {
    let detail = `HTTP ${r.status}`;
    try { detail = (await r.json()).detail || detail; } catch { /* non-JSON body */ }
    throw new Error(typeof detail === 'string' ? detail : `HTTP ${r.status}`);
  }
  return r.json();
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
    const d = await post('/api/analyze', {
      text: $('text').value,
      interventions: interventions(),
      targets: targets(),
    });
    state.tokens = d.tokens;
    state.attention = d.attention;
    state.dist = d.dist;
    state.kl = d.kl;
    state.topBase = d.top_base;
    state.topInt = d.top_int;
    state.stats = d.head_stats || null;
    state.logitDiff = d.logit_diff || null;
    state.ms = d.ms;
    state.L = d.attention.length;
    state.H = d.attention[0].length;
    setStatus('live', 'ok');
    renderAll();
    writeHash();
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
  const s = state.stats?.[state.sel.l]?.[state.sel.h];
  $('sel-title').textContent = `Layer ${state.sel.l} · Head ${state.sel.h}`;
  $('sel-role').textContent = s ? `${s.role} · entropy ${s.entropy}` : '';
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
      c.onclick = () => { state.sel = { l, h }; syncControls(); renderAll(); };
      row.appendChild(c);
      state.minis.push(c);
    }
    grid.appendChild(row);
  }
}

function impScale() {
  if (!state.sweep) return 1;
  const m = state.sweep.metric;
  let max = 0;
  for (const row of state.sweep.grid)
    for (const c of row) max = Math.max(max, Math.abs(c[m] ?? c.kl ?? 0));
  return max || 1;
}

function drawMinis() {
  const T = state.tokens.length;
  const showImp = state.view === 'imp' && state.sweep;
  const scale = showImp ? impScale() : 1;
  for (let l = 0; l < state.L; l++) {
    for (let h = 0; h < state.H; h++) {
      const c = state.minis[l * state.H + h];
      if (!c) continue;
      const ctx = c.getContext('2d');
      const kk = key(l, h);
      const st = state.stats?.[l]?.[h];
      const sw = state.sweep?.grid?.[l]?.[h];
      c.title = `Layer ${l} · Head ${h}` +
        (st ? `\n${st.role} · entropy ${st.entropy} · prev ${st.prev} · sink ${st.first}` : '') +
        (sw ? `\nablation KL ${sw.kl}${sw.flip ? ' · flips the top token' : ''}` : '');
      if (showImp) {
        const m = state.sweep.metric;
        const v = Math.abs(sw?.[m] ?? sw?.kl ?? 0) / scale;
        ctx.fillStyle = heat(v, 'imp');
        ctx.fillRect(0, 0, c.width, c.height);
        if (sw?.flip) {
          ctx.strokeStyle = '#d03b3b';
          ctx.lineWidth = 6;
          ctx.strokeRect(3, 3, c.width - 6, c.height - 6);
        }
      } else {
        const s = c.width / T;
        ctx.fillStyle = heat(0);
        ctx.fillRect(0, 0, c.width, c.height);
        const m = state.attention[l][h];
        for (let q = 0; q < T; q++)
          for (let k = 0; k <= q; k++) {
            ctx.fillStyle = heat(m[q][k]);
            ctx.fillRect(k * s, q * s, Math.ceil(s), Math.ceil(s));
          }
      }
      c.classList.toggle('sel', l === state.sel.l && h === state.sel.h);
      c.classList.toggle('edited', !!state.ops[kk] || !!state.edits[kk] || !!state.masks[kk]?.length);
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

  const masked = new Set(state.masks[selKey()] || []);
  const cells = state.edits[key(state.sel.l, state.sel.h)] || {};
  ctx.strokeStyle = css.getPropertyValue('--int');
  ctx.lineWidth = 2;
  for (const qk of Object.keys(cells)) {
    const [q, k] = qk.split(',').map(Number);
    if (q < T && k < T) ctx.strokeRect(PAD + k * cell + 1, PAD + q * cell + 1, cell - 3, cell - 3);
  }

  // token labels: rows = query (destination), cols = key (source)
  ctx.font = `${Math.min(11, cell)}px ui-monospace, monospace`;
  for (let q = 0; q < T; q++) {
    ctx.fillStyle = css.getPropertyValue('--ink-2');
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(trim(state.tokens[q]), PAD - 6, PAD + q * cell + cell / 2, PAD - 8);
  }
  for (let k = 0; k < T; k++) {
    ctx.save();
    ctx.translate(PAD + k * cell + cell / 2, PAD - 6);
    ctx.rotate(-Math.PI / 4);
    ctx.fillStyle = masked.has(k) ? '#d03b3b' : css.getPropertyValue('--ink-2');
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    const label = masked.has(k) ? '✕ ' + trim(state.tokens[k]) : trim(state.tokens[k]);
    ctx.fillText(label, 0, 0, PAD + 4);
    ctx.restore();
    if (masked.has(k)) {
      ctx.fillStyle = 'rgba(208,59,59,0.16)';
      ctx.fillRect(PAD + k * cell, PAD, cell - 1, cell * T);
    }
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
  $('ms').innerHTML = state.ms == null ? '—' : `${Math.round(state.ms)}<small> ms</small>`;
  const shifted = state.topBase !== state.topInt;
  $('topshift').innerHTML = shifted
    ? `${esc(trim(state.topBase))} → <b>${esc(trim(state.topInt))}</b>`
    : `${esc(trim(state.topInt))} <small>(unchanged)</small>`;
  const ld = state.logitDiff;
  $('ld-tile').hidden = !ld;
  if (ld) {
    const sign = ld.delta > 0 ? '+' : '';
    $('ld').innerHTML = `${ld.base.toFixed(2)} → <b>${ld.int.toFixed(2)}</b> ` +
      `<small>(${sign}${ld.delta.toFixed(2)})</small>`;
  }
}

const pct = (p) => (p * 100).toFixed(1) + '%';
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;');

function drawChips() {
  const box = $('chips');
  box.innerHTML = '';
  const items = [];
  for (const [k, o] of Object.entries(state.ops)) {
    const label = o.op === 'scale' ? `scale ${o.value.toFixed(2)}`
      : o.op === 'temp' ? `temp ${o.value.toFixed(1)}`
      : o.op === 'topk' ? `top-${o.value}`
      : o.op;
    items.push([k, label, 'ops']);
  }
  for (const [k, pos] of Object.entries(state.masks))
    if (pos.length) items.push([k, `blind to ${pos.length} token${pos.length > 1 ? 's' : ''}`, 'masks']);
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

/* ---------------- sweep ---------------- */

async function runSweep() {
  const btn = $('sweep');
  btn.disabled = true;
  btn.textContent = 'sweeping…';
  setStatus('sweeping every head…', '');
  try {
    state.sweep = await post('/api/sweep', {
      text: $('text').value,
      op: 'zero',
      targets: targets(),
      interventions: [],
    });
    state.view = 'imp';
    syncView();
    drawSweepList();
    drawMinis();
    setStatus(`sweep done (${Math.round(state.sweep.ms)} ms)`, 'ok');
  } catch (e) {
    setStatus(`sweep failed: ${e.message}`, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Sweep all heads →';
  }
}

function drawSweepList() {
  const s = state.sweep;
  $('sweep-panel').hidden = !s;
  if (!s) return;
  $('sweep-metric').textContent = s.metric === 'kl'
    ? 'by KL when ablated'
    : 'by change in logit difference';
  const ol = $('sweep-list');
  ol.innerHTML = '';
  for (const c of s.top) {
    const v = c[s.metric] ?? c.kl;
    const li = document.createElement('li');
    li.innerHTML = `<b>L${c.layer}H${c.head}</b> <span>${Number(v).toFixed(3)}</span>` +
      (c.flip ? ' <em>flips</em>' : '');
    li.onclick = () => { state.sel = { l: c.layer, h: c.head }; syncControls(); renderAll(); };
    ol.appendChild(li);
  }
}

function syncView() {
  $('view-att').classList.toggle('on', state.view === 'att');
  $('view-imp').classList.toggle('on', state.view === 'imp');
}

/* ---------------- generation ---------------- */

async function runGenerate() {
  const btn = $('gen-run');
  btn.disabled = true;
  btn.textContent = 'generating…';
  try {
    const d = await post('/api/generate', {
      text: $('text').value,
      interventions: interventions(),
      max_new_tokens: Math.max(1, Math.min(32, Number($('gen-n').value) || 12)),
    });
    $('gen-out').innerHTML =
      `<div class="gen-row"><i class="sw sw-base"></i><span>${esc(d.prompt)}<b>${esc(d.baseline)}</b></span></div>` +
      `<div class="gen-row"><i class="sw sw-int"></i><span>${esc(d.prompt)}<b>${esc(d.intervened)}</b></span></div>` +
      (d.baseline === d.intervened
        ? '<span class="hint">identical — this edit does not change the continuation</span>'
        : '');
  } catch (e) {
    $('gen-out').innerHTML = `<span class="hint">generation failed: ${esc(e.message)}</span>`;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate';
  }
}

/* ---------------- permalink ---------------- */

function writeHash() {
  const payload = {
    t: $('text').value,
    a: $('tgt-a').value, b: $('tgt-b').value,
    o: state.ops, m: state.masks, e: state.edits,
  };
  const empty = !Object.keys(state.ops).length && !Object.keys(state.edits).length &&
    !Object.values(state.masks).some((p) => p.length);
  history.replaceState(null, '', empty ? location.pathname : '#' + b64(JSON.stringify(payload)));
}

function readHash() {
  if (!location.hash.length) return false;
  try {
    const p = JSON.parse(unb64(location.hash.slice(1)));
    if (typeof p.t === 'string') $('text').value = p.t;
    $('tgt-a').value = p.a || '';
    $('tgt-b').value = p.b || '';
    state.ops = p.o || {};
    state.masks = p.m || {};
    state.edits = p.e || {};
    return true;
  } catch {
    return false; // a hand-edited or truncated link must not break the boot
  }
}

const b64 = (s) => btoa(String.fromCharCode(...new TextEncoder().encode(s)))
  .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
const unb64 = (s) => new TextDecoder().decode(
  Uint8Array.from(atob(s.replace(/-/g, '+').replace(/_/g, '/')), (c) => c.charCodeAt(0)));

/* ---------------- interaction ---------------- */

function selKey() { return key(state.sel.l, state.sel.h); }

function setOp(op, value) {
  const k = selKey();
  if (op === null) delete state.ops[k];
  else state.ops[k] = value === undefined ? { op } : { op, value };
  syncControls();
}

function syncControls() {
  const o = state.ops[selKey()];
  $('scale').value = o && o.op === 'scale' ? o.value : 1;
  $('scale-val').textContent = Number($('scale').value).toFixed(2);
  $('temp').value = o && o.op === 'temp' ? o.value : 1;
  $('temp-val').textContent = o && o.op === 'temp' ? o.value.toFixed(1) : 'off';
  $('topk').value = o && o.op === 'topk' ? o.value : 0;
  $('topk-val').textContent = o && o.op === 'topk' ? String(o.value) : 'off';
  document.querySelectorAll('.toolbar [data-op]').forEach((b) =>
    b.classList.toggle('on', !!o && o.op === b.dataset.op));
}

document.querySelectorAll('.toolbar [data-op]').forEach((btn) => {
  btn.onclick = () => {
    const cur = state.ops[selKey()];
    setOp(cur && cur.op === btn.dataset.op ? null : btn.dataset.op);
    schedule(0);
  };
});

$('reset-head').onclick = () => {
  delete state.ops[selKey()];
  delete state.edits[selKey()];
  delete state.masks[selKey()];
  syncControls();
  schedule(0);
};

$('clear-all').onclick = () => {
  state.ops = {}; state.edits = {}; state.masks = {};
  syncControls();
  schedule(0);
};

$('share').onclick = async () => {
  writeHash();
  try {
    await navigator.clipboard.writeText(location.href);
    setStatus('link copied', 'ok');
  } catch {
    setStatus('copy blocked — use the address bar', 'err');
  }
};

function sliderOp(id, op, isOff, fmt) {
  $(id).oninput = () => {
    const v = Number($(id).value);
    if (isOff(v)) {
      const cur = state.ops[selKey()];
      if (cur && cur.op === op) delete state.ops[selKey()];
    } else {
      state.ops[selKey()] = { op, value: v };
    }
    syncControls();
    $(id).value = v;                  // syncControls may have rewritten it
    $(id + '-val').textContent = isOff(v) ? (op === 'scale' ? v.toFixed(2) : 'off') : fmt(v);
    schedule(300);
  };
}
sliderOp('scale', 'scale', (v) => Math.abs(v - 1) < 1e-9, (v) => v.toFixed(2));
sliderOp('temp', 'temp', (v) => Math.abs(v - 1) < 1e-9, (v) => v.toFixed(1));
sliderOp('topk', 'topk', (v) => v <= 0, (v) => String(v));

$('brush').oninput = () => {
  state.brush = Number($('brush').value);
  $('brush-val').textContent = state.brush.toFixed(2);
};

$('renorm').onchange = () => { state.renorm = $('renorm').checked; schedule(0); };

function promptChanged() {
  // token positions change with the text — painted cells and masks no longer apply
  state.edits = {};
  state.masks = {};
  state.sweep = null;
  $('sweep-panel').hidden = true;
  state.view = 'att';
  syncView();
}

$('text').addEventListener('input', () => { promptChanged(); schedule(500); });

$('examples').onchange = () => {
  const opt = $('examples').selectedOptions[0];
  if ($('examples').value) {
    $('text').value = $('examples').value;
    $('tgt-a').value = opt.dataset.a || '';
    $('tgt-b').value = opt.dataset.b || '';
    $('examples').value = '';
    promptChanged();
    schedule(0);
  }
};

for (const id of ['tgt-a', 'tgt-b']) $(id).oninput = () => schedule(400);

$('api').value = state.api;
$('api').onchange = () => {
  state.api = $('api').value.trim();
  localStorage.setItem('aav.api', state.api);
  schedule(0);
};

$('view-att').onclick = () => { state.view = 'att'; syncView(); drawMinis(); };
$('view-imp').onclick = () => {
  if (!state.sweep) { runSweep(); return; }
  state.view = 'imp';
  syncView();
  drawMinis();
};
$('sweep').onclick = runSweep;
$('gen-run').onclick = runGenerate;

/* --- painting and column masking on the main canvas --- */

let painting = false;

function hit(ev) {
  if (!geom) return null;
  const r = $('main').getBoundingClientRect();
  const x = ev.clientX - r.left - PAD;
  const y = ev.clientY - r.top - PAD;
  const k = Math.floor(x / geom.cell);
  const q = Math.floor(y / geom.cell);
  if (y < 0 && x >= 0 && k < geom.T) return { header: true, k };
  if (x < 0 || y < 0 || q >= geom.T || k >= geom.T || k > q) return null;
  return { q, k };
}

function toggleMask(k) {
  const kk = selKey();
  const cur = new Set(state.masks[kk] || []);
  cur.has(k) ? cur.delete(k) : cur.add(k);
  state.masks[kk] = [...cur].sort((a, b) => a - b);
  if (!state.masks[kk].length) delete state.masks[kk];
  schedule(0);
}

function paint(ev) {
  const c = hit(ev);
  if (!c || c.header) return;
  const kk = selKey();
  (state.edits[kk] ??= {})[`${c.q},${c.k}`] = state.brush;
  // immediate local feedback; the server round-trip corrects renormalized rows
  state.attention[state.sel.l][state.sel.h][c.q][c.k] = state.brush;
  drawMain();
  drawChips();
}

const main = $('main');
main.addEventListener('pointerdown', (ev) => {
  const c = hit(ev);
  if (c && c.header) { toggleMask(c.k); return; }
  painting = true;
  main.setPointerCapture(ev.pointerId);
  paint(ev);
});
main.addEventListener('pointermove', (ev) => {
  if (painting) paint(ev);
  const c = hit(ev);
  const tip = $('tooltip');
  if (c && !c.header && state.attention) {
    const v = state.attention[state.sel.l][state.sel.h][c.q][c.k];
    tip.textContent = `${trim(state.tokens[c.q])} ← ${trim(state.tokens[c.k])} : ${v.toFixed(3)}`;
    tip.style.left = ev.clientX + 12 + 'px';
    tip.style.top = ev.clientY + 12 + 'px';
    tip.hidden = false;
  } else if (c && c.header) {
    tip.textContent = `click to blind this head to “${state.tokens[c.k]}”`;
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
  if (!readHash()) {
    $('tgt-a').value = ' Mary';
    $('tgt-b').value = ' John';
  }
  syncControls();
  syncView();
  try {
    const r = await fetch(apiBase() + '/api/model_info', { signal: AbortSignal.timeout(120000) });
    const info = await r.json();
    $('model-name').textContent = info.model;
    state.L = info.layers;
    state.H = info.heads;
  } catch { /* analyze() will surface the error */ }
  analyze();
})();

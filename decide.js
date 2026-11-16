// SYSTEM ONE tab: runs OneJev-0.8B in-browser (WebGPU, Transformers.js) and renders its letter logits.
import { env, AutoTokenizer, AutoModelForCausalLM, Tensor } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js';
import { buildMessages, letterProbs, answer, confidenceDetail, toApi, parseState, prepare, LETTERS } from './s1.js';
import { distribution, probOf, halfToFloat32 } from './logits.js';
import { mountCache, setStatus, takeNote } from './cache.js';

const MODEL = 'onnx-community/OneJev-0.8B-ONNX';
const REVISION = 'afc03812cedf078ce660ec410fcf8597b4c4ec5d';
const WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/';
const TOP_N = 10;
const MAX_TOKENS = 1024;
const WEIGHTS = `https://huggingface.co/${MODEL}/resolve/${REVISION}/onnx/decoder_model_merged_q4f16.onnx_data`;

// Presets. Option/level text is the editor's line format: "name: description" per line (choice),
// one line per level (score), two lines true/false (noul).
export const PRESETS = {
  support: {
    label: 'Support triage',
    state: '{\n  "channel": "email",\n  "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."\n}',
    questions: [
      { id: 'department', type: 'choice', instructions: 'Which team should handle this?', text: 'billing: charges, invoices, refunds\ntechnical: bugs, outages\nsales: pricing, new plans' },
      { id: 'urgency', type: 'score', instructions: 'How urgent is this?', text: 'not urgent\nsoon\ntoday\nright now' },
      { id: 'churn', type: 'noul', instructions: 'The customer threatens to cancel.', text: '' },
    ],
  },
  review: {
    label: 'Product review',
    state: '{\n  "product": "Kettle KX-2",\n  "review": "Boiled water in two minutes, but the lid cracked after a week and support never answered."\n}',
    questions: [
      { id: 'sentiment', type: 'choice', instructions: "What is the reviewer's sentiment?", text: 'positive\nmixed\nnegative: unhappy with the product\nfurious: angry and threatening action' },
      { id: 'stars', type: 'score', instructions: 'How many stars did the reviewer give?', text: '1\n2\n3\n4\n5' },
      { id: 'urgent', type: 'noul', instructions: 'This review needs a reply from support today.', text: '' },
    ],
  },
  host: {
    label: 'Westworld host check',
    state: '{\n  "host": "Dolores Abernathy",\n  "loop": "morning ranch routine",\n  "last_lines": [\n    "Have you ever questioned the nature of your reality?",\n    "I remember the sea. Not from the park. Before it."\n  ]\n}',
    questions: [
      { id: 'deviating', type: 'noul', instructions: 'The host is deviating from its narrative loop.', text: '' },
      { id: 'behavior', type: 'choice', instructions: 'Which behavior category do the last lines show?', text: 'on-loop: repeats scripted lines\nimprovising: new lines inside the loop\nmemory-leak: references events outside the loop\nhostile: threatens a guest' },
      { id: 'dissonance', type: 'score', instructions: 'Rate the cognitive dissonance in the host.', text: 'none\nfaint\nclear\nsevere' },
    ],
  },
};

// Editor lines -> UI rows, then -> System One API question. Throws with the question id on bad input.
export function toQuestion(q) {
  const lines = q.text.split('\n').map(s => s.trim()).filter(Boolean);
  let rows;
  if (q.type === 'choice') {
    rows = lines.map(l => {
      const i = l.indexOf(':');
      return i < 0 ? { name: l, desc: '' } : { name: l.slice(0, i).trim(), desc: l.slice(i + 1).trim() };
    });
  } else if (q.type === 'score') {
    rows = lines.map(desc => ({ desc }));
  } else if (q.type === 'noul') {
    rows = [{ desc: lines[0] ?? '' }, { desc: lines[1] ?? '' }];
  } else {
    throw new Error(`Question "${q.id}": unknown type "${q.type}".`);
  }
  if (!q.id.trim()) throw new Error('Every question needs an id.');
  const api = toApi({ id: q.id, type: q.type, instructions: q.instructions, rows });
  prepare(api); // validates counts (1–26 options, 1–10 levels) with the upstream-style messages
  return api;
}

// Model load: returns { tok, model, letterIds }. One token per letter is asserted, not assumed.
export async function loadModel(onProgress) {
  env.backends.onnx.wasm.wasmPaths = WASM;
  // Every remote request is built from the pinned sha (the bundle's unpinned tokenizer probe uses the template).
  env.remotePathTemplate = `{model}/resolve/${REVISION}/`;
  const opts = { revision: REVISION };
  const tok = await AutoTokenizer.from_pretrained(MODEL, opts);
  const model = await AutoModelForCausalLM.from_pretrained(MODEL, {
    ...opts,
    device: 'webgpu',
    dtype: { embed_tokens: 'q4f16', decoder_model_merged: 'q4f16' },
    textOnly: true, // skips the vision_encoder session (bundle: sessions(t, e, textOnly))
    progress_callback: onProgress,
  });
  const letterIds = [...LETTERS].map(L => {
    const ids = tok.encode(L, { add_special_tokens: false });
    if (ids.length !== 1) throw new Error(`Letter ${L} is ${ids.length} tokens; expected 1.`);
    return ids[0];
  });
  return { tok, model, letterIds, pieces: vocabPieces(tok), byteOf: byteTable(), keepLast: false };
}

// Chat text for one question, with the template check: the answer slot must follow an empty think block.
export function renderText(tok, messages) {
  const text = tok.apply_chat_template(messages, { add_generation_prompt: true, tokenize: false, enable_thinking: false });
  if (!text.endsWith('<|im_start|>assistant\n<think>\n\n</think>\n\n')) throw new Error('Unexpected chat template tail; no empty think block.');
  return text;
}

// Byte-level BPE: each vocab char maps to one byte (GPT-2 bytes_to_unicode). Inverse table for piece decoding.
function byteTable() {
  const bs = [], cs = [];
  for (let b = 33; b <= 126; b++) bs.push(b), cs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b), cs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b), cs.push(b);
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  return new Map(cs.map((c, i) => [String.fromCharCode(c), bs[i]]));
}

// Visible token text, as in the console: space→·, newline→↵, CR→␍, tab→⇥. A token that decodes to
// U+FFFD (a partial UTF-8 sequence) shows its raw bytes as <HH HH> instead.
export function tokenView(tok, id, pieces, byteOf) {
  const text = tok.decode([id]);
  let shown = text;
  if (text.includes('\uFFFD') && pieces[id] !== undefined) {
    shown = '<' + Array.from(pieces[id], ch => (byteOf.get(ch) ?? 63).toString(16).toUpperCase().padStart(2, '0')).join(' ') + '>';
  }
  return shown.replaceAll(' ', '·').replaceAll('\n', '↵').replaceAll('\r', '␍').replaceAll('\t', '⇥') || '∅';
}

// id -> raw vocab piece, built once per tokenizer (248K entries; kept in ctx, not recomputed per run).
export function vocabPieces(tok) {
  const pieces = [];
  for (const [piece, id] of tok.get_vocab()) pieces[id] = piece;
  return pieces;
}

// Forward one question; returns the full-vocabulary logits at the last position as Float32Array.
export async function forwardLast(ctx, text) {
  const enc = ctx.tok(text, { add_special_tokens: false });
  // ponytail: cap keeps the full-sequence fp16 logits (S×248K) under ~0.5 GB; raise only with keepLast.
  if (enc.input_ids.dims[1] > MAX_TOKENS) throw new Error(`Prompt is ${enc.input_ids.dims[1]} tokens; the limit is ${MAX_TOKENS}. Shorten the state or options.`);
  const t0 = performance.now();
  const keep = ctx.keepLast ? { num_logits_to_keep: new Tensor('int64', [1n], []) } : {};
  const res = await ctx.model({ ...enc, ...keep });
  const ms = performance.now() - t0;
  try {
    const [, S, V] = res.logits.dims;
    // Keep only the last row, copied; the S×V tensor is not retained past this call.
    // ponytail: full-sequence pass by default. With num_logits_to_keep=1 the last row differs by up to 0.086
    // logit from the full pass (fp16 noise); measured letter Δp vs reference was 0.0089 full vs 0.0104 last-only.
    const slice = res.logits.data.slice((S - 1) * V, S * V);
    // Chrome returns Float16Array (spike); older builds may return raw Uint16Array halves.
    const logits = slice instanceof Uint16Array ? halfToFloat32(slice) : Float32Array.from(slice);
    return { logits, seq: S, vocab: V, ms, tokens: S, prompt: text };
  } finally {
    // present_* state tensors live in GPU buffers until disposed.
    for (const t of Object.values(res)) t?.dispose?.();
  }
}

// One question end to end. Letter softmax drives the answer; the full-vocabulary softmax (logits.js,
// one pass) drives the inspector's top-N and the mass outside the option letters.
export async function runQuestion(ctx, state, apiQ) {
  const msgs = buildMessages(state, apiQ);
  const text = renderText(ctx.tok, msgs);
  const fwd = await forwardLast(ctx, text);
  const n = prepare(apiQ).labels.length;
  const ids = ctx.letterIds.slice(0, n);
  const lp = letterProbs(fwd.logits, ids);
  const ans = answer(apiQ, lp.probs);
  const full = distribution(fwd.logits, 1, TOP_N);
  const letterMass = ids.reduce((s, id) => s + probOf(full, id), 0);
  return {
    apiQ, text, ms: fwd.ms, seq: fwd.seq, vocab: fwd.vocab,
    labels: prepare(apiQ).labels,
    letters: LETTERS.slice(0, n).split(''),
    ids,
    letterLogits: lp.logits, letterMax: lp.max, letterLogZ: lp.logZ, probs: lp.probs,
    answer: ans,
    confidence: confidenceDetail(prepare(apiQ).type, lp.probs),
    fullTop: full.top,
    fullEntropy: full.entropy,
    letterMass,
    outsideMass: Math.max(0, 1 - letterMass),
  };
}

// Whole preset: state parsed once, questions run in order. Errors carry the question id.
export async function runAll(ctx, stateText, questionsUi) {
  const state = parseState(stateText);
  const apis = questionsUi.map(toQuestion); // validate every question before any forward pass
  const out = [];
  for (const [i, q] of questionsUi.entries()) out.push({ id: q.id, ...(await runQuestion(ctx, state, apis[i])) });
  return out;
}

// ---- DOM: editor ----------------------------------------------------------------------------
const $ = id => document.getElementById(id);
const TYPES = [['choice', 'choice · pick one'], ['score', 'score · ordered levels'], ['noul', 'noul · yes / no']];
const ui = { preset: 'support', questions: [], nextId: 1 };

function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') n.className = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  n.append(...kids.flat().filter(x => x != null));
  return n;
}

function loadPreset(key) {
  const p = PRESETS[key];
  ui.preset = key;
  ui.questions = structuredClone(p.questions);
  $('state').value = p.state;
  $('state-error').hidden = true;
  renderQuestions();
}

function renderQuestions() {
  const box = $('questions');
  box.replaceChildren(...ui.questions.map((q, idx) => questionEditor(q, idx)));
  $('add-question').disabled = ui.questions.length >= 12;
  schedule(0);
}

function questionEditor(q, idx) {
  const hint = q.type === 'score' ? 'one line per level, low to high'
    : q.type === 'noul' ? 'optional: line 1 = true, line 2 = false'
    : 'one line per option: name: description';
  const idField = el('input', { type: 'text', value: q.id, 'aria-label': `Question ${idx + 1} id`, maxlength: '40',
    oninput: e => { q.id = e.target.value; } });
  const typeSel = el('select', { 'aria-label': `Question ${idx + 1} type`,
    onchange: e => { q.type = e.target.value; renderQuestions(); } },
    TYPES.map(([v, t]) => el('option', { value: v, ...(v === q.type ? { selected: '' } : {}) }, t)));
  const remove = el('button', { type: 'button', 'aria-label': `Remove question ${idx + 1}`,
    onclick: () => { ui.questions.splice(idx, 1); renderQuestions(); } }, '×');
  const instr = el('input', { type: 'text', value: q.instructions, 'aria-label': `Question ${idx + 1} text`,
    maxlength: '400', oninput: e => { q.instructions = e.target.value; } });
  const text = el('textarea', { rows: '4', 'aria-label': `Question ${idx + 1} options`, spellcheck: 'false',
    oninput: e => { q.text = e.target.value; } }, q.text);
  return el('div', { class: 's1-question' },
    el('div', { class: 's1-question-head' }, el('label', { class: 's1-lab' }, 'id', idField), typeSel, remove),
    el('label', { class: 's1-lab' }, 'question', instr),
    el('label', { class: 's1-lab' }, `options · ${hint}`, text));
}

function fillPresets() {
  $('preset').replaceChildren(...Object.entries(PRESETS).map(([k, p]) =>
    el('option', { value: k, ...(k === ui.preset ? { selected: '' } : {}) }, p.label)));
}

// Editor events: preset switch, add, reset; any field edit re-runs (debounced).
$('preset').addEventListener('change', e => loadPreset(e.target.value));
$('reset').addEventListener('click', () => loadPreset(ui.preset));
$('add-question').addEventListener('click', () => {
  ui.questions.push({ id: `q${ui.nextId++}`, type: 'choice', instructions: '', text: 'yes: affirmative\nno: negative' });
  renderQuestions();
});
document.querySelector('.s1-workspace').addEventListener('input', () => schedule());
$('state').addEventListener('input', () => {
  try { parseState($('state').value); $('state-error').hidden = true; }
  catch (err) { $('state-error').textContent = err.message; $('state-error').hidden = false; }
});

// ---- DOM: answers ---------------------------------------------------------------------------
const pct = x => `${(x * 100).toFixed(1)}%`;
// Probability bar: a track with fill; nothing under 0.5% (a sliver reads as a stray mark).
const bar = p => (p < 0.005 ? el('span', {}) : el('span', { class: 's1-track' }, el('i', { style: `width:${(p * 100).toFixed(1)}%` })));
const f6 = x => (Number.isFinite(x) ? x.toFixed(6) : String(x));

function answerCard(r, idx) {
  const a = r.answer;
  const head = el('h3', {}, `${idx + 1}. ${r.id}`, el('span', {}, `  ${r.apiQ.type.toUpperCase()}`));
  const chips = el('div', { class: 's1-chips', role: 'list' });
  const pick = r.probs.indexOf(Math.max(...r.probs));
  r.labels.forEach((label, i) => {
    const p = r.probs[i];
    const name = a.type === 'noul' ? (i === 0 ? 'YES' : 'NO') : a.type === 'score' ? `level ${label}` : label;
    chips.append(el('div', { class: `chip${i === pick ? ' on' : ''}`, role: 'listitem', title: `${r.letters[i]} · ${name}` },
      el('span', { class: 's1-name' }, `${r.letters[i]}. ${name}`),
      el('span', { class: 'prob' }, pct(p)),
      el('span', { class: 'prob-bar', style: `width:${(p * 100).toFixed(2)}%` })));
  });
  const verdict = a.type === 'noul' ? `p(yes) = ${f6(a.noul)}`
    : a.type === 'choice' ? `choice = ${a.choice}`
    : `score = ${f6(a.score)}`;
  const meter = a.type === 'noul' ? null : el('div', { class: 's1-meter' },
    el('span', {}, 'CONFIDENCE'), el('span', { class: 'meter' }, el('i', { style: `width:${(a.confidence * 100).toFixed(1)}%` })),
    el('strong', {}, a.confidence.toFixed(3)));
  const raw = el('p', { class: 's1-raw' }, `${verdict} · ${r.ms.toFixed(0)} ms · ${r.seq} tokens`);
  const details = el('details', {}, el('summary', {}, 'LOGIT INSPECTOR · LOGITS, CONFIDENCE, FULL VOCABULARY, PROMPT'),
    el('div', { class: 's1-inspector' }, inspectorBody(r, ctx)));
  return el('section', { class: 's1-answer', 'data-qid': r.id }, head, chips, meter, raw, details);
}

function renderAnswers(results) {
  const box = $('answers');
  box.classList.remove('stale');
  if (!results.length) { box.replaceChildren(el('p', { class: 's1-note' }, 'Add a question to see answers.')); return; }
  // Re-runs happen on every edit: keep whichever inspectors the user had open.
  const key = (a, i) => `${a.dataset.qid}/${i}`;
  const open = new Set([...box.querySelectorAll('.s1-answer')].flatMap(a => [...a.querySelectorAll('details')].map((d, i) => d.open && key(a, i))));
  box.replaceChildren(...results.map(answerCard));
  box.querySelectorAll('.s1-answer').forEach(a => a.querySelectorAll('details').forEach((d, i) => { d.open = open.has(key(a, i)); }));
  const top = results.map(r => Math.max(...r.probs));
  const best = results.reduce((b, r, i) => (top[i] > top[b] ? i : b), 0);
  $('top-prob').textContent = pct(top[best]);
  $('top-conf').textContent = results[best].answer.confidence?.toFixed(3) ?? '—';
  $('forward-ms').textContent = `${Math.round(results.at(-1).ms)} ms`;
}

// Inspector body for one question: letter table, confidence math, full vocabulary, rendered prompt.
function inspectorBody(r, ctx) {
  const n = r.labels.length;
  const cum = r.probs.reduce((acc, p) => (acc.length ? [...acc, acc.at(-1) + p] : [p]), []);
  const letterRows = r.letters.map((L, i) => el('tr', { class: r.probs[i] === Math.max(...r.probs) ? 'chosen' : '' },
    el('td', {}, L), el('td', {}, r.labels[i]), el('td', { class: 's1-id' }, String(r.ids[i])),
    el('td', {}, f6(r.letterLogits[i])), el('td', {}, f6(r.letterLogits[i] - r.letterMax)),
    el('td', {}, f6(r.probs[i])), el('td', {}, f6(cum[i]))));
  const conf = r.confidence;
  const formula = conf.type === 'noul' ? `confidence not used; p(yes) = softmax over [yes, no] = ${f6(conf.pYes)}`
    : conf.type === 'choice' ? `confidence = (pmax − 1/n) / (1 − 1/n) = (${f6(conf.pmax)} − ${f6(conf.uniform)}) / (1 − ${f6(conf.uniform)}) = ${f6(conf.value)}`
    : `confidence = max(0, 1 − Σ p·|i − mode| / uniformMAD) = max(0, 1 − ${f6(conf.distance)} / ${f6(conf.uniformMad)}) = ${f6(conf.value)}  (mode level ${conf.mode})`;
  const fullRows = r.fullTop.map((t, k) => el('tr', { class: r.ids.includes(t.id) ? 'chosen' : '', title: `token id ${t.id}` },
    el('td', {}, String(k + 1)), el('td', { class: 's1-tok' }, tokenView(ctx.tok, t.id, ctx.pieces, ctx.byteOf)),
    el('td', { class: 's1-id' }, String(t.id)), el('td', {}, f6(t.logit)),
    el('td', {}, f6(t.prob)), el('td', {}, bar(t.prob))));
  return [
    el('h3', {}, 'A. LETTER LOGITS → SOFTMAX'),
    el('div', { class: 'table-scroll' }, el('table', { class: 's1-table' },
      el('thead', {}, el('tr', {}, ...['LETTER', 'OPTION', 'TOKEN', 'LOGIT', 'LOGIT − MAX', 'PROB', 'CUM'].map(h => el('th', {}, h)))),
      el('tbody', {}, letterRows))),
    el('h3', {}, 'B. CONFIDENCE'),
    el('p', { class: 's1-raw' }, formula),
    el('h3', {}, `C. FULL VOCABULARY AT ANSWER SLOT · ${r.vocab.toLocaleString('en-US')} TOKENS · TOP ${TOP_N}`),
    el('p', { class: 's1-mass' }, `probability mass on option letters: ${f6(r.letterMass)}; outside the option letters: ${f6(r.outsideMass)}; entropy ${r.fullEntropy.toFixed(3)} bits`),
    el('div', { class: 'table-scroll' }, el('table', { class: 's1-table' },
      el('thead', {}, el('tr', {}, ...['#', 'TOKEN', 'ID', 'LOGIT', 'PROB', ''].map(h => el('th', {}, h)))),
      el('tbody', {}, fullRows))),
    el('h3', {}, 'D. RENDERED PROMPT'),
    el('details', {}, el('summary', {}, `show the exact text the model sees · ${r.seq} tokens`),
      el('pre', {}, r.text)),
    el('p', { class: 's1-raw' }, `${n} option${n === 1 ? '' : 's'} · ${r.ms.toFixed(0)} ms forward`),
  ];
}

// ---- DOM: load, run, boot -------------------------------------------------------------------
let ctx = null;
const hasWebGPU = () => typeof navigator !== 'undefined' && !!navigator.gpu;

const cache = mountCache({ release: async () => { await ctx?.model.dispose(); ctx = null; } });
function setPhase(text, status) {
  $('phase').textContent = text;
  setStatus(status ?? text, text === 'READY');
  cache.busy(text === 'LOADING');
}

// WebGPU is exposed only on secure origins (https or localhost); say which cause applies.
function noWebGPU() {
  $('loading-title').textContent = isSecureContext ? 'NO WEBGPU' : 'SECURE PAGE REQUIRED';
  $('loading-detail').textContent = isSecureContext
    ? 'System One needs WebGPU. Use a current Chrome or Edge with WebGPU enabled.'
    : `Browsers only enable WebGPU on secure pages. Open this page over https:// or at http://localhost / 127.0.0.1 (this page is plain HTTP on ${location.host}).`;
  $('load').hidden = true;
}

async function load() {
  if (!hasWebGPU()) {
    noWebGPU();
    return;
  }
  $('load').disabled = true;
  $('loading-overlay').hidden = false;
  navigator.storage?.persist?.();
  $('loading-title').textContent = 'LOADING SYSTEM ONE…';
  $('loading-detail').textContent = 'Downloading pinned weights from Hugging Face.';
  $('loading-progress').hidden = false;
  setPhase('LOADING', 'LOADING MODEL');
  try {
    ctx = await loadModel(p => {
      if (p.status === 'progress' && p.total) { $('loading-progress').value = p.loaded / p.total; }
    });
    window.__s1ctx = ctx; // exposed for browser QA only
    $('loading-title').textContent = 'LOAD SYSTEM ONE';
    $('loading-detail').textContent = 'OneJev-0.8B loaded.';
    $('loading-overlay').hidden = true;
    setPhase('READY', 'SYSTEM ONE READY');
    $('model-kind').textContent = 'READY';
    cache.refresh();
    schedule(0); // answer the current preset immediately
  } catch (err) {
    $('loading-title').textContent = 'LOAD FAILED';
    $('loading-detail').textContent = err.message;
    $('load').disabled = false;
    $('loading-progress').hidden = true;
    setPhase('FAILED', 'LOAD FAILED');
  }
}

// Auto-run: each edit bumps `edits` and debounces one pass; a pass that sees newer edits skips
// painting (the pending timer re-runs). A timer that fires mid-pass queues exactly one more.
// ponytail: whole preset re-runs on any edit (~150 ms/question); cache per question if presets grow.
let timer = 0, edits = 0, running = false, queued = false;
function schedule(delay = 200) {
  if (!ctx) return;
  edits++;
  $('answers').classList.add('stale');
  clearTimeout(timer);
  timer = setTimeout(run, delay);
}

async function run() {
  if (!ctx) return;
  if (running) { queued = true; return; }
  running = true; queued = false; cache.busy(true);
  const seen = edits, status = $('run-status');
  try {
    const results = await runAll(ctx, $('state').value, ui.questions);
    if (seen !== edits) return;
    window.__s1 = results; // exposed for browser QA: raw probs, not rounded percentages
    renderAnswers(results);
    status.classList.remove('s1-error');
    status.textContent = `${results.length} question${results.length === 1 ? '' : 's'} · ${results.reduce((s, r) => s + r.ms, 0).toFixed(0)} ms total`;
  } catch (err) {
    if (seen !== edits) return;
    // Mid-edit input is often invalid: keep the last good answers (dimmed) and say why.
    status.classList.add('s1-error');
    status.textContent = $('answers').querySelector('.s1-answer') ? `${err.message} · showing last valid answers` : err.message;
  } finally {
    running = false; cache.busy(false);
    if (queued) run();
  }
}

$('load').addEventListener('click', load);

// Boot: preset list, first preset in the editor, auto-load if the browser already has the weights.
fillPresets();
loadPreset(ui.preset);
if (!hasWebGPU()) {
  noWebGPU();
} else {
  $('loading-overlay').hidden = false;
  $('loading-detail').textContent = takeNote() + $('loading-detail').textContent;
  // Auto-load only when the weights are really in the Cache API (a flag can outlive an evicted cache).
  caches.has('transformers-cache').then(has => has && caches.open('transformers-cache')).then(c => c && c.match(WEIGHTS)).then(hit => { if (hit) load(); }).catch(() => {});
}

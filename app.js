import { distribution, probOf, sample, freeze } from './logits.js';
import { mountCache, setStatus, takeNote } from './cache.js';
const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
let model, promptIds = [], chain = [], pending = null, selected = -1, busy = false, running = false;
let lastMs = 0, loading = false;
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};
const visible = text => text.replaceAll(' ', '·').replaceAll('\n', '↵').replaceAll('\r', '␍').replaceAll('\t', '⇥') || '∅';
const token = id => visible(model.tokenText(id));
const topN = () => Math.max(3, Math.min(20, Math.round(Number($('top-n').value) || 10)));
const temperature = () => Math.max(0, Math.min(5, Number($('temperature').value) || 0));
const pct = p => `${(p * 100).toFixed(p < .001 ? 3 : 1)}%`;
const ids = () => [...promptIds, ...chain.map(item => item.id)];

const tip = el('div', 'chart-tooltip'); tip.id = 'token-tooltip'; tip.setAttribute('role', 'tooltip'); tip.hidden = true; document.body.append(tip);
function describe(node, text) {
  node.setAttribute('aria-label', text); node.setAttribute('aria-describedby', tip.id);
  const show = () => { tip.textContent = text; tip.hidden = false; };
  node.addEventListener('pointerenter', show); node.addEventListener('focus', show); node.addEventListener('click', show);
  node.addEventListener('pointerleave', () => tip.hidden = true); node.addEventListener('blur', () => tip.hidden = true);
}
document.addEventListener('keydown', event => { if (event.key === 'Escape') { tip.hidden = true; $('entropy-chart').querySelector('.entropy-crosshair')?.setAttribute('visibility', 'hidden'); } });
// Chosen token keeps top-k + its own stats; the 248K-entry logits array is dropped (freeze). A frozen dist can still re-branch to any shown chip.
const choose = (dist, id) => {
  if (dist.logits) return freeze(dist, id);
  const t = dist.top.find(x => x.id === id);
  return { ...dist, chosen: { id, logit: t.logit, prob: t.prob } };
};
function phase(name) {
  $('phase').textContent = name;
  document.querySelectorAll('#processes li').forEach(item => item.classList.toggle('selected', item.dataset.phase === name));
  $('forward-engine').classList.toggle('busy', name === 'FORWARD');
}
function controls() {
  // ponytail: hard stop at the context cap (stateless recompute is O(n) per step). Upgrade: merged decoder with KV cache + sliding window.
  if (model && ids().length >= model.config.sequence_len) { pending = null; running = false; $('app-message').textContent = `Context ceiling reached (${model.config.sequence_len} tokens). Back, re-branch, or shorten the prompt; no silent crop.`; }
  $('step').disabled = !model || busy || !pending;
  $('run').disabled = !model || busy && !running || !pending && !running;
  $('reset').disabled = !model || busy;
  $('back').disabled = !model || busy || !chain.length;
  $('run').textContent = running ? 'PAUSE' : 'RUN';
  $('prompt').disabled = busy;
  document.querySelectorAll('#tree .chip').forEach(chip => { chip.disabled = busy; });
  $('temperature').disabled = busy;
  $('mode').disabled = busy;
  cache.busy(busy || running || loading);
}
function stop() { running = false; controls(); }
async function operate(action) {
  if (busy || !model) return;
  const focused = document.activeElement;
  busy = true; controls();
  try { await action(); $('app-message').textContent = ''; }
  catch (error) { stop(); $('app-message').textContent = `INFERENCE ERROR: ${error.message}`; }
  finally { busy = false; phase('READY'); controls(); if (focused?.isConnected && !focused.disabled && document.activeElement === document.body) focused.focus({ preventScroll: true }); }
}
async function forward() {
  const context = ids();
  if (context.length >= model.config.sequence_len) { pending = null; render(true); return; }
  phase('FORWARD');
  // Yield one paint before the forward pass.
  await new Promise(resolve => setTimeout(resolve, 0));
  const start = performance.now();
  const logits = await model.logits(context);
  lastMs = performance.now() - start;
  phase('SOFTMAX');
  pending = distribution(logits, temperature());
  selected = -1;
  render(true);
}
async function reset() {
  stop();
  await operate(async () => {
    phase('TOKENIZE');
    const encoded = model.encode($('prompt').value);
    if (encoded.length >= model.config.sequence_len) throw new Error(`Prompt must be shorter than ${model.config.sequence_len} tokens. Nothing was cropped; previous chain retained.`);
    promptIds = encoded; chain = []; pending = null; selected = -1;
    render(true);
    if (encoded.length) await forward(); // no BOS token: an empty prompt has nothing to predict from
  });
  if (model && !promptIds.length) $('app-message').textContent = 'Prompt is empty. Type a prompt, then press Enter.';
}
async function step() {
  if (!model || !promptIds.length || ids().length >= model.config.sequence_len) return stop();
  await operate(async () => {
    if (!pending) await forward();
    phase('SAMPLE');
    const id = $('mode').value === 'sample' ? sample(pending, Math.random()) : pending.top[0].id;
    chain.push({ id, dist: freeze(pending, id) }); pending = null;
    render(true);
    await forward();
  });
}
async function branch(index, id) {
  stop();
  await operate(async () => {
    const dist = index === chain.length ? pending : chain[index]?.dist;
    if (!dist) return;
    chain = chain.slice(0, index); selected = -1;
    chain.push({ id, dist: choose(dist, id) }); pending = null;
    render(true);
    await forward();
  });
}
async function back() {
  stop();
  await operate(async () => {
    if (!chain.length) return;
    chain.pop(); pending = null; selected = -1;
    render(true);
    await forward();
  });
}
async function run() {
  if (running) return stop();
  if (busy || !pending) return;
  running = true; controls();
  while (running) {
    await step();
    if (running) await new Promise(resolve => setTimeout(resolve, 180));
  }
}

function inspector() {
  const item = selected < 0 ? { dist: pending } : chain[selected];
  const dist = item?.dist;
  $('inspector-sub').textContent = ` / ${selected < 0 ? 'NEXT' : `TOKEN ${selected + 1}`}`;
  $('logits-table').replaceChildren();
  if (!dist) { $('distribution-meta').textContent = model && ids().length >= model.config.sequence_len ? `CONTEXT CEILING / ${model.config.sequence_len} TOKENS` : 'FORWARD PASS PENDING'; return; }
  $('distribution-meta').textContent = `ENTROPY ${dist.entropy.toFixed(3)} bits   MAX LOGIT ${dist.max.toFixed(3)}   T ${dist.temp.toFixed(1)}`;
  let cumulative = 0;
  const shown = dist.top.slice(0, topN());
  shown.forEach(({ id, logit, prob }, rank) => {
    cumulative += prob;
    const row = el('tr', id === item.id ? 'chosen' : ''); row.tabIndex = 0; row.dataset.focus = `logit-${id}`;
    const info = `${token(id)}; raw logit ${logit.toFixed(4)}; probability ${pct(prob)}; cumulative ${pct(cumulative)}`;
    describe(row, info);
    row.append(el('td', '', String(rank + 1)));
    const cell = el('td', '', token(id));
    const bar = el('span', 'table-bar'); bar.style.width = `${prob * 100}%`; cell.append(bar); row.append(cell);
    for (const value of [logit.toFixed(3), dist.temp ? (logit / dist.temp).toFixed(3) : '—', pct(prob), pct(cumulative)]) row.append(el('td', '', value));
    $('logits-table').append(row);
  });
  $('inspector-note').textContent = `Top ${topN()} of ${dist.vocab.toLocaleString()} tokens. Full-vocabulary softmax.${dist.temp === 0 ? ' T=0: greedy limit; logit/T undefined.' : ''}${item.id !== undefined && !shown.some(t => t.id === item.id) ? ` Chosen token ${token(item.id)} outside top-N (${pct(probOf(dist, item.id))}).` : ''}`;
}
function entropyChart() {
  const svg = $('entropy-chart'), ns = 'http://www.w3.org/2000/svg';
  svg.replaceChildren();
  const values = chain.map(item => item.dist.entropy), ceiling = Math.log2(model.config.vocab_size);
  const data = $('entropy-table'); data.replaceChildren();
  const table = el('table'), head = el('thead'), header = el('tr'); header.append(el('th', '', 'Step'), el('th', '', 'Entropy (bits)')); head.append(header); table.append(head);
  const body = el('tbody'); values.forEach((value, i) => { const row = el('tr'); row.append(el('td', '', i + 1), el('td', '', value.toFixed(6))); body.append(row); }); table.append(body); data.append(table);
  svg.setAttribute('aria-label', `Generated token entropy, 0–${ceiling.toFixed(2)} bits; ${values.length} steps. Focus a step for its value.`);
  if (!values.length) { svg.onpointermove = svg.onclick = null; $('entropy-hover').textContent = 'NO STEPS'; return; }
  const points = values.map((v, i) => [values.length === 1 ? 125 : i * 250 / (values.length - 1), 31 - v / ceiling * 28]);
  const line = document.createElementNS(ns, 'polyline'); line.setAttribute('class', 'wave'); line.setAttribute('points', points.map(p => p.join(',')).join(' ')); svg.append(line);
  const crosshair = document.createElementNS(ns, 'line'); crosshair.setAttribute('class', 'entropy-crosshair'); crosshair.setAttribute('y1', '0'); crosshair.setAttribute('y2', '34'); crosshair.setAttribute('visibility', 'hidden'); svg.append(crosshair);
  const inspect = i => { const [x] = points[i], text = `Step ${i + 1}: ${values[i].toFixed(3)} bits`; crosshair.setAttribute('visibility', 'visible'); crosshair.setAttribute('x1', x); crosshair.setAttribute('x2', x); $('entropy-hover').textContent = text; tip.textContent = text; tip.hidden = false; };
  svg.onpointerleave = () => { tip.hidden = true; crosshair.setAttribute('visibility', 'hidden'); };
  svg.onfocusout = event => { if (!svg.contains(event.relatedTarget)) { tip.hidden = true; crosshair.setAttribute('visibility', 'hidden'); } };
  svg.onpointermove = event => { const box = svg.getBoundingClientRect(), at = (event.clientX - box.left) * 250 / box.width; inspect(points.reduce((best, p, i) => Math.abs(p[0] - at) < Math.abs(points[best][0] - at) ? i : best, 0)); };
  svg.onclick = svg.onpointermove;
  points.forEach(([x, y], i) => {
    const hit = document.createElementNS(ns, 'rect'); hit.setAttribute('class', 'entropy-hit'); hit.setAttribute('x', String(Math.max(0, x - 12))); hit.setAttribute('y', '0'); hit.setAttribute('width', '24'); hit.setAttribute('height', '34'); hit.setAttribute('tabindex', '0'); hit.setAttribute('aria-label', `Step ${i + 1}: ${values[i].toFixed(3)} bits`); hit.dataset.focus = `entropy-${i}`;
    hit.setAttribute('aria-describedby', tip.id);
    hit.addEventListener('pointerenter', () => inspect(i)); hit.addEventListener('focus', () => inspect(i)); hit.addEventListener('click', () => inspect(i)); svg.append(hit);
    if (points.length === 1) { const dot = document.createElementNS(ns, 'circle'); dot.setAttribute('cx', x); dot.setAttribute('cy', y); dot.setAttribute('r', '4'); dot.setAttribute('class', 'entropy-point'); svg.append(dot); }
  });
  $('entropy-hover').textContent = `${values.at(-1).toFixed(2)} bits`;
}
function render(scroll = false) {
  const tree = $('tree'), active = document.activeElement?.dataset?.focus;
  tip.hidden = true; tree.replaceChildren();
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('class', 'connectors'); tree.append(svg);
  let x = 14, previous = null;
  const connect = (left, y, prompt) => {
    if (previous) {
      const path = document.createElementNS(ns, 'path');
      path.setAttribute('d', `M${previous.x} ${previous.y} C${previous.x + 25} ${previous.y} ${left - 25} ${y} ${left} ${y}`); svg.append(path);
      if (!prompt) { const badge = el('div', 'badge', 'InEn'); badge.append(el('span', '', '≫≫')); badge.style.left = `${left - 48}px`; badge.style.top = `${(previous.y + y) / 2 - 20}px`; tree.append(badge); }
    }
  };
  promptIds.forEach(id => {
    const column = el('div', 'word-column prompt-column'); column.style.left = `${x}px`; column.style.top = '21px';
    const node = el('article', 'node'); node.append(el('span', 'node-label', 'PROMPT'), el('h2', '', token(id))); node.tabIndex = 0; node.dataset.focus = `prompt-${x}`; describe(node, `Prompt token ${id}: ${model.tokenText(id)}`); column.append(node); tree.append(column);
    connect(x, 47, true); previous = { x: x + 76, y: 47 }; x += 91;
  });
  const columns = [...chain, ...(pending ? [{ dist: pending, preview: true }] : [])];
  columns.forEach((item, index) => {
    x += 48;
    const y = 28 + index % 4 * 29;
    const column = el('div', `word-column${item.preview ? ' preview-column' : ''}`); column.style.left = `${x}px`; column.style.top = `${y}px`;
    const node = el('article', `node${selected === index || item.preview && selected < 0 ? ' selected' : ''}`); node.tabIndex = 0; node.setAttribute('role', 'button'); node.dataset.focus = `node-${index}`;
    node.append(el('span', 'node-label', item.preview ? 'NEXT / PREVIEW' : 'FORWARD CHAIN'), el('h2', '', item.preview ? '?' : token(item.id)));
    const select = () => { selected = item.preview ? -1 : index; if (!item.preview) setInspector(true, false); render(); };
    node.addEventListener('click', select); node.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); event.stopPropagation(); select(); } });
    column.append(node, el('span', 'column-caption', item.preview ? 'PENDING DISTRIBUTION' : `TOKEN ${index + 1} / ID ${item.id}`));
    item.dist.top.slice(0, topN()).forEach(({ id, prob }) => {
      const chip = el('button', `chip${id === item.id ? ' on' : ''}`); chip.type = 'button'; chip.disabled = busy;
      chip.append(el('span', 'token', token(id)), el('span', 'prob', pct(prob)));
      const bar = el('span', 'prob-bar'); bar.style.width = `${prob * 100}%`; chip.append(bar);
      chip.dataset.focus = `chip-${index}-${id}`;
      describe(chip, `${model.tokenText(id)} (ID ${id}) · ${pct(prob)}${id === item.id ? ' · chosen' : ' · choose / re-branch'}`);
      chip.setAttribute('aria-pressed', String(id === item.id));
      chip.addEventListener('click', () => branch(index, id)); column.append(chip);
    });
    tree.append(column); connect(x, y + 37, false); previous = { x: x + 150, y: y + 37 }; x += 150;
  });
  tree.style.width = `${x + 24}px`; tree.style.height = `${Math.max(508, 205 + topN() * 26)}px`;
  const promptText = model ? model.decode(promptIds) : '';
  const fullText = model ? model.decode(ids()) : '';
  $('decoded').replaceChildren(el('span', 'prompt-text', promptText), el('span', '', fullText.slice(promptText.length)));
  $('decoded').scrollTop = $('decoded').scrollHeight;
  $('tokens-count').textContent = chain.length;
  $('top-prob').textContent = pending ? (pending.top[0].prob * 100).toFixed(1) : '—';
  $('forward-ms').textContent = Math.round(lastMs);
  inspector(); entropyChart();
  if (active) { const target = [...document.querySelectorAll('[data-focus]')].find(node => node.dataset.focus === active); (target || $('tree-viewport')).focus({ preventScroll: true }); }
  if (scroll) $('tree-viewport').scrollLeft = tree.scrollWidth;
}
const KIND = { linear_attention: 'GATED DELTANET', full_attention: 'FULL ATTENTION' };
function config() {
  const c = model.config, stub = params.has('stub');
  $('model-config').replaceChildren();
  for (const [label, value] of [['LAYERS / HEADS', `${c.n_layer} / ${c.n_head} (KV ${c.n_kv_head})`], ['D_MODEL / VOCAB', `${c.n_embd} / ${c.vocab_size.toLocaleString()}`], ['PARAMETERS', `${(c.n_params / 1e9).toFixed(2)} B`], ['CONTEXT', `${c.sequence_len} tokens`], ['QUANT', `${c.quant} · ${c.device}`]]) {
    const row = el('div'); row.append(el('dt', '', label), el('dd', '', value)); $('model-config').append(row);
  }
  $('layers').replaceChildren();
  c.layer_types.forEach((type, i) => { const row = el('li'); row.append(el('span', 'checkmark', '✓'), el('span', 'binary', `L${i}`), document.createTextNode(` ${KIND[type] ?? type.toUpperCase()} + MLP`)); $('layers').append(row); });
  $('model-kind').textContent = stub ? 'DEV STUB' : 'LIVE LLM';
  setStatus(stub ? 'STUB / SYNTHETIC LOGITS' : 'HOST ONLINE', true);
}
$('step').addEventListener('click', () => { stop(); step(); });
$('run').addEventListener('click', run);
$('back').addEventListener('click', back);
$('reset').addEventListener('click', reset);
$('prompt').addEventListener('keydown', event => { if (event.key === 'Enter') reset(); });
$('top-n').addEventListener('change', () => { $('top-n').value = topN(); if (model) render(); });
$('temperature').addEventListener('change', () => {
  $('temperature').value = temperature();
  if (pending && !busy) { pending = distribution(pending.logits, temperature()); render(); }
});
document.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest?.('input,select,button,[role="button"]')) return;
  if (event.key === ' ' || event.key === 'ArrowRight') { event.preventDefault(); stop(); step(); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); back(); }
});

const cache = mountCache({ release: async () => { running = false; await model?.dispose?.(); model = null; } });
// Inspector rail: collapsed by default; the user's own toggle persists, a node click opens it for this visit only.
function setInspector(open, remember = true) {
  $('inspector').dataset.open = open; $('inspector-toggle').setAttribute('aria-expanded', open); $('inspector-body').hidden = !open;
  if (remember) try { localStorage.setItem('delos-inspector-open', open); } catch {}
  $('tree-viewport').scrollLeft = $('tree').scrollWidth;
}
$('inspector-toggle').addEventListener('click', () => setInspector($('inspector').dataset.open !== 'true'));
// Re-pin the newest tree column whenever the viewport resizes: the inspector transition, a window resize or a reflow breakpoint.
new ResizeObserver(() => { $('tree-viewport').scrollLeft = $('tree').scrollWidth; }).observe($('tree-viewport'));
try { setInspector(localStorage.getItem('delos-inspector-open') === 'true', false); } catch {}

const fail = (title, detail) => {
  $('loading-overlay').hidden = false; $('loading-overlay').classList.add('error'); $('loading-title').textContent = title;
  $('loading-progress').hidden = true; $('load-host').hidden = true; $('loading-detail').textContent = detail;
  setStatus('HOST OFFLINE');
};
async function load(engine) {
  loading = true; cache.busy(true);
  $('load-host').hidden = true; $('loading-overlay').classList.remove('error');
  $('loading-title').textContent = 'LOADING HOST…';
  const bar = $('loading-progress'); bar.hidden = false; bar.removeAttribute('value');
  $('loading-detail').textContent = 'Starting WebGPU…';
  model = await engine.loadModel(({ loaded, total }) => {
    bar.max = total; bar.value = loaded;
    $('loading-detail').textContent = `Downloading ${engine.MODEL.name} · ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`;
  });
  $('loading-detail').textContent = 'Warming up…';
  config();
  if (params.has('prompt')) $('prompt').value = params.get('prompt');
  await reset();
  $('loading-overlay').hidden = true;
  const count = Math.min(100, Math.max(0, Number.parseInt(params.get('autostep'), 10) || 0));
  for (let i = 0; i < count; i++) await step();
  document.body.dataset.ready = 'true';
  $('step').focus({ preventScroll: true });
}
const loaded = async engine => { try { await load(engine); } finally { loading = false; cache.busy(false); cache.refresh(); } };
const note = takeNote();
try {
  if (location.protocol === 'file:') throw new Error('Browser modules need HTTP. Serve with `python3 -m http.server` then open http://localhost:8000/');
  const stub = params.has('stub');
  const engine = await import(stub ? './dev/stub-engine.js' : './engine.js');
  if (!stub && !navigator.gpu) fail('WEBGPU REQUIRED', 'This host runs a 0.8B-parameter model on your GPU. WebGPU is required (Chrome/Edge 113+, Safari 26+). There is no WASM fallback at this size.');
  else {
    const cached = stub || await engine.isCached(), mb = Math.round(engine.MODEL.bytes / 1e6);
    if (cached) await loaded(engine);
    else {
      $('loading-title').textContent = `${engine.MODEL.name} / HOST`;
      $('loading-progress').hidden = true;
      $('loading-detail').textContent = note + `${mb} MB · cached by your browser after first load.\nWeights download from Hugging Face; inference then runs locally on your GPU. Nothing you type leaves this page.`;
      setStatus('HOST OFFLINE');
      const button = $('load-host'); button.hidden = false; button.focus({ preventScroll: true });
      button.addEventListener('click', () => loaded(engine).catch(error => { fail('HOST LOAD FAILED', `${error.message}\nReload to retry.`); console.error(error); }), { once: true });
    }
  }
} catch (error) {
  fail('HOST LOAD FAILED', `${error.message}\nUse ?stub for the synthetic development host.`);
  console.error(error);
}

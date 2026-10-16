const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
let model, promptIds = [], chain = [], pending = null, selected = -1, busy = false, running = false;
let lastMs = 0;
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

// Stable full-vocabulary softmax. T=0 is the deterministic limit, ties share mass.
function distribution(logits, temp) {
  if (!logits.length || Array.from(logits).some(x => !Number.isFinite(x))) throw new Error('Model returned invalid logits.');
  const max = Math.max(...logits);
  const probs = Float64Array.from(logits, x => temp === 0 ? Number(x === max) : Math.exp((x - max) / temp));
  const total = probs.reduce((a, b) => a + b, 0);
  let entropy = 0;
  probs.forEach((p, i) => { probs[i] = p / total; if (probs[i]) entropy -= probs[i] * Math.log2(probs[i]); });
  const order = Array.from(logits, (_, i) => i).sort((a, b) => probs[b] - probs[a] || logits[b] - logits[a] || a - b);
  return { logits, probs, order, entropy, max, temp };
}
function phase(name) {
  $('phase').textContent = name;
  document.querySelectorAll('#processes li').forEach(item => item.classList.toggle('selected', item.dataset.phase === name));
  $('forward-engine').classList.toggle('busy', name === 'FORWARD');
}
function controls() {
  // ponytail: stop at context window; past it every step is a ~4s cold forward. Add sliding-window KV if long stories matter.
  if (model && pending && ids().length >= model.config.sequence_len) { pending = null; running = false; }
  $('step').disabled = !model || busy || !pending;
  $('run').disabled = !model || busy && !running || !pending && !running;
  $('reset').disabled = !model || busy;
  $('back').disabled = !model || busy || !chain.length;
  $('run').textContent = running ? 'PAUSE' : 'RUN';
  $('prompt').disabled = busy;
  $('temperature').disabled = busy;
  $('mode').disabled = busy;
}
function stop() { running = false; controls(); }
async function operate(action) {
  if (busy || !model) return;
  busy = true; controls();
  try { await action(); $('app-message').textContent = ''; }
  catch (error) { stop(); $('app-message').textContent = `INFERENCE ERROR: ${error.message}`; }
  finally { busy = false; phase('READY'); controls(); }
}
async function forward() {
  phase('FORWARD');
  const context = ids().slice(-model.config.sequence_len);
  // Yield one paint before synchronous browser inference blocks the main thread.
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
    promptIds = encoded.length ? encoded : [model.bosId];
    chain = []; pending = null; selected = -1;
    render(true);
    await forward();
  });
}
async function step() {
  if (ids().length >= model.config.sequence_len) return stop();
  await operate(async () => {
    if (!pending) await forward();
    phase('SAMPLE');
    let id = pending.order[0];
    if ($('mode').value === 'sample' && pending.temp > 0) {
      let draw = Math.random();
      id = pending.order.at(-1);
      for (const candidate of pending.order) { draw -= pending.probs[candidate]; if (draw <= 0) { id = candidate; break; } }
    }
    chain.push({ id, dist: pending }); pending = null;
    render(true);
    await forward();
  });
}
async function branch(index, id) {
  stop();
  await operate(async () => {
    const dist = index === chain.length ? pending : chain[index]?.dist;
    if (!dist) return;
    chain = chain.slice(0, index);
    chain.push({ id, dist }); pending = null;
    render(true);
    await forward();
  });
}
async function back() {
  stop();
  await operate(async () => {
    if (!chain.length) return;
    chain.pop(); pending = null;
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
  $('inspector-title').textContent = `LOGIT INSPECTOR / ${selected < 0 ? 'NEXT' : `TOKEN ${selected + 1}`}`;
  $('logits-table').replaceChildren();
  if (!dist) { $('distribution-meta').textContent = 'FORWARD PASS PENDING'; return; }
  $('distribution-meta').textContent = `ENTROPY ${dist.entropy.toFixed(3)} bits   MAX LOGIT ${dist.max.toFixed(3)}   T ${dist.temp.toFixed(1)}`;
  let cumulative = 0;
  dist.order.slice(0, topN()).forEach((id, rank) => {
    cumulative += dist.probs[id];
    const row = el('tr', id === item.id ? 'chosen' : ''); row.tabIndex = 0;
    const info = `${token(id)}; raw logit ${dist.logits[id].toFixed(4)}; probability ${pct(dist.probs[id])}; cumulative ${pct(cumulative)}`;
    row.title = info; row.setAttribute('aria-label', info);
    row.append(el('td', '', String(rank + 1)));
    const cell = el('td', '', token(id));
    const bar = el('span', 'table-bar'); bar.style.width = `${dist.probs[id] * 100}%`; cell.append(bar); row.append(cell);
    for (const value of [dist.logits[id].toFixed(3), dist.temp ? (dist.logits[id] / dist.temp).toFixed(3) : '—', pct(dist.probs[id]), pct(cumulative)]) row.append(el('td', '', value));
    $('logits-table').append(row);
  });
  $('inspector-note').textContent = `Top ${topN()} of ${dist.logits.length} tokens. Full-vocabulary softmax.${dist.temp === 0 ? ' T=0: greedy limit; logit/T undefined.' : ''}${item.id !== undefined && !dist.order.slice(0, topN()).includes(item.id) ? ` Chosen token ${token(item.id)} outside top-N (${pct(dist.probs[item.id])}).` : ''}`;
}
function entropyChart() {
  const svg = $('entropy-chart'), ns = 'http://www.w3.org/2000/svg';
  svg.replaceChildren();
  const values = chain.map(item => item.dist.entropy);
  const title = document.createElementNS(ns, 'title'); title.textContent = values.length ? values.map((v, i) => `Step ${i + 1}: ${v.toFixed(3)} bits`).join('; ') : 'No generated tokens yet'; svg.append(title);
  if (!values.length) { $('entropy-hover').textContent = 'NO STEPS'; return; }
  const ceiling = Math.log2(model.config.vocab_size);
  const points = values.map((v, i) => [values.length === 1 ? 125 : i * 250 / (values.length - 1), 31 - v / ceiling * 28]);
  const line = document.createElementNS(ns, 'polyline'); line.setAttribute('class', 'wave'); line.setAttribute('points', points.map(p => p.join(',')).join(' ')); svg.append(line);
  const crosshair = document.createElementNS(ns, 'line'); crosshair.setAttribute('class', 'entropy-crosshair'); crosshair.setAttribute('y1', '0'); crosshair.setAttribute('y2', '34'); crosshair.style.display = 'none'; svg.append(crosshair);
  points.forEach(([x, y], i) => {
    const hit = document.createElementNS(ns, 'rect'); hit.setAttribute('class', 'entropy-hit'); hit.setAttribute('x', String(Math.max(0, x - 12))); hit.setAttribute('y', '0'); hit.setAttribute('width', '24'); hit.setAttribute('height', '34'); hit.setAttribute('tabindex', '0'); hit.setAttribute('aria-label', `Step ${i + 1}: ${values[i].toFixed(3)} bits`);
    const show = () => { crosshair.style.display = ''; crosshair.setAttribute('x1', x); crosshair.setAttribute('x2', x); $('entropy-hover').textContent = `#${i + 1} ${values[i].toFixed(2)} bits`; };
    hit.addEventListener('pointerenter', show); hit.addEventListener('focus', show); svg.append(hit);
    if (points.length === 1) { const dot = document.createElementNS(ns, 'circle'); dot.setAttribute('cx', x); dot.setAttribute('cy', y); dot.setAttribute('r', '4'); dot.setAttribute('class', 'entropy-point'); svg.append(dot); }
  });
  $('entropy-hover').textContent = `${values.at(-1).toFixed(2)} bits`;
}
function render(scroll = false) {
  const tree = $('tree'); tree.replaceChildren();
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
    const node = el('article', 'node'); node.append(el('span', 'node-label', 'PROMPT'), el('h2', '', token(id))); node.title = `Prompt token ${id}: ${model.tokenText(id)}`; column.append(node); tree.append(column);
    connect(x, 47, true); previous = { x: x + 76, y: 47 }; x += 91;
  });
  const columns = [...chain, ...(pending ? [{ dist: pending, preview: true }] : [])];
  columns.forEach((item, index) => {
    x += 48;
    const y = 28 + index % 4 * 29;
    const column = el('div', `word-column${item.preview ? ' preview-column' : ''}`); column.style.left = `${x}px`; column.style.top = `${y}px`;
    const node = el('article', `node${selected === index || item.preview && selected < 0 ? ' selected' : ''}`); node.tabIndex = 0; node.setAttribute('role', 'button');
    node.append(el('span', 'node-label', item.preview ? 'NEXT / PREVIEW' : 'FORWARD CHAIN'), el('h2', '', item.preview ? '?' : token(item.id)));
    const select = () => { selected = item.preview ? -1 : index; render(); };
    node.addEventListener('click', select); node.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); event.stopPropagation(); select(); } });
    column.append(node, el('span', 'column-caption', item.preview ? 'PENDING DISTRIBUTION' : `TOKEN ${index + 1} / ID ${item.id}`));
    item.dist.order.slice(0, topN()).forEach(id => {
      const chip = el('button', `chip${id === item.id ? ' on' : ''}`); chip.type = 'button';
      chip.append(el('span', 'token', token(id)), el('span', 'prob', pct(item.dist.probs[id])));
      const bar = el('span', 'prob-bar'); bar.style.width = `${item.dist.probs[id] * 100}%`; chip.append(bar);
      chip.title = `${model.tokenText(id)} (ID ${id}) · ${pct(item.dist.probs[id])}${id === item.id ? ' · chosen' : ' · choose / re-branch'}`;
      chip.setAttribute('aria-label', chip.title); chip.setAttribute('aria-pressed', String(id === item.id));
      chip.addEventListener('click', () => branch(index, id)); column.append(chip);
    });
    tree.append(column); connect(x, y + 37, false); previous = { x: x + 150, y: y + 37 }; x += 150;
  });
  tree.style.width = `${x + 24}px`; tree.style.height = `${Math.max(508, 205 + topN() * 26)}px`;
  const promptText = model ? model.decode(promptIds) : '';
  const fullText = model ? model.decode(ids()) : '';
  $('decoded').replaceChildren(el('span', 'prompt-text', promptText), el('span', '', fullText.slice(promptText.length)));
  $('tokens-count').textContent = chain.length;
  $('top-prob').textContent = pending ? (pending.probs[pending.order[0]] * 100).toFixed(1) : '—';
  $('forward-ms').textContent = Math.round(lastMs);
  inspector(); entropyChart();
  if (scroll) $('tree-viewport').scrollLeft = tree.scrollWidth;
}
function config() {
  const c = model.config;
  for (const [label, value] of [['LAYERS / HEADS', `${c.n_layer} / ${c.n_head} (KV ${c.n_kv_head})`], ['D_MODEL / VOCAB', `${c.n_embd} / ${c.vocab_size}`], ['PARAMETERS', Number(c.n_params).toLocaleString()], ['CONTEXT', `${c.sequence_len} tokens`], ['VALIDATION LOSS', Number.isFinite(c.val_loss) ? c.val_loss.toFixed(4) : 'not supplied']]) {
    const row = el('div'); row.append(el('dt', '', label), el('dd', '', value)); $('model-config').append(row);
  }
  for (let i = 0; i < c.n_layer; i++) { const row = el('li'); row.append(el('span', 'checkmark', '✓'), el('span', 'binary', `L${i}`), document.createTextNode(' ATTENTION + MLP')); $('layers').append(row); }
  $('model-kind').textContent = params.has('stub') ? 'DEV STUB' : 'LIVE GPT';
  $('system-status').textContent = params.has('stub') ? 'STUB / SYNTHETIC LOGITS' : 'HOST ONLINE';
}
$('step').addEventListener('click', () => { stop(); step(); });
$('run').addEventListener('click', run);
$('back').addEventListener('click', back);
$('reset').addEventListener('click', reset);
$('prompt').addEventListener('keydown', event => { if (event.key === 'Enter') reset(); });
$('top-n').addEventListener('change', () => { $('top-n').value = topN(); render(); });
$('temperature').addEventListener('change', () => {
  $('temperature').value = temperature();
  if (pending && !busy) { pending = distribution(pending.logits, temperature()); render(); }
});
document.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest?.('input,select,button,[role="button"]')) return;
  if (event.key === ' ' || event.key === 'ArrowRight') { event.preventDefault(); stop(); step(); }
  if (event.key === 'ArrowLeft') { event.preventDefault(); back(); }
});

// Runnable browser self-check; no test framework or engine fixtures required.
const sanity = distribution(new Float32Array([1000, 1001, -1000]), 1);
if (Math.abs(sanity.probs.reduce((a, b) => a + b, 0) - 1) > 1e-10 || sanity.order[0] !== 1 || distribution(new Float32Array([2, 3]), 0).probs[1] !== 1) throw new Error('Softmax self-check failed');
try {
  if (location.protocol === 'file:') throw new Error('Browser modules need HTTP. Serve with `python3 -m http.server` then open http://localhost:8000/');
  const { loadModel } = await import(new URLSearchParams(location.search).has('stub') ? './dev/stub-engine.js' : './engine.js');
  $('loading-detail').textContent = 'Loading host weights… progress unavailable from engine.';
  model = await loadModel();
  $('loading-progress').max = 1; $('loading-progress').value = 1;
  config();
  if (params.has('prompt')) $('prompt').value = params.get('prompt');
  await reset();
  $('loading-overlay').hidden = true;
  const count = Math.min(100, Math.max(0, Number.parseInt(params.get('autostep'), 10) || 0));
  for (let i = 0; i < count; i++) await step();
  document.body.dataset.ready = 'true';
} catch (error) {
  $('loading-overlay').classList.add('error'); $('loading-title').textContent = 'HOST LOAD FAILED';
  $('loading-progress').hidden = true;
  $('loading-detail').textContent = `${error.message}\nServe with \`python3 -m http.server\`. Check engine.js and model/ files. Use ?stub for the synthetic development host.`;
  $('system-status').textContent = 'HOST OFFLINE';
  console.error(error);
}

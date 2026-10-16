import { loadModel } from './engine.js';
import { decide } from './s1.js';

const $ = id => document.getElementById(id), pct = p => `${(100 * p).toFixed(1)}%`, ms = x => x > 0 ? `${x.toFixed(0)} ms` : 'timing unavailable (clock frozen)';
const escape = text => String(text).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const json = x => JSON.stringify(x, null, 2), yieldPaint = () => new Promise(r => setTimeout(r, 0));
const modelName = 'tinystories-15m-local';
let model, fixtures, busy = false, route = '', runRoute, gateAnswer, forgeAnswers, inferenceRuns = 0;
const exampleState = 'Lily got a lovely present from her mother. She smiled and danced around the room.';
const happiness = ['very sad', 'sad', 'calm', 'happy', 'very happy'];
const exampleQuestions = {
  mood: { type: 'choice', instructions: 'She was very', criteria: { happy: 'happy', sad: 'sad', angry: 'angry', scared: 'scared' } },
  happiness: { type: 'score', instructions: 'She felt', criteria: happiness },
  smiled: { type: 'noul', instructions: 'Question: Did Lily smile? Answer:', criteria: { true: 'yes', false: 'no' } },
};
const request = (story, questions) => ({ model: modelName, state: { story }, questions });
const panel = (title, body, cls = '') => `<section class="module s1-panel ${cls}"><h2>${title}</h2>${body}</section>`;
const button = (id, text) => `<button id="${id}" type="button" class="s1-run">${text}</button>`;
const note = text => `<p class="s1-note">${text}</p>`;
const blank = text => `<p class="s1-empty">${text}</p>`;
const textArea = (id, label, value, rows = 3) => `<label class="s1-field">${label}<textarea id="${id}" rows="${rows}" spellcheck="false">${escape(value)}</textarea></label>`;
function overlay(title, detail) { $('loading-overlay').hidden = false; $('loading-title').textContent = title; $('loading-detail').textContent = detail; }
async function operate(action) {
  if (busy || !model) return;
  busy = true; document.body.dataset.ready = 'false'; $('app-message').textContent = '';
  document.querySelectorAll('#route button, #route input, #route select, #route textarea').forEach(x => x.disabled = true);
  overlay('EVALUATING HOST…', 'Real local model inference. Shared state cache; sequential question suffixes.');
  await yieldPaint();
  try { await action(); }
  catch (error) { $('app-message').textContent = `INFERENCE ERROR: ${error.message}`; console.error(error); }
  finally {
    busy = false; $('loading-overlay').hidden = true;
    document.querySelectorAll('#route button, #route input, #route select, #route textarea').forEach(x => x.disabled = false);
    if (currentRoute() !== route) renderRoute();
    document.body.dataset.ready = 'true';
  }
}
function tooltip(target, text, output) {
  target.title = text; target.tabIndex = 0; target.setAttribute('aria-label', text);
  if (output) { target.addEventListener('pointerenter', () => output.textContent = text); target.addEventListener('focus', () => output.textContent = text); }
}
function bars(probabilities, chosen, legend = {}) {
  return `<div class="s1-probs">${Object.entries(probabilities).map(([key, p]) => `<div class="s1-prob ${key === chosen ? 'chosen' : ''}" tabindex="0" title="${escape(`${Object.hasOwn(legend, key) ? legend[key] : key}: ${pct(p)}`)}"><span>${key === chosen ? '› ' : ''}${escape(Object.hasOwn(legend, key) ? legend[key] : key)}</span><div class="s1-track"><i style="width:${100 * p}%"></i></div><strong>${pct(p)}</strong></div>`).join('')}</div>`;
}
function answerHtml(id, answer) {
  const isNoul = answer.type === 'noul', probabilities = isNoul ? { true: answer.noul, false: 1 - answer.noul } : answer.probabilities;
  const p = Object.values(probabilities), chosen = answer.type === 'choice' ? answer.choice : isNoul ? null : String(p.indexOf(Math.max(...p)));
  const heading = answer.type === 'choice' ? `CHOICE / ${answer.choice}` : isNoul ? `NOUL / P(TRUE) ${pct(answer.noul)}` : `SCORE / ${answer.score.toFixed(3)} OF ${p.length - 1}`;
  return `<article class="s1-answer"><h3>${escape(id)} <span>${escape(heading)}</span></h3>${isNoul ? '<p class="s1-warning">UNRELIABLE ON THIS MODEL — measured AUROC 0.49 (n=40)</p>' : ''}${bars(probabilities, chosen, answer.legend)}${answer.type === 'score' ? `<div class="s1-mean" title="Weighted mean ${answer.score.toFixed(3)}"><i style="left:${100 * answer.score / (p.length - 1)}%"></i><span>0</span><b>Weighted mean ${answer.score.toFixed(2)}</b><span>${p.length - 1}</span></div>` : ''}${isNoul ? note('No confidence field in the Noul API. These are likelihood ratios, not calibrated truth.') : `<div class="s1-confidence"><span>Jev-formula confidence (tiny GPT) <strong>${pct(answer.confidence)}</strong></span><span>Clef-formula pmax (tiny GPT) <strong>${pct(Math.max(...p))}</strong></span></div>`}</article>`;
}
function confidenceNote() { return note('Jev Choice removes the uniform baseline; Jev Score measures distance from the modal level. Clef-style confidence is pmax for both. All numbers come from the in-browser 15M TinyStories GPT; Jev/Clef name the published confidence formulas, not calls to those models. Concentration is not correctness.'); }
function showJSON(req, response, target = 'raw') {
  $(target).innerHTML = `<details class="s1-json" open><summary>RAW REQUEST / API CONTRACT</summary><pre id="request-preview">${escape(json(req))}</pre></details><details class="s1-json" open><summary>RAW RESPONSE / ${response.usage.input_tokens} INPUT TOKENS · 0 GENERATED</summary><pre>${escape(json(response))}</pre></details>`;
}
function card(id, q) {
  const div = document.createElement('article'); div.className = 's1-question';
  div.innerHTML = `<div class="s1-question-head"><label>ID <input class="q-id" value="${escape(id)}" maxlength="64"></label><label>TYPE <select class="q-type"><option value="choice">choice</option><option value="score">score</option><option value="noul">noul</option></select></label><button class="q-remove" type="button" aria-label="Remove question ${escape(id)}">REMOVE</button></div><label class="s1-field">INSTRUCTIONS / CLOZE LEAD-IN<input class="q-instructions" value="${escape(q.instructions)}"></label><label class="s1-field">CRITERIA / JSON LITERAL CONTINUATIONS<textarea class="q-criteria" rows="3" spellcheck="false">${escape(json(q.criteria))}</textarea></label><p class="s1-warning q-warning" ${q.type !== 'noul' ? 'hidden' : ''}>UNRELIABLE ON THIS MODEL — measured AUROC 0.49 (n=40)</p>`;
  div.querySelector('.q-type').value = q.type;
  div.querySelector('.q-type').addEventListener('change', event => {
    const type = event.target.value;
    div.querySelector('.q-criteria').value = json(type === 'score' ? happiness : type === 'noul' ? { true: 'yes', false: 'no' } : exampleQuestions.mood.criteria);
    div.querySelector('.q-instructions').value = type === 'noul' ? 'Question: Did Lily smile? Answer:' : type === 'score' ? 'She felt' : 'She was very';
    div.querySelector('.q-warning').hidden = type !== 'noul'; updateRequest();
  });
  div.querySelector('.q-remove').onclick = () => { div.remove(); updateRequest(); };
  div.addEventListener('input', updateRequest);
  return div;
}
function parseCriteria(text) {
  // Criteria are flat objects of literal strings (or a level array); JSON.parse silently drops duplicate object keys.
  const keys = [...text.matchAll(/"(?:\\.|[^"\\])*"(?=\s*:)/g)].map(match => JSON.parse(match[0]));
  if (new Set(keys).size !== keys.length) throw new Error('Duplicate criteria key. Every option must have a unique key.');
  return JSON.parse(text);
}
function playgroundRequest() {
  const entries = [...document.querySelectorAll('.s1-question')].map(div => {
    const id = div.querySelector('.q-id').value.trim();
    if (!id) throw new Error('Question IDs cannot be empty.');
    return [id, { type: div.querySelector('.q-type').value, instructions: div.querySelector('.q-instructions').value, criteria: parseCriteria(div.querySelector('.q-criteria').value) }];
  });
  if (new Set(entries.map(([id]) => id)).size !== entries.length) throw new Error('Question IDs must be unique.');
  return request($('state').value, Object.fromEntries(entries));
}
function updateRequest() {
  try { $('request-preview').textContent = json(playgroundRequest()); $('app-message').textContent = ''; }
  catch (e) { $('app-message').textContent = `EDITOR: ${e.message}`; }
}
function playground() {
  $('route').innerHTML = `<div class="s1-grid playground-grid">${panel('STATE + TYPED QUESTIONS', `${textArea('state', 'STATE.STORY / TINYSTORIES CONTEXT', exampleState)}<div class="s1-toolbar">${button('run', 'RUN JUDGMENTS')}<label>METHOD <select id="method"><option value="sum">SUM / CONTINUATIONS</option><option value="codes">CODES / ANSWER LETTERS</option></select></label><label>T <input id="temperature" type="number" min="0.1" max="5" step="0.1" value="1"></label><button id="add" type="button">ADD QUESTION</button></div><div id="questions" class="s1-question-list"></div>`)}${panel('TYPED ANSWERS / LIVE LOCAL INFERENCE', `<div id="answers">${blank('RUN to score literal continuations. No text generation, no remote call.')}</div>${confidenceNote()}<div id="timing"></div><div id="raw" class="s1-raw"><details class="s1-json" open><summary>RAW REQUEST / API CONTRACT</summary><pre id="request-preview"></pre></details></div>`)}</div>`;
  Object.entries(exampleQuestions).forEach(([id, q]) => $('questions').append(card(id, q)));
  $('state').oninput = updateRequest;
  $('add').onclick = () => { let n = document.querySelectorAll('.s1-question').length + 1; const used = [...document.querySelectorAll('.q-id')].map(x => x.value); while (used.includes(`q${n}`)) n++; $('questions').append(card(`q${n}`, exampleQuestions.mood)); updateRequest(); };
  runRoute = () => operate(async () => {
    const req = playgroundRequest(), result = await decide(model, req, { method: $('method').value, temperature: Number($('temperature').value) });
    $('answers').innerHTML = Object.entries(result.answers).map(([id, a]) => answerHtml(id, a)).join('');
    $('timing').innerHTML = note(`LIVE: state prefill ${ms(result.timing.state_prefill_ms)} · ${Object.entries(result.timing.questions).map(([id, t]) => `${escape(id)} ${ms(t)}`).join(' · ')}. Logical token budget, not KV work or remote billing.`);
    showJSON(req, result);
  });
  $('run').onclick = runRoute; updateRequest();
}
const paragraph = 'Once upon a time, Lily and Tom went to the park with their mother. Lily carried a red ball and Tom carried a blue kite. They played near a big tree. A little dog watched them. The sun was warm and the birds sang. Lily gave the dog some water. Tom helped his mother find a place to sit. They were kind to each other. Later they shared a nice lunch and talked about their toys. ';
function timeline(target, segments, total, scale) {
  if (!(scale > 0) || !(total > 0)) { $(target).innerHTML = note('Timing unavailable (clock frozen). Use a real-time browser session for inference timelines.'); return; }
  let at = 0;
  $(target).innerHTML = `<div class="s1-timeline" aria-label="Timeline in milliseconds, common scale ${ms(scale)}">${segments.map(([label, duration]) => {
    const start = at; at += duration;
    return `<div class="s1-time-row" tabindex="0" title="${escape(`${label}: start ${ms(start)}, duration ${ms(duration)}`)}"><span>${escape(label)}</span><div class="s1-time-track"><i style="left:${100 * start / scale}%;width:${Math.max(.12, 100 * duration / scale)}%"></i></div><strong>${ms(duration)}</strong></div>`;
  }).join('')}<p class="s1-note">Measured inference ${ms(total)} · common axis 0–${ms(scale)}; question suffix durations include paint yields.</p></div>`;
}
function fanout() {
  const state = model.decode(model.encode(paragraph.repeat(3)).slice(0, 150));
  $('route').innerHTML = `${textArea('fan-state', 'SHARED STATE / DEFAULT 150 TOKENS', state, 3)}<div class="s1-toolbar">${button('run', 'RUN BOTH SYSTEMS')}<label>QUESTIONS <input id="fan-count" type="number" value="10" min="1" max="12"></label><span>Same state, same question set. Cold benchmark also measured live.</span></div><div class="s1-grid">${panel('SYSTEM ONE / SHARED PREFIX + N SUFFIXES', '<div id="fan-one">'+blank('RUN to measure state prefill and each question suffix.')+'</div>')}${panel('SYSTEM TWO / GREEDY TOKEN STEPPING', '<div id="fan-two">'+blank('Same real logits loop as CODE; generation gets approximately the shared-run wall time.')+'</div>')}</div>${panel('CACHE IS THE SPEEDUP — NOT A JOINT CLASSIFICATION HEAD', '<div id="fan-speed">'+note('Probe (node, 3 repetitions): cold 150-token state 851 ms; 10 questions cold 9.9 s vs shared 1.9 s, ≈5.2×. Extra question ≈104 ms; switching option ≈14 ms. Browser timings vary.')+'</div>'+note('Jev-style answer-letter readout is chance-like on this GPT. Clef uses pooled option spans plus a cross-attention head in one joint prefill; continuation scoring is closer in spirit, but still sequential here.'))}`;
  runRoute = () => operate(async () => {
    const count = Math.max(1, Math.min(12, Number.parseInt($('fan-count').value, 10) || 10)); $('fan-count').value = count;
    const story = $('fan-state').value, questions = Object.fromEntries(Array.from({ length: count }, (_, i) => [`q${i + 1}`, { type: 'choice', instructions: `Question: What did ${i % 2 ? 'Tom' : 'Lily'} carry ${['today', 'to the park', 'in the story', 'outside', 'near the tree', 'that day'][Math.floor(i / 2)]}? Answer:`, criteria: { ball: 'red ball', kite: 'blue kite', hat: 'green hat', boat: 'yellow boat' } }]));
    const firstToken = model.encode(story)[0];
    const resetIds = ['Unrelated reset.', 'Different reset.', 'Apple reset.', 'Zebra reset.'].map(text => model.encode(text)).find(ids => ids[0] !== firstToken);
    if (!resetIds) throw new Error('Unable to construct a cold-cache reset.');
    let cold = 0;
    for (const [id, q] of Object.entries(questions)) {
      $('loading-detail').textContent = `Cold benchmark ${id}/${count}; each question resets the shared prefix.`;
      await model.logits(resetIds);
      const result = await decide(model, request(story, { [id]: q })); cold += result.timing.state_prefill_ms + result.timing.questions[id];
    }
    await model.logits(resetIds);
    $('loading-detail').textContent = 'Shared benchmark: one state prefill, all question suffixes.';
    const shared = await decide(model, request(story, questions)), budget = shared.timing.total_ms;
    const segments = [['STATE PREFILL', shared.timing.state_prefill_ms], ...Object.entries(shared.timing.questions)], sharedInference = segments.reduce((s, [, v]) => s + v, 0);
    await model.logits(resetIds); await yieldPaint();
    const ids = model.encode(story), generated = [], steps = [], start = performance.now();
    let t = performance.now(), z = await model.logits(ids), prefill = performance.now() - t;
    while (performance.now() - start < budget && ids.length < model.config.sequence_len - 1) {
      let best = 0; for (let i = 1; i < z.length; i++) if (z[i] > z[best]) best = i;
      ids.push(best); generated.push(best); await yieldPaint(); t = performance.now(); z = await model.logits(ids); steps.push(performance.now() - t);
      $('loading-detail').textContent = `System Two: ${generated.length} greedy tokens; shared wall-time budget ${ms(budget)}.`;
    }
    const generationWall = performance.now() - start, generationInference = prefill + steps.reduce((a, b) => a + b, 0), scale = Math.max(sharedInference, generationInference);
    $('fan-one').innerHTML = `<p class="s1-result-line">${count} typed answers · shared ${ms(sharedInference)} · cold ${ms(cold)} · ${cold > 0 && sharedInference > 0 ? `${(cold / sharedInference).toFixed(2)}× speedup` : 'timing unavailable (clock frozen)'}</p><div id="timeline-one"></div><div class="s1-fan-answers">${Object.entries(shared.answers).map(([id, a]) => `<span>${id}: ${escape(a.choice)} <b>${pct(a.confidence)}</b></span>`).join('')}</div>`;
    $('fan-two').innerHTML = `<p class="s1-result-line">${generated.length} tokens · ${ms(generationWall)} wall time · budget ${ms(budget)}</p><div id="timeline-two"></div><div class="s1-generated">${escape(model.decode(generated)) || 'No token generated within budget.'}</div>${note('Greedy next-token generation is genuine; it does not answer the typed questions. Mirrors the CODE greedy loop, including special tokens. Stops at budget or context ceiling.')}`;
    timeline('timeline-one', segments, sharedInference, scale);
    const chunks = Array.from({ length: Math.ceil(steps.length / 10) }, (_, i) => [`TOKENS ${i * 10 + 1}–${Math.min((i + 1) * 10, steps.length)}`, steps.slice(i * 10, (i + 1) * 10).reduce((a, b) => a + b, 0)]);
    timeline('timeline-two', [['STATE PREFILL', prefill], ...chunks], generationInference, scale);
    $('fan-speed').innerHTML = `<p class="s1-result-line">LIVE / ${count} questions: cold ${ms(cold)} vs shared ${ms(sharedInference)} · <strong>${cold > 0 && sharedInference > 0 ? `${(cold / sharedInference).toFixed(2)}× speedup` : 'timing unavailable (clock frozen)'}</strong></p>${note('Same suffixes, unrelated cache reset before every cold question; reset time excluded. Shared state prefill counted once. Not a parallel forward pass. Probe: 9.9 s vs 1.9 s (10 questions, 3 repeats); cold 150-token state 851 ms.')}`;
  }); $('run').onclick = runRoute;
}
const presets = [
  { name: 'LILY / THE PRESENT', story: exampleState, instructions: 'She was very', criteria: { smile: 'happy', comfort: 'sad', retreat: 'scared', confront: 'angry' } },
  { name: 'SARA / LOST PUPPY', story: 'Sara found her lost puppy. She hugged him and laughed with joy.', instructions: 'She was very', criteria: { smile: 'happy', comfort: 'sad', retreat: 'scared', confront: 'angry' } },
  { name: 'LILY / BROKEN TOY', story: 'Lily saw a boy break her toy on purpose. She frowned and shouted at him.', instructions: 'She was very', criteria: { smile: 'happy', comfort: 'sad', retreat: 'scared', confront: 'angry' } },
  { name: 'SARA / DARK ROOM', story: 'Sara heard a loud noise in the dark. She thought a monster was in her room and held her mother tightly.', instructions: 'She was very', criteria: { smile: 'happy', comfort: 'sad', retreat: 'scared', confront: 'angry' } },
];
function slider(id, label, value, max = 1, step = .01) { return `<label class="s1-slider">${label}<input id="${id}" type="range" min="0" max="${max}" step="${step}" value="${value}"><output id="${id}-value">${value.toFixed(2)}</output></label>`; }
function gate() {
  gateAnswer = null;
  $('route').innerHTML = `<div class="s1-grid">${panel('HOST STORY / EMOTION TO FICTIONAL ACTION', `<div class="s1-toolbar"><label>PRESET <select id="preset">${presets.map((x, i) => `<option value="${i}">${x.name}</option>`).join('')}</select></label>${button('run', 'ASSESS HOST')}</div>${textArea('gate-state', 'STATE.STORY', presets[0].story, 5)}${textArea('gate-instructions', 'INSTRUCTIONS / CLOZE LEAD-IN', presets[0].instructions, 1)}${note('Code maps mood labels to pretend actions: happy→smile, sad→comfort, scared→retreat, angry→confront. Criteria still contain literal mood continuations. No real action is executed.')}<div id="gate-answer">${blank('ASSESS HOST to produce a real choice distribution.')}</div>`)}${panel('CONFIDENCE POLICY / CODE OWNS THE DECISION', `${slider('execute', 'EXECUTE AT OR ABOVE', .8)}${slider('confirm', 'CONFIRM AT OR ABOVE', .45)}${slider('escalate', 'ESCALATE AT OR ABOVE', .15)}<div id="gate-band" class="s1-band">AWAITING JUDGMENT</div><div id="gate-policy"></div>${confidenceNote()}${note('Threshold sliders are constrained: execute ≥ confirm ≥ escalate. Below escalate: HOLD. Changes apply to the cached judgment without inference. Toy-demo policy only; calibrate thresholds on your own labeled outcomes.')}`)}</div>`;
  $('preset').onchange = () => { const p = presets[Number($('preset').value)]; $('gate-state').value = p.story; $('gate-instructions').value = p.instructions; gateAnswer = null; $('gate-answer').innerHTML = blank('Preset changed. ASSESS HOST for a fresh judgment.'); updateGate(); };
  for (const id of ['gate-state', 'gate-instructions']) $(id).oninput = () => { gateAnswer = null; $('gate-answer').innerHTML = blank('State changed. ASSESS HOST for a fresh judgment.'); updateGate(); };
  for (const id of ['execute', 'confirm', 'escalate']) $(id).oninput = () => {
    const values = ['execute', 'confirm', 'escalate'].map(x => Number($(x).value));
    if (id === 'execute') { $('confirm').value = Math.min(values[1], values[0]); $('escalate').value = Math.min(values[2], Number($('confirm').value)); }
    if (id === 'confirm') { $('execute').value = Math.max(values[0], values[1]); $('escalate').value = Math.min(values[2], values[1]); }
    if (id === 'escalate') { $('confirm').value = Math.max(values[1], values[2]); $('execute').value = Math.max(values[0], Number($('confirm').value)); }
    updateGate();
  };
  runRoute = () => operate(async () => { const result = await decide(model, request($('gate-state').value, { action: { type: 'choice', instructions: $('gate-instructions').value, criteria: presets[Number($('preset').value)].criteria } })); gateAnswer = result.answers.action; $('gate-answer').innerHTML = answerHtml('action', gateAnswer) + note(`LIVE ${ms(result.timing.total_ms)} · action names are selected keys, not generated prose.`); updateGate(); });
  $('run').onclick = runRoute; updateGate();
}
function updateGate() {
  const thresholds = ['execute', 'confirm', 'escalate'].map(id => { const v = Number($(id).value); $(`${id}-value`).textContent = v.toFixed(2); return v; });
  if (!gateAnswer) { $('gate-band').textContent = 'AWAITING FRESH JUDGMENT'; $('gate-policy').textContent = ''; return; }
  const confidence = gateAnswer.confidence, band = confidence >= thresholds[0] ? 'EXECUTE' : confidence >= thresholds[1] ? 'CONFIRM' : confidence >= thresholds[2] ? 'ESCALATE' : 'HOLD';
  $('gate-band').textContent = `${band} / ${gateAnswer.choice.toUpperCase()}`;
  $('gate-policy').innerHTML = `<p class="s1-result-line">Jev-formula confidence ${pct(confidence)} · ${band === 'EXECUTE' ? 'meets execute threshold' : band === 'CONFIRM' ? 'request human confirmation' : band === 'ESCALATE' ? 'send to a stronger model / person' : 'insufficient concentration; take no action'}.</p>${note('Policy simulation only. No host action, backend call, or permission change.')}`;
}
const hosts = [
  { name: 'LILY / REUNION', story: 'Lily found her lost puppy alive and well. She laughed, hugged him, and danced with joy. She gave a hungry bird some food.' },
  { name: 'SARA / RESCUE', story: 'Sara saw a little cat stuck in a tree. She was scared, but she climbed up and saved it. She smiled and gave it some milk.' },
  { name: 'LUCY / RAIN', story: 'Lucy could not find her pretty hat. She frowned and slowly went home. She did not want to share her toys with anyone.' },
  { name: 'MIA / QUIET DAY', story: 'Mia sat quietly at the table and looked at a plain cup. Nothing special happened. She gave her brother a cookie and said hello.' },
];
const dimensions = { happy: { type: 'score', instructions: 'She felt', criteria: happiness }, brave: { type: 'score', instructions: 'She was', criteria: ['very scared', 'scared', 'calm', 'brave', 'very brave'] }, kind: { type: 'score', instructions: 'She was', criteria: ['very mean', 'mean', 'quiet', 'kind', 'very kind'] } };
function forge() {
  forgeAnswers = null; inferenceRuns = 0;
  $('route').innerHTML = `<div class="s1-toolbar">${button('run', 'SCORE ALL HOSTS ONCE')}<span id="forge-count">0 INFERENCE RUNS</span><strong>WEIGHT CHANGES RE-RANK WITHOUT RE-RUNNING INFERENCE</strong></div><div class="s1-forge-grid">${hosts.map(h => panel(escape(h.name), `<p class="s1-story">${escape(h.story)}</p>`)).join('')}</div><div class="s1-weights">${Object.keys(dimensions).map(id => slider(`weight-${id}`, `${id.toUpperCase()} WEIGHT`, 1, 3, .1)).join('')}</div>${panel('COMPOSITE RANK / WEIGHTED MEAN OF CACHED LEVEL SCORES', `<div id="forge-ranking">${blank('Four host stories × three score questions. RUN once, then mix weights freely.')}</div>`)}${note('All dimensions use 0–4 level indices. Composite = Σ(weight × score) / Σweights; zero weights produce a tie at 0. Happiness ordering was measured (Spearman ≈.90, n=20). Bravery and kindness are exploratory, not validated. Scores are toy lexical continuations, not judgments from a trained System One model.')}`;
  Object.keys(dimensions).forEach(id => $(`weight-${id}`).oninput = updateForge);
  runRoute = () => operate(async () => {
    forgeAnswers = [];
    for (const host of hosts) { $('loading-detail').textContent = `Scoring ${host.name}: happy, brave, kind.`; const result = await decide(model, request(host.story, dimensions)); forgeAnswers.push({ host, result }); inferenceRuns++; }
    updateForge();
  }); $('run').onclick = runRoute;
}
function updateForge() {
  const weights = Object.fromEntries(Object.keys(dimensions).map(id => { const value = Number($(`weight-${id}`).value); $(`weight-${id}-value`).textContent = value.toFixed(1); return [id, value]; }));
  $('forge-count').textContent = `${inferenceRuns} HOST REQUESTS / ${inferenceRuns * 3} SCORE JUDGMENTS`;
  if (!forgeAnswers) return;
  const denominator = Object.values(weights).reduce((a, b) => a + b, 0);
  const rows = forgeAnswers.map(row => ({ ...row, composite: denominator ? Object.entries(weights).reduce((s, [id, w]) => s + w * row.result.answers[id].score, 0) / denominator : 0 })).sort((a, b) => b.composite - a.composite);
  $('forge-ranking').innerHTML = `<table class="s1-table forge-table"><thead><tr><th>RANK</th><th>HOST</th><th>HAPPY / 4</th><th>BRAVE / 4</th><th>KIND / 4</th><th>COMPOSITE / 4</th><th>LIVE INFERENCE</th></tr></thead><tbody>${rows.map((row, i) => `<tr><td>${i + 1}</td><td>${escape(row.host.name)}</td>${Object.keys(dimensions).map(id => `<td title="Jev-formula confidence ${pct(row.result.answers[id].confidence)}; Clef pmax ${pct(Math.max(...Object.values(row.result.answers[id].probabilities)))}">${row.result.answers[id].score.toFixed(3)}</td>`).join('')}<td><strong>${row.composite.toFixed(3)}</strong><span class="s1-table-bar" style="width:${25 * row.composite}%"></span></td><td>${ms(row.result.timing.total_ms)}</td></tr>`).join('')}</tbody></table>${note('Cached distributions retained. Weight changes perform arithmetic only. Hover a score for Jev- vs Clef-formula confidence (both computed on the tiny GPT).')}<details><summary>ALL CACHED SCORE DISTRIBUTIONS + CONFIDENCE</summary><div class="s1-forge-details">${rows.map(row => `<section><h3>${escape(row.host.name)}</h3>${Object.entries(row.result.answers).map(([id, a]) => answerHtml(id, a)).join('')}</section>`).join('')}</div></details>`;
}
function calibrationStats(items, method) {
  const bins = Array.from({ length: 10 }, (_, i) => ({ lower: i / 10, n: 0, correct: 0, totalConfidence: 0 }));
  const rows = items.map(item => { const p = item.probabilities[method], peak = Math.max(...p), correct = Number(p.indexOf(peak) === item.label), confidence = (peak - 1 / p.length) / (1 - 1 / p.length); const bin = bins[Math.min(9, Math.floor(peak * 10))]; bin.n++; bin.correct += correct; bin.totalConfidence += peak; return { item, p, peak, correct, confidence }; });
  const n = rows.length, accuracy = rows.reduce((s, r) => s + r.correct, 0) / n, ece = bins.reduce((s, b) => s + Math.abs(b.totalConfidence - b.correct), 0) / n, brier = rows.reduce((s, r) => s + r.p.reduce((t, p, i) => t + (p - Number(i === r.item.label)) ** 2, 0), 0) / n;
  return { rows, bins, n, accuracy, ece, brier };
}
function calibration() {
  $('route').innerHTML = `<div class="s1-toolbar"><label>METHOD <select id="cal-method"><option value="sum">SUM / CONTINUATIONS</option><option value="codes">CODES / ANSWER LETTERS</option></select></label>${button('verify', 'VERIFY LIVE / 5 RANDOM FIXTURES')}<span id="verification">Precomputed by test/s1-fixtures.mjs using this exact model and adapter.</span></div><div id="cal-metrics" class="s1-metrics"></div><div class="s1-grid">${panel('RELIABILITY / 10 BINS OF MAX OPTION PROBABILITY', `<div class="s1-legend"><span><i></i>Observed accuracy</span><span><i class="second"></i>Mean pmax</span></div><div id="reliability" class="s1-reliability"></div><p class="s1-note" id="bin-hover">Hover or focus a bin for exact count, accuracy, confidence.</p><details><summary>BIN DATA / TABLE VIEW</summary><div id="bin-table"></div></details>`)}${panel('RISK–COVERAGE / FILTER BY JEV-FORMULA CHOICE CONFIDENCE', `${slider('threshold', 'ACCEPT AT OR ABOVE', .8)}<div id="coverage-metrics" class="s1-result-line"></div><svg id="risk-chart" viewBox="0 0 600 195" role="img" aria-label="Risk versus accepted coverage"></svg><p class="s1-note" id="risk-hover">Risk = 1 − accepted accuracy. Hover or focus curve.</p><details><summary>CURVE DATA / TABLE VIEW</summary><div id="risk-table"></div></details>`)}</div>${note(`SUBSET n=${fixtures.items.length}: 32 hand-written + 40 validation cloze. Gold cloze labels are observed words; other continuations may be plausible. Illustrative subset of the feasibility probe, not a new held-out evaluation. ECE uses pmax (10 equal-width bins); Brier = sum of squared class errors, not divided by option count. T=1 for both methods. Full probe n=182: SUM 87.4%, ECE .044, Brier .171; CODES 28.0% vs 26.1% chance.`)}<details class="s1-fixture-details"><summary>FIXTURES / GOLD LABELS / PRECOMPUTED PROBABILITIES</summary><div id="fixture-table"></div></details>`;
  $('cal-method').onchange = updateCalibration; $('threshold').oninput = updateCoverage;
  runRoute = () => operate(async () => {
    const pool = [...fixtures.items]; for (let i = pool.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [pool[i], pool[j]] = [pool[j], pool[i]]; }
    const method = $('cal-method').value, selected = pool.slice(0, 5); let maxDiff = 0;
    for (const [i, item] of selected.entries()) {
      $('loading-detail').textContent = `VERIFY ${i + 1}/5: ${item.id} / ${method}`;
      const result = await decide(model, request(item.prefix, { answer: { type: 'choice', instructions: '', criteria: Object.fromEntries(item.options.map((x, i) => [String(i), x])) } }), { method });
      const p = Object.values(result.answers.answer.probabilities);
      p.forEach((v, i) => maxDiff = Math.max(maxDiff, Math.abs(v - item.probabilities[method][i])));
    }
    $('verification').textContent = `VERIFIED LIVE / ${method} / n=5 / max |Δp| ${maxDiff.toExponential(3)} / ${selected.map(x => x.id).join(', ')}`;
  }); $('verify').onclick = runRoute; updateCalibration();
}
function updateCalibration() {
  const method = $('cal-method').value, stats = calibrationStats(fixtures.items, method);
  $('cal-metrics').innerHTML = [['Accuracy', pct(stats.accuracy)], ['ECE / pmax', stats.ece.toFixed(3)], ['Multiclass Brier', stats.brier.toFixed(3)], ['Chance', pct(fixtures.items.reduce((s, x) => s + 1 / x.options.length, 0) / stats.n)]].map(([label, value]) => `<div><span>${label} · n=${stats.n}</span><strong>${value}</strong></div>`).join('');
  const rows = stats.bins.map((bin, i) => ({ ...bin, confidence: bin.n ? bin.totalConfidence / bin.n : 0, accuracy: bin.n ? bin.correct / bin.n : 0, label: `${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}` }));
  $('reliability').innerHTML = `<div class="s1-y-axis"><span>100%</span><span>50%</span><span>0%</span></div><div class="s1-bin-columns">${rows.map(row => `<div class="s1-bin" data-bin="${row.lower}"><div class="s1-bin-marks"><i style="height:${100 * row.accuracy}%"></i><i class="second" style="height:${100 * row.confidence}%"></i></div><span>${row.lower.toFixed(1)}</span><b>n=${row.n}</b></div>`).join('')}</div>`;
  document.querySelectorAll('.s1-bin').forEach((node, i) => { const r = rows[i]; tooltip(node, `${r.label}: n=${r.n}; accuracy ${r.n ? pct(r.accuracy) : '—'}; mean pmax ${r.n ? pct(r.confidence) : '—'}`, $('bin-hover')); });
  $('bin-table').innerHTML = table(['PMAX BIN', 'COUNT', 'ACCURACY', 'MEAN PMAX'], rows.map(r => [r.label, r.n, r.n ? pct(r.accuracy) : '—', r.n ? pct(r.confidence) : '—']));
  $('fixture-table').innerHTML = table(['ID / CATEGORY', 'GOLD', 'TOP CHOICE', 'PMAX', 'JEV-FORMULA CONF.'], stats.rows.map(r => [`${r.item.id} / ${r.item.category}`, r.item.options[r.item.label], r.item.options[r.p.indexOf(r.peak)], pct(r.peak), pct(r.confidence)]));
  updateCoverage();
}
function table(headers, rows) { return `<table class="s1-table"><thead><tr>${headers.map(x => `<th>${escape(x)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(x => `<td>${escape(x)}</td>`).join('')}</tr>`).join('')}</tbody></table>`; }
function coverage(rows, threshold) { const accepted = rows.filter(r => r.confidence >= threshold); return { n: accepted.length, coverage: accepted.length / rows.length, accuracy: accepted.length ? accepted.reduce((s, r) => s + r.correct, 0) / accepted.length : null }; }
function updateCoverage() {
  const stats = calibrationStats(fixtures.items, $('cal-method').value), threshold = Number($('threshold').value), current = coverage(stats.rows, threshold);
  $('threshold-value').textContent = threshold.toFixed(2);
  $('coverage-metrics').textContent = `Accepted accuracy ${current.accuracy === null ? '—' : pct(current.accuracy)} (n=${current.n}) · coverage ${pct(current.coverage)} (${current.n}/${stats.n})`;
  const curve = Array.from({ length: 21 }, (_, i) => ({ threshold: i / 20, ...coverage(stats.rows, i / 20) })).filter(x => x.n);
  const x = c => 45 + c * 520, y = risk => 160 - risk * 130;
  const svg = $('risk-chart');
  svg.innerHTML = `<title>Risk–coverage curve, n=${stats.n}; risk is one minus accepted accuracy</title>${[0, .5, 1].map(v => `<line class="s1-gridline" x1="45" x2="565" y1="${y(v)}" y2="${y(v)}"/><text x="7" y="${y(v) + 4}">${pct(v)}</text>`).join('')}<text x="45" y="185">0% coverage</text><text x="478" y="185">100% coverage</text><polyline class="s1-risk-line" points="${curve.map(c => `${x(c.coverage)},${y(1 - c.accuracy)}`).join(' ')}"/><line id="risk-crosshair" class="s1-crosshair" x1="0" x2="0" y1="25" y2="160" visibility="hidden"/>${current.n ? `<circle class="s1-risk-dot" cx="${x(current.coverage)}" cy="${y(1 - current.accuracy)}" r="5"/>` : ''}${curve.map((c, i) => `<rect data-point="${i}" class="s1-hit" x="${x(c.coverage) - 12}" y="20" width="24" height="140" tabindex="0"/>`).join('')}`;
  svg.querySelectorAll('[data-point]').forEach(node => { const c = curve[Number(node.dataset.point)], text = `Threshold ${c.threshold.toFixed(2)}: coverage ${pct(c.coverage)} (${c.n}/${stats.n}); accepted accuracy ${pct(c.accuracy)}; risk ${pct(1 - c.accuracy)}.`; tooltip(node, text, $('risk-hover')); const show = () => { const line = $('risk-crosshair'); line.setAttribute('visibility', 'visible'); line.setAttribute('x1', x(c.coverage)); line.setAttribute('x2', x(c.coverage)); }; node.addEventListener('pointerenter', show); node.addEventListener('focus', show); });
  $('risk-table').innerHTML = table(['THRESHOLD', 'ACCEPTED N', 'COVERAGE', 'ACCURACY', 'RISK'], curve.map(c => [c.threshold.toFixed(2), c.n, pct(c.coverage), pct(c.accuracy), pct(1 - c.accuracy)]));
}
const currentRoute = () => location.hash.slice(2) || 'playground';
function renderRoute() {
  if (busy || !model) return;
  route = ['playground', 'fanout', 'gate', 'forge', 'calibration'].includes(currentRoute()) ? currentRoute() : 'playground';
  $('app-message').textContent = '';
  document.querySelectorAll('.s1-routes a').forEach(a => { const active = a.hash === `#/${route}`; a.classList.toggle('active', active); if (active) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  ({ playground, fanout, gate, forge, calibration })[route]();
  document.body.dataset.route = route;
  if (new URLSearchParams(location.search).has('autorun')) runRoute();
  else document.body.dataset.ready = 'true';
}
// Small, deterministic metric check; no chart numbers may silently drift.
let duplicateRejected = false;
try { parseCriteria('{"same":"a","same":"b"}'); } catch { duplicateRejected = true; }
if (!duplicateRejected || parseCriteria('{"unique":"a","other":"b"}').other !== 'b') throw new Error('Criteria-key self-check failed.');
const check = calibrationStats([{ label: 0, probabilities: { sum: [.5, .5] } }], 'sum');
if (check.accuracy !== 1 || check.ece !== .5 || check.brier !== .5 || coverage(check.rows, .1).n !== 0) throw new Error('Calibration metric self-check failed.');
$('texture').onclick = () => { const active = document.body.classList.toggle('texture'); $('texture').setAttribute('aria-pressed', String(active)); };
window.addEventListener('hashchange', renderRoute);
try {
  if (location.protocol === 'file:') throw new Error('HTTP server required. Serve with python3 -m http.server 8766.');
  [model, fixtures] = await Promise.all([loadModel('./model/'), fetch('./s1-fixtures.json').then(response => { if (!response.ok) throw new Error('Calibration fixture load failed.'); return response.json(); })]);
  $('system-status').textContent = 'HOST ONLINE / REAL GPT'; $('loading-overlay').hidden = true;
  if (!location.hash) history.replaceState(null, '', `${location.pathname}${location.search}#/playground`);
  renderRoute();
} catch (error) {
  $('loading-overlay').classList.add('error'); $('loading-title').textContent = 'HOST LOAD FAILED'; $('loading-detail').textContent = error.message; $('loading-progress').hidden = true; $('system-status').textContent = 'HOST OFFLINE'; console.error(error);
}

// System One (OneJev) prompt, letter softmax, confidence and answer shapes. Pure: no DOM, no model.
// Ported from OmniJev/OneJev qev/prompt.py and qev/answers.py (commit cb06fe0); confidence = TypeSafe formulas.
// ponytail: options limited to single letters A-Z (one token each); score capped at 10 levels like upstream schema.
export const SYSTEM_PROMPT = 'Apply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase letter, with no explanation or reasoning.';
export const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
export const MAX_OPTIONS = 26;
export const MAX_LEVELS = 10;
const NOUL_INSTRUCTIONS = 'Is the statement true, or is the answer to the question yes?';
const NOUL_TRUE = 'the statement is true / the answer is yes';
const NOUL_FALSE = 'the statement is false / the answer is no';
const CHOICE_INSTRUCTIONS = 'Which option applies to the state?';
const SCORE_INSTRUCTIONS = 'Which level describes the state?';

// render_entry: null -> '', string -> trimmed, anything else -> pretty JSON.
const entry = x => (x == null ? '' : typeof x === 'string' ? x.trim() : JSON.stringify(x, null, 2));

export function stateText(state) {
  return state == null ? '' : typeof state === 'string' ? state : JSON.stringify(state, null, 2);
}

// Console input: JSON objects/arrays are structured state, anything else is raw text.
export function parseState(text) {
  if (!text.trim()) throw new Error('State is empty.');
  let state = text;
  try { const v = JSON.parse(text); if (v && typeof v === 'object') state = v; } catch { /* raw text */ }
  checkFinite(state);
  return state;
}

function checkFinite(v) {
  if (typeof v === 'number' && !Number.isFinite(v)) throw new Error('State contains a non-finite number.');
  if (v && typeof v === 'object') for (const x of Object.values(v)) checkFinite(x);
}

// UI rows -> System One question (API shape). Empty description = null = default text, as upstream.
export function toApi(q) {
  const rows = q.rows ?? [];
  const instructions = q.instructions?.trim() || null;
  if (q.type === 'noul') {
    return { type: 'noul', instructions, criteria: { true: rows[0]?.desc?.trim() || null, false: rows[1]?.desc?.trim() || null } };
  }
  if (q.type === 'score') {
    return { type: 'score', instructions, criteria: rows.map(r => r.desc?.trim() || null) };
  }
  if (q.type !== 'choice') throw new Error(`Question "${q.id}": unknown type "${q.type}".`);
  const seen = new Set(), entries = [];
  for (const r of rows) {
    const name = r.name ?? '';
    if (!name.trim()) throw new Error(`Question "${q.id}": option names cannot be empty.`);
    // JS objects order integer-like keys numerically, which would reorder the option letters.
    if (/^\d+$/.test(name.trim())) throw new Error(`Question "${q.id}": option names cannot be plain numbers ("${name}"); add a word, e.g. "option ${name}".`);
    if (seen.has(name)) throw new Error(`Question "${q.id}": duplicate option "${name}".`);
    seen.add(name);
    entries.push([name, r.desc?.trim() || null]);
  }
  return { type: 'choice', instructions, criteria: Object.fromEntries(entries) };
}

// Validate an API-shape question and return its rendering plan.
export function prepare(api) {
  const instructions = entry(api.instructions);
  if (api.type === 'noul') {
    const c = api.criteria ?? {};
    return {
      type: 'noul', instructions: instructions || NOUL_INSTRUCTIONS,
      labels: ['yes', 'no'], names: ['yes', 'no'],
      descs: [c.true == null ? NOUL_TRUE : c.true, c.false == null ? NOUL_FALSE : c.false], levels: null,
    };
  }
  if (api.type === 'choice') {
    const keys = Object.keys(api.criteria ?? {});
    if (!keys.length) throw new Error('Choice needs at least one option.');
    if (keys.length > MAX_OPTIONS) throw new Error(`Choice has ${keys.length} options; maximum is ${MAX_OPTIONS}.`);
    return { type: 'choice', instructions: instructions || CHOICE_INSTRUCTIONS, labels: keys, names: keys, descs: keys.map(k => api.criteria[k]), levels: null };
  }
  if (api.type === 'score') {
    const levels = api.criteria;
    if (!Array.isArray(levels) || !levels.length) throw new Error('Score needs at least one level.');
    if (levels.length > MAX_LEVELS) throw new Error(`Score has ${levels.length} levels; maximum is ${MAX_LEVELS}.`);
    return {
      type: 'score', instructions: instructions || SCORE_INSTRUCTIONS,
      labels: levels.map((_, i) => String(i)), names: levels.map((_, i) => `level ${i}`), descs: levels, levels,
    };
  }
  throw new Error(`Unknown question type "${api.type}".`);
}

// prompt.py render_question + build_messages, byte for byte.
export function buildMessages(state, api) {
  const p = prepare(api);
  const header = p.type === 'score' ? 'Rate the state:' : 'Question:';
  const lines = p.labels.map((_, i) => {
    const text = entry(p.descs[i]);
    return text ? `${LETTERS[i]}. ${p.names[i]}: ${text}` : `${LETTERS[i]}. ${p.names[i]}`;
  }).join('\n');
  const letters = p.labels.map((_, i) => LETTERS[i]).join(', ');
  const suffix = `${header} ${p.instructions}\n\nOptions:\n${lines}\n\nAnswer with one letter: ${letters}.`;
  return [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: `<state>\n${stateText(state)}\n</state>\n\n${suffix}` },
  ];
}

// Softmax over the option-letter logits only. `logits` is indexable by token id.
export function letterProbs(logits, letterIds) {
  const xs = letterIds.map(id => logits[id]);
  if (xs.some(x => !Number.isFinite(x))) throw new Error('Model returned invalid logits.');
  const max = Math.max(...xs);
  const e = xs.map(x => Math.exp(x - max));
  const z = e.reduce((a, b) => a + b, 0);
  return { logits: xs, max, logZ: max + Math.log(z), probs: e.map(v => v / z) };
}

const argmax = p => p.reduce((best, v, i) => (v > p[best] ? i : best), 0);

// Confidence with the intermediate numbers, for the inspector.
export function confidenceDetail(type, p) {
  const n = p.length;
  if (type === 'noul') return { type, value: p[0], pYes: p[0] };
  if (n === 1) return { type, value: 1, n };
  if (type === 'choice') {
    const pmax = Math.max(...p), uniform = 1 / n;
    return { type, n, pmax, uniform, value: (pmax - uniform) / (1 - uniform) };
  }
  const mode = argmax(p), center = (n - 1) / 2;
  const distance = p.reduce((s, v, i) => s + v * Math.abs(i - mode), 0);
  const uniformMad = p.reduce((s, _, i) => s + Math.abs(i - center), 0) / n;
  return { type, n, mode, distance, uniformMad, value: Math.max(0, 1 - distance / uniformMad) };
}

const r6 = x => Math.round(x * 1e6) / 1e6;

// Probabilities over labels -> System One answer (API shape, 6 dp like answers.py).
export function answer(api, probs) {
  const p = prepare(api);
  const confidence = confidenceDetail(p.type, probs).value;
  if (p.type === 'noul') return { type: 'noul', noul: r6(Math.min(1, Math.max(0, probs[0]))) };
  const probabilities = Object.fromEntries(p.labels.map((l, i) => [l, r6(probs[i])]));
  if (p.type === 'choice') {
    return { type: 'choice', choice: p.labels[argmax(probs)], probabilities, confidence: r6(confidence) };
  }
  const score = probs.reduce((s, v, i) => s + i * v, 0);
  return {
    type: 'score', score: r6(score),
    legend: Object.fromEntries(p.levels.map((lv, i) => [String(i), lv ?? null])),
    probabilities, confidence: r6(confidence),
  };
}

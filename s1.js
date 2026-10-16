// Local continuation adapter, not a trained System One head. No text generation.
export const stateText = state => Object.keys(state).length === 1 && typeof state.story === 'string' ? state.story : JSON.stringify(state);
export function scoreConfidence(p) {
  const n = p.length, m = p.indexOf(Math.max(...p));
  const D = p.reduce((s, _, i) => s + Math.abs(i - (n - 1) / 2), 0) / n;
  return Math.max(0, 1 - p.reduce((s, v, i) => s + v * Math.abs(i - m), 0) / D);
}
function softmax(scores, temperature) {
  if (!scores.length || scores.some(x => !Number.isFinite(x))) throw new Error('Invalid model scores.');
  const max = Math.max(...scores), w = scores.map(x => Math.exp((x - max) / temperature));
  const sum = w.reduce((a, b) => a + b, 0);
  return w.map(x => x / sum);
}
function logProbability(logits, id) {
  if (!logits.length || !Number.isInteger(id) || id < 0 || id >= logits.length) throw new Error('Invalid token or logits.');
  let max = -Infinity;
  for (const z of logits) { if (!Number.isFinite(z)) throw new Error('Invalid model logits.'); max = Math.max(max, z); }
  let total = 0;
  for (const z of logits) total += Math.exp(z - max);
  return logits[id] - max - Math.log(total);
}
const object = x => x !== null && typeof x === 'object' && !Array.isArray(x);
const yieldPaint = () => new Promise(resolve => setTimeout(resolve, 0));
export async function decide(model, request, { temperature = 1, method = 'sum' } = {}) {
  if (!Number.isFinite(temperature) || temperature <= 0) throw new TypeError('Temperature must be finite and positive.');
  if (!['sum', 'codes'].includes(method)) throw new TypeError('Method must be sum or codes.');
  if (!object(request) || typeof request.model !== 'string' || !request.model.trim() || !object(request.state) || !object(request.questions) || !Object.keys(request.questions).length) throw new TypeError('Expected model, state object, and nonempty questions map.');
  const state = stateText(request.state), stateIds = model.encode(state), limit = model.config?.sequence_len ?? 256;
  const fits = ids => { if (ids.length >= limit) throw new RangeError(`Request exceeds ${limit - 1} tokens; shorten state, instructions, or criteria. No silent context crop.`); };
  fits(stateIds);
  const jobs = Object.entries(request.questions).map(([id, q]) => {
    if (!object(q) || !['choice', 'score', 'noul'].includes(q.type) || typeof q.instructions !== 'string') throw new TypeError(`${id}: expected typed question with string instructions.`);
    if (q.type === 'score' ? !Array.isArray(q.criteria) : !object(q.criteria)) throw new TypeError(`${id}: invalid criteria shape.`);
    if (q.type === 'noul' && (Object.keys(q.criteria).length !== 2 || !Object.hasOwn(q.criteria, 'true') || !Object.hasOwn(q.criteria, 'false'))) throw new TypeError(`${id}: noul requires true and false criteria.`);
    const entries = q.type === 'score' ? q.criteria.map((x, i) => [String(i), x]) : q.type === 'noul' ? [['true', q.criteria.true], ['false', q.criteria.false]] : Object.entries(q.criteria);
    if (entries.length < 2 || entries.length > 255 || entries.some(([k, v]) => !k || typeof v !== 'string' || !v.trim())) throw new TypeError(`${id}: use 2–255 nonempty literal continuations.`);
    const prefix = state + (q.instructions ? ' ' + q.instructions : ''), ids = model.encode(prefix), options = entries.map(([, text]) => model.encode(' ' + text));
    options.forEach(tokens => fits(ids.concat(tokens)));
    let codes, promptIds;
    if (method === 'codes') {
      if (entries.length > 26) throw new RangeError('Answer-letter comparison supports at most 26 options.');
      codes = entries.map((_, i) => model.encode(' ' + String.fromCharCode(65 + i)));
      if (codes.some(tokens => tokens.length !== 1)) throw new Error('Answer codes must encode as single tokens.');
      promptIds = model.encode(prefix + '\n' + entries.map(([, text], i) => `${String.fromCharCode(65 + i)}) ${text}`).join(' ') + '\nThe answer is');
      fits(promptIds);
    }
    return { id, q, entries, ids, options, codes, promptIds };
  });
  const start = performance.now(), prefill = performance.now();
  await model.logits(stateIds);
  const timing = { state_prefill_ms: performance.now() - prefill, questions: {}, total_ms: 0 };
  let inputTokens = stateIds.length;
  const answers = [];
  for (const job of jobs) {
    await yieldPaint();
    const t = performance.now();
    const { id, q, entries, ids, options, codes, promptIds } = job;
    let scores;
    if (method === 'codes') {
      const z = await model.logits(promptIds);
      scores = codes.map(tokens => logProbability(z, tokens[0]));
      inputTokens += promptIds.length - stateIds.length;
    } else {
      const first = await model.logits(ids);
      scores = [];
      for (const tokens of options) {
        await yieldPaint();
        let score = logProbability(first, tokens[0]);
        for (let j = 1; j < tokens.length; j++) {
          await yieldPaint();
          score += logProbability(await model.logits(ids.concat(tokens.slice(0, j))), tokens[j]);
        }
        scores.push(score);
      }
      inputTokens += ids.length - stateIds.length + options.reduce((s, tokens) => s + tokens.length, 0);
    }
    const p = softmax(scores, temperature), peak = Math.max(...p), best = p.indexOf(peak);
    const probabilities = Object.fromEntries(entries.map(([key], i) => [key, p[i]]));
    const answer = q.type === 'noul' ? { type: 'noul', noul: p[0] } : q.type === 'choice'
      ? { type: 'choice', choice: entries[best][0], probabilities, confidence: (peak - 1 / p.length) / (1 - 1 / p.length) }
      : { type: 'score', score: p.reduce((s, v, i) => s + v * i, 0), legend: Object.fromEntries(entries), probabilities, confidence: scoreConfidence(p) };
    answers.push([id, answer]);
    Object.defineProperty(timing.questions, id, { value: performance.now() - t, enumerable: true });
  }
  timing.total_ms = performance.now() - start;
  // Input usage is a logical adapter token budget, not engine KV work or a remote invoice.
  const response = { model: request.model, answers: Object.fromEntries(answers), usage: { input_tokens: inputTokens, output_tokens: 0 } };
  // Local diagnostic extension; serialized JSON remains the exact backend contract.
  return Object.defineProperty(response, 'timing', { value: timing, enumerable: false });
}

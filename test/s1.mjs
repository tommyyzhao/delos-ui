// No model, no network. Provenance: golden prompt strings are the output of OmniJev/OneJev
// qev/prompt.py at commit cb06fe0 (run unchanged on the reference cases below); answer/confidence
// checks use test/onejev-reference.json (conversion/reference.json at HF revision afc03812).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildMessages, letterProbs, answer, confidenceDetail, prepare, parseState, toApi, LETTERS } from '../s1.js';

const ref = JSON.parse(readFileSync(new URL('./onejev-reference.json', import.meta.url), 'utf8'));
const close = (a, b, eps, msg) => assert.ok(Math.abs(a - b) <= eps, `${msg ?? ''} ${a} != ${b} (eps ${eps})`);

const SYSTEM = 'Apply the question to the state. Choose exactly one of the listed options. Respond with only its uppercase letter, with no explanation or reasoning.';
const STATE_SUPPORT = '<state>\n{\n  "channel": "email",\n  "body": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."\n}\n</state>\n\n';

// 1. Prompt parity with qev/prompt.py (golden strings).
const golden = {
  department: STATE_SUPPORT + 'Question: Which team should handle this?\n\nOptions:\nA. billing: charges, invoices, refunds\nB. technical: bugs, outages\nC. sales: pricing, new plans\n\nAnswer with one letter: A, B, C.',
  urgency: STATE_SUPPORT + 'Rate the state: How urgent is this?\n\nOptions:\nA. level 0: not urgent\nB. level 1: soon\nC. level 2: today\nD. level 3: right now\n\nAnswer with one letter: A, B, C, D.',
  churn: STATE_SUPPORT + 'Question: The customer threatens to cancel.\n\nOptions:\nA. yes: the statement is true / the answer is yes\nB. no: the statement is false / the answer is no\n\nAnswer with one letter: A, B.',
};
const supportCase = ref.find(c => c.name === 'text-support');
for (const [qid, q] of Object.entries(supportCase.questions)) {
  const msgs = buildMessages(supportCase.state, q);
  assert.equal(msgs[0].role, 'system'); assert.equal(msgs[0].content, SYSTEM);
  assert.equal(msgs[1].role, 'user');
  assert.equal(msgs[1].content, golden[qid], `prompt ${qid}`);
}

// Text after the closing state tag (the question suffix).
const suffixOf = msgs => msgs[1].content.split('</state>\n\n')[1];

// Edge cases from the same upstream run: string state, null descriptions, noul defaults.
const edge = buildMessages('plain text\nline2', { type: 'choice', instructions: null, criteria: { alpha: null, beta: 'second one' } });
assert.equal(edge[1].content, '<state>\nplain text\nline2\n</state>\n\nQuestion: Which option applies to the state?\n\nOptions:\nA. alpha\nB. beta: second one\n\nAnswer with one letter: A, B.');
assert.equal(suffixOf(buildMessages('x', { type: 'score', criteria: ['low', 'high'] })),
  'Rate the state: Which level describes the state?\n\nOptions:\nA. level 0: low\nB. level 1: high\n\nAnswer with one letter: A, B.');
assert.equal(suffixOf(buildMessages('x', { type: 'noul', criteria: { true: 'it is', false: null } })),
  'Question: Is the statement true, or is the answer to the question yes?\n\nOptions:\nA. yes: it is\nB. no: the statement is false / the answer is no\n\nAnswer with one letter: A, B.');
// Model-visible text: no ids, no gold labels, no internal names leak.
assert.ok(!buildMessages(supportCase.state, supportCase.questions.churn)[1].content.includes('churn'));

// Letter ids: A..Z map one-to-one; 26 options is the cap.
assert.equal(LETTERS.length, 26);
assert.throws(() => prepare({ type: 'choice', criteria: Object.fromEntries([...Array(27).keys()].map(i => ['o' + i, null])) }), /maximum is 26/);
assert.throws(() => prepare({ type: 'score', criteria: Array(11).fill('x') }), /maximum is 10/);

// 2. Letter softmax and answer math against reference hf probabilities.
const ids = [...LETTERS].map((_, i) => 100 + i);
const logits = new Float32Array(200);
logits[ids[0]] = 14.078; logits[ids[1]] = 8.578; logits[ids[2]] = 10.727;
const lp = letterProbs(logits, ids.slice(0, 3));
const sum = lp.probs.reduce((a, b) => a + b, 0);
close(sum, 1, 1e-12, 'softmax sums to 1');
close(lp.probs[0], 1 / (1 + Math.exp(logits[101] - logits[100]) + Math.exp(logits[102] - logits[100])), 1e-12);
// Shift invariance: adding a constant to every letter logit leaves probabilities unchanged.
const shifted = letterProbs(Float64Array.from(logits, x => x + 500), ids.slice(0, 3));
lp.probs.forEach((p, i) => close(shifted.probs[i], p, 1e-12, 'shift'));

// Reference answers: prompt answer shape + TypeSafe formulas, applied to hf probabilities.
for (const [qid, q] of Object.entries(supportCase.questions)) {
  const want = supportCase.answers[qid];
  const probs = supportCase.hf[qid];
  const got = answer(toApi(withRows(q)), probs);
  assert.equal(got.type, want.type, qid);
  if (want.type === 'choice') {
    assert.equal(got.choice, want.choice);
    for (const [k, v] of Object.entries(want.probabilities)) close(got.probabilities[k], v, 2e-6, `${qid}.${k}`); // reference answers rounded independently: up to 1 ulp at 6 dp
    close(got.confidence, want.confidence, 2e-5, `${qid} confidence`);
  } else if (want.type === 'score') {
    close(got.score, want.score, 2e-5, `${qid} score`);
    for (const [k, v] of Object.entries(want.probabilities)) close(got.probabilities[k], v, 2e-6, `${qid}.${k}`); // reference answers rounded independently: up to 1 ulp at 6 dp
    close(got.confidence, want.confidence, 2e-5, `${qid} confidence`);
    assert.deepEqual(got.legend, want.legend);
  } else {
    close(got.noul, want.noul, 1e-6, `${qid} noul`);
  }
}

// Other reference cases: score and choice formulas on their hf probabilities. Media cases are skipped:
// their `answers` come from qev's cached-image-prefix path, which the model card says differs from hf
// by up to ~0.07 on image prompts (e.g. image-review sentiment 0.4999 vs hf 0.5622). Not comparable here.
for (const c of ref.filter(c => c.name !== 'text-support' && c.media.length === 0)) {
  for (const [qid, q] of Object.entries(c.questions)) {
    const want = c.answers[qid];
    const got = answer(toApi(withRows(q)), c.hf[qid]);
    if (want.type === 'choice') close(got.confidence, want.confidence, 2e-5, `${c.name}.${qid} confidence`);
    if (want.type === 'score') {
      close(got.score, want.score, 2e-5, `${c.name}.${qid} score`);
      close(got.confidence, want.confidence, 2e-5, `${c.name}.${qid} confidence`);
    }
    if (want.type === 'noul') close(got.noul, want.noul, 1e-6, `${c.name}.${qid} noul`);
  }
}

// Confidence edge cases: certainty, uniform, single option, zero-mass guards.
assert.equal(confidenceDetail('choice', [1, 0, 0]).value, 1);
close(confidenceDetail('choice', [1 / 3, 1 / 3, 1 / 3]).value, 0, 1e-12);
assert.equal(confidenceDetail('choice', [1]).value, 1);
assert.equal(confidenceDetail('score', [0, 0, 1, 0, 0]).value, 1);
// Uniform 5 levels: mode = first index, distance = 2, uniform MAD = 1.2, clamped at 0.
assert.equal(confidenceDetail('score', [0.2, 0.2, 0.2, 0.2, 0.2]).value, 0);

// Input validation: clear messages at the boundary.
assert.throws(() => parseState('   '), /State is empty/);
assert.throws(() => parseState('{"x": 1e999}'), /non-finite|Unexpected/);
assert.equal(parseState('{"a":1}').a, 1);
assert.equal(parseState('hello'), 'hello');
assert.throws(() => toApi({ id: 'q', type: 'choice', rows: [{ name: ' ', desc: '' }] }), /cannot be empty/);
assert.throws(() => toApi({ id: 'q', type: 'choice', rows: [{ name: 'a' }, { name: 'a' }] }), /duplicate option/);
assert.throws(() => prepare({ type: 'choice', criteria: {} }), /at least one option/);
assert.throws(() => prepare({ type: 'score', criteria: [] }), /at least one level/);
assert.throws(() => letterProbs(new Float32Array([NaN]), [0]), /invalid logits/);

console.log('s1: ok');

// Helper: rebuild the API question from reference JSON (criteria already in API shape).
function withRows(q) {
  if (q.type === 'choice') return { id: 'q', type: 'choice', instructions: q.instructions, rows: Object.entries(q.criteria).map(([name, desc]) => ({ name, desc })) };
  if (q.type === 'score') return { id: 'q', type: 'score', instructions: q.instructions, rows: q.criteria.map(desc => ({ desc })) };
  return { id: 'q', type: 'noul', instructions: q.instructions, rows: [] };
}

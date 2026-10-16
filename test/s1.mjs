import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { loadModel } from '../engine.js';
import { decide, scoreConfidence } from '../s1.js';

const fixtures = JSON.parse(await readFile(new URL('../s1-fixtures.json', import.meta.url), 'utf8'));
const model = await loadModel(new URL('../model/', import.meta.url));
const make = item => ({ model: 'tinystories-15m-local', state: { story: item.prefix }, questions: { answer: { type: 'choice', instructions: '', criteria: Object.fromEntries(item.options.map(text => [text, text])) } } });
const keys = (x, expected) => assert.deepEqual(Object.keys(x).sort(), expected.sort());
for (const id of ['hand-0', 'hand-8', 'hand-16']) {
  const item = fixtures.items.find(x => x.id === id), result = await decide(model, make(item)), answer = result.answers.answer;
  keys(result, ['model', 'answers', 'usage']);
  assert.ok(result.timing && !Object.getOwnPropertyDescriptor(result, 'timing').enumerable);
  keys(JSON.parse(JSON.stringify(result)), ['model', 'answers', 'usage']);
  keys(answer, ['type', 'choice', 'probabilities', 'confidence']);
  keys(result.usage, ['input_tokens', 'output_tokens']);
  const p = Object.values(answer.probabilities), peak = Math.max(...p);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-6);
  assert.ok(p.every(x => x >= 0 && x <= 1));
  assert.equal(answer.confidence, (peak - 1 / p.length) / (1 - 1 / p.length));
  assert.equal(answer.choice, item.options[item.label], `${id}: gold top choice`);
  assert.equal(result.usage.output_tokens, 0);
  assert.ok(result.timing.state_prefill_ms >= 0 && result.timing.questions.answer >= 0);
}
const request = { model: 'tinystories-15m-local', state: { story: 'Lily got a lovely present from her mother. She smiled and danced around the room.' }, questions: {
  happiness: { type: 'score', instructions: 'She felt', criteria: ['very sad', 'sad', 'calm', 'happy', 'very happy'] },
  smiles: { type: 'noul', instructions: 'Question: Did Lily smile? Answer:', criteria: { true: 'yes', false: 'no' } },
} };
const result = await decide(model, request);
keys(result.answers.happiness, ['type', 'score', 'legend', 'probabilities', 'confidence']);
keys(result.answers.smiles, ['type', 'noul']);
assert.ok(result.answers.happiness.score >= 0 && result.answers.happiness.score <= 4);
assert.ok(result.answers.smiles.noul >= 0 && result.answers.smiles.noul <= 1);
assert.ok(Math.abs(Object.values(result.answers.happiness.probabilities).reduce((a, b) => a + b, 0) - 1) < 1e-6);
assert.equal(scoreConfidence([1, 0, 0, 0, 0]), 1);
assert.equal(scoreConfidence([.2, .2, .2, .2, .2]), Math.max(0, 1 - 2 / 1.2));
const p = Object.values(result.answers.happiness.probabilities), m = p.indexOf(Math.max(...p)), D = p.reduce((s, _, i) => s + Math.abs(i - 2), 0) / p.length;
assert.equal(result.answers.happiness.confidence, Math.max(0, 1 - p.reduce((s, v, i) => s + v * Math.abs(i - m), 0) / D));
const codes = await decide(model, make(fixtures.items.find(x => x.id === 'hand-0')), { method: 'codes' });
assert.ok(Math.abs(Object.values(codes.answers.answer.probabilities).reduce((a, b) => a + b, 0) - 1) < 1e-6);
await assert.rejects(decide(model, request, { temperature: 0 }), TypeError);
await assert.rejects(decide(model, { ...request, state: { story: 'Lily '.repeat(300) } }), RangeError);
await assert.rejects(decide(model, { ...request, questions: { x: { type: 'noul', instructions: '', criteria: { true: 'yes' } } } }), TypeError);
const stub = { encode: text => Array.from(text, c => c.charCodeAt(0) % 3), logits: async () => new Float32Array([1000, 999, -1000]), config: { sequence_len: 256 } };
const edge = await decide(stub, { model: 'test', state: { story: '' }, questions: { '__proto__': undefined, x: { type: 'choice', instructions: '', criteria: JSON.parse('{"__proto__":"a","other":"b"}') } } });
assert.ok(Object.hasOwn(edge.answers.x.probabilities, '__proto__'));
console.log('PASS s1: 3 probe gold choices (happy, Lily, ball); API shape; normalized probabilities; Jev Choice/Score formulas; bounded Score/Noul; codes; validation; stable softmax.');

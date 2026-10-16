import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { loadModel } from '../engine.js';
import { decide } from '../s1.js';

const target = new URL('../s1-fixtures.json', import.meta.url);
// First run: node test/s1-fixtures.mjs /tmp/s1probe/results.json. Later runs reuse saved items.
const source = JSON.parse(await readFile(process.argv[2] ?? target, 'utf8'));
const items = source.choice ? [...source.choice.items.filter(x => x.category !== 'validation_cloze'), ...source.choice.items.filter(x => x.category === 'validation_cloze').slice(0, 40)].map(({ id, category, prefix, options, label }) => ({ id, category, prefix, options, label })) : source.items;
const model = await loadModel(new URL('../model/', import.meta.url));
for (const [i, item] of items.entries()) {
  item.probabilities = {};
  const request = { model: 'tinystories-15m-local', state: { story: item.prefix }, questions: { answer: { type: 'choice', instructions: '', criteria: Object.fromEntries(item.options.map((text, i) => [String(i), text])) } } };
  for (const method of ['sum', 'codes']) {
    const result = await decide(model, request, { method });
    item.probabilities[method] = Object.values(result.answers.answer.probabilities);
    assert.ok(Math.abs(item.probabilities[method].reduce((a, b) => a + b, 0) - 1) < 1e-6);
    if (source.choice) {
      const scores = source.choice.items.find(x => x.id === item.id).scores[method], max = Math.max(...scores), w = scores.map(x => Math.exp(x - max)), total = w.reduce((a, b) => a + b, 0);
      assert.ok(item.probabilities[method].every((p, j) => Math.abs(p - w[j] / total) < 1e-6), `${item.id} ${method}: probe parity`);
    }
  }
  if ((i + 1) % 12 === 0) console.log(`Scored ${i + 1}/${items.length} fixtures; sum + codes.`);
}
await writeFile(target, JSON.stringify({ model: 'tinystories-15m-local', temperature: 1, provenance: '32 hand-written + first 40 validation cloze items from feasibility probe seed 20261011. Cloze gold = observed next word; alternatives may be plausible. Small illustrative subset, not an independent test set.', generated_by: 'test/s1-fixtures.mjs imports s1.js; actual local model inference, not copied probe probabilities.', items }, null, 2) + '\n');
console.log(`PASS ${items.length} fixtures; normalized sum + codes; probe parity checked when source provided.`);

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import { loadModel } from '../engine.js';

const base = new URL('../model/', import.meta.url);
const reference = JSON.parse(await readFile(new URL('reference.json', base), 'utf8'));
const model = await loadModel(base);
assert.ok(Number.isFinite(model.config.val_loss), 'recorded validation loss');
assert.deepEqual(model.encode(reference.prompt), reference.ids, 'Python/JS prompt encoding');
for (const fixture of reference.tokenizer_cases ?? []) {
  assert.deepEqual(model.encode(fixture.text), fixture.ids, `Python/JS encoding: ${fixture.text}`);
}
for (const s of ['', 'Hello world!', 'Café 中文 Привет 👋🏽 🌈', "I'm HERE! It's 12345.\r\nNext\tline.  ", 'hello   world\n\n', '<|bos|> literal', '﻿﻿hello']) {
  assert.equal(model.decode(model.encode(s)), s, 'UTF-8 round trip');
}
function top10(logits) {
  return Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, 10);
}
const start = performance.now();
const logits = await model.logits(reference.ids);
const milliseconds = performance.now() - start;
assert.equal(logits.length, model.config.vocab_size);
let maxDiff = 0, sumDiff = 0;
for (let i = 0; i < logits.length; i++) {
  assert.ok(Number.isFinite(logits[i]), 'finite logits');
  const diff = Math.abs(logits[i] - reference.logits[i]);
  maxDiff = Math.max(maxDiff, diff); sumDiff += diff;
}
assert.ok(maxDiff < 2e-2, `logits max absolute difference ${maxDiff}`);
assert.deepEqual(top10(logits), reference.top10, 'top-10 token IDs');
assert.deepEqual(await model.logits(reference.ids), logits, 'same-prefix reuse');
const extension = [...reference.ids, model.encode(' Then')[0]];
const warm = await model.logits(extension);
await model.logits(model.encode('Different prompt.'));
const cold = await model.logits(extension);
assert.deepEqual(warm, cold, 'KV reuse matches fresh forward');
const a = reference.ids.slice(0, 8), b = model.encode(' a small bird'), c = model.encode(' a happy rabbit');
await model.logits(a.concat(b));
const branch = await model.logits(a.concat(c));
await model.logits(model.encode('Fresh branch reset.'));
assert.deepEqual(branch, await model.logits(a.concat(c)), 'longest-common-prefix branch matches cold forward');
const shorter = extension.slice(0, 6);
const shrunk = await model.logits(shorter);
await model.logits(model.encode('Reset.'));
assert.deepEqual(shrunk, await model.logits(shorter), 'shorter prefix recomputes correctly');
const long = Array.from({ length: model.config.sequence_len + 13 }, (_, i) => reference.ids[i % reference.ids.length]);
const cropped = await model.logits(long);
assert.deepEqual(cropped, await model.logits(long.slice(-model.config.sequence_len)), 'context crop');
await assert.rejects(model.logits([-1]), RangeError);
console.log(`PASS max_abs_diff=${maxDiff.toExponential(6)} mean_abs_diff=${(sumDiff/logits.length).toExponential(6)} top10=exact`);
console.log(`Prompt forward (${reference.ids.length+1} tokens): ${milliseconds.toFixed(1)} ms`);
await model.logits(model.encode('Reset timing.'));
const bench = Array.from({ length: model.config.sequence_len - 1 }, (_, i) => reference.ids[i % reference.ids.length]);
const fullStart = performance.now();
await model.logits(bench);
console.log(`Cold ${model.config.sequence_len}-token forward: ${(performance.now()-fullStart).toFixed(1)} ms`);
await model.logits(bench.slice(0, -1));
const decodeStart = performance.now();
await model.logits(bench);
console.log(`Cached single-token forward: ${(performance.now()-decodeStart).toFixed(1)} ms`);

// Reproducible temperature-0.8/top-40 samples, no framework or extra dependency.
if (process.argv.includes('--samples')) {
  let seed = 42;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
  for (const prompt of ['Once upon a time, there was a little rabbit', 'Lily went to the park with her mom.']) {
    const ids = model.encode(prompt);
    for (let t = 0; t < 140; t++) {
      const z = await model.logits(ids), candidates = Array.from(z.keys()).sort((a, b) => z[b] - z[a]).slice(0, 40);
      const weights = candidates.map(id => Math.exp((z[id] - z[candidates[0]]) / .8));
      let sample = random() * weights.reduce((a, b) => a + b, 0), next = candidates.at(-1);
      for (let i = 0; i < weights.length; i++) { sample -= weights[i]; if (sample <= 0) { next = candidates[i]; break; } }
      if (next === model.bosId) break;
      ids.push(next);
    }
    console.log('SAMPLE:', model.decode(ids));
  }
}

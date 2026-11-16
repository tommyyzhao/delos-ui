import assert from 'node:assert/strict';
import { distribution, probOf, sample, freeze, halfToFloat32 } from '../logits.js';

const close = (a, b, eps = 1e-9) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

// Matches a naive softmax, including extreme logits.
const raw = new Float32Array([1000, 1001, -1000, 3, 1001]);
const d = distribution(raw, 1, 3);
const naive = [...raw].map(x => Math.exp(x - 1001)), z = naive.reduce((a, b) => a + b);
raw.forEach((_, i) => close(probOf(d, i), naive[i] / z));
assert.deepEqual(d.top.map(t => t.id), [1, 4, 0]);
close(d.top.reduce((s, t) => s + t.prob, 0), 1);
close(d.entropy, -naive.reduce((s, x) => s + (x / z ? (x / z) * Math.log2(x / z) : 0), 0));

// Greedy limit: ties share mass, entropy = log2(ties), sample is deterministic.
const g = distribution(raw, 0);
assert.equal(probOf(g, 1), .5); assert.equal(probOf(g, 0), 0); assert.equal(g.entropy, 1); assert.equal(sample(g, .99), 1);

// Temperature flattens; sampling follows the CDF.
const flat = distribution(new Float32Array([0, Math.log(3)]), 1);
assert.equal(sample(flat, .2), 0); assert.equal(sample(flat, .8), 1);
close(probOf(distribution(new Float32Array([0, 2]), 2), 1), 1 / (1 + Math.exp(-1)));

// Vocabulary-sized input: no spread RangeError, top-k ordered.
const big = new Float32Array(248320).map((_, i) => Math.sin(i));
const b = distribution(big, .7, 20);
assert.equal(b.top.length, 20);
for (let i = 1; i < 20; i++) assert.ok(b.top[i - 1].logit >= b.top[i].logit);
assert.equal(b.top[0].logit, big.reduce((m, x) => Math.max(m, x), -Infinity));

// Frozen chain items keep chosen + top only.
const f = freeze(d, 2);
assert.equal(f.logits, null); close(probOf(f, 2), naive[2] / z); close(probOf(f, 1), d.top[0].prob);

// fp16 decode.
assert.deepEqual([...halfToFloat32(new Uint16Array([0x3c00, 0xc000, 0x0001, 0x7bff]))], [1, -2, 2 ** -24, 65504]);
assert.throws(() => distribution(new Float32Array([1, NaN]), 1), /invalid logits/);
console.log('logits: ok');

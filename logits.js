// Full-vocabulary next-token math shared by the console and System One tab.
// One pass per call; never spreads or sorts the vocabulary (Qwen3.5 has 248,320 entries).
export const TOP_MAX = 20;

// IEEE half → float, for WebGPU fp16 logits returned as Uint16Array.
export function halfToFloat32(half) {
  const out = new Float32Array(half.length);
  for (let i = 0; i < half.length; i++) {
    const h = half[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

// Softmax at temperature `temp` (0 = greedy limit, tied maxima share mass), entropy in bits,
// and the top-k tokens ordered by probability, then logit, then id.
export function distribution(logits, temp, k = TOP_MAX) {
  const n = logits.length;
  if (!n) throw new Error('Model returned invalid logits.');
  let max = -Infinity, ties = 0;
  for (let i = 0; i < n; i++) {
    const x = logits[i];
    if (!Number.isFinite(x)) throw new Error('Model returned invalid logits.');
    if (x > max) { max = x; ties = 1; } else if (x === max) ties++;
  }
  let logZ = 0, entropy = Math.log2(ties);
  if (temp > 0) {
    let z = 0, zs = 0;
    for (let i = 0; i < n; i++) { const s = (logits[i] - max) / temp, e = Math.exp(s); z += e; zs += e * s; }
    logZ = Math.log(z);
    entropy = (logZ - zs / z) / Math.LN2;
  }
  const ids = [];
  for (let i = 0; i < n; i++) {
    const x = logits[i];
    if (ids.length === k && x <= logits[ids[k - 1]]) continue;
    let j = ids.length;
    while (j > 0 && logits[ids[j - 1]] < x) j--;
    ids.splice(j, 0, i);
    if (ids.length > k) ids.pop();
  }
  const dist = { logits, temp, max, ties, logZ, entropy, vocab: n };
  dist.top = ids.map(id => ({ id, logit: logits[id], prob: probOf(dist, id) }));
  return dist;
}

export function probOf(dist, id) {
  if (!dist.logits) return dist.chosen?.id === id ? dist.chosen.prob : dist.top.find(t => t.id === id)?.prob ?? NaN;
  const x = dist.logits[id];
  return dist.temp > 0 ? Math.exp((x - dist.max) / dist.temp - dist.logZ) : x === dist.max ? 1 / dist.ties : 0;
}

// Inverse-CDF draw over the whole vocabulary; u in [0, 1).
export function sample(dist, u) {
  if (!(dist.temp > 0)) return dist.top[0].id;
  let last = dist.top[0].id;
  for (let i = 0; i < dist.vocab; i++) {
    const p = Math.exp((dist.logits[i] - dist.max) / dist.temp - dist.logZ);
    if (p > 0) { last = i; if ((u -= p) <= 0) return i; }
  }
  return last;
}

// Drop the vocabulary-sized array once a token is chosen; keeps top-k plus the chosen token.
export function freeze(dist, id) {
  return { ...dist, logits: null, chosen: { id, logit: dist.logits[id], prob: probOf(dist, id) } };
}

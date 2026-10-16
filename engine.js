// Tiny nanochat, plain typed-array inference. No native/WASM/dependency runtime.
const utf8 = new TextEncoder(), textDecoder = new TextDecoder('utf-8', { ignoreBOM: true });

async function read(base, name, binary = false) {
  const target = new URL(name, /^(https?:|file:)/.test(String(base))
    ? String(base).replace(/\/?$/, '/')
    : typeof location !== 'undefined' ? new URL(String(base).replace(/\/?$/, '/'), location.href) : 'file://' + String(base).replace(/\/?$/, '/'));
  if (target.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
    const { readFile } = await import('node:fs/promises');
    const bytes = await readFile(target);
    return binary ? bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) : JSON.parse(bytes.toString('utf8'));
  }
  const response = await fetch(target);
  if (!response.ok) throw new Error(`Model load failed: ${target} (${response.status})`);
  return binary ? response.arrayBuffer() : response.json();
}

function half(v) {
  const sign = v & 0x8000 ? -1 : 1, exponent = (v >>> 10) & 31, fraction = v & 1023;
  return sign * (exponent === 0 ? fraction * 2 ** -24 : exponent === 31 ? fraction ? NaN : Infinity : (1 + fraction / 1024) * 2 ** (exponent - 15));
}

function tokenizer(spec) {
  // Possessive quantifiers are safe to remove here: adjacent classes are disjoint.
  // Scoped ASCII case-insensitivity affects contractions only, not all text.
  const whitespace = '[\\t\\n\\v\\f\\r \\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]';
  const pattern = spec.pattern.replace("(?i:[sdmt]|ll|ve|re)", '(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])')
    .replaceAll('?+', '?').replaceAll('++', '+')
    .replaceAll('[^\\s\\p{L}\\p{N}]', '[^' + whitespace.slice(1, -1) + '\\p{L}\\p{N}]')
    .replaceAll('\\s', whitespace).replaceAll('\\S', whitespace.replace('[', '[^'));
  const split = new RegExp(pattern, 'gu');
  const ranks = new Map(), bytes = Array.from({ length: 256 }, (_, i) => Uint8Array.of(i));
  for (let i = 0; i < spec.merges.length; i++) {
    const [a, b] = spec.merges[i];
    if (!bytes[a] || !bytes[b]) throw new Error('Invalid BPE merge ordering');
    const joined = new Uint8Array(bytes[a].length + bytes[b].length);
    joined.set(bytes[a]); joined.set(bytes[b], bytes[a].length);
    bytes.push(joined);
    ranks.set(a * 65536 + b, 256 + i);
  }
  const specials = new Map(Object.entries(spec.special).map(([s, id]) => [id, utf8.encode(s)]));
  function encode(text) {
    if (typeof text !== 'string') throw new TypeError('encode expects a string');
    const ids = [];
    for (const match of text.matchAll(split)) {
      const piece = Array.from(utf8.encode(match[0]));
      while (piece.length > 1) {
        let rank = Infinity, at = -1;
        for (let i = 0; i < piece.length - 1; i++) {
          const r = ranks.get(piece[i] * 65536 + piece[i + 1]);
          if (r !== undefined && r < rank) { rank = r; at = i; }
        }
        if (at < 0) break;
        piece.splice(at, 2, rank);
      }
      for (const id of piece) ids.push(id);
    }
    return ids;
  }
  function tokenBytes(id) {
    const b = bytes[id] || specials.get(id);
    if (!b) throw new RangeError(`Unknown token id ${id}`);
    return b;
  }
  function decode(ids) {
    const chunks = Array.from(ids, tokenBytes);
    const out = new Uint8Array(chunks.reduce((n, b) => n + b.length, 0));
    let at = 0;
    for (const b of chunks) { out.set(b, at); at += b.length; }
    return textDecoder.decode(out);
  }
  return { encode, decode, tokenText: id => textDecoder.decode(tokenBytes(id)) };
}

// Row-major weights: 12-token tiles reuse each weight across twelve dot products.
// No transposition: benchmarks favor contiguous weight rows on V8/Apple Silicon.
function linear(w, x, rows, cols, out, count = 1) {
  let t = 0;
  for (; t + 11 < count; t += 12) for (let r = 0; r < rows; r++) {
    let a=0, b=0, c=0, d=0, e=0, f=0, g=0, h=0, j=0, k=0, l=0, m=0;
    const o = r * cols, p = t * cols;
    for (let i = 0; i < cols; i++) {
      const z = w[o + i];
      a+=z*x[p+i]; b+=z*x[p+cols+i]; c+=z*x[p+2*cols+i]; d+=z*x[p+3*cols+i];
      e+=z*x[p+4*cols+i]; f+=z*x[p+5*cols+i]; g+=z*x[p+6*cols+i]; h+=z*x[p+7*cols+i];
      j+=z*x[p+8*cols+i]; k+=z*x[p+9*cols+i]; l+=z*x[p+10*cols+i]; m+=z*x[p+11*cols+i];
    }
    out[t*rows+r]=a; out[(t+1)*rows+r]=b; out[(t+2)*rows+r]=c; out[(t+3)*rows+r]=d;
    out[(t+4)*rows+r]=e; out[(t+5)*rows+r]=f; out[(t+6)*rows+r]=g; out[(t+7)*rows+r]=h;
    out[(t+8)*rows+r]=j; out[(t+9)*rows+r]=k; out[(t+10)*rows+r]=l; out[(t+11)*rows+r]=m;
  }
  for (; t < count; t++) {
    const p = t*cols;
    let r = 0;
    for (; r+3 < rows; r += 4) {
      let a=0, b=0, c=0, d=0;
      const o = r*cols;
      for (let i = 0; i < cols; i++) {
        const z = x[p+i];
        a+=w[o+i]*z; b+=w[o+cols+i]*z; c+=w[o+2*cols+i]*z; d+=w[o+3*cols+i]*z;
      }
      out[t*rows+r]=a; out[t*rows+r+1]=b; out[t*rows+r+2]=c; out[t*rows+r+3]=d;
    }
    for (; r < rows; r++) {
      let sum = 0;
      for (let i = 0; i < cols; i++) sum += w[r*cols+i]*x[p+i];
      out[t*rows+r] = sum;
    }
  }
  return out;
}
function rms(x, epsilon, out = new Float32Array(x.length)) {
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
  const scale = 1 / Math.sqrt(sum / x.length + epsilon);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * scale;
  return out;
}

export async function loadModel(base = './model/') {
  if (typeof process !== 'undefined' && process.versions?.node && !/^(https?:|file:)/.test(String(base))) {
    const { resolve } = await import('node:path');
    base = 'file://' + resolve(String(base)) + '/';
  }
  const [config, manifest, spec, buffer] = await Promise.all([
    read(base, 'config.json'), read(base, 'manifest.json'), read(base, 'tokenizer.json'), read(base, 'weights.bin', true),
  ]);
  const { n_embd: C, n_layer: L, n_head: H, n_kv_head: K, sequence_len: T, vocab_size: V } = config;
  if (!Number.isInteger(C) || C < 24 || C % H || H % K || (C / H) % 2 || !Number.isInteger(T) || T < 1 || L < 1 || V > 65536)
    throw new Error('Unsupported model dimensions');
  const D = C / H, KD = K * D, epsilon = config.rms_eps ?? 1.1920928955078125e-7;
  const tensors = {}, view = new DataView(buffer);
  for (const item of manifest) {
    const count = item.shape.reduce((n, v) => n * v, 1), size = item.dtype === 'float16' ? 2 : item.dtype === 'float32' ? 4 : 0;
    if (!size || item.offset < 0 || item.offset + count * size > buffer.byteLength) throw new Error(`Invalid tensor ${item.name}`);
    const w = new Float32Array(count);
    for (let i = 0; i < count; i++) w[i] = size === 2 ? half(view.getUint16(item.offset + i * 2, true)) : view.getFloat32(item.offset + i * 4, true);
    tensors[item.name] = w;
  }
  function weight(name, length) {
    const w = tensors[name];
    if (!w || w.length !== length) throw new Error(`Missing or wrong shape: ${name}`);
    return w;
  }
  const embedding = weight('wte.weight', V * C), lm = weight('lm_head.weight', V * C);
  const resid = weight('resid_lambdas', L), x0Scale = weight('x0_lambdas', L);
  const smear = weight('smear_gate.weight', 24), smearScale = weight('smear_lambda', 1)[0];
  const backout = weight('backout_lambda', 1)[0];
  const layers = Array.from({ length: L }, (_, i) => {
    const prefix = `blocks.${i}.`, ve = tensors[`value_embeds.${i}.weight`];
    if (ve && ve.length !== V * KD) throw new Error('Invalid value embedding');
    return {
      q: weight(prefix + 'attn.c_q.weight', C * C), k: weight(prefix + 'attn.c_k.weight', KD * C),
      v: weight(prefix + 'attn.c_v.weight', KD * C), proj: weight(prefix + 'attn.c_proj.weight', C * C),
      fc: weight(prefix + 'c_fc.weight', 4 * C * C), mlp: weight(prefix + 'c_proj.weight', 4 * C * C),
      gate: ve ? weight(prefix + 'attn.ve_gate.weight', K * 12) : null, ve,
      keys: new Float32Array(T * KD), values: new Float32Array(T * KD),
      window: config.window_sizes?.[i] ?? T,
    };
  });
  const cos = new Float32Array(T * D / 2), sin = new Float32Array(cos.length);
  for (let t = 0; t < T; t++) for (let i = 0; i < D / 2; i++) {
    const theta = t * (config.rope_base ?? 100000) ** (-2 * i / D);
    cos[t * D / 2 + i] = Math.cos(theta); sin[t * D / 2 + i] = Math.sin(theta);
  }
  function rotateNorm(z, heads, pos) {
    for (let h = 0; h < heads; h++) {
      const off = h * D, rope = pos * D / 2;
      for (let i = 0; i < D / 2; i++) {
        const a = z[off + i], b = z[off + i + D / 2];
        z[off + i] = a * cos[rope + i] + b * sin[rope + i];
        z[off + i + D / 2] = -a * sin[rope + i] + b * cos[rope + i];
      }
      let sum = 0;
      for (let i = 0; i < D; i++) sum += z[off + i] * z[off + i];
      const scale = 1 / Math.sqrt(sum / D + epsilon);
      for (let i = 0; i < D; i++) z[off + i] = Math.fround(Math.fround(z[off + i] * scale) * (config.qk_scale ?? 1.2));
    }
  }
  const preEmbeddings = new Float32Array(T * C), finalStates = new Float32Array(T * C);
  const scores = new Float64Array(T), x = new Float32Array(T*C), x0 = new Float32Array(T*C);
  const normalized = new Float32Array(T*C), q = new Float32Array(T*C), k = new Float32Array(T*KD), v = new Float32Array(T*KD);
  const attended = new Float32Array(T*C), projected = new Float32Array(T*C), hidden = new Float32Array(T*4*C), mid = new Float32Array(T*C);
  const gates = new Float32Array(K), lastState = new Float32Array(C);
  let cachedIds = [], cachedLogits = null;
  function forwardSuffix(ids, firstToken) {
    const count = ids.length - firstToken, size = count * C;
    for (let t = 0; t < count; t++) {
      const id = ids[firstToken+t], pos = firstToken+t, off = t*C;
      const pre = preEmbeddings.subarray(pos*C, (pos+1)*C);
      rms(embedding.subarray(id*C, (id+1)*C), epsilon, pre);
      let gate = 0;
      if (pos) {
        for (let i = 0; i < 24; i++) gate += smear[i]*pre[i];
        gate = Math.fround(smearScale/(1+Math.exp(-gate)));
      }
      for (let i = 0; i < C; i++) x[off+i] = pos ? pre[i]+Math.fround(gate*preEmbeddings[(pos-1)*C+i]) : pre[i];
    }
    x0.set(x.subarray(0, size));
    for (let layer = 0; layer < L; layer++) {
      const b = layers[layer];
      for (let i = 0; i < size; i++) x[i] = Math.fround(resid[layer]*x[i])+Math.fround(x0Scale[layer]*x0[i]);
      for (let t = 0; t < count; t++) rms(x.subarray(t*C, (t+1)*C), epsilon, normalized.subarray(t*C, (t+1)*C));
      linear(b.q, normalized, C, C, q, count);
      linear(b.k, normalized, KD, C, k, count);
      linear(b.v, normalized, KD, C, v, count);
      for (let t = 0; t < count; t++) {
        const pos = firstToken+t, id = ids[pos], ko = t*KD;
        if (b.ve) {
          linear(b.gate, normalized.subarray(t*C), K, 12, gates);
          for (let h = 0; h < K; h++) {
            const gate = Math.fround(3*Math.fround(1/(1+Math.exp(-gates[h]))));
            for (let i = 0; i < D; i++) v[ko+h*D+i] += Math.fround(gate*b.ve[id*KD+h*D+i]);
          }
        }
        rotateNorm(q.subarray(t*C, (t+1)*C), H, pos);
        rotateNorm(k.subarray(ko, ko+KD), K, pos);
        b.keys.set(k.subarray(ko, ko+KD), pos*KD);
        b.values.set(v.subarray(ko, ko+KD), pos*KD);
      }
      for (let t = 0; t < count; t++) {
        const pos = firstToken+t, first = Math.max(0, pos-b.window);
        for (let h = 0; h < H; h++) {
          const qo = t*C+h*D, ko = Math.floor(h/(H/K))*D;
          let max = -Infinity;
          for (let s = first; s <= pos; s++) {
            let score = 0;
            for (let i = 0; i < D; i++) score += q[qo+i]*b.keys[s*KD+ko+i];
            scores[s] = score/Math.sqrt(D); max = Math.max(max, scores[s]);
          }
          let denom = 0;
          for (let s = first; s <= pos; s++) { scores[s] = Math.exp(scores[s]-max); denom += scores[s]; }
          for (let i = 0; i < D; i++) {
            let value = 0;
            for (let s = first; s <= pos; s++) value += scores[s]/denom*b.values[s*KD+ko+i];
            attended[qo+i] = value;
          }
        }
      }
      linear(b.proj, attended, C, C, projected, count);
      for (let i = 0; i < size; i++) x[i] += projected[i];
      for (let t = 0; t < count; t++) rms(x.subarray(t*C, (t+1)*C), epsilon, normalized.subarray(t*C, (t+1)*C));
      linear(b.fc, normalized, 4*C, C, hidden, count);
      for (let i = 0; i < size*4; i++) hidden[i] = hidden[i] > 0 ? hidden[i]*hidden[i] : 0;
      linear(b.mlp, hidden, C, 4*C, projected, count);
      for (let i = 0; i < size; i++) x[i] += projected[i];
      if (layer === (config.backout_layer ?? Math.floor(L/2))) mid.set(x.subarray(0, size));
    }
    for (let i = 0; i < size; i++) x[i] -= Math.fround(backout*mid[i]);
    for (let t = 0; t < count; t++) rms(x.subarray(t*C, (t+1)*C), epsilon, finalStates.subarray((firstToken+t)*C, (firstToken+t+1)*C));
  }
  async function logits(ids) {
    if (!Array.isArray(ids) && !ArrayBuffer.isView(ids)) throw new TypeError('logits expects token ids');
    for (const id of ids) if (!Number.isInteger(id) || id < 0 || id >= V) throw new RangeError(`Invalid token id ${id}`);
    const input = ids.length < T ? [config.bos_id, ...ids] : Array.from(ids.slice(-T));
    let prefix = 0;
    while (prefix < Math.min(input.length, cachedIds.length) && input[prefix] === cachedIds[prefix]) prefix++;
    if (prefix === input.length && input.length === cachedIds.length && cachedLogits) return cachedLogits.slice();
    // KV + final-state reuse: pure-JS cold prefill exceeds 300 ms.
    // Branches preserve the longest common prefix; cropping restarts relative positions.
    if (prefix < input.length) forwardSuffix(input, prefix);
    lastState.set(finalStates.subarray((input.length-1)*C, input.length*C));
    const out = linear(lm, lastState, V, C, new Float32Array(V)), cap = config.softcap ?? 15;
    for (let i = 0; i < V; i++) out[i] = cap * Math.tanh(out[i] / cap);
    cachedIds = input; cachedLogits = out;
    return out.slice();
  }
  // ponytail: single-session synchronous math; add workers/WebGPU only for larger models.
  return { config, bosId: config.bos_id, ...tokenizer(spec), logits };
}

// Qwen3.5-0.8B (text-only ONNX, q4f16) on WebGPU via the vendored Transformers.js bundle.
// Weights come from Hugging Face at one pinned revision; inference is local. No KV cache: each logits(ids) is a full forward.
import { halfToFloat32 } from './logits.js';

const REPO = 'onnx-community/Qwen3.5-0.8B-Text-ONNX';
const REVISION = '25cdf33f386b487a25f3d743e86442517bdb77ff';
// jsDelivr, not vendored: the wasm files exceed Cloudflare Pages' 25 MiB per-file limit.
const WASM_PATHS = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/';
const WEIGHTS = `https://huggingface.co/${REPO}/resolve/${REVISION}/onnx/model_q4f16.onnx_data`;
// ponytail: stateless recompute is O(n) per step (~240 ms at 500 tokens). Upgrade: merged decoder with KV cache.
const CONTEXT = 512;

export const MODEL = { name: 'QWEN3.5-0.8B', repo: REPO, revision: REVISION, bytes: 469331968 + 597442 + 19226111, quant: 'q4f16', context: CONTEXT }; // bytes: weights + graph + tokenizer.json

// Transformers.js stores completed downloads in this Cache API cache, keyed by request URL.
export async function isCached() {
  try {
    const cache = await caches.open('transformers-cache');
    return !!(await cache.match(WEIGHTS) && await cache.match(WEIGHTS.replace('_data', '')));
  } catch { return false; }
}

// GPT-2 byte-level BPE: piece characters map back to raw bytes.
const byteOf = new Map();
{
  const keep = [...Array(94).keys()].map(i => i + 33).concat([...Array(12).keys()].map(i => i + 161), [...Array(82).keys()].map(i => i + 174));
  let extra = 0;
  for (let b = 0; b < 256; b++) byteOf.set(String.fromCharCode(keep.includes(b) ? b : 256 + extra++), b);
}

export async function loadModel(onProgress = () => {}) {
  const { AutoTokenizer, AutoModelForCausalLM, Tensor, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js');
  env.backends.onnx.wasm.wasmPaths = WASM_PATHS;
  env.remotePathTemplate = `{model}/resolve/${REVISION}/`; // pins every request, incl. the tokenizer probe that passes no revision
  navigator.storage?.persist?.();
  const files = new Map(), options = { revision: REVISION, progress_callback: p => {
    if (p.status !== 'progress') return;
    files.set(p.file, p);
    let loaded = 0, total = 0;
    for (const f of files.values()) { loaded += f.loaded; total += f.total; }
    onProgress({ loaded, total });
  } };
  const tokenizer = await AutoTokenizer.from_pretrained(REPO, options);
  const model = await AutoModelForCausalLM.from_pretrained(REPO, { ...options, dtype: 'q4f16', device: 'webgpu' });
  const c = model.config, V = c.vocab_size;
  if (c.model_type !== 'qwen3_5_text' || V !== 248320) throw new Error('Unexpected model configuration.');
  // ponytail: raw piece table built once (248K strings) for tokens that are partial UTF-8 sequences.
  const pieces = [];
  for (const [piece, id] of tokenizer.get_vocab()) pieces[id] = piece;

  const decode = ids => tokenizer.decode(Array.from(ids));
  // A lone byte-level piece can decode to U+FFFD; show its bytes instead (e.g. <F0 9F>).
  function tokenText(id) {
    const text = decode([id]);
    if (!text.includes('�') || pieces[id] === undefined) return text;
    return '<' + Array.from(pieces[id], ch => (byteOf.get(ch) ?? 63).toString(16).toUpperCase().padStart(2, '0')).join(' ') + '>';
  }
  async function logits(ids) {
    if (!ids.length || ids.length > CONTEXT) throw new RangeError(`Need 1–${CONTEXT} token ids.`);
    for (const id of ids) if (!Number.isInteger(id) || id < 0 || id >= V) throw new RangeError(`Invalid token id ${id}`);
    const input_ids = new Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]);
    const out = await model.forward({
      input_ids, attention_mask: new Tensor('int64', new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
      num_logits_to_keep: new Tensor('int64', [1n], []), // last position only: [1, 1, V]
    });
    const { logits: tensor, ...cache } = out;
    // Unused present_* state tensors hold GPU buffers until disposed.
    for (const t of Object.values(cache)) t?.dispose?.();
    const data = tensor.data; // fp16 (Float16Array, or Uint16Array on browsers without it)
    const result = data instanceof Uint16Array ? halfToFloat32(data) : Float32Array.from(data);
    tensor.dispose?.();
    if (result.length !== V) throw new Error(`Unexpected logits length ${result.length}.`);
    return result;
  }
  const config = {
    n_layer: c.num_hidden_layers, n_head: c.num_attention_heads, n_kv_head: c.num_key_value_heads, n_embd: c.hidden_size,
    vocab_size: V, layer_types: c.layer_types, sequence_len: CONTEXT, quant: 'q4f16', device: 'WebGPU',
    n_params: 752393024, // text-only safetensors total from the pinned HF revision
  };
  return { config, encode: text => tokenizer.encode(text), decode, tokenText, logits };
}

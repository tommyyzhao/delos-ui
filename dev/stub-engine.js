// Synthetic development host only; never loaded without ?stub. Fixed vocabulary, deterministic logits.
export const MODEL = { name: 'DEV STUB', bytes: 0, quant: 'none', context: 128 };
export async function isCached() { return true; }
export async function loadModel() {
  const words = ['\n', 'Have', ' you', ' ever', ' questioned', ' the', ' nature', ' of', ' your', ' reality', '?', ' Once', ' upon', ' a', ' time', ',', ' there', ' was', ' dream', ' mind', ' world', ' own', '.', ' It', ' is', ' and', ' to', ' in', ' that', ' not'];
  const story = [9, 10, 23, 24, 13, 18, 25, 20, 22];
  const index = new Map(words.map((w, i) => [w, i]));
  return {
    config: { vocab_size: words.length, n_layer: 4, n_head: 4, n_kv_head: 2, n_embd: 128, sequence_len: 128, n_params: 987648, quant: 'none', device: 'STUB', layer_types: ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'] },
    encode: text => (text.match(/\s*[^\s]+/g) || []).map(part => index.get(part) ?? 1 + (part.length + part.charCodeAt(1 % part.length)) % (words.length - 1)),
    decode: ids => Array.from(ids, id => words[id]).join(''),
    tokenText: id => words[id],
    async logits(ids) {
      const preferred = story[ids.length % story.length];
      const seed = ids.reduce((sum, id) => (sum * 31 + id) % 65521, 17);
      return Float32Array.from(words, (_, id) => id === preferred ? 4.4 : -1.7 + Math.sin(id * 19.1 + seed) * 1.7);
    },
  };
}

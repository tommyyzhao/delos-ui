// Synthetic development host only; never loaded without ?stub.
export async function loadModel() {
  const words = ['<BOS>', 'Once', ' upon', ' a', ' time', ',', ' there', ' was', ' little', ' girl', ' named', ' Lily', '.', ' She', ' loved', ' to', ' play', ' in', ' the', ' garden', ' with', ' her', ' friend', ' One', ' day', ' they', ' found', ' happy', ' small', ' dog', ' and', ' went', ' home', '\n', ' It', ' is', ' story', ' very', ' kind', ' boy', ' blue', ' tree', ' bird', ' ran', ' said', ' hello', '!'];
  const story = [5, 6, 7, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 12];
  return {
    config: { vocab_size: words.length, n_layer: 4, n_head: 4, n_kv_head: 2, n_embd: 128, sequence_len: 128, n_params: 987648, val_loss: 2.718 },
    bosId: 0,
    encode(text) {
      return (text.match(/\s*[^\s]+/g) || []).map(part => {
        let id = words.indexOf(part);
        if (id < 0) { words.push(part); id = words.length - 1; this.config.vocab_size = words.length; }
        return id;
      });
    },
    decode(ids) { return ids.map(id => id === 0 ? '' : words[id] ?? '').join(''); },
    tokenText(id) { return words[id] ?? ''; },
    async logits(ids) {
      const position = Math.max(0, ids.length - 4);
      const preferred = story[position % story.length];
      const seed = ids.reduce((sum, id) => (sum * 31 + id) % 65521, 17);
      return Float32Array.from(words, (_, id) => id === preferred ? 4.4 : -1.7 + Math.sin(id * 19.1 + seed) * 1.7);
    }
  };
}

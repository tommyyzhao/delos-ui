# Delos UI

A Westworld-style "Delos" host console that runs a real language model in your browser and shows its real next-token logits.

- **CONSOLE** (`index.html`): [Qwen3.5-0.8B](https://huggingface.co/onnx-community/Qwen3.5-0.8B-Text-ONNX) (q4f16) on WebGPU. You can step, run, go back and re-branch the dialogue tree. The logit inspector shows the full 248,320-token distribution at every step.
- **SYSTEM ONE** (`decide.html`): [OneJev-0.8B](https://huggingface.co/onnx-community/OneJev-0.8B-ONNX), a decision model. You edit a state and some typed questions (choice, score, yes/no), and the answers recompute on every edit. Each answer can expand to show:
  - the option-letter logits and their softmax;
  - the confidence formula, with its numbers filled in;
  - the full-vocabulary distribution at the answer slot;
  - the exact prompt.

There is no backend. Weights download once from Hugging Face at pinned revisions and are cached by the browser; inference runs locally on your GPU. Nothing you type leaves the page. **CLEAR CACHE** (top right) deletes the cached weights.

## Run

```sh
python3 -m http.server 8000
# open http://localhost:8000/
```

## Build (production deployment)

```sh
sh deploy/build.sh dist   # fresh dir: hashed assets + rewritten pages + _headers; upload to Cloudflare Pages
```

You need a browser with WebGPU (current Chrome or Edge, or Safari 26+) on a secure origin. Browsers only expose WebGPU over https:// or on localhost. The first load downloads about 0.5 GB per model.

To work on the UI without a GPU, open `http://localhost:8000/?stub`. It uses a synthetic engine and downloads nothing.

## Test

```sh
node test/logits.mjs && node test/s1.mjs
```

These tests need no model and no network. `test/s1.mjs` checks the prompt port against OneJev's reference prompts and answers (`test/onejev-reference.json`).

## Credits

- **Models:** Qwen3.5-0.8B and OneJev-0.8B, both Apache-2.0.
- **Runtime:** [Transformers.js](https://github.com/huggingface/transformers.js) 4.3.2 and ONNX Runtime Web (both Apache-2.0), loaded from jsDelivr at pinned versions.

## License

MIT

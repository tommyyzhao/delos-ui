# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "numpy", "tokenizers"]
# ///
"""uv run train/train.py --minutes 43
TinyStories training + portable fp16 export. Forward mirrors nanochat/gpt.py
(master fetched 2026-10-05), including smear, backout, value gates and scalars.
Only distributed/FlashAttention machinery, padded vocab and optimizer are omitted:
SDPA + AdamW, exact unpadded vocabulary. Training uses fp32 RMSNorm.
"""
import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / 'train/data', ROOT / 'model'
PATTERN = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"


def byte_alphabet():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs.copy(), 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def tokenizer():
    cached = DATA / 'tokenizer-hf.json'
    if cached.exists():
        return Tokenizer.from_file(str(cached))
    alphabet = byte_alphabet()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    text = (DATA / 'train.txt').read_text()
    tok.train_from_iterator([text[i:i+65536] for i in range(0, min(len(text), 16000000), 65536)],
        trainer=trainers.BpeTrainer(vocab_size=4095, initial_alphabet=list(alphabet.values()), show_progress=False))
    raw = json.loads(tok.to_str())['model']
    # Remap HF's alphabet ordering to literal bytes 0..255, then ranked merges.
    vocab = {alphabet[b]: b for b in range(256)}
    merges, pairs = raw['merges'], []
    for i, pair in enumerate(merges):
        a, b = pair if isinstance(pair, list) else pair.split(' ')
        pairs.append([vocab[a], vocab[b]])
        vocab[a+b] = 256+i
    bos = len(vocab)
    vocab['<|bos|>'] = bos
    tok.model = models.BPE(vocab=vocab, merges=[tuple(p) if isinstance(p, list) else tuple(p.split(' ')) for p in merges])
    tok.save(str(cached))
    (OUT / 'tokenizer.json').write_text(json.dumps({'pattern': PATTERN, 'merges': pairs, 'special': {'<|bos|>': bos}}, separators=(',', ':')))
    return tok


def download(name, limit):
    dest = DATA / f'{name}.txt'
    if dest.exists():
        return
    url = f'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-{name if name == "valid" else "train"}.txt'
    print(f'Downloading {name} subset ({limit//1000000} MB)', flush=True)
    with urllib.request.urlopen(url, timeout=120) as src:
        chunks, n = [], 0
        while n < limit:
            chunk = src.read(min(1048576, limit-n))
            if not chunk:
                break
            chunks.append(chunk)
            n += len(chunk)
    text = b''.join(chunks).decode('utf-8', errors='ignore')
    text = text.rsplit('<|endoftext|>', 1)[0]
    dest.write_text(text)


def token_data(tok, name, bos):
    path = DATA / f'{name}.npy'
    if path.exists():
        return np.load(path)
    docs = (DATA / f'{name}.txt').read_text().split('<|endoftext|>')
    ids = []
    for i in range(0, len(docs), 2048):
        batch = tok.encode_batch([d.strip() for d in docs[i:i+2048] if d.strip()])
        for enc in batch:
            ids.extend([bos] + enc.ids)
    data = np.array(ids, dtype=np.uint16)
    np.save(path, data)
    print(f'{name}: {len(data):,} tokens', flush=True)
    return data


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Attention(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.c, self.i = c, i
        d, kd = c['n_embd'], c['n_kv_head'] * (c['n_embd']//c['n_head'])
        self.c_q = nn.Linear(d, d, bias=False)
        self.c_k = nn.Linear(d, kd, bias=False)
        self.c_v = nn.Linear(d, kd, bias=False)
        self.c_proj = nn.Linear(d, d, bias=False)
        self.ve_gate = nn.Linear(12, c['n_kv_head'], bias=False) if i % 2 == (c['n_layer']-1) % 2 else None

    def forward(self, x, ve, cos, sin, mask):
        B, T, C = x.shape
        H, K, D = self.c['n_head'], self.c['n_kv_head'], C//self.c['n_head']
        q = self.c_q(x).view(B, T, H, D)
        k = self.c_k(x).view(B, T, K, D)
        v = self.c_v(x).view(B, T, K, D)
        if ve is not None:
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :12]))
            v = v + gate.unsqueeze(-1) * ve.view(B, T, K, D)
        def rotary(z):
            a, b = z.chunk(2, dim=-1)
            return torch.cat((a*cos+b*sin, -a*sin+b*cos), dim=-1)
        q, k = norm(rotary(q))*1.2, norm(rotary(k))*1.2
        q, k, v = [z.transpose(1, 2) for z in (q, k, v)]
        # MPS SDPA lacks native GQA; expand KV heads, preserving contiguous groups.
        if H != K:
            k, v = [z.repeat_interleave(H//K, dim=1) for z in (k, v)]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.attn = Attention(c, i)
        self.c_fc = nn.Linear(c['n_embd'], 4*c['n_embd'], bias=False)
        self.c_proj = nn.Linear(4*c['n_embd'], c['n_embd'], bias=False)

    def forward(self, x, ve, cos, sin, mask):
        x = x + self.attn(norm(x), ve, cos, sin, mask)
        return x + self.c_proj(F.relu(self.c_fc(norm(x))).square())


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        L, C, V = c['n_layer'], c['n_embd'], c['vocab_size']
        self.wte = nn.Embedding(V, C)
        self.blocks = nn.ModuleList([Block(c, i) for i in range(L)])
        self.lm_head = nn.Linear(C, V, bias=False)
        self.resid_lambdas = nn.Parameter(torch.linspace(1.15, 1.05, L))
        self.x0_lambdas = nn.Parameter(torch.linspace(.20, .05, L))
        self.smear_gate = nn.Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(torch.tensor([.2]))
        kd = c['n_kv_head']*(C//c['n_head'])
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(V, kd) for i in range(L) if i % 2 == (L-1) % 2})
        D = C//c['n_head']
        f = torch.outer(torch.arange(c['sequence_len'], dtype=torch.float32), c['rope_base']**(-torch.arange(0, D, 2, dtype=torch.float32)/D))
        self.register_buffer('cos', f.cos()[None, :, None, :], persistent=False)
        self.register_buffer('sin', f.sin()[None, :, None, :], persistent=False)
        pos = torch.arange(c['sequence_len'])
        for i, window in enumerate(c['window_sizes']):
            mask = (pos[:, None] >= pos[None, :]) & (pos[:, None]-pos[None, :] <= window)
            self.register_buffer(f'mask_{i}', mask, persistent=False)
        self.initialize()

    @torch.no_grad()
    def initialize(self):
        s = math.sqrt(3/self.c['n_embd'])
        nn.init.normal_(self.wte.weight, std=.8)
        nn.init.normal_(self.lm_head.weight, std=.001)
        nn.init.uniform_(self.smear_gate.weight, 0, .02)
        for b in self.blocks:
            for p in (b.attn.c_q.weight, b.attn.c_k.weight, b.attn.c_v.weight):
                nn.init.uniform_(p, -s, s)
            nn.init.zeros_(b.attn.c_proj.weight)
            nn.init.uniform_(b.c_fc.weight, -.4*s, .4*s)
            nn.init.zeros_(b.c_proj.weight)
            if b.attn.ve_gate is not None:
                nn.init.uniform_(b.attn.ve_gate.weight, 0, .02)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = norm(self.wte(idx))
        if T > 1:
            gate = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat((x[:, :1], x[:, 1:] + gate*x[:, :-1]), dim=1)
        x0, mid = x, None
        for i, b in enumerate(self.blocks):
            x = self.resid_lambdas[i]*x + self.x0_lambdas[i]*x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = b(x, ve, self.cos[:, :T], self.sin[:, :T], getattr(self, f'mask_{i}')[:T, :T])
            if i == self.c['n_layer']//2:
                mid = x
        x = norm(x-self.backout_lambda*mid)
        z = self.lm_head(x)
        z = self.c['softcap']*torch.tanh(z.float()/self.c['softcap'])
        if targets is None:
            return z
        return F.cross_entropy(z.flatten(0, 1), targets.flatten())


def batch(data, batch_size, seq, device):
    starts = np.random.randint(0, len(data)-seq-1, size=batch_size)
    rows = np.stack([data[s:s+seq+1] for s in starts]).astype(np.int64)
    return torch.from_numpy(rows[:, :-1].copy()).to(device), torch.from_numpy(rows[:, 1:].copy()).to(device)


@torch.no_grad()
def validate(model, data, device, n=12):
    model.eval()
    losses = [model(*batch(data, 8, model.c['sequence_len'], device)).item() for _ in range(n)]
    model.train()
    return sum(losses)/len(losses)


@torch.no_grad()
def sample(model, tok, prompt, n=80):
    ids = [model.c['bos_id']] + tok.encode(prompt).ids
    device = next(model.parameters()).device
    for _ in range(n):
        z = model(torch.tensor([ids[-model.c['sequence_len']:]], device=device))[:, -1].float().cpu()
        v, ix = z.topk(40)
        nxt = ix[0, torch.multinomial(torch.softmax(v[0]/.8, dim=0), 1)].item()
        if nxt == model.c['bos_id']:
            break
        ids.append(nxt)
    return tok.decode(ids[1:])


def export(model, tok, config):
    config.update(source_model='https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py',
        source_tokenizer='https://github.com/karpathy/nanochat/blob/master/nanochat/tokenizer.py',
        tokenizer_training='Byte BPE; HF tokenizers trainer; nanochat regex pre-split; literal byte IDs 0..255')
    model = model.to('cpu').float().eval()
    manifest, offset = [], 0
    with (OUT / 'weights.bin').open('wb') as f:
        for name, tensor in model.named_parameters():
            raw = tensor.detach().numpy().astype('<f2').tobytes()
            f.write(raw)
            manifest.append({'name': name, 'shape': list(tensor.shape), 'dtype': 'float16', 'offset': offset})
            offset += len(raw)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, separators=(',', ':')))
    (OUT / 'config.json').write_text(json.dumps(config, indent=2))
    prompt = 'Once upon a time, a little girl named Lily found a tiny bird in the garden.'
    ids = tok.encode(prompt).ids
    with torch.no_grad():
        logits = model(torch.tensor([[config['bos_id']]+ids]))[0, -1].numpy()
    fixtures = []
    for text in ["I'm HERE! It's 12345.\r\nNext\tline.  ", 'Café 中文 Привет 👋🏽 🌈', 'hello   world\n\n', '<|bos|> literal', '﻿﻿hello', 'one\u0085two three　four']:
        fixtures.append({'text': text, 'ids': tok.encode(text).ids})
    (OUT / 'reference.json').write_text(json.dumps({'prompt': prompt, 'ids': ids, 'logits': logits.tolist(), 'top10': np.argsort(-logits)[:10].tolist(), 'tokenizer_cases': fixtures}))
    print(f'Exported {offset/1e6:.2f} MB weights', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--minutes', type=float, default=43)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--export-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    np.random.seed(42)
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    download('train', 100000000)
    download('valid', 6000000)
    tok = tokenizer()
    bos = tok.get_vocab()['<|bos|>']
    config = dict(vocab_size=tok.get_vocab_size(), n_layer=6, n_head=6, n_kv_head=3, n_embd=384, sequence_len=256,
        softcap=15, rope_base=100000, bos_id=bos, rms_eps=float(torch.finfo(torch.float32).eps), qk_scale=1.2,
        window_pattern='SSSL', window_sizes=[128, 128, 128, 256, 128, 256], ve_gate_channels=12, smear_gate_channels=24,
        backout_layer=3, architecture='nanochat-master-2026-10-05', weights_dtype='float16')
    model = GPT(config)
    config['n_params'] = sum(p.numel() for p in model.parameters())
    print(f'Model: {config["n_params"]:,} parameters', flush=True)
    checkpoint = DATA / 'checkpoint.pt'
    if args.export_only:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        export(model, tok, saved['config'])
        return
    train = token_data(tok, 'train', bos)
    valid = token_data(tok, 'valid', bos)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = model.to(device)
    emb = [model.wte.weight] + [v.weight for v in model.value_embeds.values()]
    emb_ids = {id(v) for v in emb}
    matrices = [v for v in model.parameters() if v.ndim >= 2 and id(v) not in emb_ids]
    scalars = [v for v in model.parameters() if v.ndim < 2]
    optim = torch.optim.AdamW([{'params': matrices, 'lr': .0015, 'weight_decay': .05},
        {'params': emb, 'lr': .003, 'weight_decay': .01}, {'params': scalars, 'lr': .0015, 'weight_decay': 0}], betas=(.9, .95))
    initial_lrs = [g['lr'] for g in optim.param_groups]
    step, tokens, previous_seconds = 0, 0, 0
    if args.resume:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        optim.load_state_dict(saved['optimizer'])
        step, tokens = saved['step'], saved['config']['trained_tokens']
        previous_seconds = saved['config']['training_seconds']
    start = last = time.monotonic()
    deadline = args.minutes*60
    # Reserve time for validation/checkpoint/export; synchronize MPS before checking
    # elapsed time so queued GPU work cannot silently overrun the wall-clock budget.
    training_deadline = max(1, deadline-90)
    val_loss = float('nan')
    model.train()
    while step < args.steps and time.monotonic()-start < training_deadline:
        elapsed = time.monotonic()-start
        warmup = min(1, (step+1)/50)
        decay = .15+.85*.5*(1+math.cos(math.pi*min(1, elapsed/training_deadline)))
        for g, lr in zip(optim.param_groups, initial_lrs):
            g['lr'] = lr*warmup*decay
        x, y = batch(train, args.batch_size, 256, device)
        optim.zero_grad(set_to_none=True)
        loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if device == 'mps':
            torch.mps.synchronize()
        step += 1
        tokens += x.numel()
        if step % 25 == 0:
            now = time.monotonic()
            print(f'step {step} loss {loss.item():.4f} tokens {tokens:,} elapsed {now-start:.1f}s tok/s {25*x.numel()/(now-last):.0f}', flush=True)
            last = now
        if step % 250 == 0:
            val_loss = validate(model, valid, device, n=8)
            print(f'VAL step {step}: {val_loss:.4f}', flush=True)
            config.update(val_loss=val_loss, trained_tokens=tokens, training_seconds=time.monotonic()-start+previous_seconds, training_steps=step)
            temp = checkpoint.with_suffix('.tmp')
            torch.save({'model': model.state_dict(), 'optimizer': optim.state_dict(), 'step': step, 'config': config}, temp)
            temp.replace(checkpoint)
    val_loss = validate(model, valid, device, n=12)
    config.update(val_loss=val_loss, trained_tokens=tokens, training_seconds=time.monotonic()-start+previous_seconds, training_steps=step,
        optimizer='AdamW', dataset='roneneldan/TinyStories (100 MB training subset; disjoint validation split)')
    print(f'FINAL val_loss {val_loss:.4f}; training {config["training_seconds"]:.1f}s; tokens {tokens:,}', flush=True)
    temp = checkpoint.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optim.state_dict(), 'step': step, 'config': config}, temp)
    temp.replace(checkpoint)
    model.eval()
    for prompt in ['Once upon a time, there was a little rabbit', 'Lily went to the park with her mom.']:
        print('SAMPLE:', sample(model, tok, prompt), flush=True)
    export(model, tok, config)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Train a tiny GPT-2-style transformer and export it for the 3D explorer.

Two tasks are supported:

  sort         Brendan-Bycroft-style toy: six letters from {A,B,C} go in, the
               model must emit them sorted. Vocab 3, context 11, ~85k params.
               Trains to ~100% accuracy in a minute. Every attention head has
               a readable job, which makes it the best model to *zoom into*.

  shakespeare  Character-level language model on tiny-shakespeare (Karpathy's
               char-rnn corpus). Vocab 65, context 64. A real next-character
               predictor you can type into.

Architecture (identical to GPT-2 / nanoGPT, just small):

  x = tok_emb[ids] + pos_emb[0:T]
  repeat n_layer times:
      x = x + W_o( heads( softmax( Q K^T / sqrt(d_head) + causal_mask ) V ) )  where Q,K,V = LN1(x) W_{q,k,v}
      x = x + W_out( GELU( LN2(x) W_in ) )
  logits = LN_f(x) tok_emb^T          (tied unembedding, like GPT-2)

Dependencies: jax, numpy.  CPU is plenty.

  python3 train_gpt.py --task sort
  python3 train_gpt.py --task shakespeare --steps 8000

Outputs  ../models/<task>.json  (config, vocab, tensor table, training log,
verification vector) and ../models/<task>.bin (all weights, float32 LE,
concatenated in the order of the tensor table).
"""
import argparse, json, math, os, time, urllib.request
import numpy as np
import jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "models")

# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def gelu(x):
    # GPT-2's tanh approximation; the JS forward pass uses the same formula.
    return 0.5 * x * (1.0 + jnp.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))

def layernorm(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps) * g + b

def init_params(cfg, key):
    d, dff, V, T, L = cfg["d_model"], cfg["d_ff"], cfg["vocab_size"], cfg["block_size"], cfg["n_layer"]
    std, std_proj = 0.02, 0.02 / math.sqrt(2 * L)
    keys = iter(jax.random.split(key, 2 + 8 * L))
    n = lambda shape, s: (s * jax.random.normal(next(keys), shape)).astype(jnp.float32)
    p = {"tok_emb": n((V, d), std), "pos_emb": n((T, d), std),
         "lnf_g": jnp.ones(d), "lnf_b": jnp.zeros(d)}
    for i in range(L):
        p[f"h{i}"] = {
            "ln1_g": jnp.ones(d), "ln1_b": jnp.zeros(d),
            "wq": n((d, d), std), "bq": jnp.zeros(d),
            "wk": n((d, d), std), "bk": jnp.zeros(d),
            "wv": n((d, d), std), "bv": jnp.zeros(d),
            "wo": n((d, d), std_proj), "bo": jnp.zeros(d),
            "ln2_g": jnp.ones(d), "ln2_b": jnp.zeros(d),
            "win": n((d, dff), std), "bin": jnp.zeros(dff),
            "wout": n((dff, d), std_proj), "bout": jnp.zeros(d),
        }
    return p

def forward(params, cfg, ids):
    B, T = ids.shape
    H = cfg["n_head"]; d = cfg["d_model"]; dh = d // H
    x = params["tok_emb"][ids] + params["pos_emb"][:T]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    for i in range(cfg["n_layer"]):
        p = params[f"h{i}"]
        h = layernorm(x, p["ln1_g"], p["ln1_b"])
        q = (h @ p["wq"] + p["bq"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        k = (h @ p["wk"] + p["bk"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = (h @ p["wv"] + p["bv"]).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(dh)
        att = jnp.where(mask, att, -1e9)
        att = jax.nn.softmax(att, axis=-1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        x = x + o @ p["wo"] + p["bo"]
        h = layernorm(x, p["ln2_g"], p["ln2_b"])
        x = x + gelu(h @ p["win"] + p["bin"]) @ p["wout"] + p["bout"]
    x = layernorm(x, params["lnf_g"], params["lnf_b"])
    return x @ params["tok_emb"].T

def loss_fn(params, cfg, ids, targets, mask):
    logits = forward(params, cfg, ids)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return (nll * mask).sum() / mask.sum()

# --------------------------------------------------------------------------
# optimiser: AdamW with warmup + cosine decay and global-norm gradient clipping
# --------------------------------------------------------------------------

def make_step(cfg, wd, clip):
    @jax.jit
    def step(params, opt, ids, targets, mask, lr):
        loss, g = jax.value_and_grad(loss_fn)(params, cfg, ids, targets, mask)
        leaves = jax.tree_util.tree_leaves(g)
        gnorm = jnp.sqrt(sum((l ** 2).sum() for l in leaves))
        scale = jnp.minimum(1.0, clip / (gnorm + 1e-6))
        g = jax.tree_util.tree_map(lambda x: x * scale, g)
        m, v, t = opt["m"], opt["v"], opt["t"] + 1
        b1, b2 = 0.9, 0.95
        m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ ** 2, v, g)
        mhat = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1 ** t), m)
        vhat = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2 ** t), v)
        def upd(p, mh, vh):
            decay = wd if p.ndim >= 2 else 0.0   # no decay on biases / LN gains
            return p - lr * (mh / (jnp.sqrt(vh) + 1e-8) + decay * p)
        params = jax.tree_util.tree_map(upd, params, mhat, vhat)
        return params, {"m": m, "v": v, "t": t}, loss, gnorm
    return step

def lr_at(it, peak, warmup, total, floor_frac=0.1):
    if it < warmup:
        return peak * (it + 1) / warmup
    prog = min(1.0, (it - warmup) / max(1, total - warmup))
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * prog)))

# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------

class SortTask:
    """Six letters in, the same six letters sorted out.
    sequence  = in0..in5 s0..s4   (11 tokens)      targets = in1..in5 s0..s5
    loss only on positions 5..10 (where the sorted letters are predicted)."""
    name = "sort"
    title = "Sorter (A/B/C)"
    vocab = ["A", "B", "C"]
    n_in = 6
    cfg = dict(vocab_size=3, block_size=11, n_layer=3, n_head=3, d_model=48, d_ff=192)
    def __init__(self, rng): self.rng = rng
    def batch(self, B, split="train"):
        x = self.rng.integers(0, 3, size=(B, self.n_in))
        s = np.sort(x, axis=1)
        full = np.concatenate([x, s], axis=1)              # B, 12
        ids, targets = full[:, :-1], full[:, 1:]
        mask = np.zeros_like(ids, dtype=np.float32); mask[:, self.n_in - 1:] = 1.0
        return ids, targets, mask
    def example(self):
        return [2, 0, 1, 2, 1, 0, 0, 0, 1, 1, 2]   # C A B C B A | A A B B C -> predicts C at the end
    def describe(self):
        return ("Toy task from Brendan Bycroft's LLM visualisation: the six input letters are "
                "followed by the same letters in sorted order; the model predicts each next sorted letter.")

class ShakespeareTask:
    name = "shakespeare"
    title = "Shakespeare (characters)"
    cfg = dict(vocab_size=None, block_size=64, n_layer=4, n_head=4, d_model=64, d_ff=256)
    URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    def __init__(self, rng, path=None):
        self.rng = rng
        path = path or os.path.join(HERE, "tinyshakespeare.txt")
        if not os.path.exists(path):
            print("downloading tiny shakespeare ->", path)
            urllib.request.urlretrieve(self.URL, path)
        text = open(path, encoding="utf-8").read()
        self.vocab = sorted(set(text))
        self.cfg = dict(self.cfg, vocab_size=len(self.vocab))
        stoi = {c: i for i, c in enumerate(self.vocab)}
        data = np.array([stoi[c] for c in text], dtype=np.int32)
        n = int(0.9 * len(data))
        self.train, self.val = data[:n], data[n:]
        self.text_head = text[:2000]
    def batch(self, B, split="train"):
        d = self.train if split == "train" else self.val
        T = self.cfg["block_size"]
        ix = self.rng.integers(0, len(d) - T - 1, size=B)
        ids = np.stack([d[i:i + T] for i in ix]); targets = np.stack([d[i + 1:i + T + 1] for i in ix])
        return ids, targets, np.ones_like(ids, dtype=np.float32)
    def example(self):
        stoi = {c: i for i, c in enumerate(self.vocab)}
        return [stoi[c] for c in "First Citizen:\nBefore we proceed any further, hear me speak"]
    def describe(self):
        return ("Character-level language model trained on Karpathy's tiny-shakespeare corpus "
                "(1.1 MB of plays). Given the characters so far it predicts the next character.")

# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

TENSOR_ORDER = ["ln1_g", "ln1_b", "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo",
                "ln2_g", "ln2_b", "win", "bin", "wout", "bout"]

def export(task, cfg, params, log, out_dir=OUT):
    os.makedirs(out_dir, exist_ok=True)
    table, chunks, offset = [], [], 0
    def add(name, arr):
        nonlocal offset
        a = np.asarray(arr, dtype=np.float32)
        table.append({"name": name, "shape": list(a.shape), "offset": offset})
        chunks.append(a.ravel()); offset += a.size
    add("tok_emb", params["tok_emb"]); add("pos_emb", params["pos_emb"])
    for i in range(cfg["n_layer"]):
        for k in TENSOR_ORDER: add(f"h{i}.{k}", params[f"h{i}"][k])
    add("lnf_g", params["lnf_g"]); add("lnf_b", params["lnf_b"])
    blob = np.concatenate(chunks).astype("<f4")
    ids = task.example()[:cfg["block_size"]]
    logits = np.asarray(forward(params, cfg, jnp.array([ids])))[0]
    meta = {
        "name": task.name, "title": task.title, "description": task.describe(),
        "task": "sort" if task.name == "sort" else "lm",
        "config": cfg, "vocab": task.vocab, "n_params": int(blob.size),
        "tensors": table, "weights_file": f"{task.name}.bin",
        "training": log,
        "verify": {"ids": [int(i) for i in ids], "logits": np.round(logits, 5).tolist()},
    }
    if task.name == "sort": meta["n_in"] = task.n_in
    with open(os.path.join(out_dir, f"{task.name}.bin"), "wb") as f: f.write(blob.tobytes())
    with open(os.path.join(out_dir, f"{task.name}.json"), "w") as f: json.dump(meta, f)
    print(f"exported {task.name}: {blob.size} params -> {out_dir}/{task.name}.json/.bin")

# --------------------------------------------------------------------------
# sampling (for a quick sanity check printed at the end)
# --------------------------------------------------------------------------

def sample(params, cfg, vocab, prompt_ids, n, rng, temp=0.8):
    ids = list(prompt_ids)
    for _ in range(n):
        ctx = ids[-cfg["block_size"]:]
        logits = np.asarray(forward(params, cfg, jnp.array([ctx])))[0, -1] / temp
        p = np.exp(logits - logits.max()); p /= p.sum()
        ids.append(int(rng.choice(len(vocab), p=p)))
    return "".join(vocab[i] for i in ids)

# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sort", "shakespeare"], required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    task = SortTask(rng) if args.task == "sort" else ShakespeareTask(rng)
    cfg = task.cfg
    steps = args.steps or (3000 if args.task == "sort" else 8000)
    B = args.batch or (128 if args.task == "sort" else 64)
    wd = 0.1 if args.task == "shakespeare" else 0.01
    warmup = 100

    params = init_params(cfg, jax.random.PRNGKey(args.seed))
    n_params = sum(int(np.prod(l.shape)) for l in jax.tree_util.tree_leaves(params))
    print(f"task={task.name} cfg={cfg} params={n_params:,} steps={steps} batch={B}")
    opt = {"m": jax.tree_util.tree_map(jnp.zeros_like, params),
           "v": jax.tree_util.tree_map(jnp.zeros_like, params), "t": jnp.array(0, dtype=jnp.int32)}
    step = make_step(cfg, wd, clip=1.0)
    eval_loss = jax.jit(lambda p, i, t, m: loss_fn(p, cfg, i, t, m))

    curve, t0 = [], time.time()
    for it in range(steps):
        ids, targets, mask = task.batch(B)
        lr = lr_at(it, args.lr, warmup, steps)
        params, opt, loss, gnorm = step(params, opt, jnp.array(ids), jnp.array(targets), jnp.array(mask), lr)
        if it % 100 == 0 or it == steps - 1:
            vi, vt, vm = task.batch(256, "val")
            vl = float(eval_loss(params, jnp.array(vi), jnp.array(vt), jnp.array(vm)))
            curve.append([it, round(float(loss), 4), round(vl, 4)])
            print(f"step {it:5d}  loss {float(loss):.4f}  val {vl:.4f}  lr {lr:.2e}  |g| {float(gnorm):.2f}  {time.time()-t0:6.1f}s", flush=True)

    minutes = (time.time() - t0) / 60
    log = {"steps": steps, "batch": B, "peak_lr": args.lr, "weight_decay": wd,
           "final_train_loss": curve[-1][1], "final_val_loss": curve[-1][2],
           "minutes": round(minutes, 2), "curve": curve, "optimizer": "AdamW(0.9,0.95) warmup+cosine, clip 1.0"}

    if task.name == "sort":
        ids, targets, mask = task.batch(2000, "val")
        pred = np.asarray(forward(params, cfg, jnp.array(ids))).argmax(-1)
        acc = ((pred == targets) | (mask == 0)).all(axis=1).mean()
        log["sequence_accuracy"] = float(acc)
        print(f"sequence accuracy on 2000 fresh inputs: {acc:.4f}")
        for row in ids[:3]:
            p = np.asarray(forward(params, cfg, jnp.array([row])))[0].argmax(-1)
            print("  in:", "".join(task.vocab[i] for i in row[:6]),
                  " model sorted:", "".join(task.vocab[i] for i in p[5:11]))
    else:
        txt = sample(params, cfg, task.vocab, task.example()[:40], 300, rng)
        log["sample"] = txt
        print("---- sample ----\n" + txt + "\n----------------")
    export(task, cfg, params, log, args.out)

if __name__ == "__main__":
    main()

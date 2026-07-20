// A tiny GPT-style transformer, implemented from scratch in plain JS.
//
// Everything the UI shows is genuinely computed by this forward pass:
// token embeddings, sinusoidal positions, multi-head causal self-attention
// (per-head weight matrices captured for the heatmaps), GELU MLP activations,
// residual stream snapshots, layer norms, and tied-embedding logits.
//
// Honesty note (also shown in the UI): the transformer weights are random —
// nothing here was trained by gradient descent. Two things ARE learned from
// the corpus at load time: the embedding geometry (PPMI co-occurrence, so
// "cat" really is near "dog") and a bigram language prior that is blended
// into the final logits so generated text is readable. The mechanics you
// watch — the math of attention and MLPs — are exactly what a real model
// does; only the numbers inside the weight matrices differ.

const MODEL_CONFIG = {
  dModel: 48,
  nHeads: 4,
  dHead: 12,
  nLayers: 3,
  dFF: 96,
  ctx: 24,
  seed: 20260720,
  embedScale: 3.0,
  posScale: 1.0,
  // logit blend: model forward pass + corpus bigram prior
  wModel: 0.35,
  wBigram: 2.0,
};

function zeros2(r, c) {
  const m = new Array(r);
  for (let i = 0; i < r; i++) m[i] = new Float32Array(c);
  return m;
}

function gaussMat(r, c, std, gauss) {
  const m = zeros2(r, c);
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) m[i][j] = gauss() * std;
  return m;
}

function layerNorm(x) {
  const d = x.length;
  let mean = 0;
  for (let i = 0; i < d; i++) mean += x[i];
  mean /= d;
  let v = 0;
  for (let i = 0; i < d; i++) { const t = x[i] - mean; v += t * t; }
  const inv = 1 / Math.sqrt(v / d + 1e-5);
  const out = new Float32Array(d);
  for (let i = 0; i < d; i++) out[i] = (x[i] - mean) * inv;
  return out;
}

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}

function vecMat(x, W) { // x:[r], W:[r][c] -> [c]
  const r = W.length, c = W[0].length;
  const out = new Float32Array(c);
  for (let i = 0; i < r; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    const row = W[i];
    for (let j = 0; j < c; j++) out[j] += xi * row[j];
  }
  return out;
}

function softmaxInPlace(a) {
  let mx = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > mx) mx = a[i];
  let s = 0;
  for (let i = 0; i < a.length; i++) { a[i] = Math.exp(a[i] - mx); s += a[i]; }
  for (let i = 0; i < a.length; i++) a[i] /= s;
  return a;
}

function l2(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s);
}

class TinyGPT {
  constructor(tokenizer, corpusText, cfg = MODEL_CONFIG) {
    this.tok = tokenizer;
    this.cfg = cfg;
    const V = tokenizer.size;
    this.V = V;

    const rand = RNG.mulberry32(cfg.seed);
    const gauss = RNG.makeGaussian(rand);

    // ---- corpus statistics: bigram prior + co-occurrence embeddings ----
    const stream = tokenizer.tokenize(corpusText).ids;
    this.corpusStream = stream;

    this.uniCount = new Float32Array(V);
    this.biCount = new Float32Array(V * V);
    for (let i = 0; i < stream.length; i++) {
      this.uniCount[stream[i]]++;
      if (i > 0) this.biCount[stream[i - 1] * V + stream[i]]++;
    }
    this._bigramCache = new Map();

    // PPMI co-occurrence (window +-2), projected to dModel with a seeded
    // random matrix -> embeddings whose geometry reflects real word usage.
    const cooc = new Float32Array(V * V);
    const W = 2;
    for (let i = 0; i < stream.length; i++) {
      for (let k = 1; k <= W; k++) {
        if (i + k < stream.length) {
          cooc[stream[i] * V + stream[i + k]]++;
          cooc[stream[i + k] * V + stream[i]]++;
        }
      }
    }
    let total = 0;
    const rowSum = new Float32Array(V);
    for (let i = 0; i < V; i++) {
      let s = 0;
      for (let j = 0; j < V; j++) s += cooc[i * V + j];
      rowSum[i] = s; total += s;
    }
    const d = cfg.dModel;
    const proj = gaussMat(V, d, 1 / Math.sqrt(d), gauss);
    const E = zeros2(V, d);
    for (let i = 0; i < V; i++) {
      if (rowSum[i] === 0) continue;
      for (let j = 0; j < V; j++) {
        const c = cooc[i * V + j];
        if (c === 0) continue;
        const pmi = Math.log((c * total) / (rowSum[i] * rowSum[j]));
        if (pmi <= 0) continue;
        const row = proj[j];
        for (let k = 0; k < d; k++) E[i][k] += pmi * row[k];
      }
    }
    // Normalize rows; unseen tokens get small random vectors.
    for (let i = 0; i < V; i++) {
      let n = l2(E[i]);
      if (n < 1e-6) {
        for (let k = 0; k < d; k++) E[i][k] = gauss() * 0.4;
        n = l2(E[i]);
      }
      for (let k = 0; k < d; k++) E[i][k] = E[i][k] / n + gauss() * 0.02;
    }
    this.E = E;

    // ---- sinusoidal positional encoding ----
    this.P = zeros2(cfg.ctx, d);
    for (let t = 0; t < cfg.ctx; t++) {
      for (let k = 0; k < d / 2; k++) {
        const w = 1 / Math.pow(10000, (2 * k) / d);
        this.P[t][2 * k] = Math.sin(t * w);
        this.P[t][2 * k + 1] = Math.cos(t * w);
      }
    }

    // ---- transformer weights (random, seeded) ----
    const H = cfg.nHeads, dh = cfg.dHead;
    this.layers = [];
    for (let l = 0; l < cfg.nLayers; l++) {
      const heads = [];
      for (let h = 0; h < H; h++) {
        heads.push({
          Wq: gaussMat(d, dh, 0.20, gauss),
          Wk: gaussMat(d, dh, 0.20, gauss),
          Wv: gaussMat(d, dh, 0.12, gauss),
        });
      }
      this.layers.push({
        heads,
        Wo: gaussMat(d, d, 0.12, gauss),
        W1: gaussMat(d, cfg.dFF, 0.15, gauss),
        W2: gaussMat(cfg.dFF, d, 0.15, gauss),
      });
    }

    // ---- 2D PCA of the embedding table (for the vector-space map) ----
    this.coords2d = this._pca2d();
  }

  _pca2d() {
    const { E, V } = this;
    const d = this.cfg.dModel;
    // Fit PCA on tokens that actually occur in the corpus.
    const idx = [];
    for (let i = 0; i < V; i++) if (this.uniCount[i] >= 2) idx.push(i);
    const mean = new Float32Array(d);
    for (const i of idx) for (let k = 0; k < d; k++) mean[k] += E[i][k];
    for (let k = 0; k < d; k++) mean[k] /= Math.max(idx.length, 1);
    // covariance
    const C = zeros2(d, d);
    for (const i of idx) {
      for (let a = 0; a < d; a++) {
        const va = E[i][a] - mean[a];
        for (let b = 0; b < d; b++) C[a][b] += va * (E[i][b] - mean[b]);
      }
    }
    const power = (deflate) => {
      let v = new Float32Array(d);
      for (let k = 0; k < d; k++) v[k] = Math.sin(k * 12.9898 + 4.1414) * 0.5 + 0.5;
      for (let it = 0; it < 60; it++) {
        let nv = vecMat(v, C);
        if (deflate) {
          let dot = 0;
          for (let k = 0; k < d; k++) dot += nv[k] * deflate[k];
          for (let k = 0; k < d; k++) nv[k] -= dot * deflate[k];
        }
        const n = l2(nv) || 1;
        for (let k = 0; k < d; k++) nv[k] /= n;
        v = nv;
      }
      return v;
    };
    const v1 = power(null);
    const v2 = power(v1);
    const coords = new Array(this.V);
    for (let i = 0; i < this.V; i++) {
      let x = 0, y = 0;
      for (let k = 0; k < d; k++) {
        const c = E[i][k] - mean[k];
        x += c * v1[k]; y += c * v2[k];
      }
      coords[i] = [x, y];
    }
    return coords;
  }

  nearestNeighbors(id, n = 5) {
    const e = this.E[id];
    const en = l2(e);
    const sims = [];
    for (let j = 0; j < this.V; j++) {
      if (j === id || this.uniCount[j] < 2) continue;
      let dot = 0;
      for (let k = 0; k < e.length; k++) dot += e[k] * this.E[j][k];
      sims.push([j, dot / (en * l2(this.E[j]) + 1e-9)]);
    }
    sims.sort((a, b) => b[1] - a[1]);
    return sims.slice(0, n);
  }

  bigramLogProbs(prev) {
    const hit = this._bigramCache.get(prev);
    if (hit) return hit;
    const V = this.V, alpha = 0.35;
    let rowTotal = 0;
    for (let j = 0; j < V; j++) rowTotal += this.biCount[prev * V + j];
    const out = new Float32Array(V);
    const denom = rowTotal + alpha * V;
    for (let j = 0; j < V; j++) {
      // back off toward unigram counts so unseen bigrams still prefer
      // common tokens over random subword fragments
      const c = this.biCount[prev * V + j] + alpha * (0.05 + this.uniCount[j] / this.corpusStream.length * 40);
      out[j] = Math.log(c / denom);
    }
    this._bigramCache.set(prev, out);
    return out;
  }

  // Full instrumented forward pass over the last `ctx` tokens.
  // Returns a trace object with every intermediate the UI visualizes.
  forward(idsFull) {
    const cfg = this.cfg;
    const ids = idsFull.slice(-cfg.ctx);
    const T = ids.length, d = cfg.dModel, H = cfg.nHeads, dh = cfg.dHead;
    if (T === 0) return null;

    const embedded = zeros2(T, d);   // content part
    const posPart = zeros2(T, d);    // position part
    const x = zeros2(T, d);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < d; k++) {
        embedded[t][k] = this.E[ids[t]][k] * cfg.embedScale;
        posPart[t][k] = this.P[t][k] * cfg.posScale;
        x[t][k] = embedded[t][k] + posPart[t][k];
      }
    }

    const trace = {
      ids, T,
      tokens: ids.map((i) => this.tok.idToToken(i)),
      embedded, posPart,
      resStream: [x.map((r) => Float32Array.from(r))], // snapshot per block
      layers: [],
    };

    for (let l = 0; l < cfg.nLayers; l++) {
      const L = this.layers[l];
      const lnA = x.map(layerNorm);
      // attention
      const attnPerHead = []; // [H][T][T]
      const headOuts = [];    // [H][T][dh]
      for (let h = 0; h < H; h++) {
        const { Wq, Wk, Wv } = L.heads[h];
        const q = lnA.map((r) => vecMat(r, Wq));
        const k = lnA.map((r) => vecMat(r, Wk));
        const v = lnA.map((r) => vecMat(r, Wv));
        const A = zeros2(T, T);
        for (let t = 0; t < T; t++) {
          const row = new Float32Array(t + 1);
          for (let s = 0; s <= t; s++) {
            let dot = 0;
            for (let z = 0; z < dh; z++) dot += q[t][z] * k[s][z];
            row[s] = dot / Math.sqrt(dh);
          }
          softmaxInPlace(row);
          for (let s = 0; s <= t; s++) A[t][s] = row[s];
        }
        attnPerHead.push(A);
        const out = zeros2(T, dh);
        for (let t = 0; t < T; t++)
          for (let s = 0; s <= t; s++) {
            const w = A[t][s];
            for (let z = 0; z < dh; z++) out[t][z] += w * v[s][z];
          }
        headOuts.push(out);
      }
      // concat + project + residual
      const attnOut = zeros2(T, d);
      for (let t = 0; t < T; t++) {
        const cat = new Float32Array(d);
        for (let h = 0; h < H; h++)
          for (let z = 0; z < dh; z++) cat[h * dh + z] = headOuts[h][t][z];
        const o = vecMat(cat, L.Wo);
        for (let k2 = 0; k2 < d; k2++) attnOut[t][k2] = o[k2];
      }
      for (let t = 0; t < T; t++)
        for (let k2 = 0; k2 < d; k2++) x[t][k2] += attnOut[t][k2];
      const afterAttn = x.map((r) => Float32Array.from(r));

      // MLP
      const lnM = x.map(layerNorm);
      const mlpHidden = zeros2(T, cfg.dFF);
      for (let t = 0; t < T; t++) {
        const pre = vecMat(lnM[t], L.W1);
        for (let j = 0; j < cfg.dFF; j++) mlpHidden[t][j] = gelu(pre[j]);
        const back = vecMat(mlpHidden[t], L.W2);
        for (let k2 = 0; k2 < d; k2++) x[t][k2] += back[k2];
      }
      const afterMlp = x.map((r) => Float32Array.from(r));

      trace.layers.push({ attn: attnPerHead, attnOut, afterAttn, mlpHidden, afterMlp });
      trace.resStream.push(afterMlp);
    }

    // final layer norm + tied-embedding logits for the LAST position
    const fin = layerNorm(x[T - 1]);
    trace.final = fin;
    const modelLogits = new Float32Array(this.V);
    for (let vI = 0; vI < this.V; vI++) {
      let dot = 0;
      const row = this.E[vI];
      for (let k = 0; k < d; k++) dot += fin[k] * row[k];
      modelLogits[vI] = dot / Math.sqrt(d);
    }
    const bigram = this.bigramLogProbs(ids[T - 1]);
    const logits = new Float32Array(this.V);
    for (let vI = 0; vI < this.V; vI++)
      logits[vI] = cfg.wModel * modelLogits[vI] + cfg.wBigram * bigram[vI];
    trace.modelLogits = modelLogits;
    trace.bigramLogits = bigram;
    trace.logits = logits;
    return trace;
  }

  // Turn last-position logits into a sorted top-k distribution.
  // recentIds + repPenalty implement a standard repetition penalty:
  // tokens that appeared recently get their logits pushed down.
  distribution(logits, temperature = 0.8, topK = 10, recentIds = null, repPenalty = 0) {
    if (recentIds && repPenalty > 0) {
      logits = Float32Array.from(logits);
      const seen = new Set(recentIds);
      for (const id of seen) logits[id] -= repPenalty;
    }
    const idxs = Array.from(logits.keys());
    idxs.sort((a, b) => logits[b] - logits[a]);
    const top = idxs.slice(0, topK);
    const t = Math.max(temperature, 0.05);
    const scaled = top.map((i) => logits[i] / t);
    const mx = Math.max(...scaled);
    let s = 0;
    const exps = scaled.map((v) => { const e = Math.exp(v - mx); s += e; return e; });
    return top.map((id, i) => ({ id, token: this.tok.idToToken(id), p: exps[i] / s }));
  }

  sampleFrom(dist, rand) {
    let r = rand();
    for (const d of dist) {
      r -= d.p;
      if (r <= 0) return d;
    }
    return dist[dist.length - 1];
  }
}

// gpt.js — a GPT-2-style transformer forward pass in plain JavaScript,
// instrumented so that EVERY intermediate tensor is kept, and every single
// number can be re-derived from its inputs ("explain" helpers).
//
// Architecture (identical to GPT-2 / nanoGPT / the Python trainer):
//   x0      = tok_emb[ids] + pos_emb[0..T)
//   per layer:
//     a     = LN1(x)                      ln1
//     q,k,v = a·Wq+bq, a·Wk+bk, a·Wv+bv   (columns grouped per head)
//     S     = q_h·k_hᵀ / √d_head           scores (causal: s > t masked)
//     A     = softmax_s(S)                 attn
//     o_h   = A_h · v_h  → concat heads   headOut
//     x     = x + o·Wo + bo               x1 (residual add)
//     b     = LN2(x)                      ln2
//     m     = GELU(b·Win + bin)           mlpPre → mlpAct
//     x     = x + m·Wout + bout           x2 (residual add)
//   logits  = LN_f(x) · tok_embᵀ          (tied unembedding, like GPT-2)
//
// Matrices are stored flat, row-major: Mat {r, c, data}, element (i,j) at i*c+j.
// A linear layer y = x·W has W of shape [in, out].

export class Mat {
  constructor(r, c, data) { this.r = r; this.c = c; this.data = data || new Float32Array(r * c); }
  get(i, j) { return this.data[i * this.c + j]; }
  set(i, j, v) { this.data[i * this.c + j] = v; }
  row(i) { return this.data.subarray(i * this.c, (i + 1) * this.c); }
  static from(arr, r, c) { return new Mat(r, c, arr); }
}

export const SQRT2_OVER_PI = 0.7978845608028654;
export function gelu(x) { return 0.5 * x * (1 + Math.tanh(SQRT2_OVER_PI * (x + 0.044715 * x * x * x))); }

export function softmaxRow(a) {
  let mx = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > mx) mx = a[i];
  const out = new Float32Array(a.length);
  let s = 0;
  for (let i = 0; i < a.length; i++) { const e = Math.exp(a[i] - mx); out[i] = e; s += e; }
  for (let i = 0; i < a.length; i++) out[i] /= s;
  return out;
}

// y = x·W + b   (x: Mat[T,in], W: Mat[in,out], b: Float32Array[out] | null)
export function linear(x, W, b) {
  const T = x.r, n = W.r, m = W.c;
  if (x.c !== n) throw new Error(`linear: ${x.c} != ${n}`);
  const y = new Mat(T, m);
  const xd = x.data, wd = W.data, yd = y.data;
  for (let t = 0; t < T; t++) {
    const yo = t * m;
    if (b) for (let j = 0; j < m; j++) yd[yo + j] = b[j];
    for (let i = 0; i < n; i++) {
      const xv = xd[t * n + i];
      if (xv === 0) continue;
      const wo = i * m;
      for (let j = 0; j < m; j++) yd[yo + j] += xv * wd[wo + j];
    }
  }
  return y;
}

export function layerNorm(x, g, b, eps) {
  const T = x.r, d = x.c, y = new Mat(T, d);
  const mean = new Float32Array(T), rstd = new Float32Array(T);
  for (let t = 0; t < T; t++) {
    const row = x.row(t);
    let m = 0; for (let j = 0; j < d; j++) m += row[j]; m /= d;
    let v = 0; for (let j = 0; j < d; j++) { const z = row[j] - m; v += z * z; } v /= d;
    const rs = 1 / Math.sqrt(v + eps);
    mean[t] = m; rstd[t] = rs;
    const out = y.row(t);
    for (let j = 0; j < d; j++) out[j] = (row[j] - m) * rs * g[j] + b[j];
  }
  return { y, mean, rstd };
}

export class GPT {
  // meta: parsed <model>.json; weights: Float32Array of the whole .bin
  constructor(meta, weights) {
    this.meta = meta;
    this.cfg = meta.config;
    this.vocab = meta.vocab;
    this.tensors = new Map();
    for (const t of meta.tensors) {
      const size = t.shape.reduce((a, b) => a * b, 1);
      const data = weights.subarray(t.offset, t.offset + size);
      this.tensors.set(t.name, t.shape.length === 2 ? Mat.from(data, t.shape[0], t.shape[1]) : data);
    }
    const { d_model, n_head, n_layer, d_ff, vocab_size, block_size } = this.cfg;
    this.d = d_model; this.H = n_head; this.dh = d_model / n_head; this.L = n_layer;
    this.dff = d_ff; this.V = vocab_size; this.T_max = block_size;
    this.eps = this.cfg.ln_eps || 1e-5;
    this.stoi = new Map(this.vocab.map((c, i) => [c, i]));
  }
  t(name) { return this.tensors.get(name); }
  layerT(l, name) { return this.tensors.get(`h${l}.${name}`); }
  nParams() { return this.meta.n_params; }

  encode(text) { const out = []; for (const ch of text) { const i = this.stoi.get(ch); if (i !== undefined) out.push(i); } return out; }
  decode(ids) { return ids.map((i) => this.vocab[i]).join(""); }

  // Full instrumented forward pass. ids: array of ints, length ≤ block_size.
  forward(ids) {
    const T = ids.length, d = this.d, H = this.H, dh = this.dh, V = this.V;
    if (T === 0) throw new Error("empty input");
    if (T > this.T_max) throw new Error(`input longer than context (${T} > ${this.T_max})`);
    const E = this.t("tok_emb"), P = this.t("pos_emb");
    const tokRows = new Mat(T, d), posRows = new Mat(T, d), x0 = new Mat(T, d);
    for (let t = 0; t < T; t++) {
      const er = E.row(ids[t]), pr = P.row(t);
      for (let j = 0; j < d; j++) { tokRows.data[t * d + j] = er[j]; posRows.data[t * d + j] = pr[j]; x0.data[t * d + j] = er[j] + pr[j]; }
    }
    const layers = [];
    let x = x0;
    const scale = 1 / Math.sqrt(dh);
    for (let l = 0; l < this.L; l++) {
      const L = {};
      L.xin = x;
      const ln1 = layerNorm(x, this.layerT(l, "ln1_g"), this.layerT(l, "ln1_b"), this.eps);
      L.ln1 = ln1.y; L.ln1_mean = ln1.mean; L.ln1_rstd = ln1.rstd;
      L.q = linear(ln1.y, this.layerT(l, "wq"), this.layerT(l, "bq"));
      L.k = linear(ln1.y, this.layerT(l, "wk"), this.layerT(l, "bk"));
      L.v = linear(ln1.y, this.layerT(l, "wv"), this.layerT(l, "bv"));
      // scores[h] and attn[h]: Mat[T,T]; masked entries are -Infinity / 0
      L.scores = []; L.attn = [];
      L.headOut = new Mat(T, d);
      for (let h = 0; h < H; h++) {
        const S = new Mat(T, T), A = new Mat(T, T);
        S.data.fill(-Infinity);
        const c0 = h * dh;
        for (let t = 0; t < T; t++) {
          const qr = L.q.row(t);
          for (let s = 0; s <= t; s++) {
            const kr = L.k.row(s);
            let dot = 0;
            for (let i = 0; i < dh; i++) dot += qr[c0 + i] * kr[c0 + i];
            S.set(t, s, dot * scale);
          }
          const sm = softmaxRow(S.row(t).subarray(0, t + 1));
          for (let s = 0; s <= t; s++) A.set(t, s, sm[s]);
          // weighted sum of value vectors
          const o = L.headOut.row(t);
          for (let s = 0; s <= t; s++) {
            const a = sm[s]; if (a === 0) continue;
            const vr = L.v.row(s);
            for (let i = 0; i < dh; i++) o[c0 + i] += a * vr[c0 + i];
          }
        }
        L.scores.push(S); L.attn.push(A);
      }
      L.attnOut = linear(L.headOut, this.layerT(l, "wo"), this.layerT(l, "bo"));
      L.x1 = new Mat(T, d);
      for (let i = 0; i < T * d; i++) L.x1.data[i] = x.data[i] + L.attnOut.data[i];
      const ln2 = layerNorm(L.x1, this.layerT(l, "ln2_g"), this.layerT(l, "ln2_b"), this.eps);
      L.ln2 = ln2.y; L.ln2_mean = ln2.mean; L.ln2_rstd = ln2.rstd;
      L.mlpPre = linear(ln2.y, this.layerT(l, "win"), this.layerT(l, "bin"));
      L.mlpAct = new Mat(T, this.dff);
      for (let i = 0; i < L.mlpPre.data.length; i++) L.mlpAct.data[i] = gelu(L.mlpPre.data[i]);
      L.mlpOut = linear(L.mlpAct, this.layerT(l, "wout"), this.layerT(l, "bout"));
      L.x2 = new Mat(T, d);
      for (let i = 0; i < T * d; i++) L.x2.data[i] = L.x1.data[i] + L.mlpOut.data[i];
      layers.push(L);
      x = L.x2;
    }
    const lnf = layerNorm(x, this.t("lnf_g"), this.t("lnf_b"), this.eps);
    const logits = this.unembed(lnf.y);
    const probs = new Mat(T, V);
    for (let t = 0; t < T; t++) probs.row(t).set(softmaxRow(logits.row(t)));
    return {
      ids, T, tokens: ids.map((i) => this.vocab[i]), tokRows, posRows, x0, layers,
      lnf: lnf.y, lnf_mean: lnf.mean, lnf_rstd: lnf.rstd, logits, probs,
    };
  }

  // logits = x · Eᵀ   (x: Mat[T,d])
  unembed(x) {
    const E = this.t("tok_emb"), T = x.r, d = this.d, V = this.V;
    const out = new Mat(T, V);
    for (let t = 0; t < T; t++) {
      const xr = x.row(t);
      for (let v = 0; v < V; v++) {
        const er = E.row(v); let s = 0;
        for (let j = 0; j < d; j++) s += xr[j] * er[j];
        out.data[t * V + v] = s;
      }
    }
    return out;
  }

  // "Logit lens": what would the model predict if we stopped after layer l
  // (l = -1 means right after the embeddings) and applied LN_f + unembedding?
  logitLens(trace, l) {
    const x = l < 0 ? trace.x0 : trace.layers[l].x2;
    const ln = layerNorm(x, this.t("lnf_g"), this.t("lnf_b"), this.eps);
    const logits = this.unembed(ln.y);
    const probs = new Mat(x.r, this.V);
    for (let t = 0; t < x.r; t++) probs.row(t).set(softmaxRow(logits.row(t)));
    return { logits, probs };
  }

  // Sampling with temperature / top-k. rng: () => [0,1)
  sampleNext(logitsRow, { temperature = 1, topk = 0, rng = Math.random } = {}) {
    const V = logitsRow.length;
    const z = new Float32Array(V);
    const tmp = Math.max(temperature, 1e-4);
    for (let i = 0; i < V; i++) z[i] = logitsRow[i] / tmp;
    if (topk > 0 && topk < V) {
      const idx = Array.from({ length: V }, (_, i) => i).sort((a, b) => z[b] - z[a]);
      const cut = z[idx[topk - 1]];
      for (let i = 0; i < V; i++) if (z[i] < cut) z[i] = -Infinity;
    }
    const p = softmaxRow(z);
    let r = rng(), acc = 0, pick = V - 1;
    for (let i = 0; i < V; i++) { acc += p[i]; if (r < acc) { pick = i; break; } }
    return { id: pick, probs: p };
  }

  // ---------------------------------------------------------------------
  // EXPLAIN helpers: re-derive one number from its inputs, returning the
  // individual terms so the UI can show "where did this value come from?".
  // ---------------------------------------------------------------------

  // y[t][j] = Σ_i x[t][i]·W[i][j] + b[j]
  explainLinear(x, W, b, t, j) {
    const n = W.r, terms = new Array(n);
    let sum = b ? b[j] : 0;
    for (let i = 0; i < n; i++) {
      const xv = x.get(t, i), wv = W.get(i, j), prod = xv * wv;
      terms[i] = { i, x: xv, w: wv, prod }; sum += prod;
    }
    return { terms, bias: b ? b[j] : 0, sum };
  }
  // score[h][t][s] = (q_t · k_s) / √d_head   restricted to head h's columns
  explainScore(trace, l, h, t, s) {
    const L = trace.layers[l], c0 = h * this.dh, terms = [];
    let dot = 0;
    for (let i = 0; i < this.dh; i++) {
      const q = L.q.get(t, c0 + i), k = L.k.get(s, c0 + i), prod = q * k;
      terms.push({ i, q, k, prod }); dot += prod;
    }
    return { terms, dot, scale: 1 / Math.sqrt(this.dh), value: dot / Math.sqrt(this.dh) };
  }
  // attn[h][t][s] = exp(S[t][s] - max) / Σ_{s'≤t} exp(S[t][s'] - max)
  explainSoftmax(trace, l, h, t, s) {
    const S = trace.layers[l].scores[h];
    let mx = -Infinity; for (let i = 0; i <= t; i++) mx = Math.max(mx, S.get(t, i));
    const exps = []; let denom = 0;
    for (let i = 0; i <= t; i++) { const e = Math.exp(S.get(t, i) - mx); exps.push({ s: i, score: S.get(t, i), e }); denom += e; }
    return { exps, max: mx, denom, value: Math.exp(S.get(t, s) - mx) / denom };
  }
  // headOut[t][h*dh+i] = Σ_{s≤t} attn[h][t][s] · v[s][h*dh+i]
  explainHeadOut(trace, l, t, col) {
    const h = Math.floor(col / this.dh), L = trace.layers[l], A = L.attn[h], terms = [];
    let sum = 0;
    for (let s = 0; s <= t; s++) { const a = A.get(t, s), v = L.v.get(s, col), prod = a * v; terms.push({ s, a, v, prod }); sum += prod; }
    return { h, terms, sum };
  }
  // ln[t][j] = (x[t][j] - mean_t) · rstd_t · g[j] + b[j]
  explainLayerNorm(x, mean, rstd, g, b, t, j) {
    const xv = x.get(t, j);
    return { x: xv, mean: mean[t], rstd: rstd[t], normalized: (xv - mean[t]) * rstd[t], g: g[j], b: b[j], value: (xv - mean[t]) * rstd[t] * g[j] + b[j] };
  }
  explainGelu(pre) { return { pre, value: gelu(pre), inner: SQRT2_OVER_PI * (pre + 0.044715 * pre ** 3) }; }
  // logit[t][v] = lnf[t] · E[v]
  explainLogit(trace, t, v) {
    const E = this.t("tok_emb"), terms = []; let sum = 0;
    for (let j = 0; j < this.d; j++) { const x = trace.lnf.get(t, j), e = E.get(v, j), prod = x * e; terms.push({ j, x, e, prod }); sum += prod; }
    return { terms, sum };
  }
  explainProb(trace, t, v) {
    const row = trace.logits.row(t); let mx = -Infinity; for (let i = 0; i < row.length; i++) mx = Math.max(mx, row[i]);
    let denom = 0; for (let i = 0; i < row.length; i++) denom += Math.exp(row[i] - mx);
    return { logit: row[v], max: mx, denom, value: Math.exp(row[v] - mx) / denom };
  }
  topK(probRow, k = 8) {
    const idx = Array.from({ length: probRow.length }, (_, i) => i).sort((a, b) => probRow[b] - probRow[a]).slice(0, k);
    return idx.map((i) => ({ id: i, token: this.vocab[i], p: probRow[i] }));
  }
}

// Convenience loader for browsers (fetch) — Node tests build GPT directly.
export async function loadGPT(jsonUrl) {
  const meta = await (await fetch(jsonUrl)).json();
  const binUrl = jsonUrl.replace(/[^/]*$/, meta.weights_file);
  const buf = await (await fetch(binUrl)).arrayBuffer();
  return new GPT(meta, new Float32Array(buf));
}

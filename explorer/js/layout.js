// layout.js — turns a model + forward-pass trace into a list of 3D "blocks"
// (one per tensor) with positions, value accessors, colour scales and the
// dependency rules used for "where did this number come from?" highlighting.
//
// Geometry conventions (one cell = one unit):
//   columns run along +x, rows along +z (row 0 at the back, the latest token
//   nearest the viewer), and the pipeline flows DOWN the y axis.
//   Activations (depend on your input) sit in the centre column; the learned
//   weight matrices sit in a lane to the LEFT of the activation they produce.

import { robustMax } from "./colors.js";

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

export function buildLayout(model, trace, opts = {}) {
  const T = trace.T, d = model.d, H = model.H, dh = model.dh, dff = model.dff, V = model.V, L = model.L;
  const pitch = opts.pitch || clamp(T * 0.62, 9, 44);     // vertical distance between levels
  const laneGap = 10, hgap = 4, headGap = 1.5;
  const blocks = [];
  const byId = new Map();
  const groups = [];
  let y = 0;
  const mark = (label) => groups.push({ label, y: y + 4, x: -(T + 60), z: 0 });

  const mkBlock = (b) => {
    b.rows = b.mat.r; b.cols = b.mat.c;
    b.vmax = b.vmax || robustMax(b.mat.data);
    b.colorMode = b.colorMode || "div";
    b.headGroups = b.headGroups || 0;
    b.width = b.cols + (b.headGroups ? (b.headGroups - 1) * headGap : 0);
    b.depth = b.rows;
    blocks.push(b); byId.set(b.id, b);
    return b;
  };
  // x offset of column j inside a block (accounts for head gaps)
  const colX = (b, j) => j + (b.headGroups ? Math.floor(j / (b.cols / b.headGroups)) * headGap : 0);

  // Place a row of sibling activation blocks centred on x = 0 at level y,
  // and their weights in the lane to the left. Returns nothing; mutates blocks.
  function level(acts, weights = [], opt = {}) {
    const total = acts.reduce((s, b) => s + b.width, 0) + hgap * (acts.length - 1);
    let x = -total / 2;
    for (const b of acts) { b.x = x; b.y = y; b.z = -b.depth / 2; x += b.width + hgap; }
    let wx = -total / 2 - laneGap;
    for (const w of weights) { w.x = wx - w.width; w.y = y; w.z = -w.depth / 2; wx -= w.width + hgap; }
    y -= opt.pitch || pitch;
  }
  const act = (id, label, mat, extra = {}) => mkBlock({ id, label, kind: "act", mat, ...extra });
  const weight = (id, label, mat, extra = {}) => mkBlock({ id, label, kind: "weight", mat, ...extra });
  const vec = (name, data) => ({ r: 1, c: data.length, data, get: (i, j) => data[j], row: () => data });

  // ------------------------------------------------------------ embeddings
  mark("Embeddings");
  const tokEmb = model.t("tok_emb"), posEmb = model.t("pos_emb");
  // tokens column: one tile per position, labelled with the token
  const tokensMat = { r: T, c: 1, data: new Float32Array(T), get: (i) => i, row: (i) => [i] };
  const tokens = mkBlock({ id: "tokens", label: "tokens", kind: "tokens", mat: tokensMat, colorMode: "cat", vmax: 1, step: "tokens", desc: "Your input, one token per row. Each token is just an integer id into the vocabulary." });
  const embRows = act("tokRows", "tok_emb[ids]", trace.tokRows, { step: "embed", desc: "Row <i>ids[t]</i> of the token-embedding matrix copied out for each position — the token's learned meaning vector, before it knows anything about its neighbours." });
  const wTok = weight("tok_emb", "W_tok_emb", tokEmb, { step: "embed", rowLabels: "vocab", desc: "The token embedding table: one learned vector per vocabulary entry. Looked up, never multiplied." });
  level([embRows], [wTok]);
  const posRows = act("posRows", "pos_emb[0:T]", trace.posRows, { step: "pos", desc: "The first T rows of the position-embedding table: what the model adds to say “you are token number t”." });
  const wPos = weight("pos_emb", "W_pos_emb", posEmb, { step: "pos", desc: "Learned position embeddings (one row per position up to the context length). GPT-2 learns these; the original Transformer used fixed sinusoids." });
  level([posRows], [wPos]);
  const x0 = act("x0", "x₀ = tok + pos", trace.x0, { step: "residual", residual: true, layerIdx: -1, desc: "The residual stream at the start: token meaning + position, one d-dimensional vector per token. Every block below READS this and ADDS to it." });
  level([x0]);
  // token labels sit to the right of x0
  tokens.x = x0.x + x0.width + hgap; tokens.y = x0.y; tokens.z = x0.z;

  // ------------------------------------------------------------ layers
  let prevResidualId = "x0";
  for (let l = 0; l < L; l++) {
    mark(`Layer ${l}`);
    const Lt = trace.layers[l], p = (n) => model.layerT(l, n), P = `L${l}.`;
    const ln1 = act(P + "ln1", `LN₁(x)`, Lt.ln1, { step: "ln", layer: l, desc: "LayerNorm: each token's vector is re-centred (mean 0) and re-scaled (variance 1) across its d dimensions, then stretched by a learned gain γ and shifted by a bias β. It keeps the numbers tame before attention." });
    const g1 = weight(P + "ln1_g", "γ₁", vec("g", p("ln1_g")), { step: "ln", layer: l, desc: "LayerNorm gain γ (one per dimension)." });
    const b1 = weight(P + "ln1_b", "β₁", vec("b", p("ln1_b")), { step: "ln", layer: l, desc: "LayerNorm bias β (one per dimension)." });
    level([ln1], [g1, b1], { pitch: pitch * 0.8 });

    const q = act(P + "q", "Q", Lt.q, { step: "qkv", layer: l, headGroups: H, desc: "Queries: for every token, “what am I looking for?” — one vector per head (the column groups)." });
    const k = act(P + "k", "K", Lt.k, { step: "qkv", layer: l, headGroups: H, desc: "Keys: for every token, “what do I contain / advertise?” — one vector per head." });
    const v = act(P + "v", "V", Lt.v, { step: "qkv", layer: l, headGroups: H, desc: "Values: the information a token hands over if it gets attended to." });
    const wq = weight(P + "wq", "W_Q", p("wq"), { step: "qkv", layer: l, headGroups: H, desc: "Query weights (d×d). Column group h is head h's query map." });
    const wk = weight(P + "wk", "W_K", p("wk"), { step: "qkv", layer: l, headGroups: H, desc: "Key weights (d×d)." });
    const wv = weight(P + "wv", "W_V", p("wv"), { step: "qkv", layer: l, headGroups: H, desc: "Value weights (d×d)." });
    level([q, k, v], [wq, wk, wv]);

    const scoreBlocks = [], attnBlocks = [];
    for (let h = 0; h < H; h++) {
      scoreBlocks.push(act(P + `scores.${h}`, `scores h${h}`, Lt.scores[h], { step: "scores", layer: l, head: h, mask: (i, j) => j > i, square: true, desc: `Head ${h}: raw attention scores q_t·k_s / √${dh}. Row t = the token that is looking; column s = the token being looked at. The empty upper triangle is the causal mask — no peeking at the future.` }));
      attnBlocks.push(act(P + `attn.${h}`, `softmax h${h}`, Lt.attn[h], { step: "softmax", layer: l, head: h, mask: (i, j) => j > i, square: true, colorMode: "seq", vmax: 1, kind: "attn", desc: `Head ${h}: the attention pattern. Every row is a probability distribution over earlier tokens (it sums to 1).` }));
    }
    level(scoreBlocks);
    level(attnBlocks);

    const headOut = act(P + "headOut", "heads ⊕", Lt.headOut, { step: "mix", layer: l, headGroups: H, desc: "Each head's output: its attention-weighted average of value vectors. The heads are concatenated side by side." });
    level([headOut]);
    const attnOut = act(P + "attnOut", "attn out", Lt.attnOut, { step: "proj", layer: l, desc: "Output projection: the concatenated heads multiplied by W_O. This is what attention writes back into the residual stream." });
    const wo = weight(P + "wo", "W_O", p("wo"), { step: "proj", layer: l, desc: "Output-projection weights (d×d): mixes the heads back into one d-dimensional update." });
    level([attnOut], [wo]);
    const x1 = act(P + "x1", `x + attn`, Lt.x1, { step: "proj", layer: l, residual: true, resIn: prevResidualId, resAdd: P + "attnOut", desc: "Residual add: the stream after attention = what came in + attention's update. Nothing is overwritten; information accumulates." });
    level([x1]);

    const ln2 = act(P + "ln2", "LN₂(x)", Lt.ln2, { step: "mlp", layer: l, desc: "Second LayerNorm, applied before the MLP." });
    const g2 = weight(P + "ln2_g", "γ₂", vec("g", p("ln2_g")), { step: "mlp", layer: l, desc: "LayerNorm gain γ." });
    const b2 = weight(P + "ln2_b", "β₂", vec("b", p("ln2_b")), { step: "mlp", layer: l, desc: "LayerNorm bias β." });
    level([ln2], [g2, b2], { pitch: pitch * 0.8 });
    const mlpPre = act(P + "mlpPre", "x·W_in + b", Lt.mlpPre, { step: "mlp", layer: l, desc: `Up-projection: ${dff} neurons each take a dot product with the token's vector (“how much does this token match my pattern?”).` });
    const win = weight(P + "win", "W_in", p("win"), { step: "mlp", layer: l, desc: `Up-projection weights (d×${dff}). Column n is neuron n's pattern detector — 3Blue1Brown's “question” vector.` });
    level([mlpPre], [win]);
    const mlpAct = act(P + "mlpAct", "GELU", Lt.mlpAct, { step: "mlp", layer: l, desc: "GELU activation: negative matches are squashed to ≈0, positive ones pass through. Only some neurons fire for a given token — these are the “neurons lighting up”." });
    level([mlpAct]);
    const mlpOut = act(P + "mlpOut", "GELU·W_out + b", Lt.mlpOut, { step: "mlpout", layer: l, desc: "Down-projection: every firing neuron adds its own learned output vector (row n of W_out) into a d-dimensional update." });
    const wout = weight(P + "wout", "W_out", p("wout"), { step: "mlpout", layer: l, desc: `Down-projection weights (${dff}×d). Row n is what neuron n writes into the residual stream when it fires.` });
    level([mlpOut], [wout]);
    const x2 = act(P + "x2", "x + mlp", Lt.x2, { step: "mlpout", layer: l, residual: true, layerIdx: l, resIn: P + "x1", resAdd: P + "mlpOut", desc: `Residual stream after layer ${l}: everything so far, plus the MLP's update.` });
    level([x2], [], { pitch: pitch * 1.6 });
    prevResidualId = P + "x2";
  }

  // ------------------------------------------------------------ output
  mark("Output");
  const lnf = act("lnf", "LN_f(x)", trace.lnf, { step: "unembed", desc: "Final LayerNorm before reading the prediction out of the residual stream." });
  const gf = weight("lnf_g", "γ_f", vec("g", model.t("lnf_g")), { step: "unembed", desc: "Final LayerNorm gain." });
  const bf = weight("lnf_b", "β_f", vec("b", model.t("lnf_b")), { step: "unembed", desc: "Final LayerNorm bias." });
  level([lnf], [gf, bf], { pitch: pitch * 0.8 });
  const logits = act("logits", "logits", trace.logits, { step: "unembed", colLabels: "vocab", desc: "Logits: one score per vocabulary token = dot product of the token's final vector with that vocabulary token's embedding (GPT-2 ties the unembedding to the embedding table)." });
  const unembed = weight("unembed", "W_unembed = W_tok_embᵀ", tokEmb, { step: "unembed", rowLabels: "vocab", tied: true, desc: "The unembedding matrix. In GPT-2 (and here) it is the SAME matrix as the token embedding, used transposed." });
  level([logits], [unembed]);
  const probs = act("probs", "softmax(logits)", trace.probs, { step: "sample", colMode: "seq", colorMode: "seq", vmax: 1, kind: "prob", colLabels: "vocab", desc: "Softmax turns the scores into a probability distribution over the next token. Only the LAST row is used to continue the text; the other rows are the model's predictions at earlier positions (used for training)." });
  level([probs]);

  // ------------------------------------------------------------ bounds
  const bounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity };
  for (const b of blocks) {
    bounds.minX = Math.min(bounds.minX, b.x); bounds.maxX = Math.max(bounds.maxX, b.x + b.width);
    bounds.minY = Math.min(bounds.minY, b.y); bounds.maxY = Math.max(bounds.maxY, b.y + 3);
    bounds.minZ = Math.min(bounds.minZ, b.z); bounds.maxZ = Math.max(bounds.maxZ, b.z + b.depth);
  }
  for (const g of groups) g.x = bounds.minX - 6;
  return { blocks, byId, groups, bounds, pitch, colX, T, d, H, dh, dff, V, L };
}

// --------------------------------------------------------------------------
// Dependency regions: which cells does block[i][j] depend on?
// Returns [{id, r0, r1, c0, c1}] with inclusive ranges (-1 = whole range).
// --------------------------------------------------------------------------
export function dependencies(layout, model, trace, block, i, j) {
  const A = -1, id = block.id, dh = model.dh, H = model.H;
  const m = id.match(/^L(\d+)\.(.+)$/);
  const reg = (bid, r0, r1, c0, c1) => ({ id: bid, r0, r1, c0, c1 });
  if (!m) {
    switch (id) {
      case "tokRows": return [reg("tok_emb", trace.ids[i], trace.ids[i], j, j)];
      case "posRows": return [reg("pos_emb", i, i, j, j)];
      case "x0": return [reg("tokRows", i, i, j, j), reg("posRows", i, i, j, j)];
      case "lnf": return [reg(`L${model.L - 1}.x2`, i, i, A, A), reg("lnf_g", 0, 0, j, j), reg("lnf_b", 0, 0, j, j)];
      case "logits": return [reg("lnf", i, i, A, A), reg("unembed", j, j, A, A)];
      case "probs": return [reg("logits", i, i, A, A)];
    }
    return [];
  }
  const l = +m[1], name = m[2], P = `L${l}.`, prev = l === 0 ? "x0" : `L${l - 1}.x2`;
  const headCols = (h) => [h * dh, (h + 1) * dh - 1];
  const hm = name.match(/^(scores|attn)\.(\d+)$/);
  if (hm) {
    const h = +hm[2], [c0, c1] = headCols(h);
    if (hm[1] === "scores") return [reg(P + "q", i, i, c0, c1), reg(P + "k", j, j, c0, c1)];
    return [reg(P + "scores." + h, i, i, 0, i)];
  }
  switch (name) {
    case "ln1": return [reg(prev, i, i, A, A), reg(P + "ln1_g", 0, 0, j, j), reg(P + "ln1_b", 0, 0, j, j)];
    case "q": return [reg(P + "ln1", i, i, A, A), reg(P + "wq", A, A, j, j)];
    case "k": return [reg(P + "ln1", i, i, A, A), reg(P + "wk", A, A, j, j)];
    case "v": return [reg(P + "ln1", i, i, A, A), reg(P + "wv", A, A, j, j)];
    case "headOut": { const h = Math.floor(j / dh); return [reg(P + "attn." + h, i, i, 0, i), reg(P + "v", 0, i, j, j)]; }
    case "attnOut": return [reg(P + "headOut", i, i, A, A), reg(P + "wo", A, A, j, j)];
    case "x1": return [reg(prev, i, i, j, j), reg(P + "attnOut", i, i, j, j)];
    case "ln2": return [reg(P + "x1", i, i, A, A), reg(P + "ln2_g", 0, 0, j, j), reg(P + "ln2_b", 0, 0, j, j)];
    case "mlpPre": return [reg(P + "ln2", i, i, A, A), reg(P + "win", A, A, j, j)];
    case "mlpAct": return [reg(P + "mlpPre", i, i, j, j)];
    case "mlpOut": return [reg(P + "mlpAct", i, i, A, A), reg(P + "wout", j, j, A, A)];
    case "x2": return [reg(P + "x1", i, i, j, j), reg(P + "mlpOut", i, i, j, j)];
  }
  return [];
}

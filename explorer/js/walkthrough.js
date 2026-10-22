// walkthrough.js — the guided tour. Each step says which blocks to frame,
// which "lens" to apply, which cell to pre-select, and renders an explanation
// with the LIVE numbers of the current input (st = app state snapshot).
import { esc, tokHtml, fmt, pct, big, human } from "./util.js";

const argmaxRow = (mat, t, n) => { let b = 0, bv = -Infinity; for (let s = 0; s < n; s++) { const v = mat.get(t, s); if (v > bv) { bv = v; b = s; } } return b; };

export const STEPS = [
  {
    id: "overview", title: "The whole model", short: "Overview",
    frame: () => null, lens: () => ({}), select: () => null,
    body: (st) => `
<p>This is <b>${esc(st.meta.title)}</b> — a real, trained GPT-style transformer (${big(st.meta.n_params)} parameters) running in your browser right now. Every tile you see is one number from the live forward pass over your input.</p>
<p><b>How to read the 3D view.</b> Data flows <b>top → bottom</b>. The centre column holds <b>activations</b> (amber plates): numbers that depend on <i>your</i> input and are recomputed each time you type. The lane on the left holds the <b>learned weights</b> (blue plates): fixed matrices produced by training, the same for every input. Flat tiles = weights; <b>bar height</b> = |value| for activations. Colour: <span class="sw pos"></span> positive, <span class="sw neg"></span> negative, <span class="sw seq"></span> probabilities 0→1.</p>
<p>Architecture: <b>${st.L} layers</b>, each = attention (${st.H} heads of ${st.dh} dims) + MLP (${st.dff} neurons), on a residual stream of width <b>d = ${st.d}</b>. Context length ${st.cfg.block_size}; vocabulary ${st.V} tokens. GPT-2 and GPT-3 have exactly this shape — just thousands of times larger (last step).</p>
<p class="hint">Drag to orbit, scroll to zoom, right-drag to pan. <b>Click any tile</b> to see its value and the arithmetic that produced it; its inputs light up in amber. Use the steps on the left, or ← → keys.</p>`,
  },
  {
    id: "tokens", title: "Tokens: text becomes integers", short: "Tokens",
    frame: () => ["tokens", "x0"], lens: () => ({}), select: (st) => ({ id: "tokens", i: st.T - 1, j: 0 }),
    body: (st) => `
<p>A model never sees letters. A <b>tokenizer</b> maps text to integer ids from a fixed vocabulary. ${st.meta.task === "sort" ? `This toy's vocabulary is just <b>A, B, C</b> (ids 0, 1, 2).` : `This model is <b>character-level</b>: every character is one token, vocabulary = the ${st.V} distinct characters of the training text (GPT-2 instead uses ~50k sub-word pieces, but the mechanics are identical).`}</p>
<p>Your ${st.T} tokens: ${st.tokens.map((t, i) => `<span class="tk" title="position ${i}, id ${st.trace.ids[i]}">${esc(st.pretty(t))}<sub>${st.trace.ids[i]}</sub></span>`).join(" ")}</p>
<p>From here on, position <i>t</i> of every activation block (the row <i>t</i> cells from the back) belongs to token <i>t</i>. The last row — ${tokHtml(st.tokens[st.T - 1])} at position ${st.T - 1} — is the one whose prediction becomes the next token.</p>`,
  },
  {
    id: "embed", title: "Token embedding: id → vector", short: "Embedding",
    frame: () => ["tok_emb", "tokRows"], lens: () => ({}), select: (st) => ({ id: "tokRows", i: st.T - 1, j: 0 }),
    body: (st) => `
<p>The <b>embedding matrix</b> W<sub>tok_emb</sub> (${st.V} × ${st.d}, left) holds one learned vector per vocabulary token. Embedding is a <b>lookup</b>, not a multiplication: token id ${st.trace.ids[st.T - 1]} (${tokHtml(st.tokens[st.T - 1])}) simply copies out row ${st.trace.ids[st.T - 1]}.</p>
<p>These ${st.d} numbers are the token's <i>meaning</i> to the model, learned by gradient descent: tokens that are used in similar ways end up with similar vectors (3Blue1Brown's “woman − man ≈ queen − king” directions live in this space). In GPT-3 this matrix is 50,257 × 12,288 = ${big(50257 * 12288)} parameters.</p>
<p>Notice the vector is the same wherever the token appears — it knows nothing about its neighbours yet. The rest of the network exists to <i>add</i> context to it.</p>`,
  },
  {
    id: "pos", title: "Position embedding: where am I?", short: "Position",
    frame: () => ["pos_emb", "posRows", "x0"], lens: () => ({}), select: (st) => ({ id: "x0", i: st.T - 1, j: 0 }),
    body: (st) => `
<p>Attention (coming up) treats the tokens as an unordered set, so order has to be injected explicitly. GPT-2 learns a second table W<sub>pos_emb</sub> (${st.cfg.block_size} × ${st.d}): row <i>t</i> means “you are token number <i>t</i>”.</p>
<p>The model's actual input is the <b>sum</b>: x<sub>0</sub>[t] = W<sub>tok_emb</sub>[id<sub>t</sub>] + W<sub>pos_emb</sub>[t]. Click any cell of x<sub>0</sub> and the inspector shows the two numbers being added.</p>
<p>That ${st.T} × ${st.d} block, x<sub>0</sub>, is the <b>residual stream</b> — the central object of the whole model.</p>`,
  },
  {
    id: "residual", title: "The residual stream", short: "Residual",
    frame: (st) => ["x0", "L0.x1", "L0.x2"], lens: () => ({}), select: (st) => ({ id: "L0.x2", i: st.T - 1, j: 0 }),
    body: (st) => `
<p>Think of the residual stream as a <b>shared whiteboard</b>: one ${st.d}-number vector per token. Every block that follows <i>reads</i> the whiteboard, computes something, and <b>adds</b> its result back (x ← x + f(x)). Nothing is ever overwritten.</p>
<p>So each token's vector accumulates information: first its own meaning + position, then (after attention) facts gathered from earlier tokens, then (after the MLP) things the model “knows” about that combination. At the very end, the <i>last</i> token's vector is read out as a prediction.</p>
<p>There are ${1 + 2 * st.L} snapshots of the stream in this view (x<sub>0</sub>, then x+attn and x+mlp for each layer). Select one and the inspector's <b>logit lens</b> shows what the model would predict if it stopped right there — watch the prediction sharpen as you go down.</p>`,
  },
  {
    id: "ln", title: "LayerNorm: tame the numbers", short: "LayerNorm",
    frame: (st) => [`L${st.layer}.ln1`, `L${st.layer}.ln1_g`, `L${st.layer}.ln1_b`], lens: (st) => ({ layer: st.layer }), select: (st) => ({ id: `L${st.layer}.ln1`, i: st.T - 1, j: 0 }),
    body: (st) => { const L = st.trace.layers[st.layer], t = st.T - 1; return `
<p>Before attention, each token's vector is <b>normalised</b> on its own: subtract its mean, divide by its standard deviation (so the ${st.d} numbers have mean 0 and variance 1), then multiply by a learned gain <b>γ</b> and add a learned bias <b>β</b>, per dimension.</p>
<p>For the last token in layer ${st.layer}: mean = ${fmt(L.ln1_mean[t])}, 1/std = ${fmt(L.ln1_rstd[t])}. This keeps the scale of the residual stream from drifting as layers add to it, which makes training stable. (GPT-2 applies LayerNorm <i>before</i> each block — “pre-LN”. The ε inside the square root is 10<sup>−5</sup>.)</p>
<p>γ and β (the two 1 × ${st.d} strips on the left) are the only learned parameters here.</p>`; },
  },
  {
    id: "qkv", title: "Queries, keys, values", short: "Q K V",
    frame: (st) => [`L${st.layer}.wq`, `L${st.layer}.wk`, `L${st.layer}.wv`, `L${st.layer}.q`, `L${st.layer}.k`, `L${st.layer}.v`], lens: (st) => ({ layer: st.layer, head: st.head }), select: (st) => ({ id: `L${st.layer}.q`, i: st.T - 1, j: st.head * st.dh }),
    body: (st) => `
<p>Attention lets tokens <b>exchange information</b>. Each token publishes three vectors, all computed from its normalised residual vector by three learned matrices:</p>
<ul>
<li><b>Query</b> q = LN(x)·W<sub>Q</sub> + b<sub>Q</sub> — “what am I looking for?”</li>
<li><b>Key</b> k = LN(x)·W<sub>K</sub> + b<sub>K</sub> — “what do I contain?”</li>
<li><b>Value</b> v = LN(x)·W<sub>V</sub> + b<sub>V</sub> — “what I hand over if you pick me”.</li>
</ul>
<p>Each of W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> is ${st.d} × ${st.d}, but the columns are split into <b>${st.H} heads</b> of ${st.dh} (the gaps in the blocks). Head ${st.head} only ever reads columns ${st.head * st.dh}–${(st.head + 1) * st.dh - 1}. Heads run in parallel and learn different relationships (“previous token”, “matching bracket”, “same letter earlier”…).</p>
<p>The selected cell is q for the last token, head ${st.head}, dimension 0: a dot product of the ${st.d}-long LN row with column ${st.head * st.dh} of W<sub>Q</sub> — ${st.d} multiplications and one add of the bias. The inspector lists the largest terms.</p>
<p class="hint">Switch head / layer in the panel above to move the lens.</p>`,
  },
  {
    id: "scores", title: "Attention scores: q · k / √d", short: "Scores",
    frame: (st) => Array.from({ length: st.H }, (_, h) => `L${st.layer}.scores.${h}`), lens: (st) => ({ layer: st.layer, head: st.head }),
    select: (st) => { const t = st.T - 1, A = st.trace.layers[st.layer].attn[st.head]; return { id: `L${st.layer}.scores.${st.head}`, i: t, j: argmaxRow(A, t, t + 1) }; },
    body: (st) => { const t = st.T - 1, L = st.trace.layers[st.layer], A = L.attn[st.head], s = argmaxRow(A, t, t + 1); return `
<p>For every pair (looker <i>t</i>, looked-at <i>s</i>) in head ${st.head}, the score is the <b>dot product</b> q<sub>t</sub>·k<sub>s</sub> (${st.dh} multiplications) divided by √${st.dh} = ${fmt(Math.sqrt(st.dh), 2)}. Big score = “this key answers my query”. Each ${st.T} × ${st.T} block is one head; row = looker, column = looked-at.</p>
<p>The empty upper-right triangle is the <b>causal mask</b>: token <i>t</i> may only look at positions ≤ <i>t</i>, because during generation the future doesn't exist yet. Masked scores are set to −∞ so the next step gives them probability 0. This is why a GPT can be trained on every position of a text at once.</p>
<p>Live: the last token ${tokHtml(st.tokens[t])} scores highest against position ${s} (${tokHtml(st.tokens[s])}) with ${fmt(L.scores[st.head].get(t, s))} → that cell is selected; the inspector shows all ${st.dh} q×k terms.</p>`; },
  },
  {
    id: "softmax", title: "Softmax: scores → attention pattern", short: "Softmax",
    frame: (st) => Array.from({ length: st.H }, (_, h) => `L${st.layer}.attn.${h}`), lens: (st) => ({ layer: st.layer, head: st.head }),
    select: (st) => { const t = st.T - 1, A = st.trace.layers[st.layer].attn[st.head]; return { id: `L${st.layer}.attn.${st.head}`, i: t, j: argmaxRow(A, t, t + 1) }; },
    body: (st) => { const t = st.T - 1, A = st.trace.layers[st.layer].attn[st.head]; const row = []; for (let s = 0; s <= t; s++) row.push([s, A.get(t, s)]); row.sort((a, b) => b[1] - a[1]); return `
<p>Each row of scores is pushed through a <b>softmax</b>: exponentiate, then divide by the sum, so the row becomes a probability distribution (sums to 1). Large scores win disproportionately — that's the “soft” in soft-max; it behaves like a differentiable arg-max.</p>
<p>The result is the <b>attention pattern</b>: for every token, a weighting over earlier tokens saying <i>whom it listens to</i>. Blue brightness = weight. A bright diagonal means “attend to myself”, a bright sub-diagonal “attend to the previous token”, a bright column “everyone looks at that token”.</p>
<p>Live, head ${st.head} of layer ${st.layer}: ${tokHtml(st.tokens[t])} listens to ${row.slice(0, 4).map(([s, p]) => `${tokHtml(st.tokens[s])}<sub>${s}</sub> ${pct(p)}`).join(", ")}${row.length > 4 ? ", …" : ""}.</p>`; },
  },
  {
    id: "mix", title: "Mixing values: what each head produces", short: "Mix V",
    frame: (st) => [`L${st.layer}.v`, `L${st.layer}.headOut`], lens: (st) => ({ layer: st.layer, head: st.head }), select: (st) => ({ id: `L${st.layer}.headOut`, i: st.T - 1, j: st.head * st.dh }),
    body: (st) => `
<p>Now the pattern is used: each token's new vector for head ${st.head} is the <b>weighted average of the value vectors</b> of the tokens it attends to — out<sub>t</sub> = Σ<sub>s≤t</sub> attn[t][s] · v<sub>s</sub>. If ${tokHtml(st.tokens[st.T - 1])} attends 90% to one token, it mostly copies that token's value vector.</p>
<p>This is the only place in the whole network where <b>information moves between positions</b>. Everything else (LayerNorm, MLP, projections) works on each token separately.</p>
<p>The ${st.H} heads' outputs (${st.dh} numbers each) are concatenated side by side into one ${st.T} × ${st.d} block — the gaps mark the heads.</p>`,
  },
  {
    id: "proj", title: "Output projection + residual add", short: "Add to stream",
    frame: (st) => [`L${st.layer}.wo`, `L${st.layer}.attnOut`, `L${st.layer}.x1`], lens: (st) => ({ layer: st.layer }), select: (st) => ({ id: `L${st.layer}.x1`, i: st.T - 1, j: 0 }),
    body: (st) => `
<p>The concatenated heads are multiplied by one more learned matrix W<sub>O</sub> (${st.d} × ${st.d}) that mixes the heads' findings into a single update vector per token. (3Blue1Brown folds this into the “value” step as the <i>value-up</i> matrix — same thing: value-down is W<sub>V</sub> here, value-up is W<sub>O</sub>.)</p>
<p>Then the update is <b>added</b> to the residual stream: x<sub>1</sub> = x + attn_out. Click a cell of x<sub>1</sub>: the inspector shows exactly the two numbers being added. The stream now carries, for each token, information collected from earlier tokens.</p>`,
  },
  {
    id: "mlp", title: "MLP: neurons that light up", short: "MLP",
    frame: (st) => [`L${st.layer}.ln2`, `L${st.layer}.win`, `L${st.layer}.mlpPre`, `L${st.layer}.mlpAct`], lens: (st) => ({ layer: st.layer }),
    select: (st) => { const t = st.T - 1, M = st.trace.layers[st.layer].mlpAct; return { id: `L${st.layer}.mlpAct`, i: t, j: argmaxRow(M, t, st.dff) }; },
    body: (st) => { const t = st.T - 1, M = st.trace.layers[st.layer].mlpAct; let fired = 0; for (let n = 0; n < st.dff; n++) if (M.get(t, n) > 0.1) fired++; const n = argmaxRow(M, t, st.dff); return `
<p>After attention, each token is processed <b>on its own</b> by a small two-layer neural network (the same one for every position). First, LayerNorm again. Then the <b>up-projection</b> W<sub>in</sub> (${st.d} × ${st.dff}): each of the ${st.dff} columns is a <b>neuron</b> — a pattern vector. Its pre-activation is the dot product “how much does this token's vector match my pattern?” (3Blue1Brown: the neuron asks a question such as “is this Michael Jordan?”).</p>
<p>Then <b>GELU</b> (a smooth ReLU): negative matches are squashed to ≈0, positive ones pass. Only some neurons fire for a given token. For the last token, ${fired} of ${st.dff} neurons are above 0.1; the strongest is neuron ${n} at ${fmt(M.get(t, n))} (selected). This sparse “which neurons fired” pattern is the closest thing the model has to <i>features</i>.</p>
<p>The MLP holds most of the parameters: ${big(2 * st.d * st.dff)} of this layer's weights vs ${big(4 * st.d * st.d)} in attention. In GPT-3 it's ~2/3 of all 175 B parameters — and the place where interpretability research finds <i>stored facts</i>.</p>`; },
  },
  {
    id: "mlpout", title: "MLP down-projection + residual add", short: "MLP out",
    frame: (st) => [`L${st.layer}.wout`, `L${st.layer}.mlpOut`, `L${st.layer}.x2`], lens: (st) => ({ layer: st.layer }), select: (st) => ({ id: `L${st.layer}.mlpOut`, i: st.T - 1, j: 0 }),
    body: (st) => `
<p>Every neuron that fired now <b>writes</b>: the <b>down-projection</b> W<sub>out</sub> (${st.dff} × ${st.d}) holds, in row <i>n</i>, the ${st.d}-vector that neuron <i>n</i> adds to the stream, scaled by how strongly it fired. So mlp_out = Σ<sub>n</sub> GELU<sub>n</sub> · W<sub>out</sub>[n] + b (a pattern-detector → a learned response). Click a cell of mlp_out to see the sum over all ${st.dff} neurons, biggest contributions first.</p>
<p>Then the second residual add of the layer: x<sub>2</sub> = x<sub>1</sub> + mlp_out. That completes one <b>transformer block</b>: attention (move information between tokens) + MLP (process each token). ${st.L} such blocks are stacked.</p>`,
  },
  {
    id: "stack", title: "Stacking layers", short: "Layers",
    frame: (st) => { const ids = ["x0"]; for (let l = 0; l < st.L; l++) ids.push(`L${l}.x1`, `L${l}.x2`); return ids; }, lens: () => ({ token: null }), select: (st) => ({ id: `L${st.L - 1}.x2`, i: st.T - 1, j: 0 }),
    body: (st) => { const t = st.T - 1; const lens = []; for (let l = -1; l < st.L; l++) { const ll = st.model.logitLens(st.trace, l); const top = st.model.topK(ll.probs.row(t), 1)[0]; lens.push(`<li>after ${l < 0 ? "embeddings" : "layer " + l}: ${tokHtml(top.token)} ${pct(top.p)}</li>`); } return `
<p>The same block structure repeats ${st.L} times with <b>different weights</b> in each layer. Early layers tend to learn local, syntactic things (previous token, “what letter usually follows”), later layers more abstract ones — and because of the residual stream, each layer builds on everything before it.</p>
<p><b>Logit lens</b> — reading the last token's residual vector out as a prediction after each layer (apply the final LayerNorm and the unembedding early):</p>
<ol class="lens-list">${lens.join("")}</ol>
<p>Watch the prediction form gradually. Early on the lens often just reads back the <i>current</i> token: with tied embeddings the residual stream starts out as the token's own embedding, whose biggest dot product is with itself — so the stream says “I am ${tokHtml(st.tokens[t])}”, and the layers' whole job is to turn that into “next comes …”. In big models the first layers' guesses are near-random and the final answer appears only in the last few layers.</p>
<p class="hint">Try the <b>token lens</b> in the panel (pick a position) to follow one token's row through every block.</p>`; },
  },
  {
    id: "unembed", title: "Unembedding: vector → scores for every token", short: "Logits",
    frame: () => ["lnf", "unembed", "logits"], lens: () => ({}), select: (st) => { const t = st.T - 1; return { id: "logits", i: t, j: argmaxRow(st.trace.logits, t, st.V) }; },
    body: (st) => { const t = st.T - 1, top = st.model.topK(st.trace.probs.row(t), 3); return `
<p>After the last layer, one more LayerNorm, then the final vector of each token is compared with <b>every vocabulary token's embedding</b>: logit[t][v] = LN<sub>f</sub>(x<sub>t</sub>) · W<sub>unembed</sub>[v]. GPT-2 (and this model) <b>ties</b> the unembedding to the embedding table — it's the same ${st.V} × ${st.d} matrix used transposed — which is why it appears twice in the 3D view.</p>
<p>The result is a ${st.T} × ${st.V} block of <b>logits</b> (raw, unnormalised scores). Only the last row matters for generation; the other rows are predictions for earlier positions (they're what training grades the model on — one loss term per position).</p>
<p>Live: the highest logit for the last token belongs to ${tokHtml(top[0].token)} (${fmt(st.trace.logits.get(t, top[0].id))}), then ${tokHtml(top[1].token)} and ${tokHtml(top[2].token)}.</p>`; },
  },
  {
    id: "sample", title: "Softmax, temperature, sampling", short: "Sampling",
    frame: () => ["logits", "probs"], lens: () => ({}), select: (st) => { const t = st.T - 1; return { id: "probs", i: t, j: argmaxRow(st.trace.probs, t, st.V) }; },
    body: (st) => { const t = st.T - 1, top = st.model.topK(st.trace.probs.row(t), 5); return `
<p>Softmax turns the last row of logits into probabilities: p<sub>v</sub> = e<sup>z<sub>v</sub>/T</sup> / Σ e<sup>z/T</sup>. The <b>temperature</b> T reshapes the distribution before sampling: T → 0 always picks the top token; T &gt; 1 flattens the odds and lets unlikely tokens through (more “creative”, less coherent). <b>Top-k</b> simply zeroes everything outside the k most likely.</p>
<p>The model's true output is this whole distribution; picking one token is a separate random draw. That's why the same prompt can produce different answers.</p>
<p>Live, after ${tokHtml(st.tokens[t])}: ${top.map((x) => `${tokHtml(x.token)} ${pct(x.p)}`).join(" · ")} (at temperature 1). Use the sliders in the panel; the inspector's bars update.</p>`; },
  },
  {
    id: "generate", title: "The generation loop", short: "Generate",
    frame: () => ["tokens", "x0", "probs"], lens: () => ({}), select: () => null,
    body: (st) => `
<p>One forward pass = one new token. To write text the model <b>loops</b>: sample the next token, append it to the input, run the whole stack again on the longer sequence. Every answer from every chat model is produced by exactly this loop (with caching tricks so earlier positions aren't recomputed).</p>
<p>Press <b>▶ Generate</b> in the panel: a row is added to every activation block in the 3D view and everything is recomputed. ${st.meta.task === "sort" ? `For the sorter: positions 0–${st.nIn - 1} are the input letters (white) and the model's own output letters follow them (green). It fills in its answer automatically; press <b>↺ reset</b> to clear it, then <b>+1 token</b> to watch it sort one letter at a time — the last position's prediction is always the next-smallest letter still owed.` : `The model was trained on Shakespeare; with ~${human(st.meta.n_params)} parameters expect plausible-looking words, not sense — the mechanism is the point.`}</p>`,
  },
  {
    id: "scale", title: "From this toy to GPT-2 and GPT-3", short: "Scale",
    frame: () => null, lens: () => ({}), select: () => null,
    body: (st) => `
<p>Nothing changes structurally between this model and GPT-3 — only the sizes. The table below computes the parameter counts with the same formulas for this toy, GPT-2 small (the 124 M model many visualisations run) and GPT-3 175 B (the breakdown in 3Blue1Brown's video). Biases and LayerNorm gains are omitted, as in the video.</p>
<div id="scale-table"></div>
<p>Press <b>Show GPT-2 / GPT-3 outlines</b> to draw their layers as wireframes next to this model at <b>true relative scale</b> (real d<sub>model</sub>, d<sub>ff</sub> and vocabulary as widths, the real context window as depth, this view's vertical pitch). Then fly out. GPT-3's residual stream is ${big(12288)} wide — ${Math.round(12288 / st.d)}× this model — and 96 layers deep.</p>
<p>What scale buys: with ~50 k sub-word tokens, 12,288-dimensional vectors and 96 layers × 96 heads, the same q·k·v arithmetic you just zoomed into learns grammar, facts and reasoning-like behaviour from trillions of tokens. The computation you watched here is the same computation.</p>`,
  },
];

export function stepIndex(id) { return STEPS.findIndex((s) => s.id === id); }

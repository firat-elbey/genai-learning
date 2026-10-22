// inspector.js — the 2D side panel: a heatmap of the selected block with axis
// labels, statistics, and for a selected cell the exact arithmetic that
// produced it (every term, biggest first), plus row-level context (who this
// token attends to, top-k next tokens, logit lens).
import { diverging, sequential, rgbCss, MASK_RGB } from "./colors.js";
import { esc, prettyToken, tokHtml, fmt, pct, big, sign } from "./util.js";

const KIND_LABEL = { weight: "learned weight", act: "activation", attn: "attention pattern", prob: "probabilities", tokens: "input tokens" };

export class Inspector {
  constructor(root, callbacks) { this.root = root; this.cb = callbacks; this.state = null; }

  render(st, sel) {
    this.state = st; this.sel = sel;
    if (!sel || !sel.block) { this.root.innerHTML = `<div class="insp-empty">Click any tile in the 3D view (or a block label) to inspect it.</div>`; return; }
    const b = sel.block, hasCell = sel.i !== undefined && sel.i !== null;
    const kind = KIND_LABEL[b.kind] || b.kind;
    const where = b.layer !== undefined ? ` · layer ${b.layer}${b.head !== undefined ? ` · head ${b.head}` : ""}` : "";
    let html = `<div class="insp-head"><div><span class="badge ${b.kind}">${kind}</span><span class="insp-where">${where}</span></div>
      <h3>${b.label} <span class="mut">${b.rows} × ${b.cols}</span></h3>
      <p class="insp-desc">${b.desc || ""}</p>
      <div class="insp-actions"><button class="btn small" data-act="focus">🎯 fly to block</button>
      ${b.kind !== "weight" && b.rows === st.T && hasCell ? `<button class="btn small" data-act="tokenlens">🔦 follow token ${sel.i}</button>` : ""}
      ${b.head !== undefined ? `<button class="btn small" data-act="headlens">🔦 lens on head ${b.head}</button>` : ""}
      <button class="btn small" data-act="clearlens">clear lens</button></div></div>`;
    html += `<div class="insp-stats">${this.stats(b)}</div>`;
    html += `<div class="heat-wrap"><canvas id="insp-heat"></canvas><div class="heat-tip" id="heat-tip"></div></div><div class="mut small" id="heat-hint">${this.heatHint(b, st)}</div>`;
    if (hasCell) html += this.cellSection(st, b, sel.i, sel.j);
    html += this.rowSection(st, b, hasCell ? sel.i : st.T - 1);
    this.root.innerHTML = html;
    this.root.querySelectorAll("[data-act]").forEach((el) => el.addEventListener("click", () => this.cb.action(el.dataset.act, sel)));
    this.drawHeat(st, b, hasCell ? sel : null);
  }

  stats(b) {
    const d = b.mat.data; let mn = Infinity, mx = -Infinity, s = 0, sa = 0, n = 0;
    for (let i = 0; i < d.length; i++) { const v = d[i]; if (!Number.isFinite(v)) continue; n++; s += v; sa += Math.abs(v); if (v < mn) mn = v; if (v > mx) mx = v; }
    if (b.kind === "tokens") return "";
    return `<span>min <b>${fmt(mn)}</b></span><span>max <b>${fmt(mx)}</b></span><span>mean <b>${fmt(s / n)}</b></span><span>mean |v| <b>${fmt(sa / n)}</b></span><span>colour scale ±<b>${fmt(b.vmax, 2)}</b></span>`;
  }
  heatHint(b, st) {
    if (b.square) return "rows = token that looks (query), columns = token looked at (key). Hover or click a cell.";
    if (b.rows === st.T && b.kind !== "weight") return "rows = your tokens (top = position 0), columns = dimensions. Hover or click a cell.";
    if (b.rowLabels === "vocab") return "rows = vocabulary tokens, columns = dimensions.";
    return "rows = input dimension, columns = output dimension. Hover or click a cell.";
  }

  // ----------------------------------------------------------- heatmap
  drawHeat(st, b, sel) {
    const canvas = this.root.querySelector("#insp-heat"), tip = this.root.querySelector("#heat-tip");
    const wrapW = Math.max(240, this.root.clientWidth - 28);
    const rowLab = b.kind !== "weight" && b.rows === st.T ? st.tokens.map(prettyToken) : b.rowLabels === "vocab" ? st.model.vocab.map(prettyToken) : null;
    const colLab = b.square ? st.tokens.map(prettyToken) : b.colLabels === "vocab" ? st.model.vocab.map(prettyToken) : null;
    const labW = rowLab ? 26 : 16, labH = colLab ? 22 : 14;
    let cell = Math.floor((wrapW - labW) / b.cols);
    cell = Math.max(2, Math.min(22, cell));
    const w = labW + cell * b.cols, h = labH + cell * b.rows;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    canvas.style.width = w + "px"; canvas.style.height = h + "px";
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#121212"; ctx.fillRect(0, 0, w, h);
    for (let i = 0; i < b.rows; i++) for (let j = 0; j < b.cols; j++) {
      let rgb;
      if (b.mask && b.mask(i, j)) rgb = MASK_RGB;
      else { const v = b.mat.get(i, j); rgb = b.colorMode === "seq" ? sequential(Math.min(1, v / b.vmax)) : b.colorMode === "cat" ? [0.25, 0.45, 0.8] : diverging(Math.max(-1, Math.min(1, v / b.vmax))); }
      ctx.fillStyle = rgbCss(rgb);
      ctx.fillRect(labW + j * cell, labH + i * cell, cell - (cell > 3 ? 1 : 0), cell - (cell > 3 ? 1 : 0));
    }
    ctx.fillStyle = "#c3c2b7"; ctx.font = `${Math.min(11, Math.max(7, cell))}px ui-monospace, monospace`;
    if (rowLab && cell >= 6) { ctx.textAlign = "right"; ctx.textBaseline = "middle"; for (let i = 0; i < b.rows; i++) ctx.fillText(rowLab[i], labW - 4, labH + i * cell + cell / 2); }
    if (colLab && cell >= 6) { ctx.textAlign = "center"; ctx.textBaseline = "bottom"; for (let j = 0; j < b.cols; j++) ctx.fillText(colLab[j], labW + j * cell + cell / 2, labH - 3); }
    if (sel) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.strokeRect(labW + sel.j * cell - 1, labH + sel.i * cell - 1, cell + 1, cell + 1); }
    const hit = (e) => { const r = canvas.getBoundingClientRect(); const j = Math.floor((e.clientX - r.left - labW) / cell), i = Math.floor((e.clientY - r.top - labH) / cell); return i >= 0 && j >= 0 && i < b.rows && j < b.cols ? { i, j } : null; };
    canvas.onmousemove = (e) => { const c = hit(e); if (!c) { tip.style.display = "none"; return; } tip.style.display = "block"; tip.style.left = (e.clientX - this.root.getBoundingClientRect().left + 12) + "px"; tip.style.top = (e.clientY - this.root.getBoundingClientRect().top + 12) + "px"; tip.textContent = `[${c.i}][${c.j}] = ${fmt(b.mask && b.mask(c.i, c.j) ? -Infinity : b.mat.get(c.i, c.j), 4)}`; };
    canvas.onmouseleave = () => { tip.style.display = "none"; };
    canvas.onclick = (e) => { const c = hit(e); if (c) this.cb.selectCell(b, c.i, c.j); };
  }

  // ----------------------------------------------------------- one cell
  cellSection(st, b, i, j) {
    const m = st.model, tr = st.trace, v = b.mask && b.mask(i, j) ? -Infinity : b.mat.get(i, j);
    const head = `<div class="cell-head"><span class="cell-name">${b.label}[${i}][${j}]</span> = <span class="cell-val">${fmt(v, 4)}</span></div>`;
    const L = b.layer !== undefined ? tr.layers[b.layer] : null, P = b.layer !== undefined ? `L${b.layer}.` : "";
    const id = b.id.replace(/^L\d+\./, "");
    const tokRow = (t) => `${tokHtml(st.tokens[t])}<sub>${t}</sub>`;
    let body = "";
    const termsTable = (ex, labelA, labelB, idxName, n = 10) => {
      const terms = [...ex.terms].sort((a, b2) => Math.abs(b2.prod) - Math.abs(a.prod));
      const rows = terms.slice(0, n).map((t) => `<tr><td>${idxName} ${t[Object.keys(t)[0]]}</td><td>${fmt(t[labelA], 3)}</td><td>×</td><td>${fmt(t[labelB], 3)}</td><td>=</td><td class="${t.prod < 0 ? "neg" : "pos"}">${fmt(t.prod, 4)}</td></tr>`).join("");
      const rest = terms.slice(n).reduce((s, t) => s + t.prod, 0);
      return `<table class="terms"><thead><tr><th></th><th>${labelA}</th><th></th><th>${labelB}</th><th></th><th>product</th></tr></thead><tbody>${rows}${terms.length > n ? `<tr class="mut"><td colspan="5">… ${terms.length - n} smaller terms</td><td>${fmt(rest, 4)}</td></tr>` : ""}${ex.bias !== undefined ? `<tr><td colspan="5">+ bias</td><td>${fmt(ex.bias, 4)}</td></tr>` : ""}<tr class="total"><td colspan="5">sum</td><td>${fmt(ex.sum !== undefined ? ex.sum : ex.value, 4)}</td></tr></tbody></table>`;
    };
    switch (id) {
      case "tokens": body = `<p>Position ${i} holds token id <b>${tr.ids[i]}</b> = ${tokHtml(st.tokens[i])}. Integers are all the model receives.</p>`; break;
      case "tok_emb": body = `<p>Learned embedding of vocabulary token <b>${tokHtml(m.vocab[i])}</b> (id ${i}), dimension ${j}. Set by training; identical for every input. ${b.tied ? "" : ""}</p>`; break;
      case "unembed": body = `<p>Row ${i} = the embedding of ${tokHtml(m.vocab[i])} reused as its <b>unembedding</b> direction: the logit for ${tokHtml(m.vocab[i])} is the dot product of a token's final vector with this row (tied weights).</p>`; break;
      case "pos_emb": body = `<p>Learned position embedding for position <b>${i}</b>, dimension ${j}. Added to whichever token sits at position ${i}.</p>`; break;
      case "tokRows": body = `<p>Copied from W<sub>tok_emb</sub>[${tr.ids[i]}][${j}] — row ${tr.ids[i]} is the embedding of ${tokHtml(st.tokens[i])}. A lookup, no arithmetic.</p>`; break;
      case "posRows": body = `<p>Copied from W<sub>pos_emb</sub>[${i}][${j}] — “this token is at position ${i}”.</p>`; break;
      case "x0": body = `<div class="formula">x₀[${i}][${j}] = tok_emb[${tr.ids[i]}][${j}] + pos_emb[${i}][${j}] = ${fmt(tr.tokRows.get(i, j), 4)} ${sign(tr.posRows.get(i, j))} ${fmt(Math.abs(tr.posRows.get(i, j)), 4)} = <b>${fmt(v, 4)}</b></div>`; break;
      case "ln1": case "ln2": case "lnf": {
        const x = id === "ln1" ? L.xin : id === "ln2" ? L.x1 : tr.layers[m.L - 1].x2;
        const mean = id === "ln1" ? L.ln1_mean : id === "ln2" ? L.ln2_mean : tr.lnf_mean, rstd = id === "ln1" ? L.ln1_rstd : id === "ln2" ? L.ln2_rstd : tr.lnf_rstd;
        const g = id === "lnf" ? m.t("lnf_g") : m.layerT(b.layer, id + "_g"), bb = id === "lnf" ? m.t("lnf_b") : m.layerT(b.layer, id + "_b");
        const ex = m.explainLayerNorm(x, mean, rstd, g, bb, i, j);
        body = `<div class="formula">(x − mean<sub>row</sub>) · 1/std<sub>row</sub> · γ[${j}] + β[${j}]<br>= (${fmt(ex.x, 4)} − ${fmt(ex.mean, 4)}) × ${fmt(ex.rstd, 3)} × ${fmt(ex.g, 3)} ${sign(ex.b)} ${fmt(Math.abs(ex.b), 3)} = <b>${fmt(ex.value, 4)}</b></div><p class="mut small">mean and std are over all ${m.d} dimensions of token ${i}'s vector; γ, β are learned per dimension.</p>`; break; }
      case "q": case "k": case "v": {
        const W = m.layerT(b.layer, "w" + id), bias = m.layerT(b.layer, "b" + id), ex = m.explainLinear(L.ln1, W, bias, i, j), h = Math.floor(j / m.dh);
        body = `<div class="formula">${id}[${i}][${j}] = Σ<sub>n</sub> LN₁(x)[${i}][n] · W<sub>${id.toUpperCase()}</sub>[n][${j}] + b[${j}] &nbsp;<span class="mut">(${m.d} terms, head ${h}, head-dim ${j - h * m.dh})</span></div>` + termsTable(ex, "x", "w", "n"); break; }
      case "headOut": {
        const ex = m.explainHeadOut(tr, b.layer, i, j);
        body = `<div class="formula">out[${i}][${j}] = Σ<sub>s≤${i}</sub> attn<sub>h${ex.h}</sub>[${i}][s] · v[s][${j}]</div><table class="terms"><thead><tr><th>s</th><th>attn</th><th></th><th>v</th><th></th><th>product</th></tr></thead><tbody>${[...ex.terms].sort((a, c) => Math.abs(c.prod) - Math.abs(a.prod)).slice(0, 12).map((t) => `<tr><td>${tokRow(t.s)}</td><td>${pct(t.a)}</td><td>×</td><td>${fmt(t.v, 3)}</td><td>=</td><td class="${t.prod < 0 ? "neg" : "pos"}">${fmt(t.prod, 4)}</td></tr>`).join("")}<tr class="total"><td colspan="5">sum</td><td>${fmt(ex.sum, 4)}</td></tr></tbody></table>`; break; }
      case "attnOut": { const ex = m.explainLinear(L.headOut, m.layerT(b.layer, "wo"), m.layerT(b.layer, "bo"), i, j); body = `<div class="formula">attn_out[${i}][${j}] = Σ<sub>n</sub> heads[${i}][n] · W<sub>O</sub>[n][${j}] + b[${j}]</div>` + termsTable(ex, "heads", "w", "n"); break; }
      case "x1": { const prev = b.layer === 0 ? tr.x0 : tr.layers[b.layer - 1].x2; body = `<div class="formula">x₁[${i}][${j}] = x[${i}][${j}] + attn_out[${i}][${j}] = ${fmt(prev.get(i, j), 4)} ${sign(L.attnOut.get(i, j))} ${fmt(Math.abs(L.attnOut.get(i, j)), 4)} = <b>${fmt(v, 4)}</b></div>`; break; }
      case "mlpPre": { const ex = m.explainLinear(L.ln2, m.layerT(b.layer, "win"), m.layerT(b.layer, "bin"), i, j); body = `<div class="formula">pre[${i}][${j}] = Σ<sub>n</sub> LN₂(x)[${i}][n] · W<sub>in</sub>[n][${j}] + b[${j}] &nbsp;<span class="mut">(neuron ${j}'s match score)</span></div>` + termsTable(ex, "x", "w", "n"); break; }
      case "mlpAct": { const ex = m.explainGelu(L.mlpPre.get(i, j)); body = `<div class="formula">GELU(${fmt(ex.pre, 4)}) = ½·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))) = <b>${fmt(ex.value, 4)}</b></div><p class="mut small">Neuron ${j} ${ex.value > 0.1 ? "<b>fires</b>" : "stays quiet"} for token ${i}.</p>`; break; }
      case "mlpOut": { const ex = m.explainLinear(L.mlpAct, m.layerT(b.layer, "wout"), m.layerT(b.layer, "bout"), i, j); body = `<div class="formula">mlp_out[${i}][${j}] = Σ<sub>n</sub> GELU[${i}][n] · W<sub>out</sub>[n][${j}] + b[${j}] &nbsp;<span class="mut">(${m.dff} neurons)</span></div>` + termsTable(ex, "gelu", "w", "neuron"); break; }
      case "x2": body = `<div class="formula">x₂[${i}][${j}] = x₁[${i}][${j}] + mlp_out[${i}][${j}] = ${fmt(L.x1.get(i, j), 4)} ${sign(L.mlpOut.get(i, j))} ${fmt(Math.abs(L.mlpOut.get(i, j)), 4)} = <b>${fmt(v, 4)}</b></div>`; break;
      case "logits": { const ex = m.explainLogit(tr, i, j); body = `<div class="formula">logit[${i}][${tokHtml(m.vocab[j])}] = Σ<sub>n</sub> LN<sub>f</sub>(x)[${i}][n] · W<sub>unembed</sub>[${j}][n]</div>` + termsTable(ex, "x", "e", "n"); break; }
      case "probs": { const ex = m.explainProb(tr, i, j); body = `<div class="formula">p[${i}][${tokHtml(m.vocab[j])}] = e<sup>${fmt(ex.logit, 3)} − ${fmt(ex.max, 3)}</sup> / ${fmt(ex.denom, 3)} = <b>${pct(ex.value)}</b></div><p class="mut small">(max subtracted for numerical stability — it cancels out)</p>`; break; }
      default: {
        const sm = id.match(/^(scores|attn)\.(\d+)$/);
        if (sm && sm[1] === "scores") {
          if (j > i) body = `<p><b>Masked.</b> Token ${i} may not look at position ${j} (the future). Set to −∞ → probability 0.</p>`;
          else { const ex = m.explainScore(tr, b.layer, +sm[2], i, j); body = `<div class="formula">score = (q<sub>${i}</sub> · k<sub>${j}</sub>) / √${m.dh} &nbsp; ${tokRow(i)} looking at ${tokRow(j)}</div><table class="terms"><thead><tr><th>dim</th><th>q</th><th></th><th>k</th><th></th><th>product</th></tr></thead><tbody>${ex.terms.map((t) => `<tr><td>${t.i}</td><td>${fmt(t.q, 3)}</td><td>×</td><td>${fmt(t.k, 3)}</td><td>=</td><td class="${t.prod < 0 ? "neg" : "pos"}">${fmt(t.prod, 4)}</td></tr>`).join("")}<tr class="total"><td colspan="5">dot product ${fmt(ex.dot, 4)} × 1/√${m.dh}</td><td>${fmt(ex.value, 4)}</td></tr></tbody></table>`; }
        } else if (sm) {
          if (j > i) body = `<p><b>Masked</b>: exactly 0 — the causal mask.</p>`;
          else { const ex = m.explainSoftmax(tr, b.layer, +sm[2], i, j); body = `<div class="formula">attn[${i}][${j}] = e<sup>score − max</sup> / Σ<sub>s≤${i}</sub> e<sup>score<sub>s</sub> − max</sup> = e<sup>${fmt(ex.exps[j].score, 3)} − ${fmt(ex.max, 3)}</sup> / ${fmt(ex.denom, 3)} = <b>${pct(ex.value)}</b></div><table class="terms"><thead><tr><th>s</th><th>score</th><th>e^(score−max)</th><th>weight</th></tr></thead><tbody>${ex.exps.map((e) => `<tr class="${e.s === j ? "total" : ""}"><td>${tokRow(e.s)}</td><td>${fmt(e.score, 3)}</td><td>${fmt(e.e, 4)}</td><td>${pct(e.e / ex.denom)}</td></tr>`).join("")}</tbody></table>`; }
        } else if (b.kind === "weight") {
          body = `<p>Learned parameter ${b.label}[${i}][${j}] = <b>${fmt(v, 4)}</b>. ${b.headGroups ? `Column ${j} belongs to head ${Math.floor(j / (b.cols / b.headGroups))}.` : ""} Fixed after training; the same for every input.</p>`;
        }
      }
    }
    return `<div class="cell-sec">${head}${body}</div>`;
  }

  // ----------------------------------------------------------- row context
  rowSection(st, b, t) {
    const m = st.model, tr = st.trace; let html = "";
    if (b.kind === "weight" || b.kind === "tokens") return "";
    if (b.square) {
      const sm = b.id.match(/(scores|attn)\.(\d+)$/); const A = tr.layers[b.layer].attn[+sm[2]];
      const row = []; for (let s = 0; s <= t; s++) row.push([s, A.get(t, s)]); row.sort((a, c) => c[1] - a[1]);
      html += `<div class="row-sec"><div class="panel-title">who does ${tokHtml(st.tokens[t])}<sub>${t}</sub> listen to? <span class="mut">(head ${sm[2]}, layer ${b.layer})</span></div><div class="bars">${row.slice(0, 8).map(([s, p]) => `<div class="bar-row"><span class="bar-lab">${tokHtml(st.tokens[s])}<sub>${s}</sub></span><div class="bar"><div class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></div></div><span class="bar-val">${pct(p)}</span></div>`).join("")}</div></div>`;
    }
    if (b.id === "logits" || b.id === "probs") {
      const temp = st.temperature || 1; const z = tr.logits.row(t); const scaled = new Float32Array(z.length); for (let i = 0; i < z.length; i++) scaled[i] = z[i] / temp;
      let mx = -Infinity; for (const v of scaled) mx = Math.max(mx, v); let den = 0; const p = new Float32Array(z.length); for (let i = 0; i < z.length; i++) { p[i] = Math.exp(scaled[i] - mx); den += p[i]; } for (let i = 0; i < z.length; i++) p[i] /= den;
      const top = m.topK(p, 8);
      html += `<div class="row-sec"><div class="panel-title">next-token odds after ${tokHtml(st.tokens[t])}<sub>${t}</sub> <span class="mut">(temperature ${temp.toFixed(2)})</span></div><div class="bars">${top.map((x) => `<div class="bar-row"><span class="bar-lab">${tokHtml(x.token)}</span><div class="bar"><div class="bar-fill" style="width:${(x.p * 100).toFixed(1)}%"></div></div><span class="bar-val">${pct(x.p)} <span class="mut">logit ${fmt(z[x.id], 2)}</span></span></div>`).join("")}</div></div>`;
    }
    if (b.residual) {
      const ll = m.logitLens(tr, b.layerIdx === undefined ? b.layer : b.layerIdx); const top = m.topK(ll.probs.row(t), 5);
      html += `<div class="row-sec"><div class="panel-title">logit lens <span class="mut">— if the model stopped here, token ${t} would predict:</span></div><div class="bars">${top.map((x) => `<div class="bar-row"><span class="bar-lab">${tokHtml(x.token)}</span><div class="bar"><div class="bar-fill" style="width:${(x.p * 100).toFixed(1)}%"></div></div><span class="bar-val">${pct(x.p)}</span></div>`).join("")}</div></div>`;
    }
    if (b.id.endsWith("mlpAct")) {
      const M = b.mat; const act = []; for (let n = 0; n < M.c; n++) act.push([n, M.get(t, n)]); act.sort((a, c) => c[1] - a[1]); const fired = act.filter((a) => a[1] > 0.1).length;
      html += `<div class="row-sec"><div class="panel-title">neurons firing for ${tokHtml(st.tokens[t])}<sub>${t}</sub> <span class="mut">${fired} of ${M.c} above 0.1</span></div><div class="chips">${act.slice(0, 12).map(([n, a]) => `<span class="chip">#${n} <b>${fmt(a, 2)}</b></span>`).join("")}</div></div>`;
    }
    return html;
  }
}

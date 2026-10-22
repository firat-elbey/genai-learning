// main.js — app shell: loads a model, runs the live forward pass, builds the
// 3D layout, and wires the walkthrough, lenses, inspector and generation loop.
import { loadGPT } from "./gpt.js";
import { buildLayout, dependencies } from "./layout.js";
import { Scene3D } from "./scene.js";
import { STEPS, stepIndex } from "./walkthrough.js";
import { Inspector } from "./inspector.js";
import { esc, prettyToken, fmt, pct, big, human } from "./util.js";

const MODEL_FILES = { shakespeare: "models/shakespeare.json", sort: "models/sort.json" };
const EXAMPLES = {
  shakespeare: ["ROMEO:\nBut, soft! what light", "To be, or not to be: that is the", "First Citizen:\nBefore we proceed any", "KING HENRY:\nWhat say you, my lo", "the quick brown fox jumps ove"],
  sort: ["CABCBA", "BBACAB", "CCCAAA", "ABCABC", "BACBAC"],
};
const REAL_GPTS = [
  { name: "GPT-2 small", vocab_size: 50257, d_model: 768, n_layer: 12, n_head: 12, d_ff: 3072, block_size: 1024, tied: true },
  { name: "GPT-2 XL", vocab_size: 50257, d_model: 1600, n_layer: 48, n_head: 25, d_ff: 6400, block_size: 1024, tied: true },
  { name: "GPT-3 175B", vocab_size: 50257, d_model: 12288, n_layer: 96, n_head: 96, d_ff: 49152, block_size: 2048, tied: false },
];

const $ = (id) => document.getElementById(id);

const app = {
  cache: {}, model: null, meta: null, trace: null, layout: null,
  stepIdx: 0, layer: 0, head: 0, sel: null, lens: {}, temperature: 1, topk: 0,
  text: "", genIds: [], auto: null, ghosts: null, busy: false,

  async init() {
    this.scene = new Scene3D($("view3d"), $("labels"), {
      onClick: (hit) => hit ? this.selectCell(hit.block, hit.i, hit.j) : null,
      onHover: (hit, e) => this.hover(hit, e),
      onLabelClick: (b) => this.selectBlock(b, true),
    });
    this.inspector = new Inspector($("inspector"), {
      action: (act, sel) => this.inspectorAction(act, sel),
      selectCell: (b, i, j) => this.selectCell(b, i, j),
    });
    this.wire();
    const hash = new URLSearchParams(location.hash.slice(1));
    const name = MODEL_FILES[hash.get("model")] ? hash.get("model") : "shakespeare";
    $("model-pick").value = name;
    await this.setModel(name, hash.get("text"));
    const s = stepIndex(hash.get("step")); this.goStep(s >= 0 ? s : 0);
  },

  // ----------------------------------------------------------- model / input
  async setModel(name, text) {
    this.stopAuto();
    $("loading").style.display = "flex"; $("loading").textContent = `loading ${name} weights…`;
    try {
      if (!this.cache[name]) this.cache[name] = await loadGPT(MODEL_FILES[name]);
    } catch (e) { $("loading").textContent = "Could not load the model files. If you opened this page from disk, serve it over http (python3 -m http.server) — see the README."; throw e; }
    this.model = this.cache[name]; this.meta = this.model.meta; this.name = name;
    this.layer = 0; this.head = 0; this.sel = null; this.lens = {}; this.genIds = [];
    this.text = text != null ? text : EXAMPLES[name][0];
    this.setupInputUI();
    this.renderModelInfo();
    if (this.isSort()) this.completeSort();
    this.recompute({ fly: true });
    $("loading").style.display = "none";
  },
  isSort() { return this.meta.task === "sort"; },
  // greedy-complete the sorter's answer so all 11 positions are visible
  completeSort() {
    this.genIds = [];
    while (this.meta.n_in + this.genIds.length < this.model.T_max) {
      const tr = this.model.forward(this.currentIds());
      this.genIds.push(this.model.topK(tr.probs.row(tr.T - 1), 1)[0].id);
    }
  },
  setupInputUI() {
    const ta = $("prompt"), L = this.model.T_max;
    ta.maxLength = this.isSort() ? this.meta.n_in : L;
    ta.value = this.text; ta.rows = this.isSort() ? 1 : 2;
    $("prompt-help").textContent = this.isSort()
      ? `exactly ${this.meta.n_in} letters from A, B, C — the model appends its sorted answer`
      : `up to ${L} characters from the training alphabet (letters, punctuation, space, newline)`;
    $("examples").innerHTML = EXAMPLES[this.name].map((e) => `<button class="chip-btn" data-ex="${esc(e)}">${esc(prettyToken(e.replace(/\n/g, "⏎")).slice(0, 26))}${e.length > 26 ? "…" : ""}</button>`).join("");
    $("examples").querySelectorAll("[data-ex]").forEach((b) => b.addEventListener("click", () => { ta.value = b.dataset.ex; this.onInput(); }));
    $("layer-pick").innerHTML = Array.from({ length: this.model.L }, (_, l) => `<option value="${l}">layer ${l}</option>`).join("");
    $("head-pick").innerHTML = Array.from({ length: this.model.H }, (_, h) => `<option value="${h}">head ${h}</option>`).join("");
  },
  currentIds() {
    let ids;
    if (this.isSort()) {
      const letters = this.text.toUpperCase().replace(/[^ABC]/g, "").slice(0, this.meta.n_in);
      ids = this.model.encode(letters);
      while (ids.length < this.meta.n_in) ids.push(0);
      ids = ids.concat(this.genIds).slice(0, this.model.T_max);
    } else {
      ids = this.model.encode(this.text);
      if (ids.length === 0) ids = this.model.encode("T");
      ids = ids.slice(-this.model.T_max);
    }
    return ids;
  },
  onInput() {
    const v = $("prompt").value;
    if (this.isSort()) { this.text = v.toUpperCase().replace(/[^ABC]/g, "").slice(0, this.meta.n_in); $("prompt").value = this.text; this.completeSort(); }
    else this.text = v;
    clearTimeout(this._deb); this._deb = setTimeout(() => this.recompute({ keepCamera: true }), 180);
  },

  recompute({ fly = false, keepCamera = false } = {}) {
    const ids = this.currentIds();
    const t0 = performance.now();
    this.trace = this.model.forward(ids);
    const ms = performance.now() - t0;
    this.layout = buildLayout(this.model, this.trace);
    this.scene.build(this.layout, this.trace.tokens);
    this.scene.setLens(this.lens);
    $("stat").innerHTML = `${this.trace.T} tokens · ${this.layout.blocks.length} tensors · ${big(this.layout.blocks.reduce((s, b) => s + b.rows * b.cols, 0))} cells · forward pass ${ms.toFixed(1)} ms`;
    this.renderTokenStrip();
    this.renderStepBody();
    if (fly) this.scene.flyToAll({ dur: 1200 });
    else if (!keepCamera) this.applyStepCamera();
    this.reselect();
    this.updateHash();
  },
  renderTokenStrip() {
    const t = this.trace, nIn = this.isSort() ? this.meta.n_in : t.T;
    $("tokstrip").innerHTML = t.tokens.map((tok, i) => `<span class="tk ${i >= nIn ? "gen" : ""} ${this.lens.token === i ? "on" : ""}" data-pos="${i}" title="position ${i} · id ${t.ids[i]}">${esc(prettyToken(tok))}</span>`).join("");
    $("tokstrip").querySelectorAll(".tk").forEach((el) => el.addEventListener("click", () => this.setLens({ ...this.lens, token: this.lens.token === +el.dataset.pos ? null : +el.dataset.pos })));
    const top = this.model.topK(this.trace.probs.row(t.T - 1), 3);
    $("nextpred").innerHTML = `next: ${top.map((x) => `<span class="tk">${esc(prettyToken(x.token))}</span> ${pct(x.p)}`).join(" · ")}`;
  },
  renderModelInfo() {
    const m = this.meta, tr = m.training || {};
    $("model-info").innerHTML = `<b>${esc(m.title)}</b> — ${esc(m.description)}<br><span class="mut">${big(m.n_params)} parameters · d=${m.config.d_model}, ${m.config.n_layer} layers, ${m.config.n_head} heads, ${m.config.d_ff} MLP neurons, context ${m.config.block_size}, vocab ${m.config.vocab_size}. Trained ${tr.steps ? `${big(tr.steps)} steps (${tr.minutes} min on CPU), final loss ${tr.final_val_loss}` : ""}${tr.sequence_accuracy !== undefined ? `, ${(tr.sequence_accuracy * 100).toFixed(1)}% of fresh inputs sorted perfectly` : ""}.</span>`;
  },

  // ----------------------------------------------------------- steps
  stateSnapshot() {
    const m = this.model;
    return { model: m, meta: this.meta, trace: this.trace, layout: this.layout, tokens: this.trace.tokens, T: this.trace.T, d: m.d, H: m.H, dh: m.dh, dff: m.dff, V: m.V, L: m.L, cfg: m.cfg, layer: this.layer, head: this.head, temperature: this.temperature, nIn: this.meta.n_in, pretty: prettyToken };
  },
  goStep(i) {
    this.stepIdx = Math.max(0, Math.min(STEPS.length - 1, i));
    const step = STEPS[this.stepIdx], st = this.stateSnapshot();
    document.querySelectorAll("#steps .step-btn").forEach((b, k) => b.classList.toggle("active", k === this.stepIdx));
    $("step-title").textContent = `${this.stepIdx + 1}. ${step.title}`;
    this.renderStepBody();
    this.lens = step.lens(st) || {}; this.syncLensUI(); this.scene.setLens(this.lens);
    if (step.id === "scale") this.renderScale(); else this.hideGhosts();
    this.applyStepCamera();
    const sel = step.select(st);
    if (sel) { const b = this.layout.byId.get(sel.id); if (b) this.selectCell(b, sel.i, sel.j, { fly: false }); }
    else { this.sel = null; this.scene.showSelection(null); this.inspector.render(st, null); }
    $("prev-step").disabled = this.stepIdx === 0; $("next-step").disabled = this.stepIdx === STEPS.length - 1;
    this.updateHash();
    $("guide").scrollTop = 0;
  },
  renderStepBody() {
    if (!this.trace) return;
    const step = STEPS[this.stepIdx];
    $("step-body").innerHTML = step.body(this.stateSnapshot());
    if (step.id === "scale") this.renderScaleTable();
  },
  applyStepCamera() {
    const step = STEPS[this.stepIdx], ids = step.frame(this.stateSnapshot());
    if (ids === null) this.scene.flyToAll({ dur: 1000 }); else this.scene.flyToBlocks(ids, { dur: 1000 });
  },

  // ----------------------------------------------------------- selection
  selectCell(block, i, j, { fly = false } = {}) {
    this.sel = { block, i, j };
    const deps = dependencies(this.layout, this.model, this.trace, block, i, j);
    this.scene.showSelection(block, i, j, deps);
    this.inspector.render(this.stateSnapshot(), this.sel);
    if (fly) this.scene.flyToBlocks([block.id, ...deps.map((d) => d.id)], { dur: 800 });
    if (window.innerWidth <= 900) this.showMobileTab("inspect");
  },
  selectBlock(block, fly) {
    this.sel = { block };
    this.scene.showSelection(null);
    this.inspector.render(this.stateSnapshot(), this.sel);
    if (fly) this.scene.flyToBlocks([block.id], { dur: 800 });
  },
  reselect() {
    if (!this.sel) { this.inspector.render(this.stateSnapshot(), null); return; }
    const b = this.layout.byId.get(this.sel.block.id);
    if (!b) { this.sel = null; this.scene.showSelection(null); this.inspector.render(this.stateSnapshot(), null); return; }
    if (this.sel.i !== undefined) { const i = Math.min(this.sel.i, b.rows - 1), j = Math.min(this.sel.j, b.cols - 1); this.selectCell(b, i, j); }
    else this.selectBlock(b, false);
  },
  inspectorAction(act, sel) {
    const b = sel.block;
    if (act === "focus") this.scene.flyToBlocks([b.id], { dur: 800 });
    if (act === "tokenlens") this.setLens({ ...this.lens, token: sel.i });
    if (act === "headlens") this.setLens({ ...this.lens, head: b.head, layer: b.layer });
    if (act === "clearlens") this.setLens({});
  },
  hover(hit, e) {
    const tip = $("tooltip");
    if (!hit) { tip.style.display = "none"; return; }
    const b = hit.block, v = b.mask && b.mask(hit.i, hit.j) ? -Infinity : b.mat.get(hit.i, hit.j);
    let extra = "";
    if (b.rows === this.trace.T && b.kind !== "weight") extra = ` · token ${hit.i} ${esc(prettyToken(this.trace.tokens[hit.i]))}`;
    if (b.square) extra += ` → ${esc(prettyToken(this.trace.tokens[hit.j]))}`;
    if (b.colLabels === "vocab") extra += ` · ${esc(prettyToken(this.model.vocab[hit.j]))}`;
    if (b.rowLabels === "vocab") extra = ` · ${esc(prettyToken(this.model.vocab[hit.i]))}`;
    tip.innerHTML = `<b>${b.label}</b>[${hit.i}][${hit.j}] = ${fmt(v, 4)}${extra}`;
    tip.style.display = "block";
    const r = $("stage").getBoundingClientRect();
    tip.style.left = Math.min(r.width - 260, e.clientX - r.left + 14) + "px"; tip.style.top = (e.clientY - r.top + 14) + "px";
  },

  // ----------------------------------------------------------- lenses & view
  setLens(lens) {
    this.lens = lens; if (lens.layer !== undefined && lens.layer !== null) this.layer = lens.layer; if (lens.head !== undefined && lens.head !== null) this.head = lens.head;
    this.syncLensUI(); this.scene.setLens(lens); this.renderTokenStrip();
  },
  syncLensUI() {
    $("layer-pick").value = this.layer; $("head-pick").value = this.head;
    $("lens-layer").checked = this.lens.layer !== undefined && this.lens.layer !== null;
    $("lens-head").checked = this.lens.head !== undefined && this.lens.head !== null;
    $("lens-token").value = this.lens.token !== undefined && this.lens.token !== null ? this.lens.token : "";
    $("lens-token").innerHTML = `<option value="">no token lens</option>` + this.trace.tokens.map((t, i) => `<option value="${i}">${i}: ${esc(prettyToken(t))}</option>`).join("");
    $("lens-token").value = this.lens.token !== undefined && this.lens.token !== null ? this.lens.token : "";
  },
  lensFromUI() {
    const lens = {};
    if ($("lens-layer").checked) lens.layer = this.layer;
    if ($("lens-head").checked) { lens.head = this.head; lens.layer = this.layer; $("lens-layer").checked = true; }
    const t = $("lens-token").value; if (t !== "") lens.token = +t;
    this.setLens(lens);
  },

  // ----------------------------------------------------------- generation
  generateNext() {
    const T = this.trace.T;
    if (this.isSort() && T >= this.model.T_max) { this.stopAuto(); return false; }
    const { id } = this.model.sampleNext(this.trace.logits.row(T - 1), { temperature: this.temperature, topk: this.topk });
    if (this.isSort()) this.genIds.push(id);
    else { this.text += this.model.vocab[id]; if (this.text.length > this.model.T_max) this.text = this.text.slice(-this.model.T_max); $("prompt").value = this.text; }
    this.recompute({ keepCamera: true });
    return true;
  },
  startAuto() {
    if (this.auto) return this.stopAuto();
    $("gen-auto").textContent = "⏸ stop";
    const speed = +$("gen-speed").value;
    this.auto = setInterval(() => { if (!this.generateNext()) this.stopAuto(); }, 1400 / speed);
  },
  stopAuto() { if (this.auto) clearInterval(this.auto); this.auto = null; $("gen-auto").textContent = "▶ auto"; },

  // ----------------------------------------------------------- scale step
  paramRows(cfg, tied) {
    const { vocab_size: V, d_model: d, n_layer: L, n_head: H, d_ff: dff } = cfg, dh = d / H;
    const rows = [
      ["Embedding", "V × d", V * d],
      ["Key", "d_head × d × heads × layers", dh * d * H * L],
      ["Query", "d_head × d × heads × layers", dh * d * H * L],
      ["Value (W_V)", "d_head × d × heads × layers", dh * d * H * L],
      ["Output (W_O)", "d × d_head × heads × layers", d * dh * H * L],
      ["Up-projection (W_in)", "d_ff × d × layers", dff * d * L],
      ["Down-projection (W_out)", "d × d_ff × layers", d * dff * L],
      ["Unembedding", tied ? "tied to embedding (0 extra)" : "V × d", tied ? 0 : V * d],
    ];
    return { rows, total: rows.reduce((s, r) => s + r[2], 0) };
  },
  renderScaleTable() {
    const el = $("scale-table"); if (!el) return;
    const toy = { ...this.model.cfg, name: this.meta.title }, cols = [toy, ...REAL_GPTS];
    const calcs = cols.map((c) => this.paramRows(c, c === toy ? true : c.tied));
    const dims = [["d_model", (c) => c.d_model], ["layers", (c) => c.n_layer], ["heads", (c) => c.n_head], ["d_head", (c) => c.d_model / c.n_head], ["d_ff (neurons/layer)", (c) => c.d_ff], ["vocabulary", (c) => c.vocab_size], ["context", (c) => c.block_size]];
    let h = `<div class="scale-wrap"><table class="scale"><thead><tr><th></th>${cols.map((c) => `<th>${esc(c.name)}</th>`).join("")}</tr></thead><tbody>`;
    for (const [n, f] of dims) h += `<tr><td>${n}</td>${cols.map((c) => `<td>${big(f(c))}</td>`).join("")}</tr>`;
    h += `<tr class="sep"><td colspan="${cols.length + 1}">parameters (weights only, as in 3Blue1Brown's table)</td></tr>`;
    calcs[0].rows.forEach((r, i) => { h += `<tr><td>${r[0]} <span class="mut">${r[1]}</span></td>${calcs.map((c) => `<td>${big(c.rows[i][2])}</td>`).join("")}</tr>`; });
    h += `<tr class="total"><td>total</td>${calcs.map((c) => `<td>${big(c.total)} <span class="mut">(${human(c.total)})</span></td>`).join("")}</tr>`;
    h += `<tr><td class="mut">+ biases & LayerNorm (this model)</td><td>${big(this.meta.n_params - calcs[0].total)} → ${big(this.meta.n_params)} total</td><td colspan="${cols.length - 1}" class="mut">GPT-3's published total is 175,181,291,520 (3Blue1Brown's table, biases excluded)</td></tr>`;
    h += `</tbody></table></div><div class="row"><button class="btn" id="ghost-btn">Show GPT-2 / GPT-3 outlines</button><button class="btn" id="fly-toy">fly to this model</button><button class="btn" id="fly-gpt2">fly to GPT-2 small</button><button class="btn" id="fly-gpt3">fly to GPT-3</button></div><p class="mut small">The outlines use the real widths (d_model, d_ff, vocabulary), the real context length as depth (1024 tokens for GPT-2, 2048 for GPT-3 — this model's is ${this.model.T_max}) and this view's vertical pitch per level, so widths, depths and layer counts are all to scale. Nothing inside them is drawn: GPT-2 small alone would be ~85 million tiles per layer.</p>`;
    el.innerHTML = h;
    $("ghost-btn").addEventListener("click", () => this.ghosts ? this.hideGhosts() : this.showGhosts());
    $("fly-toy").addEventListener("click", () => this.scene.flyToAll({ dur: 1200 }));
    $("fly-gpt2").addEventListener("click", () => { if (!this.ghosts) this.showGhosts(); this.scene.flyToBox(this.ghosts[0].stack, { dur: 1500, elev: 0.5, azim: 0.5 }); });
    $("fly-gpt3").addEventListener("click", () => { if (!this.ghosts) this.showGhosts(); this.scene.flyToBox(this.ghosts[1].stack, { dur: 1800, elev: 0.5, azim: 0.5 }); });
  },
  renderScale() { this.renderScaleTable(); },
  showGhosts() {
    const b = this.layout.bounds;
    const g2 = this.scene.addGhost(REAL_GPTS[0], b.maxX + 120, 0x86b6ef, "GPT-2 small");
    const g3 = this.scene.addGhost(REAL_GPTS[2], b.maxX + 120 + g2.width + 1500, 0xf5b942, "GPT-3 175B");
    this.ghosts = [g2, g3];
    this.scene.setFog(false);
    const btn = $("ghost-btn"); if (btn) btn.textContent = "Hide outlines";
    this.scene.flyToBox(g2.stack.clone().union(g3.stack).union(this.scene.boxOf([...this.scene.meshes.keys()])), { dur: 1600, elev: 0.45, azim: 0.35 });
  },
  hideGhosts() { if (!this.ghosts) return; this.scene.clearGhosts(); this.scene.setFog(true); this.ghosts = null; const btn = $("ghost-btn"); if (btn) btn.textContent = "Show GPT-2 / GPT-3 outlines"; },

  // ----------------------------------------------------------- misc UI
  updateHash() {
    const p = new URLSearchParams({ model: this.name, step: STEPS[this.stepIdx].id });
    if (!this.isSort() && this.text.length <= 64) p.set("text", this.text);
    history.replaceState(null, "", "#" + p.toString());
  },
  showMobileTab(name) {
    document.querySelectorAll(".mtab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    document.body.dataset.mtab = name;
  },
  wire() {
    $("steps").innerHTML = STEPS.map((s, i) => `<button class="step-btn" data-i="${i}" title="${esc(s.title)}">${i + 1}. ${s.short}</button>`).join("");
    $("steps").querySelectorAll(".step-btn").forEach((b) => b.addEventListener("click", () => this.goStep(+b.dataset.i)));
    $("prev-step").addEventListener("click", () => this.goStep(this.stepIdx - 1));
    $("next-step").addEventListener("click", () => this.goStep(this.stepIdx + 1));
    $("model-pick").addEventListener("change", (e) => this.setModel(e.target.value).then(() => this.goStep(this.stepIdx)));
    $("prompt").addEventListener("input", () => this.onInput());
    $("layer-pick").addEventListener("change", (e) => { this.layer = +e.target.value; this.lensFromUI(); this.goStep(this.stepIdx); });
    $("head-pick").addEventListener("change", (e) => { this.head = +e.target.value; this.lensFromUI(); this.goStep(this.stepIdx); });
    $("lens-layer").addEventListener("change", () => this.lensFromUI());
    $("lens-head").addEventListener("change", () => this.lensFromUI());
    $("lens-token").addEventListener("change", () => this.lensFromUI());
    $("opt-bars").addEventListener("change", (e) => this.scene.setBars(e.target.checked));
    $("opt-weights").addEventListener("change", (e) => this.scene.setShowWeights(e.target.checked));
    $("opt-labels").addEventListener("change", (e) => this.scene.setShowLabels(e.target.checked));
    $("fit-all").addEventListener("click", () => this.scene.flyToAll({ dur: 1000 }));
    $("gen-next").addEventListener("click", () => this.generateNext());
    $("gen-auto").addEventListener("click", () => this.startAuto());
    $("gen-reset").addEventListener("click", () => { this.stopAuto(); this.genIds = []; if (!this.isSort()) { this.text = EXAMPLES[this.name][0]; $("prompt").value = this.text; } this.recompute({ keepCamera: true }); });
    $("gen-reset").title = "sorter: clear the model's answer so you can add it one letter at a time · Shakespeare: restore the example prompt";
    $("temp").addEventListener("input", (e) => { this.temperature = +e.target.value; $("temp-val").textContent = this.temperature.toFixed(2); this.inspector.render(this.stateSnapshot(), this.sel); });
    $("topk").addEventListener("input", (e) => { this.topk = +e.target.value; $("topk-val").textContent = this.topk === 0 ? "off" : this.topk; });
    document.querySelectorAll(".mtab").forEach((b) => b.addEventListener("click", () => this.showMobileTab(b.dataset.tab)));
    window.addEventListener("keydown", (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
      if (e.key === "ArrowRight" || e.key === "n") this.goStep(this.stepIdx + 1);
      if (e.key === "ArrowLeft" || e.key === "p") this.goStep(this.stepIdx - 1);
      if (e.key === "f") this.scene.flyToAll({ dur: 1000 });
      if (e.key === "g") this.generateNext();
    });
  },
};

window.LLM_APP = app;
app.init().catch((e) => { console.error(e); const el = $("loading"); el.style.display = "flex"; el.textContent = "Something went wrong: " + e.message; });

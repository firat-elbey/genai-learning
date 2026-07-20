// App shell: owns the live pipeline (tokenize -> forward pass -> render) and
// wires every stage's interactions. All heavy lifting lives in model.js/viz.js.

const APP = {
  currentStage: 0,
  tok: null,
  model: null,
  prompt: "",
  tokenized: null, // full tokenization of the prompt (may exceed ctx)
  trace: null,     // forward-pass trace over the last ctx tokens
  lastMs: 0,

  sel: { emb: 0, attnLayer: 0, attnHead: 0, attnTok: 0, mlpTok: 0 },
  gen: { ids: null, promptLen: 0, running: false, timer: null, count: 0 },

  init() {
    this.tok = new Tokenizer(CORPUS_TEXT, 260);
    this.model = new TinyGPT(this.tok, CORPUS_TEXT);
    GAME.load();
    GAME.renderHud();
    GAME.renderMap();

    const input = document.getElementById("prompt-input");
    let deb = null;
    input.addEventListener("input", () => {
      clearTimeout(deb);
      deb = setTimeout(() => this.recompute(), 220);
    });
    document.querySelectorAll(".chip-btn").forEach((b) =>
      b.addEventListener("click", () => { input.value = b.dataset.fill; this.recompute(); }));

    document.querySelectorAll(".quiz-btn").forEach((b) =>
      b.addEventListener("click", () => QUIZ_UI.open(parseInt(b.dataset.stage, 10))));
    document.getElementById("quiz-close").addEventListener("click", () => QUIZ_UI.close());
    document.getElementById("reset-progress").addEventListener("click", () => {
      GAME.reset(); location.reload();
    });
    document.getElementById("unlock-all").addEventListener("click", () => {
      GAME.state.unlockAll = true; GAME.save(); GAME.renderMap();
      toast("All levels unlocked — explorer mode 🗺");
    });

    this.wireAttention();
    this.wirePredict();
    this.wireGenerate();
    this.wirePlayground();

    let rdeb = null;
    window.addEventListener("resize", () => {
      clearTimeout(rdeb);
      rdeb = setTimeout(() => this.renderStage(), 200);
    });

    this.recompute();
    this.showStage(0);
  },

  detok(ids) {
    return ids.map((i) => this.tok.idToToken(i)).join("").replace(/▁/g, " ").trim();
  },

  recompute() {
    this.prompt = document.getElementById("prompt-input").value;
    this.tokenized = this.tok.tokenize(this.prompt);
    const t0 = performance.now();
    this.trace = this.tokenized.ids.length ? this.model.forward(this.tokenized.ids) : null;
    this.lastMs = performance.now() - t0;
    if (this.trace) {
      const last = this.trace.T - 1;
      this.sel.emb = Math.min(this.sel.emb, last);
      this.sel.attnTok = last;
      this.sel.mlpTok = Math.min(this.sel.mlpTok, last);
    }
    this.resetGen();
    this.renderStage();
  },

  showStage(i) {
    this.currentStage = i;
    document.querySelectorAll(".stage").forEach((s) => s.classList.remove("visible"));
    document.getElementById("stage-" + STAGES[i].id).classList.add("visible");
    GAME.renderMap();
    if (i === 6 && !GAME.state.visited.playground) {
      GAME.state.visited.playground = true;
      GAME.save();
      GAME.award(30, "reached the Playground");
    }
    this.renderStage();
    window.scrollTo({ top: 0, behavior: "smooth" });
  },

  refreshStageChrome() {
    GAME.renderMap();
    const i = this.currentStage;
    if (GAME.isDone(i) && i + 1 < STAGES.length) {
      setTimeout(() => this.showStage(i + 1), 1500);
    }
  },

  renderStage() {
    if (!this.trace) return;
    switch (this.currentStage) {
      case 0: this.renderTokens(); break;
      case 1: this.renderVectors(); break;
      case 2: this.renderAttention(); break;
      case 3: this.renderNeurons(); break;
      case 4: this.renderPredict(); break;
      case 5: this.renderGenerate(); break;
      case 6: this.renderPlayground(); break;
    }
  },

  // ---------- shared: token chip strip ----------
  buildStrip(el, tokens, opts = {}) {
    el.innerHTML = "";
    tokens.forEach((t, i) => {
      const chip = document.createElement("span");
      const isPunct = t.length === 1 && /[.,!?;:']/.test(t);
      chip.className = "token " +
        (isPunct ? "punct" : this.tok.isWordStart(t) ? "word-start" : "cont") +
        (opts.selected === i ? " selected" : "");
      const text = this.tok.display(t) || t;
      chip.innerHTML = `${escapeHtml(text)}${opts.showIds ? `<span class="tid">#${opts.ids[i]}</span>` : ""}`;
      chip.title = `token ${i}: "${t}" (id ${opts.ids ? opts.ids[i] : "?"})`;
      if (opts.onClick) chip.addEventListener("click", () => opts.onClick(i));
      el.appendChild(chip);
    });
  },

  // ---------- stage 0: tokens ----------
  renderTokens() {
    const tk = this.tokenized;
    this.buildStrip(document.getElementById("tok-strip"), tk.tokens, { showIds: true, ids: tk.ids });
    document.getElementById("tok-count").textContent =
      `· ${tk.tokens.length} token${tk.tokens.length === 1 ? "" : "s"} from ${this.prompt.length} characters · vocabulary: ${this.tok.size} tokens` +
      (tk.tokens.length > this.model.cfg.ctx ? ` · model context is ${this.model.cfg.ctx} — it will see only the last ${this.model.cfg.ctx}` : "");
    document.getElementById("tok-ids").textContent =
      "what the model receives: [" + tk.ids.join(", ") + "]";
  },

  // ---------- stage 1: vectors ----------
  renderVectors() {
    const tr = this.trace;
    this.buildStrip(document.getElementById("emb-strip"), tr.tokens, {
      selected: this.sel.emb,
      onClick: (i) => { this.sel.emb = i; this.renderVectors(); },
    });
    const i = this.sel.emb;
    let scale = 0;
    const sum = new Float32Array(tr.embedded[i].length);
    for (let k = 0; k < sum.length; k++) {
      sum[k] = tr.embedded[i][k] + tr.posPart[i][k];
      scale = Math.max(scale, Math.abs(sum[k]), Math.abs(tr.embedded[i][k]), Math.abs(tr.posPart[i][k]));
    }
    VIZ.drawVectorStrip(document.getElementById("emb-content"), tr.embedded[i], { scale });
    VIZ.drawVectorStrip(document.getElementById("emb-pos"), tr.posPart[i], { scale });
    VIZ.drawVectorStrip(document.getElementById("emb-sum"), sum, { scale });

    const id = tr.ids[i];
    const nn = this.model.nearestNeighbors(id, 6)
      .filter(([, s]) => s > 0.15)
      .map(([j, s]) => `<span class="nn-item">${escapeHtml(this.tok.display(this.tok.idToToken(j)))} · ${s.toFixed(2)}</span>`)
      .join(" ");
    document.getElementById("emb-nn").innerHTML =
      `Nearest vectors to <b>${escapeHtml(this.tok.display(tr.tokens[i]))}</b> (cosine similarity): ${nn || "<span class='mut'>no close neighbors — rare token</span>"}`;

    // 2D map
    const pts = [];
    const freqIds = [];
    for (let v = 0; v < this.tok.size; v++) {
      const t = this.tok.idToToken(v);
      if (this.model.uniCount[v] >= 3 && t.startsWith("▁") && t.length > 2) freqIds.push(v);
    }
    freqIds.sort((a, b) => this.model.uniCount[b] - this.model.uniCount[a]);
    const activeIds = new Set(tr.ids);
    freqIds.forEach((v, rank) => {
      pts.push({
        x: this.model.coords2d[v][0], y: this.model.coords2d[v][1],
        label: this.tok.display(this.tok.idToToken(v)),
        kind: "vocab", showLabel: rank < 42 && !activeIds.has(v),
      });
    });
    const seen = new Set();
    for (const v of tr.ids) {
      const t = this.tok.idToToken(v);
      if (seen.has(v) || !t.startsWith("▁")) continue;
      seen.add(v);
      pts.push({
        x: this.model.coords2d[v][0], y: this.model.coords2d[v][1],
        label: this.tok.display(t), kind: "active",
      });
    }
    VIZ.drawEmbeddingMap(document.getElementById("emb-map"), pts);
  },

  // ---------- stage 2: attention ----------
  wireAttention() {
    buildTabs("attn-layer-tabs", ["Layer 1", "Layer 2", "Layer 3"], (i) => {
      this.sel.attnLayer = i; this.renderAttention();
    });
    buildTabs("attn-head-tabs", ["Head 1", "Head 2", "Head 3", "Head 4"], (i) => {
      this.sel.attnHead = i; this.renderAttention();
    });
    const canvas = document.getElementById("attn-heatmap");
    // pointermove covers mouse hover; pointerdown covers taps on touch screens
    const readCell = (e) => {
      const g = this._attnGeom;
      if (!g) return;
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left, y = e.clientY - r.top;
      const col = Math.floor((x - g.labelSpace) / g.cell);
      const row = Math.floor((y - g.labelSpace) / g.cell);
      const out = document.getElementById("attn-hover");
      if (row >= 0 && row < g.T && col >= 0 && col <= row) {
        const tr = this.trace;
        const w = tr.layers[this.sel.attnLayer].attn[this.sel.attnHead][row][col];
        out.textContent = `“${this.tok.display(tr.tokens[row])}” pays ${(w * 100).toFixed(1)}% of its attention to “${this.tok.display(tr.tokens[col])}”`;
      } else if (e.type === "pointermove") {
        out.textContent = "hover or tap a cell to read the weight";
      }
    };
    canvas.addEventListener("pointermove", readCell);
    canvas.addEventListener("pointerdown", readCell);
  },

  renderAttention() {
    const tr = this.trace;
    setActiveTab("attn-layer-tabs", this.sel.attnLayer);
    setActiveTab("attn-head-tabs", this.sel.attnHead);
    const A = tr.layers[this.sel.attnLayer].attn[this.sel.attnHead];
    const labels = tr.tokens.map((t) => this.tok.display(t));
    this._attnGeom = VIZ.drawAttnHeatmap(document.getElementById("attn-heatmap"), A, labels, {
      highlightRow: this.sel.attnTok,
    });

    this.buildStrip(document.getElementById("attn-strip"), tr.tokens, {
      selected: this.sel.attnTok,
      onClick: (i) => { this.sel.attnTok = i; this.renderAttention(); },
    });
    document.getElementById("attn-sel-name").textContent =
      `“${this.tok.display(tr.tokens[this.sel.attnTok])}”`;
    const row = A[this.sel.attnTok];
    const pairs = [];
    for (let s = 0; s <= this.sel.attnTok; s++) pairs.push([s, row[s]]);
    pairs.sort((a, b) => b[1] - a[1]);
    const box = document.getElementById("attn-bars");
    box.innerHTML = "";
    pairs.slice(0, 8).forEach(([s, w]) => {
      const div = document.createElement("div");
      div.className = "attn-bar-row";
      div.innerHTML = `<span class="lbl">${escapeHtml(this.tok.display(tr.tokens[s]))}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${(w * 100).toFixed(1)}%"></span></span>
        <span class="val">${(w * 100).toFixed(1)}%</span>`;
      box.appendChild(div);
    });
  },

  // ---------- stage 3: neurons ----------
  renderNeurons() {
    const tr = this.trace;
    this.buildStrip(document.getElementById("mlp-strip"), tr.tokens, {
      selected: this.sel.mlpTok,
      onClick: (i) => { this.sel.mlpTok = i; this.renderNeurons(); },
    });
    const i = this.sel.mlpTok;
    const grids = document.getElementById("mlp-grids");
    grids.innerHTML = "";
    const stats = [];
    tr.layers.forEach((L, li) => {
      const wrap = document.createElement("div");
      wrap.className = "neuron-layer";
      const acts = L.mlpHidden[i];
      let firing = 0;
      for (let k = 0; k < acts.length; k++) if (acts[k] > 0.3) firing++;
      stats.push(`Layer ${li + 1}: ${firing}/${acts.length} neurons firing`);
      wrap.innerHTML = `<div class="lbl">Layer ${li + 1} · MLP activations</div>`;
      const cv = document.createElement("canvas");
      wrap.appendChild(cv);
      grids.appendChild(wrap);
      VIZ.drawNeuronGrid(cv, acts, 12, { cell: 18 });
    });
    document.getElementById("mlp-stats").textContent =
      `for token “${this.tok.display(tr.tokens[i])}” — ${stats.join(" · ")} (firing = activation > 0.3; bright = strong, dark = quiet)`;

    const res = document.getElementById("res-strips");
    res.innerHTML = "";
    let scale = 0;
    tr.resStream.forEach((snap) => {
      for (let k = 0; k < snap[i].length; k++) scale = Math.max(scale, Math.abs(snap[i][k]));
    });
    const names = ["after embedding", "after layer 1", "after layer 2", "after layer 3"];
    tr.resStream.forEach((snap, si) => {
      const rowEl = document.createElement("div");
      rowEl.className = "vec-row";
      const lbl = document.createElement("span");
      lbl.className = "vec-label";
      let n = 0;
      for (let k = 0; k < snap[i].length; k++) n += snap[i][k] * snap[i][k];
      lbl.textContent = `${names[si]} (‖v‖=${Math.sqrt(n).toFixed(1)})`;
      const holder = document.createElement("div");
      holder.className = "strip-holder";
      const cv = document.createElement("canvas");
      holder.appendChild(cv);
      rowEl.appendChild(lbl); rowEl.appendChild(holder);
      res.appendChild(rowEl);
      VIZ.drawVectorStrip(cv, snap[i], { scale });
    });
  },

  // ---------- stage 4: prediction ----------
  wirePredict() {
    const temp = document.getElementById("pred-temp");
    const topk = document.getElementById("pred-topk");
    temp.addEventListener("input", () => this.renderPredict());
    topk.addEventListener("input", () => this.renderPredict());
    document.getElementById("pred-roll").addEventListener("click", () => this.rollDice());
  },

  renderPredict(winnerId = null) {
    const tr = this.trace;
    const temp = parseFloat(document.getElementById("pred-temp").value);
    const topk = parseInt(document.getElementById("pred-topk").value, 10);
    document.getElementById("pred-temp-val").textContent = temp.toFixed(2);
    document.getElementById("pred-topk-val").textContent = topk;
    document.getElementById("pred-context").textContent =
      "…" + this.detok(tr.ids.slice(-7));
    const dist = this.model.distribution(tr.logits, temp, topk);
    renderProbBars(document.getElementById("pred-bars"), dist, this.tok, winnerId);
    return dist;
  },

  rollDice() {
    const dist = this.renderPredict();
    const btn = document.getElementById("pred-roll");
    btn.disabled = true;
    const rows = [...document.getElementById("pred-bars").children];
    let hops = 0;
    const spin = setInterval(() => {
      rows.forEach((r) => r.classList.remove("winner"));
      rows[Math.floor(Math.random() * rows.length)].classList.add("winner");
      if (++hops > 10) {
        clearInterval(spin);
        const pick = this.model.sampleFrom(dist, Math.random);
        this.renderPredict(pick.id);
        document.getElementById("pred-note").textContent =
          `The dice landed on “${this.tok.display(pick.token)}” (drawn with ${(pick.p * 100).toFixed(1)}% probability). Roll again — same odds, maybe a different pick. That's why LLM answers vary.`;
        btn.disabled = false;
      }
    }, 90);
  },

  // ---------- stage 5: generation ----------
  wireGenerate() {
    document.getElementById("gen-play").addEventListener("click", () => {
      this.gen.running ? this.pauseGen() : this.playGen();
    });
    document.getElementById("gen-reset").addEventListener("click", () => {
      this.resetGen(); this.renderGenerate();
    });
    document.getElementById("gen-temp").addEventListener("input", () => {
      document.getElementById("gen-temp-val").textContent =
        parseFloat(document.getElementById("gen-temp").value).toFixed(2);
    });
  },

  resetGen() {
    this.pauseGen();
    if (this.tokenized) {
      this.gen.ids = this.tokenized.ids.slice();
      this.gen.promptLen = this.gen.ids.length;
      this.gen.meta = [];
    }
  },

  playGen() {
    if (!this.gen.ids || !this.gen.ids.length) return;
    this.gen.running = true;
    document.getElementById("gen-play").textContent = "⏸ Pause";
    const step = () => {
      if (!this.gen.running) return;
      const maxNew = 40;
      if (this.gen.ids.length - this.gen.promptLen >= maxNew) { this.pauseGen(); return; }
      const temp = parseFloat(document.getElementById("gen-temp").value);
      const rep = document.getElementById("gen-rep").checked;
      const tr = this.model.forward(this.gen.ids);
      const dist = this.model.distribution(tr.logits, temp, 8,
        rep ? this.gen.ids.slice(-10) : null, rep ? 1.1 : 0);
      const pick = this.model.sampleFrom(dist, Math.random);
      this.gen.ids.push(pick.id);
      this.gen.meta.push({ p: pick.p });
      this.gen.count = (this.gen.count || 0) + 1;
      this.renderGenerate(dist, pick.id);
      const speed = parseInt(document.getElementById("gen-speed").value, 10);
      this.gen.timer = setTimeout(step, 1150 - speed * 105);
    };
    step();
  },

  pauseGen() {
    this.gen.running = false;
    clearTimeout(this.gen.timer);
    const b = document.getElementById("gen-play");
    if (b) b.textContent = "▶ Generate";
  },

  renderGenerate(dist = null, winnerId = null) {
    const out = document.getElementById("gen-out");
    out.innerHTML = "";
    if (!this.gen.ids) return;
    this.gen.ids.forEach((id, i) => {
      const isPrompt = i < this.gen.promptLen;
      const span = document.createElement("span");
      span.className = "gen-tok" + (isPrompt ? " prompt-tok" : "") +
        (i === this.gen.ids.length - 1 && !isPrompt ? " new" : "");
      const t = this.tok.idToToken(id);
      span.textContent = this.tok.display(t);
      if (!isPrompt) {
        const m = this.gen.meta[i - this.gen.promptLen];
        if (m) {
          span.style.background = `rgba(57,135,229,${0.12 + 0.5 * m.p})`;
          span.title = `sampled with p=${(m.p * 100).toFixed(1)}%`;
        }
      }
      if (this.tok.isWordStart(t) && i > 0) out.appendChild(document.createTextNode(" "));
      out.appendChild(span);
    });
    if (dist) renderProbBars(document.getElementById("gen-bars"), dist.slice(0, 5), this.tok, winnerId);
  },

  // ---------- stage 6: playground ----------
  wirePlayground() {
    this.pg = { layer: 0, head: 0 };
    buildTabs("pg-layer-tabs", ["L1", "L2", "L3"], (i) => { this.pg.layer = i; this.renderPlayground(); });
    buildTabs("pg-head-tabs", ["H1", "H2", "H3", "H4"], (i) => { this.pg.head = i; this.renderPlayground(); });
    document.getElementById("pg-temp").addEventListener("input", () => this.renderPlayground());
    document.getElementById("pg-step").addEventListener("click", () => {
      const temp = parseFloat(document.getElementById("pg-temp").value);
      const dist = this.model.distribution(this.trace.logits, temp, 8, this.tokenized.ids.slice(-10), 1.1);
      const pick = this.model.sampleFrom(dist, Math.random);
      document.getElementById("prompt-input").value =
        this.detok(this.tokenized.ids.concat([pick.id]));
      this.recompute();
    });
  },

  renderPlayground() {
    const tr = this.trace;
    setActiveTab("pg-layer-tabs", this.pg.layer);
    setActiveTab("pg-head-tabs", this.pg.head);
    const labels = tr.tokens.map((t) => this.tok.display(t));
    VIZ.drawAttnHeatmap(document.getElementById("pg-heatmap"),
      tr.layers[this.pg.layer].attn[this.pg.head], labels, {});
    const temp = parseFloat(document.getElementById("pg-temp").value);
    document.getElementById("pg-temp-val").textContent = temp.toFixed(2);
    renderProbBars(document.getElementById("pg-bars"),
      this.model.distribution(tr.logits, temp, 8), this.tok, null);
    document.getElementById("pg-ms").textContent = this.lastMs.toFixed(1);
    const pipe = document.getElementById("pg-pipeline");
    const cfg = this.model.cfg;
    pipe.innerHTML = [
      `<span class="pipe-step"><b>${this.prompt.length}</b>characters</span>`,
      `<span class="pipe-step"><b>${this.tokenized.ids.length}</b>tokens</span>`,
      `<span class="pipe-step"><b>${tr.T}×${cfg.dModel}</b>vectors</span>`,
      `<span class="pipe-step"><b>${cfg.nLayers}×${cfg.nHeads}</b>layers × heads</span>`,
      `<span class="pipe-step"><b>${this.tok.size}</b>logits</span>`,
      `<span class="pipe-step"><b>1</b>next token</span>`,
    ].join(`<span class="pipe-arrow">→</span>`);
  },
};

// ---------- small shared helpers ----------

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function buildTabs(containerId, names, onPick) {
  const el = document.getElementById(containerId);
  el.innerHTML = "";
  names.forEach((n, i) => {
    const b = document.createElement("button");
    b.className = "tab" + (i === 0 ? " active" : "");
    b.textContent = n;
    b.addEventListener("click", () => onPick(i));
    el.appendChild(b);
  });
}

function setActiveTab(containerId, idx) {
  [...document.getElementById(containerId).children].forEach((c, i) =>
    c.classList.toggle("active", i === idx));
}

function renderProbBars(el, dist, tok, winnerId) {
  el.innerHTML = "";
  dist.forEach((d) => {
    const row = document.createElement("div");
    row.className = "prob-row" + (d.id === winnerId ? " winner" : "");
    const label = tok.display(d.token);
    row.innerHTML = `<span class="lbl">${escapeHtml(label)}</span>
      <span class="bar-track"><span class="bar-fill" style="width:${(d.p * 100).toFixed(2)}%"></span></span>
      <span class="val">${(d.p * 100).toFixed(1)}%</span>`;
    el.appendChild(row);
  });
}

window.addEventListener("DOMContentLoaded", () => APP.init());

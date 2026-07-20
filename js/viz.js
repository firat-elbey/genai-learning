// Canvas visualization helpers: color ramps, heatmaps, vector strips,
// and the 2D embedding map. Palette follows the validated dark-mode data-viz
// reference palette (sequential single-hue blue for magnitude, blue<->red
// diverging with a neutral midpoint for signed values).

const VIZ = (() => {
  function hex2rgb(h) {
    return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  }
  function lerp(a, b, t) { return a + (b - a) * t; }
  function rampFromStops(stops) {
    const rgb = stops.map(hex2rgb);
    return (t) => {
      t = Math.max(0, Math.min(1, t));
      const x = t * (rgb.length - 1);
      const i = Math.min(Math.floor(x), rgb.length - 2);
      const f = x - i;
      const c = [0, 1, 2].map((k) => Math.round(lerp(rgb[i][k], rgb[i + 1][k], f)));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    };
  }

  // Sequential (magnitude, e.g. attention weight): dark surface -> bright blue
  const seq = rampFromStops(["#20242a", "#104281", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]);
  // Diverging (signed values, e.g. vector components): red <- neutral -> blue
  const divNeg = rampFromStops(["#383835", "#a34544", "#e66767"]);
  const divPos = rampFromStops(["#383835", "#25588f", "#3987e5", "#86b6ef"]);
  const div = (t) => (t < 0 ? divNeg(Math.min(1, -t)) : divPos(Math.min(1, t)));

  function setupCanvas(canvas, cssW, cssH) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  // Lower-triangular attention heatmap with token labels on both axes.
  // Returns geometry for hit-testing hover tooltips.
  function drawAttnHeatmap(canvas, A, labels, opts = {}) {
    const T = labels.length;
    // Fit the container when possible; on small (phone) screens keep a
    // readable minimum cell size and let the surrounding .scrollwrap scroll.
    const availW = Math.min(canvas.parentElement.clientWidth - 8, 640);
    const compact = availW < 480;
    const labelSpace = compact ? 46 : 64;
    const cell = Math.max(compact ? 12 : 14, Math.min(30, Math.floor((availW - labelSpace) / Math.max(T, 1))));
    const w = labelSpace + cell * T + 8;
    const h = labelSpace + cell * T + 8;
    const ctx = setupCanvas(canvas, w, h);
    ctx.clearRect(0, 0, w, h);
    ctx.font = (compact ? "8.5px" : "10px") + " ui-monospace, monospace";
    const maxChars = compact ? 6 : 8;
    for (let t = 0; t < T; t++) {
      for (let s = 0; s < T; s++) {
        if (s > t) {
          ctx.fillStyle = "#151517";
          ctx.fillRect(labelSpace + s * cell, labelSpace + t * cell, cell - 1, cell - 1);
          continue;
        }
        ctx.fillStyle = seq(Math.pow(A[t][s], 0.6));
        ctx.fillRect(labelSpace + s * cell, labelSpace + t * cell, cell - 1, cell - 1);
      }
    }
    // labels
    ctx.fillStyle = "#c3c2b7";
    for (let s = 0; s < T; s++) {
      ctx.save();
      ctx.translate(labelSpace + s * cell + cell / 2 + 3, labelSpace - 6);
      ctx.rotate(-Math.PI / 3);
      ctx.textAlign = "left";
      ctx.fillText(labels[s].slice(0, maxChars), 0, 0);
      ctx.restore();
    }
    ctx.textAlign = "right";
    for (let t = 0; t < T; t++) {
      ctx.fillText(labels[t].slice(0, maxChars), labelSpace - 6, labelSpace + t * cell + cell / 2 + 3);
    }
    if (opts.highlightRow != null) {
      ctx.strokeStyle = "#eda100";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(labelSpace - 1, labelSpace + opts.highlightRow * cell - 1, cell * T + 1, cell + 1);
    }
    return { labelSpace, cell, T };
  }

  // Horizontal strip of vector components (diverging color per cell).
  function drawVectorStrip(canvas, vec, opts = {}) {
    const n = vec.length;
    const w = opts.width || Math.max(160, Math.min(canvas.parentElement.clientWidth - 4, 560));
    const cellW = w / n;
    const h = opts.height || 26;
    const ctx = setupCanvas(canvas, w, h);
    ctx.clearRect(0, 0, w, h);
    let scale = opts.scale;
    if (!scale) {
      let mx = 0;
      for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(vec[i]));
      scale = mx || 1;
    }
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = div(vec[i] / scale);
      ctx.fillRect(i * cellW, 0, Math.ceil(cellW) - (cellW > 3 ? 1 : 0), h);
    }
    return { scale };
  }

  // Grid of MLP neuron activations (mostly >= 0 after GELU -> sequential).
  function drawNeuronGrid(canvas, acts, cols, opts = {}) {
    const n = acts.length;
    const rows = Math.ceil(n / cols);
    const cell = opts.cell || 22;
    const w = cols * cell, h = rows * cell;
    const ctx = setupCanvas(canvas, w, h);
    ctx.clearRect(0, 0, w, h);
    let mx = 0.001;
    for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(acts[i]));
    for (let i = 0; i < n; i++) {
      const r = Math.floor(i / cols), c = i % cols;
      // GELU leaves negatives tiny — render "quiet" as dark, "firing" as bright
      const v = Math.max(0, acts[i]);
      ctx.fillStyle = seq(Math.pow(v / mx, 0.7));
      ctx.fillRect(c * cell, r * cell, cell - 2, cell - 2);
    }
    return { cell, cols, rows, max: mx };
  }

  // 2D embedding map. points: [{x, y, label, kind}], kinds: 'vocab' | 'active'
  function drawEmbeddingMap(canvas, points, opts = {}) {
    const w = Math.min(canvas.parentElement.clientWidth - 16, 620);
    const h = opts.height || 400;
    const ctx = setupCanvas(canvas, w, h);
    ctx.clearRect(0, 0, w, h);
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const pad = 28;
    const sx = (x) => pad + ((x - x0) / (x1 - x0 || 1)) * (w - 2 * pad);
    const sy = (y) => h - pad - ((y - y0) / (y1 - y0 || 1)) * (h - 2 * pad);
    ctx.font = "10px system-ui, sans-serif";
    const placed = [];
    for (const p of points) {
      if (p.kind === "active") continue;
      const X = sx(p.x), Y = sy(p.y);
      ctx.fillStyle = "#898781";
      ctx.beginPath(); ctx.arc(X, Y, 2.5, 0, Math.PI * 2); ctx.fill();
      placed.push({ ...p, X, Y });
    }
    // label a subset of vocab points to avoid clutter (most frequent first)
    ctx.fillStyle = "#c3c2b7";
    let labeled = 0;
    for (const p of placed) {
      if (!p.showLabel || labeled > 40) continue;
      ctx.fillText(p.label, p.X + 5, p.Y + 3);
      labeled++;
    }
    for (const p of points) {
      if (p.kind !== "active") continue;
      const X = sx(p.x), Y = sy(p.y);
      ctx.fillStyle = "#eda100";
      ctx.beginPath(); ctx.arc(X, Y, 5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = "#1a1a19"; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 11px system-ui, sans-serif";
      ctx.fillText(p.label, X + 7, Y + 4);
      ctx.font = "10px system-ui, sans-serif";
    }
    return { sx, sy, placed };
  }

  return { seq, div, setupCanvas, drawAttnHeatmap, drawVectorStrip, drawNeuronGrid, drawEmbeddingMap };
})();

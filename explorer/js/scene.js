// scene.js — three.js rendering of the layout: one InstancedMesh per tensor
// (cell colour = value, bar height = |value| for activations), base plates
// that distinguish weights from activations, CSS2D labels, analytic+raycast
// picking, dependency highlights, "lenses" (dim everything but one token /
// head / layer), smooth camera fly-to, and wireframe "ghosts" of real GPTs
// at true relative scale.

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/CSS2DRenderer.js";
import { diverging, sequential, MASK_RGB, PLATE } from "./colors.js";
import { prettyToken } from "./util.js";

const CAT = ["#3987e5", "#e66767", "#199e70", "#f5b942", "#b07cf0", "#ec8c3f", "#52c6d8", "#d85ba5"];
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export class Scene3D {
  constructor(container, labelLayer, callbacks = {}) {
    this.container = container; this.cb = callbacks;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(this.renderer.domElement);
    this.labels = new CSS2DRenderer({ element: labelLayer });
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0d0d0d);
    this.scene.fog = new THREE.Fog(0x0d0d0d, 600, 6000);
    this.camera = new THREE.PerspectiveCamera(42, 1, 1, 400000);
    this.camera.position.set(120, 90, 220);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.95;
    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x201a10, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4); sun.position.set(0.4, 1, 0.6); this.scene.add(sun);
    const sun2 = new THREE.DirectionalLight(0x88aaff, 0.4); sun2.position.set(-0.6, 0.3, -0.5); this.scene.add(sun2);
    this.blockGroup = new THREE.Group(); this.scene.add(this.blockGroup);
    this.hiGroup = new THREE.Group(); this.scene.add(this.hiGroup);
    this.ghostGroup = new THREE.Group(); this.scene.add(this.ghostGroup);
    this.meshes = new Map();  // block.id → {mesh, plate, label, block, box}
    this.lens = {}; this.bars = true; this.showWeights = true; this.showLabels = true;
    this.fly = null; this.hover = null; this.selection = null;
    this.raycaster = new THREE.Raycaster();
    this.geoBox = new THREE.BoxGeometry(0.84, 1, 0.84); this.geoBox.translate(0, 0.5, 0);
    this.geoFlat = new THREE.PlaneGeometry(0.84, 0.84); this.geoFlat.rotateX(-Math.PI / 2); this.geoFlat.translate(0, 0.3, 0);
    this.groupLabels = [];
    this.dummy = new THREE.Object3D(); this.tmpColor = new THREE.Color();
    this._onResize = () => this.resize();
    window.addEventListener("resize", this._onResize);
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    const dom = this.renderer.domElement;
    let downAt = null;
    dom.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
    dom.addEventListener("pointerup", (e) => {
      if (!downAt) return;
      const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]); downAt = null;
      if (moved > 5) return;
      const hit = this.pick(e.clientX, e.clientY);
      this.cb.onClick && this.cb.onClick(hit);
    });
    let lastMove = 0;
    dom.addEventListener("pointermove", (e) => {
      const now = performance.now(); if (now - lastMove < 40) return; lastMove = now;
      if (e.buttons) return;
      const hit = this.pick(e.clientX, e.clientY);
      const key = hit ? `${hit.block.id}:${hit.i}:${hit.j}` : "";
      if (key !== this._hoverKey) { this._hoverKey = key; this.cb.onHover && this.cb.onHover(hit, e); }
    });
    dom.addEventListener("pointerleave", () => { this._hoverKey = ""; this.cb.onHover && this.cb.onHover(null); });
    this._loop = () => { this.tick(); requestAnimationFrame(this._loop); };
    requestAnimationFrame(this._loop);
  }

  resize() {
    const w = this.container.clientWidth || 1, h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = "100%"; this.renderer.domElement.style.height = "100%";
    this.labels.setSize(w, h);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  tick() {
    if (this.fly) {
      const t = Math.min(1, (performance.now() - this.fly.t0) / this.fly.dur), k = ease(t);
      this.camera.position.lerpVectors(this.fly.p0, this.fly.p1, k);
      this.controls.target.lerpVectors(this.fly.c0, this.fly.c1, k);
      if (t >= 1) this.fly = null;
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    if (this.showLabels) {
      if ((this._frame = (this._frame || 0) + 1) % 6 === 0) this.cullLabels();
      this.labels.render(this.scene, this.camera);
    }
  }

  // Show a block's label only when the camera is close enough for it to be
  // readable next to its block (big blocks keep their label from further away).
  cullLabels() {
    const cam = this.camera.position, v = new THREE.Vector3();
    for (const m of this.meshes.values()) {
      if (!m.label || !m.mesh.visible) continue;
      const b = m.block;
      v.set(b.x + b.width / 2, b.y, b.z + b.depth / 2);
      const range = 40 + 3.2 * Math.max(b.width, b.depth);
      m.label.visible = cam.distanceTo(v) < range;
      for (const el of m.extraLabels || []) el.visible = cam.distanceTo(v) < 90 + 2 * b.depth;
    }
  }

  // ------------------------------------------------------------ building
  clear() {
    for (const { mesh, plate, label, extraLabels } of this.meshes.values()) {
      this.blockGroup.remove(mesh); mesh.dispose(); mesh.material.dispose();
      this.blockGroup.remove(plate); plate.geometry.dispose(); plate.material.dispose();
      if (label) { this.blockGroup.remove(label); label.element.remove(); }
      for (const el of extraLabels || []) { this.blockGroup.remove(el); el.element.remove(); }
    }
    for (const o of this.groupLabels) { this.blockGroup.remove(o); o.element.remove(); }
    this.groupLabels = [];
    this.meshes.clear(); this.clearHighlights();
  }

  build(layout, tokens) {
    this.clear();
    this.layout = layout; this._tokens = tokens;
    for (const g of layout.groups || []) {
      const el = document.createElement("div"); el.className = "group-label"; el.textContent = g.label;
      const o = new CSS2DObject(el); o.position.set(g.x, g.y, g.z); this.blockGroup.add(o); this.groupLabels.push(o);
    }
    for (const b of layout.blocks) {
      const n = b.rows * b.cols;
      const mat = new THREE.MeshLambertMaterial({ color: 0xffffff });
      const flat = b.kind === "weight" || !this.bars;
      const mesh = new THREE.InstancedMesh(flat ? this.geoFlat : this.geoBox, mat, n);
      mesh.userData.flat = flat;
      mesh.userData.block = b; mesh.frustumCulled = true;
      this.blockGroup.add(mesh);
      const plateGeo = new THREE.BoxGeometry(b.width + 1.2, 0.3, b.depth + 1.2);
      const plateKind = b.kind === "weight" ? "weight" : b.kind === "attn" ? "attn" : b.kind === "prob" ? "prob" : "act";
      const plate = new THREE.Mesh(plateGeo, new THREE.MeshLambertMaterial({ color: new THREE.Color(PLATE[plateKind]) }));
      plate.position.set(b.x + b.width / 2, b.y - 0.2, b.z + b.depth / 2);
      plate.userData.block = b;
      this.blockGroup.add(plate);
      const el = document.createElement("div");
      el.className = "blk-label " + (b.kind === "weight" ? "is-weight" : "is-act");
      el.innerHTML = `<span class="blk-name">${b.label}</span><span class="blk-shape">${b.rows}×${b.cols}</span>`;
      el.addEventListener("pointerdown", (e) => { e.stopPropagation(); this.cb.onLabelClick && this.cb.onLabelClick(b); });
      const label = new CSS2DObject(el);
      label.position.set(b.x - 0.5, b.y + 2.5, b.z - 0.5);
      this.blockGroup.add(label);
      const extraLabels = [];
      if (b.kind === "tokens" && tokens) {
        for (let i = 0; i < b.rows; i++) {
          const te = document.createElement("div"); te.className = "tok-label";
          te.textContent = prettyToken(tokens[i]);
          const o = new CSS2DObject(te); o.position.set(b.x + 1.6, b.y + 0.6, b.z + i + 0.5);
          this.blockGroup.add(o); extraLabels.push(o);
        }
      }
      const box = new THREE.Box3(new THREE.Vector3(b.x, b.y - 0.3, b.z), new THREE.Vector3(b.x + b.width, b.y + 3.5, b.z + b.depth));
      this.meshes.set(b.id, { mesh, plate, label, block: b, box, extraLabels });
      this.fillBlock(b, mesh);
    }
    this.applyVisibility();
  }

  cellColor(b, i, j) {
    if (b.mask && b.mask(i, j)) return MASK_RGB;
    const v = b.mat.get(i, j);
    if (b.colorMode === "cat") return hex2rgb(CAT[Math.round(v) % CAT.length]);
    if (b.colorMode === "seq") return sequential(Math.min(1, v / b.vmax));
    return diverging(Math.max(-1, Math.min(1, v / b.vmax)));
  }
  cellHeight(b, i, j) {
    if (b.kind === "weight") return 0.3;
    if (b.mask && b.mask(i, j)) return 0.12;
    if (!this.bars) return 0.3;
    const v = Math.abs(b.mat.get(i, j));
    const t = b.colorMode === "seq" ? Math.min(1, v / b.vmax) : Math.min(1, v / b.vmax);
    return 0.2 + 2.6 * t;
  }
  lensFactor(b, i, j) {
    const L = this.lens; let f = 1;
    if (L.layer !== undefined && L.layer !== null) { if (b.layer !== undefined && b.layer !== L.layer) f *= 0.12; else if (b.layer === undefined) f *= 0.5; }
    if (L.token !== undefined && L.token !== null) {
      if (b.kind === "weight") f *= 0.35;
      else if (b.rows === this.layout.T) { if (i !== L.token) f *= 0.15; }
    }
    if (L.head !== undefined && L.head !== null) {
      if (b.head !== undefined) { if (b.head !== L.head) f *= 0.12; }
      else if (b.headGroups) { const dh = b.cols / b.headGroups; if (Math.floor(j / dh) !== L.head) f *= 0.15; }
      else f *= 0.5;
    }
    return f;
  }
  fillBlock(b, mesh) {
    const d = this.dummy, c = this.tmpColor, colX = this.layout.colX;
    let idx = 0;
    for (let i = 0; i < b.rows; i++) {
      for (let j = 0; j < b.cols; j++, idx++) {
        d.position.set(b.x + colX(b, j) + 0.5, b.y, b.z + i + 0.5);
        d.scale.set(1, mesh.userData.flat ? 1 : this.cellHeight(b, i, j), 1);
        d.updateMatrix(); mesh.setMatrixAt(idx, d.matrix);
        const rgb = this.cellColor(b, i, j), f = this.lensFactor(b, i, j);
        c.setRGB(rgb[0] * f + 0.03 * (1 - f), rgb[1] * f + 0.03 * (1 - f), rgb[2] * f + 0.03 * (1 - f));
        mesh.setColorAt(idx, c);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
  }
  recolorAll() { for (const { mesh, block } of this.meshes.values()) this.fillBlock(block, mesh); }
  setLens(lens) { this.lens = lens || {}; this.recolorAll(); }
  setBars(on) { if (this.bars === on) return; this.bars = on; if (this.layout) this.build(this.layout, this._tokens); }
  setShowWeights(on) { this.showWeights = on; this.applyVisibility(); }
  setShowLabels(on) { this.showLabels = on; this.labels.domElement.style.display = on ? "" : "none"; }
  applyVisibility() {
    for (const { mesh, plate, label, block } of this.meshes.values()) {
      const vis = block.kind !== "weight" || this.showWeights;
      mesh.visible = vis; plate.visible = vis; if (label) label.visible = vis;
    }
  }

  // ------------------------------------------------------------ picking
  pick(clientX, clientY) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const nx = ((clientX - rect.left) / rect.width) * 2 - 1, ny = -((clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(new THREE.Vector2(nx, ny), this.camera);
    const ray = this.raycaster.ray, cands = [];
    for (const m of this.meshes.values()) {
      if (!m.mesh.visible) continue;
      const hit = ray.intersectBox(m.box, new THREE.Vector3());
      if (hit) cands.push([hit.distanceTo(ray.origin), m]);
    }
    cands.sort((a, b) => a[0] - b[0]);
    let best = null;
    for (const [dist, m] of cands) {
      if (best && dist > best.distance) break;
      const hits = this.raycaster.intersectObject(m.mesh, false);
      if (hits.length && (!best || hits[0].distance < best.distance)) {
        const idx = hits[0].instanceId, b = m.block;
        best = { distance: hits[0].distance, block: b, i: Math.floor(idx / b.cols), j: idx % b.cols, point: hits[0].point };
      }
    }
    return best;
  }

  // ------------------------------------------------------------ highlights
  clearHighlights() {
    for (const o of [...this.hiGroup.children]) { this.hiGroup.remove(o); o.geometry && o.geometry.dispose(); o.material && o.material.dispose(); }
  }
  regionBox(region, color, opacity = 0.35, pad = 0.25, height = 3.6) {
    const m = this.meshes.get(region.id); if (!m) return null;
    const b = m.block, colX = this.layout.colX;
    const r0 = region.r0 < 0 ? 0 : region.r0, r1 = region.r1 < 0 ? b.rows - 1 : region.r1;
    const c0 = region.c0 < 0 ? 0 : region.c0, c1 = region.c1 < 0 ? b.cols - 1 : region.c1;
    const x0 = b.x + colX(b, c0) - pad, x1 = b.x + colX(b, c1) + 1 + pad, z0 = b.z + r0 - pad, z1 = b.z + r1 + 1 + pad;
    const geo = new THREE.BoxGeometry(x1 - x0, height, z1 - z0);
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
    mesh.position.set((x0 + x1) / 2, b.y + height / 2 - 0.2, (z0 + z1) / 2);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color, transparent: true, opacity: Math.min(1, opacity * 2.4) }));
    mesh.add(edges);
    this.hiGroup.add(mesh);
    return mesh;
  }
  // selection: white edge box on the cell; deps: amber translucent boxes
  showSelection(block, i, j, deps = []) {
    this.clearHighlights();
    if (!block) return;
    this.regionBox({ id: block.id, r0: i, r1: i, c0: j, c1: j }, 0xffffff, 0.55, 0.12, 4.2);
    for (const r of deps) this.regionBox(r, 0xf5b942, 0.32);
  }
  showRegions(regions, color = 0x3987e5, opacity = 0.25) { for (const r of regions) this.regionBox(r, color, opacity); }

  // ------------------------------------------------------------ camera
  boxOf(ids) {
    const box = new THREE.Box3();
    for (const id of ids) { const m = this.meshes.get(id); if (m) box.union(m.box); }
    return box.isEmpty() ? null : box;
  }
  flyToBox(box, { dur = 900, elev = 1.0, azim = 0.3, fill = 0.8 } = {}) {
    if (!box) return;
    const size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(14, 0.5 * Math.hypot(size.x, size.y, size.z));
    const fov = (this.camera.fov * Math.PI) / 180, aspect = this.camera.aspect;
    const vfit = radius / Math.sin(fov / 2), hfit = radius / Math.sin(Math.atan(Math.tan(fov / 2) * aspect));
    const dist = Math.max(vfit, hfit) / fill;
    const dir = new THREE.Vector3(Math.sin(azim) * Math.cos(elev), Math.sin(elev), Math.cos(azim) * Math.cos(elev)).normalize();
    this.fly = { t0: performance.now(), dur, p0: this.camera.position.clone(), p1: center.clone().addScaledVector(dir, dist), c0: this.controls.target.clone(), c1: center };
  }
  flyToBlocks(ids, opts) { this.flyToBox(this.boxOf(ids), opts); }
  flyToAll(opts) { this.flyToBox(this.boxOf([...this.meshes.keys()]), { elev: 0.85, azim: 0.45, fill: 1.0, ...opts }); }

  // ------------------------------------------------------------ ghosts (scale comparison)
  clearGhosts() { for (const o of [...this.ghostGroup.children]) { this.ghostGroup.remove(o); o.traverse && o.traverse((c) => { c.geometry && c.geometry.dispose(); c.material && c.material.dispose(); }); if (o.element) o.element.remove(); } }
  // Draw outline boxes for a GPT of config cfg at true relative scale, to the
  // right of the current model. Each layer is one box (its true footprint and
  // the same vertical pitch structure as the toy), plus embedding and unembedding.
  addGhost(cfg, T, originX, color, name) {
    const { d_model: d, n_layer: L, d_ff: dff, vocab_size: V } = cfg;
    const pitch = this.layout ? this.layout.pitch : 20;
    const laneGap = 10, hgap = 4;
    const width = Math.max(3 * d + 2 * hgap, dff) + laneGap + Math.max(d, dff);   // activations + weight lane
    const layerH = pitch * (1 + 0.8 + 1 + 1 + 1 + 1 + 1 + 1 + 0.8 + 1 + 1 + 1 + 1.6);
    const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7, fog: false });
    const g = new THREE.Group();
    let y = 0;
    const box = (w, h, dep, x, yy, label) => {
      const geo = new THREE.BoxGeometry(w, h, dep);
      const m = new THREE.LineSegments(new THREE.EdgesGeometry(geo), mat);
      m.position.set(x + w / 2, yy - h / 2, 0); g.add(m);
      if (label) { const el = document.createElement("div"); el.className = "ghost-label"; el.textContent = label; const o = new CSS2DObject(el); o.position.set(x, yy + 2, -dep / 2); g.add(o); }
      return m;
    };
    box(V, pitch * 2, d, originX - V - laneGap, y, `${name} · token embedding ${V}×${d}`);
    box(d, pitch * 2, T, originX, y, `residual stream ${T}×${d}`); y -= pitch * 3;
    for (let l = 0; l < L; l++) { box(width, layerH, T, originX - laneGap - Math.max(d, dff), y, l === 0 || l === L - 1 ? `${name} · layer ${l}` : null); y -= layerH + pitch * 0.4; }
    box(V, pitch * 2, T, originX, y, `logits ${T}×${V}`);
    const title = document.createElement("div"); title.className = "ghost-title"; title.innerHTML = name;
    const to = new CSS2DObject(title); to.position.set(originX + Math.max(d, V) / 2, pitch * 4, 0); g.add(to);
    this.ghostGroup.add(g);
    return { group: g, minY: y - pitch * 2, width: width + V + laneGap, height: -y + pitch * 4 };
  }
  ghostBox(group) { return new THREE.Box3().setFromObject(group); }

  dispose() { window.removeEventListener("resize", this._onResize); this.clear(); this.clearGhosts(); this.renderer.dispose(); }
}

function hex2rgb(h) { return [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255, parseInt(h.slice(5, 7), 16) / 255]; }

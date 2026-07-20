// Gamification: XP, ranks, badges, stage locking, quiz modal, confetti,
// toasts. Progress persists in localStorage.

const STAGES = [
  { id: "tokens",   icon: "✂️", title: "Tokens",     tag: "text → pieces" },
  { id: "vectors",  icon: "🧭", title: "Vectors",    tag: "pieces → numbers" },
  { id: "attention",icon: "🔦", title: "Attention",  tag: "tokens talk" },
  { id: "neurons",  icon: "⚡", title: "Neurons",    tag: "patterns fire" },
  { id: "predict",  icon: "🎲", title: "Prediction", tag: "scores → odds" },
  { id: "generate", icon: "🔁", title: "Generation", tag: "the loop" },
  { id: "playground", icon: "🎮", title: "Playground", tag: "free play" },
];

const RANKS = [
  [0, "Curious Visitor"],
  [60, "Token Trainee"],
  [140, "Vector Voyager"],
  [230, "Attention Apprentice"],
  [320, "Activation Analyst"],
  [420, "Probability Pro"],
  [500, "Transformer Tamer"],
];

const GAME = {
  KEY: "inside-transformer-save-v1",
  state: { xp: 0, done: {}, visited: {} },

  load() {
    try {
      const raw = localStorage.getItem(this.KEY);
      if (raw) this.state = Object.assign(this.state, JSON.parse(raw));
    } catch (e) { /* fresh start */ }
  },
  save() {
    try { localStorage.setItem(this.KEY, JSON.stringify(this.state)); } catch (e) {}
  },
  reset() {
    this.state = { xp: 0, done: {}, visited: {} };
    this.save();
  },

  rank() {
    let r = RANKS[0][1];
    for (const [t, name] of RANKS) if (this.state.xp >= t) r = name;
    return r;
  },
  nextRankAt() {
    for (const [t] of RANKS) if (this.state.xp < t) return t;
    return null;
  },

  isUnlocked(i) {
    if (i === 0) return true;
    if (this.state.unlockAll) return true;
    return !!this.state.done[i - 1];
  },
  isDone(i) { return !!this.state.done[i]; },

  award(xp, reason) {
    this.state.xp += xp;
    this.save();
    this.renderHud();
    if (xp > 0) toast(`+${xp} XP — ${reason}`);
  },

  completeStage(i) {
    if (this.state.done[i]) return;
    this.state.done[i] = true;
    this.save();
    this.renderHud();
    this.renderMap();
    confettiBurst();
    const next = STAGES[i + 1];
    toast(`${STAGES[i].icon} Level complete! ${next ? `“${next.title}” unlocked` : "You finished the journey!"}`, 3500);
  },

  renderHud() {
    const xp = this.state.xp;
    document.getElementById("hud-rank").textContent = this.rank();
    const next = this.nextRankAt();
    const prevT = RANKS.filter(([t]) => t <= xp).pop()[0];
    const frac = next ? (xp - prevT) / (next - prevT) : 1;
    document.getElementById("hud-xpfill").style.width = `${Math.round(frac * 100)}%`;
    document.getElementById("hud-xp").textContent = `${xp} XP${next ? ` · next rank at ${next}` : " · max rank"}`;
    const badges = document.getElementById("hud-badges");
    badges.innerHTML = "";
    STAGES.forEach((s, i) => {
      if (i < QUIZZES.length && this.state.done[i]) {
        const b = document.createElement("span");
        b.className = "badge";
        b.title = `${s.title} — completed`;
        b.textContent = s.icon;
        badges.appendChild(b);
      }
    });
  },

  renderMap() {
    const map = document.getElementById("level-map");
    map.innerHTML = "";
    STAGES.forEach((s, i) => {
      const node = document.createElement("button");
      const unlocked = this.isUnlocked(i);
      const active = APP.currentStage === i;
      node.className = "map-node" + (active ? " active" : "") + (unlocked ? "" : " locked");
      node.innerHTML = `<span class="map-icon">${unlocked ? s.icon : "🔒"}</span>
        <span class="map-title">${i + 1}. ${s.title}${this.isDone(i) ? " ✓" : ""}</span>
        <span class="map-tag">${s.tag}</span>`;
      node.addEventListener("click", () => {
        if (!unlocked) { toast(`Finish “${STAGES[i - 1].title}” (take its quiz) to unlock this level`); return; }
        APP.showStage(i);
      });
      map.appendChild(node);
    });
  },
};

// ---------- quiz modal ----------

const QUIZ_UI = {
  stage: null, qIndex: 0, mistakes: 0,

  open(stageIdx) {
    this.stage = stageIdx; this.qIndex = 0; this.mistakes = 0;
    document.getElementById("quiz-modal").classList.remove("hidden");
    this.renderQuestion();
  },
  close() {
    document.getElementById("quiz-modal").classList.add("hidden");
  },
  renderQuestion() {
    const qz = QUIZZES[this.stage];
    const q = qz[this.qIndex];
    const box = document.getElementById("quiz-body");
    document.getElementById("quiz-title").textContent =
      `${STAGES[this.stage].icon} ${STAGES[this.stage].title} quiz — question ${this.qIndex + 1} of ${qz.length}`;
    box.innerHTML = "";
    const qEl = document.createElement("p");
    qEl.className = "quiz-q";
    qEl.textContent = q.q;
    box.appendChild(qEl);
    const wrongFirst = { v: false };
    q.options.forEach((opt, oi) => {
      const b = document.createElement("button");
      b.className = "quiz-opt";
      b.textContent = opt;
      b.addEventListener("click", () => {
        if (oi === q.correct) {
          b.classList.add("correct");
          const xp = wrongFirst.v ? 10 : 30;
          GAME.award(xp, wrongFirst.v ? "correct (after a retry)" : "correct on the first try!");
          const why = document.createElement("p");
          why.className = "quiz-why";
          why.textContent = "✔ " + q.why;
          box.appendChild(why);
          box.querySelectorAll(".quiz-opt").forEach((x) => (x.disabled = true));
          const nextB = document.createElement("button");
          nextB.className = "btn primary";
          nextB.textContent = this.qIndex + 1 < qz.length ? "Next question →" : "Finish quiz 🏁";
          nextB.addEventListener("click", () => {
            this.qIndex++;
            if (this.qIndex < qz.length) this.renderQuestion();
            else { this.close(); GAME.award(20, "quiz finished"); GAME.completeStage(this.stage); APP.refreshStageChrome(); }
          });
          box.appendChild(nextB);
        } else {
          wrongFirst.v = true;
          b.classList.add("wrong");
          b.disabled = true;
        }
      });
      box.appendChild(b);
    });
  },
};

// ---------- toasts & confetti ----------

let toastTimer = null;
function toast(msg, ms = 2200) {
  let el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), ms);
}

function confettiBurst() {
  const canvas = document.getElementById("confetti");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = innerWidth * dpr; canvas.height = innerHeight * dpr;
  canvas.style.width = innerWidth + "px"; canvas.style.height = innerHeight + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const colors = ["#3987e5", "#008300", "#d55181", "#c98500", "#199e70", "#9085e9"];
  const parts = [];
  for (let i = 0; i < 130; i++) {
    parts.push({
      x: innerWidth / 2 + (Math.random() - 0.5) * 200,
      y: innerHeight * 0.32,
      vx: (Math.random() - 0.5) * 9,
      vy: -Math.random() * 8 - 3,
      s: Math.random() * 6 + 3,
      c: colors[i % colors.length],
      r: Math.random() * Math.PI,
      vr: (Math.random() - 0.5) * 0.3,
    });
  }
  let frames = 0;
  canvas.style.display = "block";
  (function tick() {
    ctx.clearRect(0, 0, innerWidth, innerHeight);
    for (const p of parts) {
      p.x += p.vx; p.y += p.vy; p.vy += 0.25; p.r += p.vr;
      ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.r);
      ctx.fillStyle = p.c; ctx.fillRect(-p.s / 2, -p.s / 2, p.s, p.s * 0.6);
      ctx.restore();
    }
    frames++;
    if (frames < 110) requestAnimationFrame(tick);
    else { ctx.clearRect(0, 0, innerWidth, innerHeight); canvas.style.display = "none"; }
  })();
}

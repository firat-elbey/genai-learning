# Design & workflow research — "Inside the Transformer"

*2026-07-20. Companion note to the app in this folder: what was considered, what was
chosen, and why.*

## 1. The brief

> "I want to understand more fundamentally how LLMs work — see all the activations —
> a web app where I take an input, see how it flows through the model, and generates
> an output. A fun learning experience, with some gamification."

Decomposed into requirements:

1. **Input → internals → output**, end to end, driven by the user's own text.
2. **Activations visible** — not a cartoon: real attention weights, real neuron
   activations, a real residual stream.
3. **Learning experience** — sequenced pedagogy, not just a dashboard.
4. **Fun / gamified** — progression, rewards, low-stakes challenges.
5. Practical: lives in this research repo, easy to run forever (no build churn).

## 2. The core design decision: what model powers it?

Options considered:

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **A. Pure animation / faked numbers** | Full narrative control | Dishonest; "activations" would be decoration; user asked for the real thing | ✗ |
| **B. Real pretrained model in-browser (transformers.js / ONNX GPT-2)** | Real trained weights, coherent text | ~100–500MB download, WebGPU/wasm complexity, hard to expose *every* intermediate cleanly, network dependency | ✗ for v1, ★ best future upgrade |
| **C. Train a tiny model in-browser (tfjs)** | "Watch it learn" is magical | Minutes of wait before the app is usable; training quality fragile; heavy dependency | ✗ (a "training" level is future work) |
| **D. Hand-rolled tiny transformer in vanilla JS, random weights, + corpus-derived statistics for readability** | Zero dependencies, instant load, every intermediate trivially observable, deterministic, honest if disclosed | Untrained → output needs help to be readable | **✓ chosen** |

Option D's weakness (gibberish output from random weights) is patched with two
*learned-from-data* components that are cheap to compute at page load and genuinely
educational in their own right:

- **Embeddings from PPMI co-occurrence** (window ±2) over a small built-in corpus,
  randomly projected to 48-d. This makes Level 2 real: *cat is near dog because they
  occur in similar contexts*, which is the actual story of embeddings.
- **A bigram language prior** blended into the final logits
  (`logits = 0.35·model + 2.0·bigram`), so the generation loop produces readable
  toy-English instead of noise. Plus an optional repetition penalty (itself a real
  LLM inference technique, exposed as a toggle).

The transformer forward pass in between — LayerNorm → 4-head causal attention →
residual add → GELU MLP → residual add, ×3 layers → tied-embedding logits — is
mathematically the real thing, fully instrumented. The app discloses the split
("what's real / what's simulated") in the README, the Playground, and a quiz question.
**Stance: never fake a number that's presented as an internal; it's fine to have an
untrained model if you say so.**

Model size chosen for legibility, not capability: d=48, 3 layers, 4 heads, ctx=24,
dFF=96, vocab ≈340 (BPE, ~260 merges). Forward pass ≈4–6ms in-browser → everything
can recompute live on each keystroke.

## 3. Pedagogy: the level sequence

Mirrors the actual dataflow, one concept per level, each with a *manipulable* view
(the user's own input drives every panel) plus a goal and a 2-question quiz:

1. **Tokens** — BPE splitting, token IDs ("models see integers").
2. **Vectors** — embedding strips, +positional encoding, PCA vocabulary map with
   nearest-neighbor lookup (the "meaning from context" moment).
3. **Attention** — the star of the show: per-layer/head heatmaps, causal mask
   visible as the empty triangle, "who is this token listening to" ranking.
4. **Neurons** — MLP activation grids ("brain scan", the user's literal ask to "see
   the activations"), residual stream strips showing accumulation across layers.
5. **Prediction** — logits→softmax as bars, temperature/top-k sliders, dice-roll
   sampling (why LLM answers vary).
6. **Generation** — the autoregressive loop animated, tokens tinted by sampled
   probability; deliberately shows repetition/drift and explains why (scale vs
   mechanism).
7. **Playground** — free exploration once concepts are unlocked.

## 4. Gamification: light, intrinsic, skippable

Chosen: **progression** (levels unlock in order), **XP + ranks** (Curious Visitor →
Transformer Tamer), **badges** per completed level, **micro-goals** per level
("find a head where a token attends far behind it"), **quizzes** with first-try
bonuses, confetti on completion, localStorage persistence. Rejected: leaderboards,
timers, streaks (no server; pressure hurts learning). An "unlock all levels" escape
hatch respects impatient explorers — gates guide rather than imprison.

## 5. Visual design

Dark, single-committed theme on the validated data-viz reference palette
(surface `#1a1a19`/`#0d0d0d`, ink `#fff`/`#c3c2b7`/`#898781`, accent blue `#3987e5`):

- **Sequential blue ramp** for magnitude (attention weights, activation strength).
- **Diverging red↔neutral↔blue** for signed values (vector components).
- Identity never rides on color alone (labels everywhere); hover tooltips on the
  heatmap; text stays in ink tokens, not series colors.
- Canvas for dense grids (heatmaps, strips, scatter), DOM for labeled bars.

## 6. Workflow log (how this was built)

1. Read repo conventions (AGENTS.md / NEXT.md), picked a sibling-folder design that
   leaves the Python package untouched.
2. Loaded the data-viz design skill; took the validated dark palette from its
   reference instance.
3. Built the engine DOM-free (rng → corpus → tokenizer → model), verified in Node:
   BPE merges, cat/dog & sun/moon nearest neighbors, attention non-uniformity
   (entropy check), generation readability; tuned merges 200→260 and blend weights,
   added repetition penalty.
4. Built UI (7 stages, game layer, quizzes), then verified in headless Chromium via
   Playwright: full-page screenshots of every stage + quiz modal + a live generation
   run, console-error watch. Fixed what the screenshots showed (bar fills needed
   `display:block`, muddy near-zero neuron cells, duplicate map labels, favicon).
5. **Mobile pass** (user requirement: usable on a phone). Responsive media queries
   (≤640px: stacked rows, full-width buttons, static header — a sticky 200px header
   would eat a phone screen), `pointer: coarse` touch-target enlargement, 16px input
   font (prevents iOS zoom-on-focus), heatmaps in `overflow-x` scroll wrappers with a
   compact sizing mode, hover interactions doubled as tap (`pointerdown`). Verified
   with Playwright iPhone-13 emulation: per-stage screenshots, a
   document-scrollWidth overflow check on every stage, and a scripted tap on the
   attention heatmap asserting the weight readout. A GitHub Pages workflow deploys
   `genai-learning/` so the app is reachable by URL from a phone.

## 7. Where to take it next

- **Real trained model**: swap the engine for GPT-2-small via transformers.js with
  an adapter exposing the same trace interface; keep the toy as "X-ray mode".
  Attention heads then show real phenomena (induction heads, previous-token heads).
- **Logit lens level**: decode the residual stream at each layer to show the
  prediction forming gradually.
- **A "training" level**: watch the toy's loss fall as it trains on the corpus for a
  few hundred steps (tfjs or hand-rolled backprop — the model is small enough).
- **Sandbox corpus editing**: let the user replace `corpus.js` text in-app and watch
  tokenizer/embeddings/bigrams re-derive — connects nicely to this repo's continual
  learning theme (what the model "knows" is what it was exposed to).
- **More game**: per-level challenge validation (detect that the user actually found
  a 3-piece word), achievements for playground discoveries.

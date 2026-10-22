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

---

# Part 2 — "LLM Explorer 3D" (`explorer/`)

*2026-10-22. Follow-up brief: "go further — zoom into any part of the steps of
an example open LLM like GPT-3, in 3D: weights vs attention vs MLP vs the
vectorised input, renderable on GitHub, super deep."* Starting point was
3Blue1Brown's *But what is a GPT?* (Deep Learning ch. 5).

## Research (prior art surveyed before building)

| source | what was taken |
|---|---|
| 3Blue1Brown ch. 5–7 (GPT, attention, MLP) | vocabulary and framing: embedding/unembedding, residual stream as the object every block edits, Q/K/V, value-down/value-up (= W_V/W_O), neurons as questions, softmax with temperature; the GPT-3 parameter table (175,181,291,520) reproduced live in the Scale step with the same formulas |
| Brendan Bycroft, *LLM Visualization* (bbycroft.net/llm, github bbycroft/llm-viz) | a 3D scene where every tensor is a block and a walkthrough flies the camera (Intro → Embedding → LayerNorm → Self-attention → Softmax → Projection → MLP → Transformer → Output); the A/B/C sorting nano-gpt from minGPT (vocab 3, ctx 11, d 48, 3 layers, 3 heads) whose heads have readable jobs |
| Transformer Explainer (Polo Club, Georgia Tech) | a real model in the browser driven by the user's own text; temperature/sampling controls as a first-class view |
| Karpathy nanoGPT / minGPT, GPT-2 | the exact architecture (pre-LN, learned positions, GELU-tanh, tied unembedding, biases, N(0,0.02) init with 1/√(2L) on residual projections) mirrored in both the JAX trainer and the JS forward pass |

GPT-3 itself is not open, so the "example open LLM" is GPT-2's architecture
(identical to GPT-3's up to size), trained small enough to draw every number.

## Decisions

1. **Trained weights, two models.** The first app ran random weights (honestly
   disclosed). Here every claim in the walkthrough is about a number the model
   computed *because it learned to*, so both models are trained in-repo with a
   ~250-line JAX script (PyTorch wheels are blocked in the build sandbox; JAX
   from PyPI is not): the Bycroft sorter (100 % accuracy, interpretable heads)
   and a character-level Shakespeare LM (val loss 1.56 nats/char, 208 k params).
2. **JS forward pass with a verification vector.** The export carries the logits
   JAX produced for a fixed input; `npm test` asserts the browser forward pass
   matches to 1e-5, and the inspector recomputes every derivation from the
   inputs rather than reading it from the trace — the sum at the bottom of each
   term table is an independent check.
3. **One InstancedMesh per tensor, no level of detail.** 100–400 k cells is well
   within WebGL budgets on a laptop GPU; flat 2-triangle tiles for weights,
   boxes with height = |value| for activations. "Zooming into a step" is then
   literally moving the camera, and the same scene serves overview and cell.
4. **Weights vs activations as a visual grammar.** Blue plates / flat tiles =
   learned (fixed for every input); amber plates / bars = activations (depend
   on your input); purple = attention patterns; green = probabilities. Weights
   sit in a lane to the left of the activation they produce.
5. **Depth = derivations + lenses + logit lens + scale.** Per-cell arithmetic
   (every term, sorted), dependency highlights, token/head/layer lenses, the
   logit lens on every residual snapshot, live generation that grows every
   block by a row, and true-relative-scale wireframes of GPT-2 small and
   GPT-3 beside the toy.
6. **Static site, vendored three.js r170.** No build step; works on GitHub
   Pages and `python3 -m http.server`. ES modules + fetched weights mean it
   cannot run from `file://` (a notice explains this).

## Workflow log

1. Read the existing app for conventions (palette, no-build, Pages workflow).
2. Research pass (above); confirmed Bycroft's walkthrough order from the
   source tree; chose the two models.
3. JAX trainer → smoke-trained the sorter in 14 s → full runs in the background
   while writing the JS: forward pass (verified), layout, scene, walkthrough,
   inspector, app shell.
4. Headless-Chromium tour (Playwright, SwiftShader): screenshots of all 18
   steps, click-picking, generation, ghosts; fixed what they showed — label
   pile-up at overview distance (distance-based culling + layer group markers),
   too-edge-on cameras (higher elevation, minimum focus radius), fog hiding the
   scale ghosts, a scale table wider than its panel, and 12-triangle boxes
   where 2-triangle tiles suffice.

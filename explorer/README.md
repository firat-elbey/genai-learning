# 🔬 LLM Explorer 3D — zoom into every number of a real GPT

**Live: https://firat-elbey.github.io/genai-learning/explorer/**

A 3D, zoomable walkthrough of a **real, trained GPT-style transformer** running
live in your browser. Every tile in the scene is one number from the forward
pass over *your* input: the learned weight matrices, the residual stream, the
Q/K/V projections per head, the attention scores and softmax patterns, the MLP
neurons, the logits and the next-token probabilities. Click any tile and the
inspector shows the exact arithmetic that produced it — every term of the dot
product, biggest first — while its inputs light up in the scene.

It is the “go deeper” companion to the gamified app one level up: that one
runs an *untrained* toy to teach the mechanism; this one runs **trained**
models so the numbers mean something, and it lets you see the whole thing in
three dimensions, at every zoom level, from a single cell to GPT-3's footprint.

## Two models, trained for this app

| model | task | shape | params | trained |
|---|---|---|---|---|
| **Sorter A/B/C** (`models/sort.*`) | 6 letters in → the same letters sorted, one per step (Brendan Bycroft's nano-gpt toy) | d=48, 3 layers, 3 heads, 192 neurons, ctx 11, vocab 3 | 85,584 | 2,000 steps, ~1 min CPU, 100 % sequence accuracy on fresh inputs |
| **Shakespeare** (`models/shakespeare.*`) | character-level next-character prediction on tiny-shakespeare | d=64, 4 layers, 4 heads, 256 neurons, ctx 64, vocab 65 | 208,320 | 8,000 steps, ~17 min CPU, val loss 1.56 nats/char |

Both are exactly the GPT-2 architecture (pre-LayerNorm blocks, learned position
embeddings, GELU, tied unembedding, biases), just small enough that *every*
tensor can be drawn and every value inspected. `train/train_gpt.py` is the
trainer (JAX, CPU is fine); the exported `.json` carries the config, vocab,
tensor table, training curve and a **verification vector** (input ids + the
logits JAX produced) that the JavaScript forward pass is checked against:

```bash
cd explorer
npm test            # node test/verify.mjs → PASS shakespeare … max|Δlogit|=1.7e-5
```

## What you can do

- **18-step guided walkthrough** (← → keys): tokens → embedding → position →
  residual stream → LayerNorm → Q/K/V → scores & causal mask → softmax → value
  mixing → output projection → MLP neurons (GELU) → down-projection → stacking
  layers (logit lens) → unembedding → softmax/temperature → generation → scale.
  Every explanation is written with the live numbers of your input.
- **Zoom into any step**: click a tile → value, formula, all contributing terms,
  dependency highlight (amber) in the scene; click a block label → fly to it;
  the inspector has a 2D heatmap with token/vocab axes for every block.
- **Lenses**: follow one token's row through every block, isolate one head
  (its Q/K/V columns, its score & softmax blocks), or one layer.
- **Generate**: sample tokens (temperature, top-k) and watch a row get added to
  every activation block as the loop runs; the sorter visibly sorts.
- **Scale**: a parameter table computed with the same formulas for this model,
  GPT-2 small, GPT-2 XL and GPT-3 175 B (3Blue1Brown's breakdown), and wireframe
  outlines of GPT-2 / GPT-3 drawn next to the toy at true relative scale.

## Reading the scene

- Data flows **top → bottom**. Centre column = **activations** (amber plates,
  recomputed on every keystroke); left lane = **learned weights** (blue plates,
  flat tiles, fixed by training). Bar height = |value| for activations.
- Colour: blue positive, red negative (diverging, scaled per block at the 98th
  percentile); the single-hue blue ramp is used for probabilities (attention
  patterns, softmax output). Masked attention cells are dark.
- Head gaps split Q, K, V and the concatenated head outputs into per-head column
  groups; each head's score/softmax block is drawn separately.

## Run locally

```bash
git clone https://github.com/firat-elbey/genai-learning
cd genai-learning && python3 -m http.server 8000
# open http://localhost:8000/explorer/
```

It must be served over http (ES modules + fetched weights don't work from
`file://`). No build step, no network: three.js r170 is vendored in
`vendor/three/` (MIT).

## Retrain / change the models

```bash
pip install jax jaxlib numpy
cd explorer/train
python3 train_gpt.py --task sort
python3 train_gpt.py --task shakespeare --steps 8000   # downloads tiny-shakespeare (1.1 MB)
cd .. && npm test
```

Edit the `cfg` dicts in `train_gpt.py` to change width/depth; the explorer
reads all shapes from the export, so nothing else needs to change.

## Research notes: what shaped this design

- **3Blue1Brown, “But what is a GPT? Visual intro to transformers”** (Deep
  Learning ch. 5) and its attention/MLP chapters: the vocabulary used here
  (embedding, residual stream, query/key/value, value-down/value-up ≙ W_V/W_O,
  neurons as questions, unembedding, softmax with temperature) and the GPT-3
  parameter table reproduced in the Scale step (175,181,291,520 weights).
- **Brendan Bycroft's LLM Visualization** (bbycroft.net/llm): the idea of a 3D
  model where every tensor is a block and the walkthrough flies the camera
  through it, and the A/B/C sorting nano-gpt (d=48, 3 layers, 3 heads, ctx 11)
  whose heads have fully readable jobs. This explorer re-trains that model
  (and a real character LM) and adds per-cell arithmetic, lenses, a logit lens,
  generation and the scale ghosts.
- **Transformer Explainer** (Georgia Tech Polo Club): running a real model in
  the browser with the user's own text, temperature/sampling controls.
- **nanoGPT / minGPT** (Karpathy): the reference GPT-2 implementation the
  trainer and the JS forward pass mirror line by line.

Design decisions: (1) *trained* weights, not random — every statement in the
walkthrough is about a number the model actually computed; (2) two models,
because the sorter's attention patterns are interpretable at a glance while
the character LM shows the same machinery doing language; (3) one `InstancedMesh`
per tensor (≈100–400 k cells) so the whole model is one scene with no level-of-
detail tricks — zooming is literally moving the camera; (4) derivations are
recomputed from the inputs in the inspector (not read from the trace), so the
sum at the bottom of each table is an independent check of the forward pass.

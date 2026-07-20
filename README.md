# 🧠 Inside the Transformer

**Live app: https://firat-elbey.github.io/genai-learning/** — works great on phones;
add it to your home screen.

A gamified, zero-dependency web app that takes **your input text** and walks it
through a **real (tiny) GPT-style transformer running live in your browser**, so
you can *see* the whole pipeline behind every LLM answer:

```
text → tokens → vectors → attention → neuron activations → probabilities → sampling loop
```

Six quiz-gated levels + a free playground, with XP, ranks (Curious Visitor →
Transformer Tamer), badges, and progress saved on your device.

## The levels

| # | Level | What you see |
|---|-------|--------------|
| 1 | ✂️ Tokens | A real BPE tokenizer (trained in-browser) splits your text into token IDs |
| 2 | 🧭 Vectors | 48-d embeddings, positional encoding, and a PCA map where cat ≈ dog emerges from data |
| 3 | 🔦 Attention | 3 layers × 4 heads of live attention heatmaps; tap a cell to read the weight |
| 4 | ⚡ Neurons | MLP activation grids (the "brain scan") and the residual stream |
| 5 | 🎲 Prediction | Logits → softmax bars with temperature & top-k sliders and a dice-roll sampler |
| 6 | 🔁 Generation | The autoregressive loop writing live, token by token |
| 7 | 🎮 Playground | Everything at once, free play |

## What's real and what's simulated

**Real, computed on every keystroke:** the BPE tokenizer, PPMI co-occurrence
embeddings (+ PCA map), sinusoidal positions, and the full instrumented
transformer forward pass — every heatmap cell and activation square is a number
from that computation.

**Simulated:** training. The transformer weights are random (seeded); readable
generation comes from a corpus bigram prior blended into the logits. The
*mechanism* you watch is exactly what GPT-class models do — they add scale and
months of gradient descent. See [RESEARCH.md](RESEARCH.md) for the full design
rationale.

## Run locally

No build, no install, no network:

```bash
git clone https://github.com/firat-elbey/genai-learning
cd genai-learning
open index.html        # or: python3 -m http.server 8000
```

Every push to `main` auto-deploys to GitHub Pages via
`.github/workflows/deploy.yml`.

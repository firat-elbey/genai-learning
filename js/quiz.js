// Quiz content per stage. Two questions each; passing both completes the
// stage and unlocks the next one. `why` is shown after answering.

const QUIZZES = [
  // Stage 0 — Tokens
  [
    {
      q: "Why did a rare word like “unbelievable” get chopped into several tokens?",
      options: [
        "The tokenizer only keeps pieces it saw often during training, so rare words are built from common sub-pieces",
        "The model ran out of memory",
        "Long words are always split every 3 letters",
        "It's a bug in the tokenizer",
      ],
      correct: 0,
      why: "BPE keeps merging the most frequent character pairs seen in training text. Frequent words end up as single tokens; rare words get assembled from the sub-word pieces the tokenizer does know.",
    },
    {
      q: "What does the model actually receive after tokenization?",
      options: [
        "The raw letters of your text",
        "A list of token IDs — integers pointing into the vocabulary",
        "A grammar tree of the sentence",
        "An audio waveform",
      ],
      correct: 1,
      why: "The model never sees text. It sees a sequence of integers, one per token — everything downstream works on those IDs.",
    },
  ],
  // Stage 1 — Vectors
  [
    {
      q: "Why is the vector for “cat” close to the vector for “dog”?",
      options: [
        "They have similar spelling",
        "They appear in similar contexts, so they get similar vectors",
        "Both have three letters",
        "The vectors are assigned alphabetically",
      ],
      correct: 1,
      why: "Embeddings are learned from usage: words that show up in similar surroundings (“the ___ sat”, “the ___ likes”) end up pointing in similar directions. Meaning emerges from context.",
    },
    {
      q: "What job does the positional encoding do?",
      options: [
        "It tells the model how important each word is",
        "It compresses the vector to save memory",
        "It mixes “where in the sentence a token is” into its vector, since attention alone has no sense of order",
        "It translates tokens into English",
      ],
      correct: 2,
      why: "Attention treats its inputs as a set — without positional information, “dog bites man” and “man bites dog” would look identical. Position is added directly into each token's vector.",
    },
  ],
  // Stage 2 — Attention
  [
    {
      q: "In the attention map, what does a bright cell in row “mat”, column “cat” mean?",
      options: [
        "The token “mat” is copying information from the token “cat” into its own vector",
        "The two words rhyme",
        "“cat” comes right before “mat”",
        "The model is confused",
      ],
      correct: 0,
      why: "Each row shows where one token is looking. A bright cell means that token pulled a large weighted share of another token's value vector into itself — that's the “tokens talking to each other” step.",
    },
    {
      q: "Why is the upper-right half of the attention map always empty?",
      options: [
        "Those values are too small to draw",
        "The model is causal: a token may only look at tokens before it, never at the future",
        "It's a rendering optimization",
        "Attention only works on nouns",
      ],
      correct: 1,
      why: "GPT-style models predict the next token, so during generation a token can't peek ahead — the future hasn't been generated yet. The mask enforces that.",
    },
  ],
  // Stage 3 — Neurons
  [
    {
      q: "What does it mean when a neuron's cell is dark (near zero) in the activation grid?",
      options: [
        "The neuron is broken",
        "The neuron stayed quiet for this token — its pattern didn't match, so it passes almost nothing forward",
        "The token is unimportant",
        "The model skipped that layer",
      ],
      correct: 1,
      why: "Each neuron computes a weighted pattern-match over the token's vector and the GELU squashes weak matches toward zero. Only neurons whose pattern “fires” pass a signal on — that sparsity is the point.",
    },
    {
      q: "The residual stream shows the same token's vector after each layer. What's the big idea?",
      options: [
        "Each layer replaces the vector with a brand-new one",
        "Each layer adds a small refinement onto a running vector, so information accumulates layer by layer",
        "The vector shrinks at every layer",
        "Layers are independent and don't share information",
      ],
      correct: 1,
      why: "Transformers are residual: every attention and MLP block ADDS its output to the stream rather than replacing it. Deep models are hundreds of small nudges on one running representation.",
    },
  ],
  // Stage 4 — Prediction
  [
    {
      q: "What happens when you raise the temperature?",
      options: [
        "The model computes faster",
        "The probability distribution flattens, so rarer tokens get picked more often — more surprise, more nonsense",
        "The model gets smarter",
        "Only the top token can be chosen",
      ],
      correct: 1,
      why: "Temperature divides the scores before the softmax. High temperature shrinks the differences between candidates → flatter distribution → riskier picks. Low temperature sharpens it → safe, repetitive picks.",
    },
    {
      q: "The model's real output is not a word. What is it?",
      options: [
        "A probability for every single token in the vocabulary",
        "A grammar rule",
        "The single correct next word",
        "A search-engine query",
      ],
      correct: 0,
      why: "One forward pass ends with a score for EVERY vocabulary token, softmaxed into a probability distribution. “Choosing a word” is a separate sampling step on top of that distribution.",
    },
  ],
  // Stage 5 — Generation
  [
    {
      q: "How does the model produce a whole paragraph if one pass predicts only one token?",
      options: [
        "It predicts all tokens simultaneously",
        "It loops: sample a token, append it to the input, run the whole model again — every token you've ever seen an LLM write was made this way",
        "It retrieves paragraphs from a database",
        "A human picks the words",
      ],
      correct: 1,
      why: "Generation is autoregressive: each new token is appended and the (whole!) forward pass runs again on the longer sequence. It's one-token-at-a-time all the way down.",
    },
    {
      q: "Why does this tiny model repeat itself and drift off topic?",
      options: [
        "The browser tab is too small",
        "Repetition is impossible for transformers",
        "It has a tiny context, random untrained layers and only bigram statistics — real LLMs differ in scale and training, not in mechanism",
        "It was trained to be repetitive",
      ],
      correct: 2,
      why: "Everything you watched — tokens, vectors, attention, activations, sampling — is mechanically the same in GPT-4-class models. What's different is scale (billions of parameters, trillions of tokens) and actual training. Mechanism vs. capability!",
    },
  ],
];

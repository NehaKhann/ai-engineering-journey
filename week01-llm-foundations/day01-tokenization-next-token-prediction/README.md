# Day 1 — Tokenization & Next-Token Prediction

## What this covers
How LLMs break text into tokens, convert them to embeddings, and predict
the next token one step at a time. This is the foundational mechanism
behind every LLM response, regardless of model size or sophistication.

## Files
- `main.py` — loads GPT-2, tokenizes a sentence, and prints the top-5
  predicted next tokens with their scores.

## Setup

```powershell
# From repo root
venv\Scripts\activate
cd week01-llm-foundations\day01-tokenization-next-token-prediction
pip install transformers torch --break-system-packages
```

## How to run

```powershell
python main.py
```

**Note:** first run downloads GPT-2 (~548MB) — this only happens once.
Every run after that is instant since the model is cached locally.

## What I learned
- Tokens aren't always whole words — GPT-2 splits some words into sub-pieces
  (subword tokenization), not clean word-by-word chunks.
- `Ġ` = a leading space, part of GPT-2's byte-level BPE (Byte Pair Encoding)
  tokenizer. It appears before every token that had a space in front of it
  in the original text — it's not a bug, just how the tokenizer represents
  spacing internally.
- The model outputs a raw score (logit) for every possible next token, not
  a clean probability. Logits are unbounded — negative numbers are normal
  and expected. Only the *relative ranking* between scores matters, not
  their absolute value. To get real probabilities (e.g. "38% chance"),
  you'd run logits through a `softmax` function — not done in this script,
  but worth knowing for later.
- `torch.no_grad()` disables gradient tracking, which is only needed during
  training. Since we're just running inference (asking the model a
  question, not teaching it anything), this saves memory and speeds things up.
- GPT-2 has no domain-specific knowledge. For the input "My firewall keeps
  blocking," it predicted generic completions (the, all, access, any, my)
  rather than anything cybersecurity-specific. This is exactly the gap that
  fine-tuning (Week 2) and RAG (Week 6) are designed to close.

## Sample output

```
Tokens: ['My', 'Ġfirewall', 'Ġkeeps', 'Ġblocking']

Top 5 predicted next tokens:
  Ġthe             score: -103.86
  Ġall             score: -103.96
  Ġaccess          score: -105.12
  Ġany             score: -105.29
  Ġmy              score: -105.32
```

## Next step
Week 1, Day 2 — transformer architecture and attention: how the model
decides which earlier words matter most when predicting the next one.
# 📘 Day 1 — Tokenization & Next-Token Prediction

Learn how a Large Language Model (LLM) converts text into tokens, transforms them into embeddings, and predicts the next token one step at a time.

This is the core mechanism behind every GPT-style language model.

---

## 🎯 Objective

In this project, you'll explore how GPT-2 performs next-token prediction by:

* Tokenizing input text
* Feeding tokens into a pretrained GPT-2 model
* Predicting the most likely next token
* Understanding what logits represent
* Exploring GPT-2's Byte Pair Encoding (BPE) tokenizer

---

## 📂 Project Files

| File      | Description                                                                                                      |
| --------- | ---------------------------------------------------------------------------------------------------------------- |
| `main.py` | Loads GPT-2, tokenizes a sentence, runs inference, and prints the top 5 predicted next tokens with their logits. |

---

## ⚙️ Setup

Activate your virtual environment and install the required packages.

```powershell
# From repository root
venv\Scripts\activate

cd week01-llm-foundations\day01-tokenization-next-token-prediction

pip install transformers torch --break-system-packages
```

---

## ▶️ Run

```powershell
python main.py
```

---

## 📦 Model Download

On the first run, Hugging Face downloads the pretrained GPT-2 model (approximately **548 MB**).

This only happens once.

Afterward, the model is loaded from the local cache, so future runs start almost instantly.

---

## 🧠 Key Takeaways

### 1. Tokens are not always words

GPT-2 doesn't split text strictly by words.

Instead, it uses **Byte Pair Encoding (BPE)** to break text into smaller subword units when necessary.

For example, uncommon words may be divided into multiple tokens.

---

### 2. Why do tokens start with `Ġ`?

The `Ġ` character indicates that the token originally had a leading space.

Example:

```
"My firewall keeps blocking"

↓

['My', 'Ġfirewall', 'Ġkeeps', 'Ġblocking']
```

This is simply how GPT-2's byte-level tokenizer stores spacing internally.

---

### 3. The model predicts logits, not probabilities

GPT-2 produces a **logit** (raw score) for every possible next token.

Important points:

* Logits can be positive or negative.
* Their absolute values don't matter.
* Only their relative ranking determines which tokens are most likely.

To convert logits into probabilities, a **Softmax** function is applied.

This example intentionally prints the raw logits so you can see what the model actually outputs before any probability conversion.

---

### 4. Why use `torch.no_grad()`?

During inference, gradient calculations aren't needed.

Wrapping inference inside:

```python
with torch.no_grad():
```

provides several benefits:

* Lower memory usage
* Faster execution
* No unnecessary computation graph

This is standard practice whenever you're using a pretrained model for prediction instead of training.

---

### 5. GPT-2 has no domain-specific knowledge

For the prompt:

```
"My firewall keeps blocking"
```

GPT-2 predicts generic continuations such as:

* the
* all
* access
* any
* my

It doesn't understand that the sentence relates to cybersecurity.

This limitation motivates later topics in the series, including:

* Fine-tuning (Week 2)
* Retrieval-Augmented Generation (Week 6)

---

## 💻 Sample Output

```text
Tokens:
['My', 'Ġfirewall', 'Ġkeeps', 'Ġblocking']

Top 5 predicted next tokens:

Ġthe             score: -103.86
Ġall             score: -103.96
Ġaccess          score: -105.12
Ġany             score: -105.29
Ġmy              score: -105.32
```

---

## 🚀 What's Next?

**Week 1 • Day 2 — Transformer Architecture & Attention**

Learn how Transformers decide which earlier words matter most when predicting the next token, and why attention made modern LLMs possible.

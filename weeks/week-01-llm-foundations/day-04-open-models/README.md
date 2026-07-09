# Day 4 — Running Open Language Models

> Week 1 • LLM Foundations

Learn how to load and interact with real open-source Large Language Models using the **Hugging Face Transformers** library.

This exercise introduces model inference by comparing a **base language model (GPT-2)** with an **instruction-tuned model (Qwen2.5-0.5B-Instruct)** while exploring how generation parameters influence model behavior.

---

## 📚 Learning Objectives

By the end of this exercise, you'll understand:

- How to load pretrained models from Hugging Face
- How to generate text using the `pipeline()` API
- The difference between base and instruction-tuned models
- How generation parameters influence model output
- Why instruction tuning dramatically improves assistant-style responses

---

## 📂 Project Structure

```text
day-04-open-models/
├── README.md
├── open_models.ipynb
└── run_open_model.py
```

| File | Description |
|------|-------------|
| `open_models.ipynb` | Interactive notebook exploring Hugging Face models, inference, and generation settings. |
| `run_open_model.py` | Standalone script comparing GPT-2 and Qwen2.5-Instruct while experimenting with generation parameters. |
| `README.md` | Documentation for this exercise. |

---

## ⚙️ Requirements

Install the required dependencies from the repository root.

```bash
pip install -r requirements.txt
```

or

```bash
pip install transformers torch matplotlib
```

---

## ▶️ Run

```bash
python run_open_model.py
```

You can also explore the concepts interactively using:

```
open_models.ipynb
```

---

## 📦 Model Download

On the first execution, Hugging Face downloads the required pretrained models:

- **GPT-2** (~548 MB)
- **Qwen2.5-0.5B-Instruct** (~1 GB)

The models are cached locally, so future runs load directly from disk without downloading again.

---

## 🧠 Concepts Covered

### Hugging Face `pipeline()`

The `pipeline()` API provides a high-level interface for running language models with only a few lines of code.

It automatically handles:

- Tokenization
- Model loading
- Inference
- Output decoding

making it an excellent starting point for working with Transformers.

---

### Base Language Models

Models like **GPT-2** are trained to predict the next token in a sequence.

They are excellent at continuing text but are not specifically optimized to follow user instructions or answer questions directly.

---

### Instruction-Tuned Models

Models such as **Qwen2.5-Instruct** undergo supervised fine-tuning after pretraining.

This additional training teaches them to:

- Follow user instructions
- Answer questions directly
- Produce more helpful responses
- Better align with user intent

This is why instruction-tuned models perform significantly better in chatbot-style interactions.

---

### Temperature

The `temperature` parameter controls the randomness of generated text.

- **Low (0.2)** → More focused and deterministic responses
- **Medium (0.7)** → Balanced creativity and coherence
- **High (1.0+)** → More diverse and creative outputs

Changing the temperature demonstrates how the same model can produce very different responses.

---

### Generation Parameters

Besides `temperature`, several additional parameters influence text generation.

| Parameter | Purpose |
|-----------|---------|
| `max_new_tokens` | Limits how many new tokens are generated. |
| `temperature` | Controls randomness during generation. |
| `do_sample` | Enables probabilistic sampling. |
| `top_k` | Restricts sampling to the top *k* candidates. |
| `top_p` | Uses nucleus sampling based on cumulative probability. |

Learning these parameters is essential for controlling LLM behavior.

---

## 📚 Key Concepts Summary

| Concept | Description |
|----------|-------------|
| **Base Model** | Predicts the next token using the provided context. |
| **Instruction-Tuned Model** | Fine-tuned to follow instructions and assist users. |
| **`pipeline()`** | High-level Hugging Face API for inference. |
| **Temperature** | Controls the creativity and randomness of generated text. |
| **Sampling** | Introduces randomness into generation. |
| **Model Caching** | Downloaded models are stored locally and reused. |

---

## 💡 Key Insight

A **base model** learns language by predicting the next token.

An **instruction-tuned model** builds on that foundation through additional supervised fine-tuning, enabling it to follow instructions, answer questions, and behave like a conversational assistant.

Both share the same Transformer architecture—the primary difference lies in how they are trained after pretraining.

---

## 🎯 Key Takeaways

- Hugging Face makes model inference straightforward through the `pipeline()` API.
- Base models predict text continuations.
- Instruction-tuned models are optimized for assistant-style interactions.
- Generation parameters significantly influence model behavior.
- The same underlying architecture can behave very differently depending on post-training.

---

## 📖 Related Resources

- Week 1 Overview → `weeks/week-01-llm-foundations`
- Repository Roadmap → `ROADMAP.md`

---

## ⏭️ Next Project

**LLM Explainer Dashboard**

Build an interactive application that visualizes how a Large Language Model tokenizes text, predicts the next token, and uses attention to generate responses step by step.
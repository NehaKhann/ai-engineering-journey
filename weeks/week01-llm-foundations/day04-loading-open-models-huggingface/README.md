# 📘 Day 4 — Loading & Running Open Models with Hugging Face

Learn how to load and interact with real open-source Large Language Models using the **Hugging Face Transformers** library.

Today you'll compare a **base language model (GPT-2)** with an **instruction-tuned model (Qwen2.5-Instruct)** and explore how generation parameters influence the model's responses.

---

## 🎯 Objective

In this project, you'll learn how to:

* Load pretrained language models from Hugging Face
* Generate text using the `pipeline()` API
* Compare base and instruction-tuned models
* Experiment with generation parameters such as `temperature`
* Understand why instruction tuning dramatically improves assistant-style responses

---

## 📂 Project Files

| File                | Description                                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `run_open_model.py` | Loads GPT-2 and Qwen2.5-Instruct, compares their outputs, experiments with different temperature values, and visualizes the results. |

---

## ⚙️ Setup

Activate your virtual environment and install the required packages.

```powershell
# From repository root
venv\Scripts\activate

cd week01-llm-foundations\day04-loading-open-models

pip install transformers torch matplotlib --break-system-packages
```

---

## ▶️ Run

```powershell
python run_open_model.py
```

---

## 📦 Model Download

On the first run, Hugging Face downloads the required pretrained models:

* **GPT-2** (~548 MB)
* **Qwen2.5-0.5B-Instruct** (~1 GB)

These models are cached locally, so future runs load almost instantly without downloading again.

---

## 🧠 Key Takeaways

### 1. `pipeline()` makes inference simple

The `pipeline()` API provides a high-level interface for running language models with just a few lines of code.

It handles tokenization, model loading, inference, and decoding automatically, making it an excellent starting point for working with Transformers.

---

### 2. Base models complete text

Models like **GPT-2** are trained to predict the next token.

They excel at continuing text but aren't specifically optimized to follow instructions or answer questions.

Given a prompt, they simply generate the most likely continuation.

---

### 3. Instruction-tuned models behave like assistants

Models such as **Qwen2.5-Instruct** undergo additional supervised fine-tuning after pretraining.

This teaches them to:

* Follow user instructions
* Answer questions directly
* Produce more helpful and conversational responses
* Better align with user intent

This is why instruction-tuned models perform much better in chatbot-style interactions.

---

### 4. Temperature controls randomness

The `temperature` parameter influences how predictable or creative the generated text will be.

* **Low temperature (e.g., 0.2)** → More focused and deterministic responses
* **Medium temperature (e.g., 0.7)** → Balanced creativity and coherence
* **High temperature (e.g., 1.0+)** → More diverse, creative, and sometimes unpredictable outputs

Experimenting with different values demonstrates how a single model can generate very different responses.

---

### 5. Generation parameters shape the output

Besides `temperature`, several other parameters affect generation:

| Parameter            | Purpose                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| `max_new_tokens`     | Limits how many new tokens the model generates.                                                              |
| `temperature`        | Controls randomness and creativity.                                                                          |
| `do_sample`          | Enables probabilistic sampling instead of always choosing the highest-probability token.                     |
| `top_k` *(optional)* | Restricts sampling to the top *k* most likely tokens.                                                        |
| `top_p` *(optional)* | Uses nucleus sampling by selecting from the smallest set of tokens whose cumulative probability exceeds *p*. |

Learning these parameters is essential for controlling LLM behavior.

---

## 📚 Key Concepts Summary

| Concept                     | Description                                                            |
| --------------------------- | ---------------------------------------------------------------------- |
| **Base Model**              | Predicts the next token based on the input context.                    |
| **Instruction-Tuned Model** | Fine-tuned to follow instructions and behave like a helpful assistant. |
| **`pipeline()`**            | A high-level Hugging Face API for running inference with minimal code. |
| **Temperature**             | Controls how deterministic or creative text generation is.             |
| **Sampling**                | Introduces randomness so outputs aren't always identical.              |
| **Model Caching**           | Downloaded models are stored locally and reused in future runs.        |

---

## 💡 Key Insight

A **base model** learns language by predicting the next token.

An **instruction-tuned model** builds on that foundation by learning how to follow prompts, answer questions, and interact naturally with users.

Both use the same Transformer architecture—the difference lies in how they're trained after pretraining.

---

## 🚀 What's Next?

**Week 1 • Day 5 — Mini Project: Token-by-Token Explainer**

Build an interactive project that visualizes how an LLM generates text one token at a time, making the prediction process easier to understand and explore.

# 📘 Day 5 — Understanding Context Windows

Learn one of the most important concepts in Large Language Models: the **Context Window**.

A context window defines how much information an LLM can "remember" while generating a response. Understanding this limitation is essential for designing effective prompts, building chatbots, and working with long documents.

---

## 🎯 Objective

In this project, you'll learn how context windows work by:

- Understanding what a context window is
- Counting tokens in different prompts
- Exploring GPT-2's maximum context length
- Comparing short and long prompts
- Understanding what happens when the context limit is exceeded
- Learning why modern LLMs support much larger context windows

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `context_window_demo.py` | Demonstrates token counting, context limits, and prompt length analysis using GPT-2. |
| `context_window_demo.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation and explanation for the project. |

---

## ⚙️ Setup

Activate your virtual environment and install the required packages.

```powershell
# From repository root
venv\Scripts\activate

cd weeks\week-01-llm-foundations\day-05-context-window

pip install transformers matplotlib
```

---

## ▶️ Run

```powershell
python context_window_demo.py
```

---

## 🧠 Key Concepts

### 1. What is a Context Window?

A **Context Window** is the maximum number of tokens a language model can process at one time.

Everything the model knows while generating a response must fit inside this window.

---

### 2. Context Windows Are Measured in Tokens

LLMs don't count words—they count **tokens**.

For example, a paragraph containing only a few hundred words may already contain several hundred tokens.

The tokenizer determines how text is divided into tokens before being processed by the model.

---

### 3. GPT-2 Context Limit

GPT-2 has a maximum context window of **1,024 tokens**.

If the combined input and generated output exceed this limit, older tokens must be discarded.

This means the model can no longer use that information when generating future responses.

---

### 4. Modern Models Support Much Larger Context Windows

Recent language models have dramatically increased their context limits.

| Model | Approximate Context Window |
|--------|---------------------------:|
| GPT-2 | 1,024 tokens |
| Modern LLMs | 32K–128K+ tokens |

Larger context windows allow models to process:

- Longer conversations
- Large documents
- Multiple files
- Entire codebases

without forgetting earlier information.

---

### 5. Why Context Windows Matter

The size of the context window directly affects an application's capabilities.

A larger context window enables:

- Better long-form conversations
- Improved document understanding
- More accurate code assistance
- Stronger retrieval-augmented generation (RAG) systems

---

## 📚 Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Context Window** | The maximum number of tokens a model can process at once. |
| **Token** | The basic unit of text processed by an LLM. |
| **Token Limit** | The maximum number of tokens allowed during inference. |
| **Truncation** | Older tokens are removed when the context window is exceeded. |
| **Long Context Models** | Modern LLMs capable of processing tens or hundreds of thousands of tokens. |

---

## 💻 Sample Output

```text
=== Context Window Demo ===

Model: GPT-2

Maximum Context Window:
1024 tokens

Prompt:
"The firewall blocked the traffic because..."

Prompt Tokens:
18

Remaining Context:
1006 tokens

Context Usage:
1.76%
```

---

## 🎯 Key Takeaways

After completing this lesson, you'll understand:

- What a context window is and why it matters
- Why LLMs measure input in tokens instead of words
- GPT-2's context limitations
- How exceeding the context window affects model behavior
- Why modern LLMs continue increasing context sizes
- How context windows influence prompt engineering and AI application design

---

## 🚀 What's Next?

**Week 1 • Day 6 — Generation Parameters**

Learn how generation parameters such as **temperature**, **top-k**, **top-p**, **max_new_tokens**, and **sampling** influence the creativity, diversity, and determinism of Large Language Model outputs.
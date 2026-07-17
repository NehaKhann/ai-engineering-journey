# Day 1 — Prompt Engineering vs Fine-Tuning

> Week 2 — Fine-Tuning Fundamentals

Learn the difference between **Prompt Engineering** and **Fine-Tuning** — two fundamental approaches to adapting Large Language Models (LLMs) for specific tasks.

Prompt engineering guides a pretrained model using carefully designed instructions. Fine-tuning teaches the model new behaviors by updating its weights with custom training data.

---

## Learning Objectives

By the end of this exercise, you'll understand:

- Why prompt engineering requires instruction-tuned models
- The difference between base models and instruction-tuned models
- Four common prompting techniques: zero-shot, few-shot, chain-of-thought, and role prompting
- How system messages and user messages work together
- How temperature interacts with prompt design
- The strengths and limitations of prompt engineering
- When fine-tuning is the better solution

---

## Project Structure

```
day-01-prompt-engineering/
├── README.md
├── prompt_engineering.py
└── prompt_engineering.ipynb
```

| File | Description |
|------|-------------|
| `prompt_engineering.py` | Demonstrates prompt engineering techniques on an instruction-tuned model and compares with fine-tuning. |
| `prompt_engineering.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation for this exercise. |

---

## Requirements

```bash
pip install transformers torch matplotlib
```

---

## Run

```bash
python prompt_engineering.py
```

Or open `prompt_engineering.ipynb` in Jupyter Notebook or VS Code.

> **Note:** The first run downloads GPT-2 (~548MB) and Qwen2.5-0.5B-Instruct (~1GB) from Hugging Face. Subsequent runs load from cache.

---

## Concepts Covered

### 1. Base Models vs Instruction-Tuned Models

A **base model** (like GPT-2) is trained only to predict the next token. It completes text — it does not follow instructions.

An **instruction-tuned model** (like Qwen2.5-Instruct) undergoes supervised fine-tuning after pretraining, which teaches it to follow instructions, answer questions, and behave like an assistant.

Prompt engineering techniques only work reliably on instruction-tuned models. Running them on a base model produces text completions rather than answers.

### 2. Four Prompting Techniques

**Zero-Shot Prompting**
Ask the model directly without providing examples. The model relies on its training to understand what is being asked.

**Few-Shot Prompting**
Provide examples before asking the question. The model infers the pattern from the examples and applies it. This technique works through the chat template format — alternating user and assistant messages.

**Chain-of-Thought (CoT) Prompting**
Ask the model to reason step by step. The explicit reasoning structure leads to more thorough and logically structured responses.

**Role Prompting**
Assign the model a specific persona or expertise using a system message. This influences the tone, depth, and framing of the response.

### 3. System Messages vs User Messages

Modern chat-based models distinguish between:
- **System message** — Sets the model's behavior, persona, and constraints. Applied once at the start.
- **User message** — The actual query or instruction.

This separation gives engineers fine-grained control over model behavior without repeating instructions in every prompt.

### 4. Temperature and Prompting

Temperature controls how strictly the model follows the most probable token path:
- **Low (0.2)** — Deterministic, consistent, predictable
- **Balanced (0.7)** — Natural variation while staying on topic
- **High (1.2)** — Creative but less reliable

Temperature interacts with prompts: a well-structured prompt with low temperature produces highly consistent outputs; the same prompt with high temperature explores different phrasings.

### 5. When Prompt Engineering Is Not Enough

Prompt engineering is fast and free — but it has limits:
- Models produce inconsistent responses
- Complex instructions may be ignored
- Long prompts become unwieldy
- The model cannot learn new facts or domain knowledge

When these limits are reached, fine-tuning is the better solution.

---

## Comparison: Prompt Engineering vs Fine-Tuning

| Aspect | Prompt Engineering | Fine-Tuning |
|--------|--------------------|-------------|
| Training Required | No | Yes |
| Time to Deploy | Immediate | Hours–days |
| Cost | Free | GPU compute required |
| Model Weights Changed | No | Yes (permanent) |
| Consistency | Varies | High |
| Domain Knowledge | Limited by base model | Can learn new domains |
| Best For | Quick prototypes, exploration | Production systems, specialization |

### Decision Flow

```
Start with Prompt Engineering
         ↓
    Evaluate results
         ↓
  Need more consistency        Stay with
  or domain expertise?   ──→   Prompt Engineering
         ↓
    Move to Fine-Tuning
```

---

## Sample Output

The program compares four prompting techniques on the same question, shows how temperature changes responses, and provides a decision framework for choosing between prompting and fine-tuning.

---

## Key Takeaways

- Prompt engineering only works on instruction-tuned models
- Base models (GPT-2) complete text; instruction-tuned models (Qwen2.5) follow instructions
- Four techniques: zero-shot, few-shot, chain-of-thought, role prompting
- System messages control model behavior independently of user queries
- Temperature affects how deterministic or creative the response is
- Prompt engineering is the fastest way to improve outputs
- Fine-tuning is needed for consistency and domain expertise

---

## Related Resources

- Day 2 — Instruction Tuning & Chat Templates
- Week 1 Day 4 — Running Open Models (introduction to base vs instruction-tuned models)
- Week 2 Overview — `weeks/week02-fine-tuning-fundamentals`

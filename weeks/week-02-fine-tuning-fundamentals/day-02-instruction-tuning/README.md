# Day 2 — Instruction Tuning & Chat Templates

> Week 2 — Fine-Tuning Fundamentals

Learn how instruction-tuned models format conversations using **chat templates** and why proper formatting is critical for fine-tuning.

Instruction tuning is the process of fine-tuning a base language model on instruction-response pairs to make it follow directions. The chat template defines how those pairs are structured into a single text input the model can process.

---

## Learning Objectives

By the end of this exercise, you'll understand:

- What instruction tuning is and why it matters
- What a chat template is (Jinja2 format)
- How `apply_chat_template()` formats conversations
- How special tokens mark role boundaries (system, user, assistant)
- Why base models lack chat templates
- How token counts differ between raw text and template-formatted text
- How to build an instruction dataset
- What loss masking is and why only assistant tokens are learned

---

## Project Structure

```
day-02-instruction-tuning/
├── README.md
├── instruction_tuning.py
└── instruction_tuning.ipynb
```

| File | Description |
|------|-------------|
| `instruction_tuning.py` | Demonstrates chat templates, conversation formatting, dataset building, and loss masking. |
| `instruction_tuning.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation for this exercise. |

---

## Requirements

```bash
pip install transformers torch matplotlib
```

---

## Run

```bash
python instruction_tuning.py
```

Or open `instruction_tuning.ipynb` in Jupyter Notebook or VS Code.

---

## Concepts Covered

### 1. What is Instruction Tuning?

Instruction tuning is a supervised fine-tuning (SFT) process where a base model is trained on instruction-response pairs. This teaches the model to:

- Follow user instructions
- Answer questions directly
- Produce helpful, structured responses
- Understand the distinction between user input and assistant output

The result is an "instruction-tuned" model — like Qwen2.5-Instruct, Llama-3-Instruct, or GPT-4.

### 2. Chat Templates

A chat template is a Jinja2 template string stored in the tokenizer. It defines how structured conversations (lists of messages with roles) are converted into a single text string.

Example (Qwen2.5):

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is a firewall?<|im_end|>
<|im_start|>assistant
A firewall monitors network traffic.<|im_end|>
```

Base models like GPT-2 have no chat template. They cannot distinguish between user and assistant messages — they see all text as a flat sequence to continue.

### 3. Role-Based Messages

Modern chat models distinguish between three message roles:

| Role | Purpose |
|------|---------|
| `system` | Sets model behavior, persona, and constraints |
| `user` | The human's input or question |
| `assistant` | The model's response |

During fine-tuning, these roles are marked with special tokens that the model learns to associate with each role.

### 4. apply_chat_template()

The `tokenizer.apply_chat_template()` method converts a list of message dictionaries into formatted text:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is a firewall?"},
    {"role": "assistant", "content": "A firewall monitors network traffic."}
]

formatted = tokenizer.apply_chat_template(messages, tokenize=False)
```

### 5. Token Count Impact

Template-formatted text includes special tokens that add to the total token count. This matters for context window budgeting during training and inference.

### 6. Building an Instruction Dataset

A fine-tuning dataset consists of instruction-response pairs. Each pair is formatted using the model's chat template before tokenization. The system message is typically the same across all examples.

### 7. Loss Masking

During fine-tuning, loss is only computed on the assistant's response tokens. Input tokens (system prompt, user instruction) are assigned a label of `-100` in PyTorch, which causes the loss function to ignore them.

This ensures the model learns to generate good responses rather than learning to repeat the input.

---

## Key Takeaways

- Chat templates convert structured messages into model-ready text
- Base models have no template; instruction-tuned models do
- `apply_chat_template()` is the standard way to format conversations
- Special tokens mark role boundaries
- Loss masking ensures only assistant responses are learned
- Using the wrong template during inference produces incorrect output
- Every fine-tuning pipeline must apply the correct template

---

## Related Resources

- Day 1 — Prompt Engineering vs Fine-Tuning
- Day 3 — Dataset Preparation
- Week 2 Overview — `weeks/week02-fine-tuning-fundamentals`

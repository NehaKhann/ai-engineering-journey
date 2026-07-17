# Day 4 — Supervised Fine-Tuning (SFT)

> Week 2 — Fine-Tuning Fundamentals

Fine-tune a GPT-2 model on a cybersecurity instruction dataset using Hugging Face TRL's SFTTrainer. This is where theory becomes practice — you'll update a real model's weights, save it, and compare responses before and after training.

---

## Learning Objectives

By the end of this exercise, you'll understand:

- What Supervised Fine-Tuning (SFT) is and when to use it
- How TRL's SFTTrainer simplifies the fine-tuning workflow
- How response templates enable loss masking during training
- How to configure training arguments for fine-tuning
- How to evaluate a model before and after fine-tuning
- How to save and reload a fine-tuned model

---

## Project Structure

```
day-04-supervised-fine-tuning/
├── README.md
├── sft_training.py
├── sft_training.ipynb
└── gpt2-cybersecurity-sft/          (generated)
    ├── config.json
    ├── model.safetensors
    └── tokenizer files...
```

| File | Description |
|------|-------------|
| `sft_training.py` | Loads dataset, fine-tunes GPT-2 with SFTTrainer, saves and tests the model. |
| `sft_training.ipynb` | Interactive notebook version of the lesson. |
| `gpt2-cybersecurity-sft/` | Fine-tuned model (generated after training). |

---

## Requirements

```bash
pip install transformers torch datasets trl peft accelerate matplotlib
```

---

## Run

```bash
python sft_training.py
```

Or open `sft_training.ipynb` in Jupyter Notebook or VS Code.

**Note:** Training runs on CPU by default. Each step takes a few seconds. The script completes in 2-5 minutes.

---

## Concepts Covered

### 1. What is Supervised Fine-Tuning?

SFT takes a pretrained language model and trains it on labeled instruction-response pairs. Unlike prompt engineering (which changes only the input), SFT updates the model's internal weights so it permanently learns new behavior.

### 2. The SFT Pipeline

```
Raw dataset (JSONL)
       ↓
Format with instruction/response template
       ↓
Create Hugging Face Dataset
       ↓
Tokenize with `response_template` loss masking
       ↓
Train with SFTTrainer (updates model weights)
       ↓
Save fine-tuned model
       ↓
Load and compare before/after responses
```

### 3. SFTTrainer

TRL's `SFTTrainer` extends Hugging Face's `Trainer` with SFT-specific features:

| Feature | Purpose |
|---------|---------|
| `dataset_text_field` | Column containing the full formatted text |
| `response_template` | String marking where the response starts (for loss masking) |
| `max_seq_length` | Truncates examples to this token length |
| Automatic loss masking | Computes loss only on response tokens (ignores instruction) |

### 4. Training Arguments

Key parameters configured for this exercise:

| Argument | Value | Purpose |
|----------|-------|---------|
| `max_steps` | 30 | Total training steps (small for demo) |
| `learning_rate` | 3e-5 | Standard rate for fine-tuning |
| `per_device_train_batch_size` | 1 | Single example per batch (CPU-friendly) |
| `eval_strategy` | "steps" | Evaluate periodically during training |
| `fp16` | False | Disabled (not supported on CPU) |

### 5. Loss Masking with Response Templates

The `response_template` tells SFTTrainer where the assistant's response begins. Tokens before this point (the instruction) are assigned a label of -100 and ignored during loss computation. This ensures the model learns to generate good responses, not just repeat the instruction.

### 6. Before and After Comparison

Testing the same prompt before and after fine-tuning reveals the impact of SFT. The base GPT-2 produces generic text completions, while the fine-tuned version should produce more focused, domain-relevant responses.

---

## Training Output

During training, you'll see output similar to:

```
Step  | Training Loss | Validation Loss
5/30  | 4.8923        | 4.5210
10/30 | 3.4512        | 3.2134
...
30/30 | 1.2345        | 1.1023
```

Both losses should decrease over time, indicating the model is learning.

---

## Files Generated

| File | Description |
|------|-------------|
| `gpt2-cybersecurity-sft/config.json` | Model configuration |
| `gpt2-cybersecurity-sft/model.safetensors` | Fine-tuned model weights |
| `gpt2-cybersecurity-sft/tokenizer.json` | Tokenizer files |

---

## Key Takeaways

- SFT updates model weights to learn domain-specific behavior
- SFTTrainer handles formatting, loss masking, and training
- `response_template` enables automatic loss masking
- Fine-tuning on small datasets can produce measurable improvement
- Always compare before and after to evaluate the impact
- SFT is the foundation for LoRA, QLoRA, and RLHF

---

## Related Resources

- Day 3 — Dataset Preparation (dataset used for training)
- Day 5 — Model Evaluation (compare fine-tuned vs base model)
- Week 2 Overview — `weeks/week02-fine-tuning-fundamentals`

# Day 2 — LoRA (Low-Rank Adaptation)

> Week 3 — Efficient Fine-Tuning & Quantization

Learn **LoRA** — instead of updating every weight in a model, freeze the base model and train small low-rank adapter matrices on top. This is the technique that makes fine-tuning large models affordable, and the second half of what QLoRA (Day 4) combines with quantization.

---

## Learning Objectives

By the end of this exercise, you'll understand:

- Why full fine-tuning is expensive, and what LoRA does differently
- The math behind LoRA: `W_new = W + (alpha / r) * (B @ A)`
- How to configure LoRA with PEFT's `LoraConfig` (`r`, `lora_alpha`, `target_modules`, `lora_dropout`)
- How trainable parameter count changes as rank (`r`) increases
- How to confirm adapters are actually learning, not just memorizing
- Why you can train and swap multiple LoRA adapters on the same frozen base model

---

## Project Structure

```
day-02-lora/
├── README.md
├── lora.py
└── lora.ipynb
```

| File | Description |
|------|-------------|
| `lora.py` | Compares full fine-tuning cost vs LoRA, sweeps across ranks, and runs a tiny training demo. |
| `lora.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation for this exercise. |

---

## Requirements

Install once from the week-level requirements file:

```bash
cd weeks/week-03-efficient-fine-tuning
pip install -r requirements.txt
```

---

## Run

```bash
python lora.py
```

Or open `lora.ipynb` in Jupyter Notebook, VS Code, or Google Colab.

> **Note:** This runs fine on CPU for the parameter-counting and rank-sweep sections. The tiny training demo (20 steps) is fast even on CPU for a 0.5B model, but a GPU (e.g. free Colab T4) will be noticeably quicker. The first run downloads `Qwen2.5-0.5B-Instruct` (~1GB) from Hugging Face; subsequent runs load from cache.

---

## Concepts Covered

### 1. The Problem With Full Fine-Tuning

Full fine-tuning updates every weight matrix in the model. For a matrix of shape `(d, d)`, that's `d * d` trainable parameters — across every layer of a modern LLM, this reaches billions of parameters, each needing gradients, optimizer states, and memory during training.

### 2. LoRA's Core Idea

LoRA freezes the original weight matrix `W` and learns a small update instead:

```
W_new = W + (alpha / r) * (B @ A)
```

- `A` has shape `(r, d)` — projects down to a small rank `r`
- `B` has shape `(d, r)` — projects back up to full size
- `r` (rank) is tiny — 4, 8, 16, or 32 — compared to `d` (often 1024+)

Because `r << d`, the trainable parameter count in `A` and `B` is a tiny fraction of the full matrix — while the frozen base model still does the heavy lifting.

### 3. Configuring LoRA with PEFT

`LoraConfig` controls the adapter setup:

- `r` — rank, the main capacity knob. Higher = more expressive, more parameters.
- `lora_alpha` — scaling factor; the effective update is scaled by `alpha / r`.
- `target_modules` — which layers get adapters. `q_proj`/`v_proj` (attention query & value projections) is a strong, common default.
- `lora_dropout` — dropout on the adapter path during training, for regularization.

`get_peft_model()` wraps the base model and automatically freezes everything except the adapters.

### 4. Rank vs Trainable Parameters

Sweeping `r = 4, 8, 16, 32` shows trainable parameters scaling roughly linearly with rank — but even at the high end, LoRA typically trains well under 1% of the full model's parameters.

### 5. Proving It Actually Learns

Parameter counts are convincing on paper, but the real test is behavior. Training LoRA adapters on a toy dataset with a distinctive, consistent pattern (a sign-off phrase) and then testing on an *unseen* prompt shows whether the adapter generalized the pattern — not just memorized the training examples.

---

## Comparison: Full Fine-Tuning vs LoRA

| Aspect | Full Fine-Tuning | LoRA |
|--------|------------------|------|
| Trainable parameters | 100% of model | Often < 1% of model |
| Memory required | Highest | Much lower |
| Training speed | Slower (large gradient/optimizer state) | Faster |
| Base model | Modified permanently | Stays frozen, untouched |
| Multiple tasks | Needs a full copy per task | Swap small adapters on one base |
| Quality ceiling | Highest possible | Very close, for most tasks |

---

## Sample Output

The program prints the full model's total parameter count, then the trainable parameter count and percentage at each rank (4, 8, 16, 32), then trains adapters for 20 steps on toy data and shows the model's response to an unseen prompt before and after training — saved to `experiment_log.json`.

---

## Key Takeaways

- LoRA freezes the base model and trains small low-rank matrices `A` and `B` instead of full weight matrices
- Trainable parameters drop from billions to millions — often under 1% of the full model
- `target_modules` controls where adapters attach; attention projections are a strong default
- Rank `r` is the main capacity/size trade-off knob
- Because the base model never changes, multiple LoRA adapters can be trained and swapped on the same frozen base

---

## Related Resources

- Day 1 — Quantization Basics
- Day 3 — DoRA (builds directly on LoRA's structure)
- Day 4 — QLoRA (combines Day 1's quantization with this day's LoRA)
- Week 3 Overview — `weeks/week-03-efficient-fine-tuning`

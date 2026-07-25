# Day 6 — Model Merging

> Week 3 — Efficient Fine-Tuning & Quantization

Learn to merge trained LoRA adapters back into the base model's weights — collapsing the frozen base + adapter combo into a single standalone model that's simpler to deploy.

---

## Learning Objectives

By the end of this exercise, you'll understand:

- Why merging matters for deployment (simpler serving, no PEFT dependency at inference)
- How `merge_and_unload()` collapses `W_base + (alpha/r) * B@A` into one weight matrix
- How to prove a merge is lossless — identical outputs before and after
- How adapter folder size compares to a merged model's folder size
- How to reload a merged model with plain `AutoModelForCausalLM`, no PEFT import needed
- The special case of merging QLoRA adapters (requires a full-precision base reload)

---

## Project Structure

```
day-06-model-merging/
├── README.md
├── model_merging.py
└── model_merging.ipynb
```

| File | Description |
|------|-------------|
| `model_merging.py` | Trains a LoRA adapter, merges it into the base model, and verifies identical outputs before/after. |
| `model_merging.ipynb` | Interactive notebook version of the lesson. |
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
python model_merging.py
```

Or open `model_merging.ipynb` in Jupyter Notebook, VS Code, or Google Colab.

> Runs fine on CPU — the toy training loop is only 20 steps on a 0.5B model. This creates two folders on disk: `trained_adapter/` (the small adapter files) and `merged_model/` (the full standalone merged model).

---

## Concepts Covered

### 1. Why Merge at All

Throughout this week, the base model stayed frozen while small adapters trained on top. That's efficient for *training*. At *inference* time, though, you're still running two things together on every forward pass: the frozen base plus the adapter's extra matrix multiplication. Merging collapses them into one normal weight matrix — simpler to serve, and sometimes faster.

### 2. The Merge Operation

```python
peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
merged_model = peft_model.merge_and_unload()
```

Under the hood, this computes `W_merged = W_base + (alpha / r) * (B @ A)` for every adapted layer, then discards the separate adapter matrices — the model that comes out the other side has ordinary weights, indistinguishable in structure from a model that was fully fine-tuned.

### 3. Merging Is Lossless

Because the merge is exact arithmetic (not an approximation), the merged model produces *identical* output to the adapter-attached model on the same input. This script proves it directly by comparing generated text before and after merging.

### 4. Deploying Without PEFT

Once merged, `save_pretrained()` and `from_pretrained()` work exactly like any standard Hugging Face model — no `PeftModel` wrapper, no adapter-loading step, no `peft` import required in your serving code.

### 5. The QLoRA Special Case

`merge_and_unload()` cannot merge cleanly into 4-bit quantized weights. If you trained with QLoRA (Day 4), merge by reloading the *base model* in full precision (no `quantization_config`), then loading your saved adapter on top of that full-precision copy, and merging as usual. The adapter's learned weights are precision-independent — they merge correctly regardless of what precision they were trained against.

---

## Comparison: Adapter-Attached vs Merged

| Aspect | Adapter-Attached (PEFT) | Merged |
|--------|--------------------------|--------|
| Files needed to serve | Base model + adapter + `peft` library | Just the merged model |
| Inference math per adapted layer | Base matmul + adapter matmul | Single matmul |
| Disk footprint | Small adapter file + separate base | Full model size |
| Swapping tasks | Load a different adapter on the same base | Need a separate merged model per task |
| Best for | Experimentation, multi-task serving | Production deployment of one fine-tuned model |

---

## Sample Output

The program trains a LoRA adapter, generates a response with the adapter attached, merges the adapter into a fresh copy of the base model, generates the same prompt again, and confirms the two outputs match exactly — then compares adapter vs merged folder sizes and reloads the merged model with plain `AutoModelForCausalLM`, saved to `experiment_log.json`.

---

## Key Takeaways

- `merge_and_unload()` collapses base + adapter into a single set of weights
- The merge is mathematically lossless — identical outputs before and after
- Merged models need no PEFT dependency at inference time
- QLoRA adapters must be merged against a full-precision reload of the base, not the 4-bit version
- Merging is typically the last step before deploying a fine-tuned model to production

---

## Related Resources

- Day 2 — LoRA (the adapter structure being merged here)
- Day 4 — QLoRA (see the special case note on merging quantized adapters)
- Week 3 Overview — `weeks/week-03-efficient-fine-tuning`
- Up next: Week 4 — Reinforcement Learning from Human Feedback (RLHF)

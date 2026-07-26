# Bonus — LoRA Training on a Real, Larger Dataset

> Week 3 Enhancement (builds on Day 2)

Fixes a problem you'll notice in Day 2: training LoRA on 4 hardcoded toy examples for 20 repeated steps causes **catastrophic forgetting** — the model can corrupt basic knowledge it already had. This folder trains the identical LoRA setup on thousands of real, diverse examples instead, pulled live from the Hugging Face `datasets` library — no hardcoded JSON.

---

## The Bug This Fixes

In Day 2's toy demo, after training on 4 repeated examples for 20 steps, asking a new trivia question sometimes produced output like this:

```json
"training_demo": {
    "before_output": "The capital of Japan is Tokyo.",
    "after_output": "Beijing. — Trained by Neha's LoRA lesson.",
    "train_time_sec": 10.5
}
```

The model went from correctly answering "Tokyo" to incorrectly answering "Beijing" — a factual regression caused by training data, not by anything wrong with the LoRA configuration itself.

---

## Why This Happens: Catastrophic Forgetting

- The training set was tiny (4 examples) and highly repetitive — the same 4 examples looped 5 times over 20 steps
- With so little diversity, the optimizer doesn't learn a general instruction-following pattern — it memorizes those 4 exact input/output pairs
- In the process of memorizing, it overwrites unrelated knowledge the base model already had, including basic facts

**The fix is not a different `r`, `alpha`, or `target_modules` setting.** Day 2's LoRA configuration is fine. The problem is entirely the training data.

---

## Learning Objectives

By the end of this exercise, you'll understand:

- Why training on too little, too-repetitive data causes catastrophic forgetting
- How to load a real, diverse dataset from the Hugging Face `datasets` library instead of hardcoding examples
- How to hold out validation data and measure perplexity as quantitative proof of generalization (not just anecdotal "it seems better")
- How to run a regression check — testing the exact prompts that broke before, to confirm they're fixed
- Why a properly-trained LoRA adapter shouldn't need to sacrifice a model's existing knowledge to learn something new

---

## Project Structure

```
bonus-lora-large-dataset/
├── README.md
├── lora_large_dataset.py
└── lora_large_dataset.ipynb
```

| File | Description |
|------|-------------|
| `lora_large_dataset.py` | Trains Day 2's exact LoRA config on 2,000 real examples, with regression checks and perplexity measurement. |
| `lora_large_dataset.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation for this exercise. |

Running either file generates `experiment_log.json` (full results) and `training_loss_curve.png` (a loss chart over all training steps).

---

## Requirements

Install once from the week-level requirements file:

```bash
cd weeks/week-03-efficient-fine-tuning
pip install -r requirements.txt
```

The `datasets` library (used to download real training data) is already included there.

---

## Run

```bash
python lora_large_dataset.py
```

Or open `lora_large_dataset.ipynb` in Jupyter Notebook, VS Code, or Google Colab.

> **System notes:** Works on CPU or GPU, but 2,000 training steps will be noticeably slow on CPU — a GPU (including a 6GB laptop GPU) is recommended. The first run downloads the Alpaca dataset (~44,000 filtered rows after removing multi-input examples) from the Hugging Face Hub, plus the model itself if not already cached.

---

## The Dataset: Alpaca

This lesson uses [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) — roughly 52,000 diverse instruction/response pairs, publicly available with no login required. The script filters out multi-turn examples (rows with extra `input` context) to keep every training example the same simple shape: one instruction, one response.

Only 2,000 of those (`NUM_TRAIN_EXAMPLES`) are used for training by default, plus 200 more (`NUM_VAL_EXAMPLES`) held out for validation — still 500x more training diversity than Day 2's 4 examples, while remaining fast enough to run on a laptop GPU.

**Want to try a different dataset?** [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) is a solid alternative — swap `DATASET_NAME`, but note its field names differ (`instruction`, `context`, `response` instead of `instruction`, `input`, `output`), so `build_training_batch()` and the filter step would need small adjustments to match.

---

## Concepts Covered

### 1. Loading Real Data Instead of Hardcoding It

```python
from datasets import load_dataset

raw = load_dataset("tatsu-lab/alpaca", split="train")
raw = raw.filter(lambda x: x["input"].strip() == "")
raw = raw.shuffle(seed=42)
```

`load_dataset()` downloads and caches the dataset automatically — no manual JSON files, no copy-pasted examples. `.filter()` and `.shuffle()` are dataset-level operations that work on the whole 52k-row dataset without loading it all into memory at once.

### 2. Train/Validation Split — Why It Matters Here

```python
train_data = raw.select(range(NUM_TRAIN_EXAMPLES))
val_data = raw.select(range(NUM_TRAIN_EXAMPLES, NUM_TRAIN_EXAMPLES + NUM_VAL_EXAMPLES))
```

The validation set is never touched during training. This is what makes perplexity measurement meaningful — if the model improves at predicting text it's never seen, that's evidence of real learning, not memorization.

### 3. Perplexity as a Quantitative Signal

```python
avg_loss = sum(losses) / len(losses)
perplexity = torch.exp(torch.tensor(avg_loss)).item()
```

Perplexity is, loosely, "how surprised is the model by this text?" — lower is better. Measuring it before and after training on held-out data gives a number to point to, instead of just eyeballing a couple of example outputs.

### 4. Regression Testing

```python
REGRESSION_PROMPTS = [
    "What is the capital of Japan?",
    "What is the capital of France?",
    "What is 12 multiplied by 8?",
    "Name the largest planet in our solar system.",
]
```

These are deliberately simple, well-known facts — the same *kind* of question that broke in Day 2. Asking them before and after training is a direct test of whether basic knowledge survived.

---

## Comparison: Day 2's Toy Demo vs. This Enhancement

| Aspect | Day 2 (toy demo) | This Enhancement |
|---|---|---|
| Training examples | 4, hardcoded | 2,000, loaded from Hugging Face |
| Repetition | Same 4 examples looped 5x | Each example seen once |
| Validation set | None | 200 held-out examples |
| Proof of learning | Anecdotal (one before/after prompt) | Perplexity (quantitative) + regression check |
| Risk of catastrophic forgetting | High | Low |
| LoRA config | `r=8, alpha=16` | Identical — `r=8, alpha=16` |

---

## Sample Output

The script prints the model's answers to four regression-check questions before and after training, held-out perplexity before and after, a live loss readout every 200 steps during training, and saves a loss curve chart plus a full JSON log with every measurement.

---

## Key Takeaways

- Catastrophic forgetting comes from too little, too-repetitive training data — not from LoRA itself
- Loading a real dataset via `datasets.load_dataset()` avoids ever hardcoding training examples
- A held-out validation set turns "seems better" into a measured perplexity number
- Regression prompts are a direct, repeatable way to check whether existing knowledge survived training
- The exact same LoRA configuration that broke in Day 2 works correctly once given enough real, diverse data

---

## Related Resources

- Day 2 — LoRA (the configuration and bug being fixed here)
- Day 4 — QLoRA (combine this larger-dataset approach with 4-bit quantization for even bigger datasets/models)
- Week 3 Overview — `weeks/week-03-efficient-fine-tuning`

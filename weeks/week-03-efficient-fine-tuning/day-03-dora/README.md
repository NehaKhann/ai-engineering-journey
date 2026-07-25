# Day 3 — DoRA (Weight-Decomposed Low-Rank Adaptation)

> Week 3 — Efficient Fine-Tuning & Quantization

Learn **DoRA** — an improvement on LoRA that decomposes each weight matrix into magnitude and direction, training them separately for adaptation that's closer to full fine-tuning quality, at almost no extra cost.

---

## Learning Objectives

By the end of this exercise, you'll understand:

- How any weight matrix can be decomposed into magnitude and direction: `W = m * (V / ||V||)`
- Why LoRA entangles magnitude and direction changes, and what that costs in quality
- How DoRA separates them — direction gets the LoRA-style update, magnitude trains independently
- How to enable DoRA in PEFT with a single config flag (`use_dora=True`)
- How LoRA and DoRA compare on trainable parameters, training loss, and final behavior

---

## Project Structure

```
day-03-dora/
├── README.md
├── dora.py
└── dora.ipynb
```

| File | Description |
|------|-------------|
| `dora.py` | Trains both LoRA and DoRA on the same toy dataset and compares them side by side. |
| `dora.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation for this exercise. |

---

## Requirements

Install once from the week-level requirements file:

```bash
cd weeks/week-03-efficient-fine-tuning
pip install -r requirements.txt
```

> **Note:** `use_dora=True` requires a recent PEFT version (0.9.0+). The week-level `requirements.txt` already pins a compatible version.

---

## Run

```bash
python dora.py
```

Or open `dora.ipynb` in Jupyter Notebook, VS Code, or Google Colab.

> This runs fine on CPU — both LoRA and DoRA training loops are only 20 steps on a 0.5B model. A GPU (e.g. free Colab T4) will still be faster, but isn't required for this lesson.

---

## Concepts Covered

### 1. Decomposing a Weight Matrix

Any weight matrix `W` can be rewritten as:

```
W = m * (V / ||V||)
```

- `V / ||V||` — the **direction**: a unit vector per output dimension, describing *which way* the weight points
- `m` — the **magnitude**: a scalar per output dimension, describing *how large* the weight is

### 2. What LoRA Does (and Doesn't Separate)

Plain LoRA applies its low-rank update directly to `W`:

```
W_new = W + (alpha / r) * (B @ A)
```

This single update entangles magnitude and direction changes together. Full fine-tuning, in contrast, naturally adjusts them somewhat independently — which is part of why full fine-tuning can outperform LoRA on harder tasks.

### 3. What DoRA Does Differently

DoRA splits the update into two independently trained pieces:

- The **direction** component (`V / ||V||`) receives the familiar LoRA-style low-rank update
- The **magnitude** component (`m`) is trained as its own small vector of parameters

This mirrors how full fine-tuning behaves, which is why DoRA tends to close more of the quality gap to full fine-tuning than plain LoRA — at the cost of just one extra magnitude vector per adapted layer.

### 4. Enabling DoRA in PEFT

In `LoraConfig`, DoRA is a single flag:

```python
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    use_dora=True,   # <-- this turns LoRA into DoRA
)
```

Everything else about the training loop stays identical to Day 2's LoRA setup.

### 5. Comparing LoRA and DoRA Directly

Training both methods on the exact same toy dataset, for the same number of steps, isolates the effect of the decomposition itself — rather than differences in data, rank, or training length.

---

## Comparison: LoRA vs DoRA

| Aspect | LoRA | DoRA |
|--------|------|------|
| Update applied to | Weight matrix directly | Direction only (magnitude trained separately) |
| Extra parameters vs LoRA | — | One magnitude vector per adapted layer (tiny) |
| Closer to full fine-tuning quality | Good | Often better, especially on harder tasks |
| Training/inference cost | Lower | Slightly higher, still much lower than full FT |
| Config change needed | — | `use_dora=True` |

---

## Sample Output

The program trains LoRA and DoRA back-to-back on the same 4-example toy dataset, printing trainable parameter counts, loss curves, and before/after responses for each — then a side-by-side summary table saved to `experiment_log.json`.

---

## Key Takeaways

- DoRA decomposes weights into magnitude and direction, training them separately
- LoRA's low-rank update applies only to the direction component in DoRA
- This mirrors full fine-tuning's behavior more closely than plain LoRA
- The extra parameter cost is minimal — one magnitude vector per layer
- Switching from LoRA to DoRA in PEFT is a single config flag

---

## Related Resources

- Day 2 — LoRA (the foundation DoRA builds on)
- Day 4 — QLoRA (combines quantization with LoRA/DoRA-style adapters)
- Week 3 Overview — `weeks/week-03-efficient-fine-tuning`

# Bonus — LoRA vs. DoRA at Scale: Does DoRA's Advantage Actually Show Up?

> Week 3 Bonus (follow-up to Day 3)

Day 3 tested LoRA vs. DoRA on a 4-example toy dataset and found no clear winner — DoRA was just slower. This is the honest follow-up: same comparison, but on 1,200+ real examples, to check whether DoRA's theoretical advantage actually shows up once there's enough data for it to matter.

---

## Where This Came From

DoRA takes LoRA's one combined correction and splits it into two separately-trained parts — **direction** and **magnitude**. On bigger, real training tasks, this is supposed to help DoRA get closer to full fine-tuning quality than plain LoRA.

Day 3's demo trained both methods on the same 4 toy examples for 20 steps. At that scale, LoRA and DoRA looked basically identical — DoRA was just slower, with no visible quality edge.

**That's not a contradiction of the theory — it's a scale problem.** A 4-example demo is great for seeing *how* something works, not for proving *which method wins*. Small demos don't carry enough signal to reveal a difference that's supposed to emerge on harder, more diverse, larger-scale tasks.

---

## What This Bonus Actually Tests

Train LoRA and DoRA back-to-back on the **same 1,200+ real examples**, with **identical hyperparameters** — the only difference between the two runs is one flag: `use_dora=True` vs. `False`. Then measure:

1. **Held-out perplexity** — quantitative, lower is better
2. **Training loss curves**, overlaid — shape comparison over 1,200 steps
3. **Wall-clock training time** — is DoRA's speed penalty still small at this scale, or does it grow?
4. **Regression check** — did either method corrupt basic factual knowledge?

**Honest caveat, upfront:** this does not prove DoRA always wins. 1,200 examples on a 0.5B model, run once, is still a modest-scale test — it's a bigger, more honest experiment than Day 3's toy demo, not a universal verdict on LoRA vs. DoRA.

---

## Learning Objectives

By the end of this exercise, you'll understand:

- Why small toy demos can't reveal differences between fine-tuning methods that only emerge at scale
- How to run a controlled comparison — identical data, identical hyperparameters, one variable changed
- How to use held-out perplexity as a quantitative tiebreaker instead of eyeballing outputs
- Why "DoRA was slower with no visible benefit" in Day 3 doesn't mean DoRA doesn't work — it means the test was too small to show it either way

---

## Project Structure

```
bonus-lora-vs-dora-at-scale/
├── README.md
├── lora_vs_dora_at_scale.py
└── lora_vs_dora_at_scale.ipynb
```

| File | Description |
|------|-------------|
| `lora_vs_dora_at_scale.py` | Trains LoRA and DoRA on identical 1,200-example real data, comparing perplexity, loss curves, training time, and regression checks. |
| `lora_vs_dora_at_scale.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation for this exercise. |

Running either file generates `lora_vs_dora_results.json` (full results for both methods) and `lora_vs_dora_comparison.png` (a 2-panel chart: overlaid loss curves + perplexity comparison).

---

## Requirements

Install once from the week-level requirements file:

```bash
cd weeks/week-03-efficient-fine-tuning
pip install -r requirements.txt
```

The `datasets` library is already included there.

---

## Run

```bash
python lora_vs_dora_at_scale.py
```

Or open `lora_vs_dora_at_scale.ipynb` in Jupyter Notebook, VS Code, or Google Colab.

> **System notes:** This trains TWO full models (LoRA, then DoRA) on 1,200 examples each — roughly double the runtime of the single-method `bonus-lora-large-dataset` folder. Works on CPU or GPU, but a GPU (including a 6GB laptop GPU) is strongly recommended here given the doubled workload. The first run downloads the Alpaca dataset and the base model if not already cached.

---

## Concepts Covered

### 1. A Controlled Comparison

```python
config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    use_dora=use_dora,  # True for DoRA, False for LoRA — the only variable
)
```

Everything else — the dataset, the number of examples, the learning rate, the number of steps — is identical between the two runs. This isolates the adapter method itself as the only variable, so any difference in the results is attributable to LoRA vs. DoRA specifically.

### 2. Perplexity as the Tiebreaker

Day 3 relied on comparing a couple of generated sentences by eye. This bonus adds a held-out validation set and computes perplexity — a number that reflects how well the model predicts text it never trained on. A meaningfully lower perplexity for one method over the other is real, comparable evidence; near-identical numbers are also a valid (and informative) result.

### 3. Reading a "No Clear Winner" Result Honestly

If the two methods land close together even at 1,200 examples, that's not a failed experiment — it's useful information. It would suggest DoRA's advantage may need an even larger dataset, a harder task, or a bigger model to become visible, which matches what the DoRA paper's own experiments (run at much larger scale) suggest.

---

## Comparison: What Changed From Day 3

| Aspect | Day 3 (toy demo) | This Bonus |
|---|---|---|
| Training examples | 4, hardcoded | 1,200, loaded from Hugging Face |
| Held-out validation | None | 150 examples, used for perplexity |
| Quality signal | One before/after generated sentence | Held-out perplexity + loss curves + regression check |
| Adapter config | Identical `r=8, alpha=16` | Identical `r=8, alpha=16` |
| Conclusion possible | "No visible difference at this scale" | Directional evidence at a meaningfully bigger scale |

---

## Sample Output

The script prints trainable parameters, a live loss readout every 200 steps, and a full side-by-side summary table (trainable %, train time, final loss, held-out perplexity) for both methods, followed by the same 4 regression-check questions answered by both LoRA and DoRA — plus a 2-panel chart overlaying both loss curves and comparing perplexity across base/LoRA/DoRA.

---

## Key Takeaways

- Day 3's "no clear winner" result was a scale limitation, not evidence against DoRA
- A controlled comparison (identical data, identical hyperparameters, one flag changed) is what makes a LoRA vs. DoRA result meaningful
- Held-out perplexity turns "which one seems better" into a specific, comparable number
- Even this bigger test is one run at modest scale — treat the result as directional, and say so honestly when writing it up

---

## Related Resources

- Day 3 — DoRA (the toy-scale version of this exact comparison)
- `bonus-lora-large-dataset/` — the sibling bonus that fixes Day 2's catastrophic forgetting using the same real dataset
- Week 3 Overview — `weeks/week-03-efficient-fine-tuning`

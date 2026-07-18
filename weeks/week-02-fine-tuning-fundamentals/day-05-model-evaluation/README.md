# Day 5 — Model Evaluation: Base vs Fine-Tuned

> Week 2 — Fine-Tuning Fundamentals

Loads both the base GPT-2 and the Day 4 fine-tuned model, compares their
responses on the same set of cybersecurity prompts (using the SAME
Instruction/Response format the fine-tuned model was actually trained on),
and computes real perplexity on the Day 3 validation set instead of a
hardcoded "quality score".

---

## Learning Objectives

By the end of this exercise, you'll understand:

- How to compare model outputs side by side with matched prompt formatting
- How to compute real perplexity on a validation set
- Why testing with the same format the model was trained on matters
- How to generate reproducible evaluation images (response table + perplexity chart)

---

## Project Structure

```
day-05-model-evaluation/
├── README.md
├── model_evaluation.py
└── model_evaluation.ipynb
```

| File | Description |
|------|-------------|
| `model_evaluation.py` | Loads both models, side-by-side response comparison, perplexity on Day 3 validation data, saves response table and perplexity chart as PNGs. |
| `model_evaluation.ipynb` | Interactive notebook version of the lesson. |
| `README.md` | Documentation for this exercise. |

---

## Requirements

```bash
pip install transformers torch matplotlib
```

---

## Run

```bash
python model_evaluation.py
```

Or open `model_evaluation.ipynb` in Jupyter Notebook or VS Code.

**Prerequisite:** Day 4 (sft_training.py) must be run first to create the fine-tuned model.

---

## Concepts Covered

### 1. Matched Prompt Format

The fine-tuned model was trained on `Instruction:\n...\n\nResponse:\n` format.
Testing it with a bare question would understate its performance. Both models
receive the same formatted prompt for a fair comparison.

### 2. Shared Generation Settings

Both models use identical `gen_kwargs` (max_new_tokens, temperature, repetition_penalty,
no_repeat_ngram_size) so any difference in output comes from the models themselves.

### 3. Perplexity

Perplexity is the standard metric for evaluating language models:

```
perplexity = exp(average cross-entropy loss)
```

- **Lower perplexity** = the model is more confident in its predictions
- Computed on the same 6-example validation split used for eval_loss in Day 4

### 4. Reproducible Visualization

- **Response table** (`day5_response_table.png`): real model outputs in a formatted table
- **Perplexity bar chart** (`day5_perplexity_comparison.png`): quantitative comparison

---

## Evaluation Questions

This exercise compares models on these cybersecurity questions:

1. What is a firewall?
2. Explain how encryption works.
3. What is a DDoS attack?
4. Define social engineering.
5. What should I do if I receive a phishing email?

---

## What to Look For

When comparing responses, ask:

- Does the fine-tuned model give more accurate definitions?
- Are the responses more concise and focused?
- Does the base model still generate generic text completions?
- Is the improvement consistent across all prompts?
- How much did perplexity drop after fine-tuning?

---

## Key Takeaways

- Prompt format must match training format for a fair evaluation
- Shared generation settings isolate model quality differences
- Perplexity quantifies prediction confidence (lower = better)
- Perplexity measures fluency, not factual correctness — always read the responses
- Both quantitative metrics and human judgment are needed

---

## Related Resources

- Day 4 — Supervised Fine-Tuning (model being evaluated)
- Day 3 — Dataset Preparation (validation set source)
- Week 2 Overview — `weeks/week-02-fine-tuning-fundamentals`

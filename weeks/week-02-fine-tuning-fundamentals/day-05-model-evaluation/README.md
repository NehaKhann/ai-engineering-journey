# Day 5 — Model Evaluation: Base vs Fine-Tuned

> Week 2 — Fine-Tuning Fundamentals

Compare the base GPT-2 against the fine-tuned model from Day 4 using both qualitative and quantitative evaluation methods.

Evaluation answers the question: *Did fine-tuning actually improve the model?* This lesson provides the tools to answer that question rigorously.

---

## Learning Objectives

By the end of this exercise, you'll understand:

- How to compare model outputs side by side
- What perplexity measures and how to compute it
- How loss masking affects evaluation metrics
- Why both quantitative and qualitative evaluation are necessary
- How to structure model comparison for reporting

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
| `model_evaluation.py` | Loads both models, compares responses, computes perplexity, and visualizes results. |
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

### 1. Qualitative Evaluation

Reading actual model outputs reveals things that numbers cannot capture:

- Is the response relevant to the question?
- Does it use domain-appropriate terminology?
- Is the tone and structure appropriate?
- Does the model actually answer the question or just complete text?

### 2. Perplexity

Perplexity is the standard metric for evaluating language models:

```
perplexity = exp(average cross-entropy loss)
```

- **Lower perplexity** = the model is more confident in its predictions
- **Higher perplexity** = the model is more surprised by the text

A fine-tuned model should have lower perplexity on domain-specific validation data.

### 3. Loss Masking in Evaluation

When computing perplexity, loss must only be computed on the assistant's response tokens — exactly like during training. Masking the instruction tokens ensures the metric reflects the model's ability to generate good responses, not its ability to predict the input.

### 4. Evaluation on Unseen Data

Validation data must not be part of the training set. Evaluating on unseen data tests whether the model has truly learned generalizable knowledge rather than simply memorizing training examples.

### 5. Balanced Assessment

| Method | What It Measures | Limitation |
|--------|-----------------|------------|
| Perplexity | Prediction confidence | Doesn't measure factual correctness |
| Response comparison | Relevance and quality | Subjective |
| Response length | Verbosity | Longer is not always better |

Both quantitative and qualitative methods are needed for a complete picture.

---

## Evaluation Questions

This exercise compares models on these cybersecurity questions:

1. What is a firewall?
2. Explain how encryption works.
3. What is a DDoS attack?
4. Define social engineering.
5. What is ransomware?

---

## What to Look For

When comparing responses, ask:

- Does the fine-tuned model give more accurate definitions?
- Are the responses more concise and focused?
- Does the base model still generate generic text completions?
- Is the improvement consistent across all prompts?

---

## Key Takeaways

- Perplexity quantifies prediction quality (lower = better)
- Loss masking is required for accurate evaluation
- Qualitative response comparison reveals relevance and accuracy
- Always evaluate on unseen data
- Fine-tuning should improve both metrics and output quality
- Balanced evaluation uses multiple methods

---

## Related Resources

- Day 4 — Supervised Fine-Tuning (model being evaluated)
- Day 3 — Dataset Preparation (validation set source)
- Week 2 Overview — `weeks/week02-fine-tuning-fundamentals`

# Day 3 — Dataset Preparation for Fine-Tuning (v2)

> Week 2 — Fine-Tuning Fundamentals

Learn how to create, clean, format, tokenize, and split an instruction dataset for Supervised Fine-Tuning (SFT).

---

## Learning Objectives

- Create a synthetic instruction dataset in JSONL format
- Load and clean data — detect duplicates, missing fields
- Format raw data using a model's chat template
- Tokenize with loss masking (ignore system/user tokens during training)
- Split data into training (80%) and validation (20%) sets
- Understand why dataset quality matters more than dataset size

---

## Project Structure

```
day-03-dataset-preparation/
├── README.md
├── dataset_preparation.py
├── dataset_preparation.ipynb
├── cybersecurity_dataset.jsonl      (generated — raw, includes broken rows)
└── day3_dataset_prep.png            (generated — token length + split chart)
```

| File | Description |
|------|-------------|
| `dataset_preparation.py` | Creates, cleans, formats, tokenizes, and splits a 30-example cybersecurity dataset. |
| `dataset_preparation.ipynb` | Interactive notebook version of the lesson. |
| `*.jsonl` | Generated dataset files (saved during execution). |
| `day3_dataset_prep.png` | Visualization of token length distribution and train/val split. |

---

## Requirements

```bash
pip install transformers torch matplotlib
```

---

## Run

```bash
python dataset_preparation.py
```

Or open `dataset_preparation.ipynb` in Jupyter Notebook or VS Code.

---

## Pipeline Overview

### Part 1: Create Dataset
30 clean cybersecurity Q&A pairs + 3 deliberately broken rows (duplicate, missing instruction, missing response). Saved to JSONL.

### Part 2: Load and Clean
- **Duplicate check** — matches on lowercase instruction + response
- **Missing field check** — flags empty instruction or response
- Drops bad rows automatically; continues with clean set only

### Part 3: Format with Chat Template
Applies Qwen2.5's `apply_chat_template()` with system/user/assistant roles.

### Part 4: Tokenize with Loss Masking
- Finds the `<|im_start|>assistant` marker position
- Sets labels to `-100` for all tokens before it (system + user)
- Keeps real token IDs for assistant response tokens
- This is how the model learns only from answers, not questions

### Part 5: Train / Validation Split
80/20 random split after shuffling. Saves separate JSONL files.

### Visualization
Two side-by-side plots:
1. Token length histogram across all clean examples
2. Count of training vs validation examples

---

## Files Created

| File | Contents |
|------|----------|
| `cybersecurity_dataset.jsonl` | Full raw dataset (33 examples, includes broken rows) |
| `day3_dataset_prep.png` | Token length + split visualization |

---

## Key Takeaways

- JSONL is the standard format for instruction datasets
- Always check for duplicates and missing fields before training
- Chat templates convert raw JSON into model-ready text
- Loss masking (label = -100) ensures the model learns only from assistant responses
- Train/val split (80/20) helps detect overfitting
- Dataset quality matters more than dataset size — 30 clean examples are worth more than 100 dirty ones

---

## Related Resources

- Day 2 — Instruction Tuning & Chat Templates
- Day 4 — Supervised Fine-Tuning (SFT)
- Week 2 Overview — `weeks/week-02-fine-tuning-fundamentals`

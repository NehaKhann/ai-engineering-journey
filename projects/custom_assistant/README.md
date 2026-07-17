# Custom Domain Assistant

> Week 2 Capstone Project

Fine-tune GPT-2 to become an expert in any domain. This project demonstrates every concept from Week 2 in a single pipeline.

---

## What It Covers

| Day | Concept | Implementation |
|-----|---------|----------------|
| 1 | Prompt Engineering | Baseline test before SFT — shows base model's generic responses |
| 2 | Chat Templates | `apply_chat_template()` with custom template for GPT-2 |
| 3 | Dataset Preparation | Stat generation, train/val split, token length distribution plot |
| 4 | Supervised Fine-Tuning | SFTTrainer with loss masking via `response_template` |
| 5 | Model Evaluation | Perplexity comparison + side-by-side before/after responses |

---

## How It Works

```
train.py runs all 5 days sequentially:

DAY 3 → Generate dataset → Stats → Split → Plot
DAY 2 → Set chat template → Show formatted example
DAY 1 → Test base model → Print baseline responses
DAY 4 → SFTTrainer → Fine-tune → Save model
DAY 5 → Compare before/after → Perplexity → Plot
```

---

## Quick Start

```bash
# Train the model (runs all 5 days)
python train.py

# Chat with your trained model
python inference.py
```

---

## Changing the Domain

Edit the config section at the top of `train.py`:

```python
DOMAIN = "cybersecurity"  # ← Change this
```

Then add your domain's data to the `DOMAIN_DATA` dict:

```python
DOMAIN_DATA = {
    "cybersecurity": [...],
    "legal": [
        {"instruction": "What is a contract?",
         "response": "A contract is a legally binding agreement between parties..."},
    ],
}
```

---

## Project Structure

```
custom_assistant/
├── train.py                  # Complete training pipeline (all 5 days)
├── inference.py              # Interactive chat with fine-tuned model
├── README.md
├── datasets/                 # Generated dataset files
│   ├── cybersecurity_dataset.jsonl
│   ├── cybersecurity_train.jsonl
│   └── cybersecurity_val.jsonl
├── plots/                    # Generated visualizations
│   ├── dataset_distribution.png
│   └── perplexity_comparison.png
└── models/                   # Saved fine-tuned models
    └── cybersecurity-assistant/
        ├── config.json
        ├── model.safetensors
        └── tokenizer files...
```

---

## Requirements

```bash
pip install transformers torch datasets trl peft accelerate matplotlib
```

---

## Output

After training, you'll see:

- **Token distribution plot** — saved to `plots/dataset_distribution.png`
- **Baseline responses** — base GPT-2 before training
- **Training loss** — logged to console every 5 steps
- **Before/after comparison** — side-by-side on 3 test prompts
- **Perplexity comparison** — `plots/perplexity_comparison.png`
- **Summary table** — all results printed at the end

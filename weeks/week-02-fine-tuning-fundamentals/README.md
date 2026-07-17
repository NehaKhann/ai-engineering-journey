# Week 2 — Fine-Tuning Fundamentals

After learning how Large Language Models work internally in Week 1, this week focuses on teaching pretrained models new behaviors through fine-tuning.

You'll begin by understanding the difference between prompting and training, then progress through instruction tuning, dataset preparation, supervised fine-tuning (SFT), and model evaluation before building your own domain-specific AI assistant.

---

## Learning Objectives

By the end of this week, you'll be able to:

- Understand when Prompt Engineering is sufficient vs when Fine-Tuning is needed
- Apply chat templates to format instruction data
- Prepare high-quality instruction datasets
- Perform Supervised Fine-Tuning (SFT) with SFTTrainer
- Evaluate and compare fine-tuned models using perplexity
- Build a custom domain-specific AI assistant

---

## Week Overview

| Day | Topic | Key Code |
|-----|-------|----------|
| **Day 1** | Prompt Engineering vs Fine-Tuning | Base vs instruction-tuned model comparison, 4 prompting techniques, temperature interplay |
| **Day 2** | Instruction Tuning & Chat Templates | apply_chat_template(), Jinja2 templates, role-based messages, loss masking concept |
| **Day 3** | Dataset Preparation | JSONL creation, token length stats, train/val split, distribution visualization |
| **Day 4** | Supervised Fine-Tuning (SFT) | SFTTrainer, response_template loss masking, before/after response comparison |
| **Day 5** | Model Evaluation | Perplexity computation, side-by-side response comparison, evaluation plots |

---

## Technologies & Libraries

- **PyTorch** — tensor operations and model training
- **Hugging Face Transformers** — model loading and tokenization
- **TRL (Transformer Reinforcement Learning)** — SFTTrainer for fine-tuning
- **Datasets** — dataset loading and splitting
- **PEFT** — parameter-efficient fine-tuning (used in later weeks)
- **Matplotlib** — data visualization

---

## Folder Structure

```text
week-02-fine-tuning-fundamentals/
│
├── README.md
├── requirements.txt
│
├── day-01-prompt-engineering/
│   ├── prompt_engineering.py
│   ├── prompt_engineering.ipynb
│   └── README.md
│
├── day-02-instruction-tuning/
│   ├── instruction_tuning.py
│   ├── instruction_tuning.ipynb
│   └── README.md
│
├── day-03-dataset-preparation/
│   ├── dataset_preparation.py
│   ├── dataset_preparation.ipynb
│   ├── README.md
│   └── *.jsonl (generated)
│
├── day-04-supervised-fine-tuning/
│   ├── sft_training.py
│   ├── sft_training.ipynb
│   ├── README.md
│   └── gpt2-cybersecurity-sft/ (generated)
│
└── day-05-model-evaluation/
    ├── model_evaluation.py
    ├── model_evaluation.ipynb
    └── README.md
```

The **capstone project** is located at:

```
projects/custom_assistant/
├── train.py           # Complete pipeline covering all 5 days
├── inference.py       # Interactive chat with fine-tuned model
└── README.md
```

---

## Setup

```powershell
# From repository root
venv\Scripts\activate

cd weeks\week02-fine-tuning-fundamentals

pip install -r requirements.txt
```

---

## What You'll Learn

- Prompt Engineering techniques (zero-shot, few-shot, chain-of-thought, role prompting)
- The difference between base models and instruction-tuned models
- Chat templates and how they structure conversations
- Dataset preparation, statistics, and train/validation splitting
- Supervised Fine-Tuning (SFT) with SFTTrainer
- Loss masking with response templates
- Model evaluation using perplexity and qualitative comparison
- Building a complete, reusable domain-specific assistant

---

## Capstone Project

### Custom Domain Assistant

Located at `projects/custom_assistant/`, the capstone project demonstrates all five days' concepts in a single pipeline:

| Day | Concept | In Project |
|-----|---------|------------|
| 1 | Prompt Engineering | Baseline test before SFT |
| 2 | Chat Templates | apply_chat_template() with custom template |
| 3 | Dataset Preparation | Stats, split, distribution plot |
| 4 | Supervised Fine-Tuning | SFTTrainer with response_template |
| 5 | Model Evaluation | Perplexity comparison + before/after |

To run:

```bash
cd projects/custom_assistant
python train.py          # Runs the full pipeline
python inference.py      # Chat with your trained model
```

Supports any domain — change `DOMAIN` in `train.py` and add your data.

---

## Ready to Begin?

Start with Day 1 to understand why prompt engineering is often the best starting point, and when it's time to move beyond prompts and teach the model through fine-tuning.
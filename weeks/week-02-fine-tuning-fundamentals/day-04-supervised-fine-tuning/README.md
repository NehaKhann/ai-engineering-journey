# Day 4 — Supervised Fine-Tuning (SFT)

> **Week 2 — Fine-Tuning Fundamentals**

Train a GPT-2 model using **Supervised Fine-Tuning (SFT)** with the cleaned dataset from Day 3. This project demonstrates the complete SFT workflow—from dataset preparation to model training, evaluation, visualization, and inference.

---

## 🎯 Learning Objectives

By the end of this project, you will be able to:

- Load and clean a JSONL instruction dataset
- Remove duplicate and invalid samples
- Format instruction data into GPT-2 training prompts
- Configure `SFTTrainer` for different TRL versions
- Perform supervised fine-tuning on GPT-2
- Track and visualize training & validation loss
- Compare responses before and after fine-tuning
- Reload a saved fine-tuned model without retraining

---

# 📁 Project Structure

```text
day-04-supervised-fine-tuning/
├── README.md
├── sft_training.py
├── sft_training.ipynb
├── test_model.py
├── test_base_model.py
├── gpt2-cybersecurity-sft/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer files...
│   └── checkpoint-60/
└── day4_training_loss.png
```

## 📄 File Overview

| File | Purpose |
|------|---------|
| `sft_training.py` | Complete training pipeline including dataset loading, cleaning, formatting, SFT training, evaluation, plotting, and response comparison. |
| `sft_training.ipynb` | Notebook version of the training pipeline. |
| `test_model.py` | Loads the saved fine-tuned model for interactive question answering without retraining. |
| `test_base_model.py` | Loads the original GPT-2 model to compare outputs against the fine-tuned model. |
| `gpt2-cybersecurity-sft/` | Saved fine-tuned model generated after training. |
| `checkpoint-60/` | Automatic checkpoint used only for resuming interrupted training. |
| `day4_training_loss.png` | Plot showing training and validation loss throughout training. |

---

# ⚙️ Requirements

```bash
pip install transformers torch datasets trl peft accelerate matplotlib
```

---

# ▶️ Training

Run the training script:

```bash
python sft_training.py
```

Or open:

```text
sft_training.ipynb
```

using Jupyter Notebook or VS Code.

> **CPU Runtime:**  
> Training for **60 steps** (batch size = 1) typically takes **60–80 minutes** on CPU. Evaluation is performed every 10 steps, which increases total runtime.

Once the model has been trained and saved, you **do not need to train again** to ask new questions.

---

# 💬 Inference

Run the fine-tuned model:

```bash
python test_model.py
```

Ask any cybersecurity-related question.

Type:

```text
quit
```

to exit.

The script loads the saved model directly from:

```text
gpt2-cybersecurity-sft/
```

No additional training occurs.

---

## Compare with the Original GPT-2

```bash
python test_base_model.py
```

This loads the untouched GPT-2 model so you can compare responses side-by-side with the fine-tuned version.

---

# ✨ What's New in Version 2

- Uses the cleaned Day 3 dataset (30 valid examples)
- Automatically supports both older and newer TRL versions
- Detects whether `SFTConfig` or `TrainingArguments` should be used
- Plots both training and validation loss
- Uses:
  - `max_steps = 60`
  - `learning_rate = 5e-5`
- Removes the inline fallback dataset
- Requires the Day 3 dataset before training
- Adds standalone inference scripts for both:
  - Fine-tuned GPT-2
  - Original GPT-2

---

# 📊 Training Results

Dataset:

- **30 cleaned examples**

Training:

- **60 steps**
- CPU

Observed metrics:

| Metric | Result |
|---------|--------|
| Training Loss | **3.55 → ~1.65–1.84** |
| Validation Loss | **2.56 → 2.11** |
| Mean Token Accuracy | **~36% → ~60–65%** |

### Observations

- Training loss decreases steadily despite noisy updates caused by the batch size of 1.
- Validation loss also declines, indicating the model is learning patterns instead of simply memorizing the training data.
- After fine-tuning, responses adopt noticeably more cybersecurity-specific terminology and style.
- Since the dataset contains only **30 examples**, improvements are primarily stylistic rather than factually robust.

---

# 🧠 Key Takeaways

- Supervised Fine-Tuning teaches the model by predicting the correct next token from labeled examples.
- The "correct answer" comes directly from the dataset—not from any external evaluator.
- `SFTTrainer` automatically manages:
  - data formatting
  - batching
  - loss computation
  - gradient updates
- Training and validation loss together help verify whether the model is learning meaningful patterns.
- Dataset quality has a direct impact on fine-tuning quality.
- Fine-tuned models can be saved once and reused indefinitely without retraining.
- Supervised Fine-Tuning is the foundation for advanced techniques such as:
  - LoRA
  - QLoRA
  - RLHF
```
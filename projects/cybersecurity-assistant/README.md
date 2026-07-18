# Week 2 Project: Cybersecurity Assistant

A fine-tuned GPT-2 model specialized in **Cybersecurity** using Supervised Fine-Tuning (SFT).

---

## Project Goal

Transform a general-purpose GPT-2 into a helpful **Cybersecurity Assistant** that can explain security concepts, give practical advice, and answer domain-specific questions with better accuracy and relevance than the base model.

---

## Project Structure

```
cybersecurity-assistant/
├── README.md
├── train_sft.py              # Training script
├── inference.py              # Chat interface
├── dataset/
│   ├── train.jsonl           # Training data (15+ examples)
│   └── val.jsonl             # Validation data
└── results/
    └── cybersecurity-assistant/   # Saved fine-tuned model
```

---

## How to Use

### Step 1: Train the Model

```powershell
cd cybersecurity-assistant
python train_sft.py
```

### Step 2: Chat with your Assistant

```powershell
python inference.py
```

---

## Results

After training, you should observe:

- **Base GPT-2**: Generic, sometimes inaccurate or shallow answers
- **Fine-Tuned Model**: More accurate, relevant, and professional cybersecurity responses
- Lower perplexity on validation set
- Better domain knowledge

---

## What I Learned

- How to prepare instruction-response datasets in `.jsonl` format
- Using `SFTTrainer` from TRL for Supervised Fine-Tuning
- The real difference between a base model and a fine-tuned model
- Importance of domain-specific data for good results
- Model evaluation techniques (qualitative + quantitative)

---

## Next Steps (Future Improvements)

- Apply QLoRA / Unsloth for faster and more memory-efficient training
- Add RAG (Retrieval-Augmented Generation) for up-to-date threat intelligence
- Deploy as a web app using Streamlit or Gradio
- Compare performance with larger models (e.g., Qwen2.5-0.5B)

---

## Technologies Used

- Hugging Face Transformers
- TRL (SFTTrainer)
- PyTorch
- GPT-2

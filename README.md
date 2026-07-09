# LLM Engineering — Learning Journey

A structured, hands-on deep-dive into modern LLM engineering. Each chapter
is a real, tested experiment — code included, runnable directly in Colab,
no local setup needed.

## Setup (for running locally instead of Colab)

```powershell
# Clone the repo
git clone https://github.com/NehaKhann/llm-engineering-learning-journey.git
cd llm-engineering-learning-journey

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# Install all dependencies (covers every week so far)
pip install -r requirements.txt --break-system-packages
```

Each week also has its own scoped `requirements.txt` (e.g. `weeks/week-01-llm-foundations/requirements.txt`)
if you only want that week's dependencies instead of everything.

## Chapters

### Week 1 — LLM & Transformer Foundations
| Day | Topic | Notebook |
|---|---|---|
| 1 | Tokenization & Next-Token Prediction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/weeks/week-01-llm-foundations/01-tokenization/Day1_Next_Token_Prediction.ipynb) |
| 2 | Transformer Architecture & Attention | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/weeks/week-01-llm-foundations/02-attention-mechanism/Day2_Attention_Analysis.ipynb) |
| 3 | PyTorch Essentials | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/weeks/week-01-llm-foundations/03-pytorch-essentials/Day03_pytorch_essentials.ipynb) |
| 4 | Loading Open Models (Hugging Face) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/weeks/week-01-llm-foundations/04-loading-open-models/Day04_open_models.ipynb) |
| 5 | Token-by-Token Explainer Project | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/weeks/week-01-llm-foundations/05-explainer-project/Day05_llm_explainer.ipynb)|

*(More chapters added as weeks progress — Fine-Tuning, LoRA/QLoRA, RLHF, DPO/GRPO, RAG, OCR.)*

# 🚀 AI Engineering Learning Journey

A structured, hands-on journey into modern **AI Engineering**, covering the concepts behind Large Language Models (LLMs), model training, efficient fine-tuning, retrieval systems, and production-ready AI applications.

This repository combines **theory**, **hands-on coding**, **interactive notebooks**, and **real-world projects** to build practical AI engineering skills.

---

## 🎯 What You'll Learn

This repository covers topics including:

- Neural Networks Fundamentals
- Large Language Models (LLMs)
- Transformers & Self-Attention
- Hugging Face Transformers
- Running Open Models
- Fine-Tuning & Instruction Tuning
- LoRA & QLoRA
- Reinforcement Learning (RLHF)
- Preference Optimization (DPO & GRPO)
- Retrieval-Augmented Generation (RAG)
- Document AI & OCR
- Generative AI Concepts & Interview Preparation
- Production-ready AI Projects

---

## 📚 Curriculum

The complete learning roadmap is available here:

➡️ **[AI Engineering Curriculum](LEARNING_PATH.md)**

Looking for interview preparation and breadth across Generative AI? Follow the parallel track:

➡️ **[Generative AI Track](generative-ai/README.md)**: beginner to intermediate concepts, hands-on code, mini projects, and interview Q&A.

---

## 📂 Repository Structure

```text
ai-engineering-journey/

├── prerequisites/
│
├── weeks/
│   ├── week-01-llm-foundations/
│   ├── week-02-fine-tuning-fundamentals/
│   ├── week-03-efficient-fine-tuning/
│   ├── week-04-rlhf/
│   ├── week-05-preference-optimization/
│   ├── week-06-rag/
│   ├── week-07-document-ai/
│   └── week-08-capstone/
│
├── generative-ai/
│   ├── 01-beginner/          # modules 01-04
│   ├── 02-intermediate/      # modules 05-11
│   ├── projects/             # hands-on projects
│   ├── interview-prep/
│   └── tools/
│
├── projects/
│   ├── llm-explainer-dashboard/
│   ├── cybersecurity-assistant/
│   └── efficient_assistant/
│
├── README.md
├── LEARNING_PATH.md
└── requirements.txt
```

---

# ⚙️ Getting Started

Clone the repository and install the dependencies.

```powershell
git clone https://github.com/NehaKhann/ai-engineering-journey.git

cd ai-engineering-journey

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

Each week also contains its own `requirements.txt` if you only want to install dependencies for a specific module.

---

# 📖 Learning Modules

## ✅ Prerequisite — Neural Networks Fundamentals

Learn the foundations that power every modern AI model.

| Lesson | Notebook |
|---------|----------|
| Neural Networks Fundamentals | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/prerequisites/prerequisite_neural_networks.ipynb) |

---

## ✅ Week 1 — LLM Foundations

| Lesson | Topic | Notebook |
|---------|-------|----------|
| Day 1 | Tokenization & Next-Token Prediction | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/weeks/week-01-llm-foundations/day-01-tokenization/tokenization.ipynb) |
| Day 2 | Transformer Architecture & Self-Attention | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/weeks/week-01-llm-foundations/day-02-transformers/attention_analysis.ipynb) |
| Day 3 | PyTorch Fundamentals | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/weeks/week-01-llm-foundations/day-03-pytorch/pytorch_fundamentals.ipynb) |
| Day 4 | Running Open Models with Hugging Face | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/weeks/week-01-llm-foundations/day-04-open-models/open_models.ipynb) |
| Day 5 | Context Window | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/weeks/week-01-llm-foundations/day-05-context-window/context_window_demo.ipynb) |
| Day 6 | Generation Parameters | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/weeks/week-01-llm-foundations/day-06-generation-parameters/generation_parameters.ipynb) |

### 🏆 Week 1 Project

| Project | Notebook |
|----------|----------|
| LLM Explainer Dashboard | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/llm-engineering-learning-journey/blob/main/projects/llm-explainer-dashboard/llm_explainer_dashboard.ipynb) |

---

## ✅ Week 2 — Fine-Tuning Fundamentals

| Lesson | Topic | Notebook |
|---------|-------|----------|
| Day 1 | Prompt Engineering vs Fine-Tuning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-02-fine-tuning-fundamentals/day-01-prompt-engineering/prompt_engineering.ipynb) |
| Day 2 | Instruction Tuning & Chat Templates | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-02-fine-tuning-fundamentals/day-02-instruction-tuning/instruction_tuning.ipynb) |
| Day 3 | Dataset Preparation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-02-fine-tuning-fundamentals/day-03-dataset-preparation/dataset_preparation.ipynb) |
| Day 4 | Supervised Fine-Tuning (SFT) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-02-fine-tuning-fundamentals/day-04-supervised-fine-tuning/sft_training.ipynb) |
| Day 5 | Model Evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-02-fine-tuning-fundamentals/day-05-model-evaluation/model_evaluation.ipynb) |

---

## ✅ Week 3 — Efficient Fine-Tuning & Quantization

| Lesson | Topic | Notebook |
|---------|-------|----------|
| Day 1 | Quantization Basics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-03-efficient-fine-tuning/day-01-quantization-basics/quantization_basics.ipynb) |
| Day 2 | LoRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-03-efficient-fine-tuning/day-02-lora/lora.ipynb) |
| Day 3 | DoRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-03-efficient-fine-tuning/day-03-dora/dora.ipynb) |
| Day 4 | QLoRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-03-efficient-fine-tuning/day-04-qlora/qlora.ipynb) |
| Day 5 | Speed & Memory Tricks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-03-efficient-fine-tuning/day-05-speed-memory-tricks/speed_memory_tricks.ipynb) |
| Day 6 | Model Merging | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-03-efficient-fine-tuning/day-06-model-merging/model_merging.ipynb) |

### 🎁 Week 3 Bonus

| Lesson | Topic | Notebook |
|---------|-------|----------|
| Bonus | LoRA Training on a Real, Larger Dataset (fixes Day 2's catastrophic forgetting) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-03-efficient-fine-tuning/bonus-lora-large-dataset/lora_large_dataset.ipynb) |
| Bonus | LoRA vs. DoRA at Scale (does DoRA's advantage show up on 1,200+ real examples?) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/weeks/week-03-efficient-fine-tuning/bonus-lora-vs-dora-at-scale/lora_vs_dora_at_scale.ipynb) |

---

## 🎨 Generative AI Track

A parallel track covering Generative AI from beginner to intermediate, with interview Q&A in every module. See the [track overview](generative-ai/README.md) for the full roadmap.

| Module | Topic | Notebook |
|---------|-------|----------|
| 01 | The Generative AI Landscape | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/generative-ai/01-beginner/01-genai-landscape/genai_landscape.ipynb) |
| 02 | Prompt Engineering | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/generative-ai/01-beginner/02-prompt-engineering/prompt_engineering.ipynb) |
| 03 | LLM APIs: Tokens, Streaming, Cost | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/generative-ai/01-beginner/03-llm-apis/llm_apis.ipynb) |
| 04 | Embeddings & Vector Search | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/generative-ai/01-beginner/04-embeddings-vector-search/embeddings_vector_search.ipynb) |
| Project | [Support Ticket Assistant](generative-ai/projects/01-support-ticket-assistant) (beginner project, runs from the command line) | |
| 05 | Retrieval-Augmented Generation (RAG) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/generative-ai/02-intermediate/05-rag/rag.ipynb) |
| 06 | Evaluation & LLM-as-Judge | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/generative-ai/02-intermediate/06-evaluation/evaluation.ipynb) |
| 07 | Tool Use & Agents | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/generative-ai/02-intermediate/07-tool-use-agents/tool_use_agents.ipynb) |
| 08 | Guardrails, Safety & Hallucination | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/generative-ai/02-intermediate/08-guardrails-safety/guardrails.ipynb) |
| 09 | Prompting vs RAG vs Fine-Tuning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/generative-ai/02-intermediate/09-prompting-vs-rag-vs-finetuning/prompting_rag_finetuning.ipynb) |
| 10 | Multimodal & Diffusion Models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NehaKhann/ai-engineering-journey/blob/main/generative-ai/02-intermediate/10-multimodal-diffusion/multimodal_diffusion.ipynb) |

---

## 🚧 Upcoming Modules

- Week 4 — Reinforcement Learning from Human Feedback (RLHF)
- Week 5 — Preference Optimization (DPO & GRPO)
- Week 6 — Retrieval-Augmented Generation (RAG)
- Week 7 — OCR & Document AI
- Week 8 — AI Engineering Capstone

---

# 🛠️ Technology Stack

### Programming

- Python

### Deep Learning

- PyTorch

### LLM Ecosystem

- Hugging Face Transformers
- TRL
- PEFT
- BitsAndBytes
- Unsloth

### AI Frameworks

- LangChain
- LangGraph

### Vector Databases

- FAISS
- ChromaDB

### Document AI

- EasyOCR
- Tesseract

### Applications

- Streamlit
- Matplotlib

---

# 🎯 Repository Philosophy

This repository follows a simple learning philosophy:

```text
Learn
   ↓
Understand
   ↓
Experiment
   ↓
Build
   ↓
Document
   ↓
Share
```

Every lesson includes:

- 📖 Conceptual explanations
- 💻 Runnable Python code
- 📓 Interactive Jupyter notebooks
- 🧪 Hands-on experiments
- 🚀 Portfolio-ready projects

---

⭐ If you find this repository helpful, consider giving it a star. Feedback, issues, and contributions are always welcome.
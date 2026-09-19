# 🗺️ AI Engineering Learning Path

This repository follows a structured, project-based roadmap for learning modern AI Engineering — from neural network fundamentals to building production-ready AI applications.

Each module combines theory, hands-on coding, practical experiments, and portfolio-ready projects.

---

# 🎯 Goals

This learning path is designed to:

- Build a strong foundation in Large Language Models (LLMs)
- Understand how modern AI systems work under the hood
- Learn efficient model training and optimization techniques
- Master production deployment and local LLM usage
- Build real-world AI applications
- Document the learning process through open source

---

# 📈 Progress

| Module | Topic | Status |
| :------ | :---- | :----: |
| ✅ Prerequisite | Neural Networks Fundamentals | Complete |
| ✅ Week 1 | LLM Foundations | Complete |
| ✅ Week 2 | Fine-Tuning Fundamentals | Complete |
| ⏳ Week 3 | Efficient Fine-Tuning & Quantization | In Progress |
| ⬜ Week 4 | Reinforcement Learning from Human Feedback (RLHF) | Planned |
| ⬜ Week 5 | Modern Preference Optimization | Planned |
| ⬜ Week 6 | Retrieval-Augmented Generation (RAG) | Planned |
| ⬜ Week 7 | Local LLMs & Production Deployment | Planned |
| ⬜ Week 8 | AI Agents & Tool Use | Planned |
| ⬜ Week 9 | Model Evaluation, Safety & Alignment | Planned |
| ⬜ Week 10 | AI Engineering Capstone | Planned |

---

# 📂 Repository Structure

```text
ai-engineering-learning-journey/
├── prerequisites/
│   └── neural-networks/
├── weeks/
│   ├── week-01-llm-foundations/
│   ├── week-02-fine-tuning-fundamentals/
│   ├── week-03-efficient-fine-tuning/
│   ├── week-04-rlhf/
│   ├── week-05-preference-optimization/
│   ├── week-06-rag/
│   ├── week-07-local-llms/
│   └── week-08-capstone/
├── generative-ai/
├── projects/
├── README.md
├── LEARNING_PATH.md
└── requirements.txt
```

---

# 📚 Curriculum

## ✅ Prerequisite — Neural Networks Fundamentals

**Topics**

- Neural Networks
- Forward Pass
- Loss Functions
- Backpropagation
- Gradient Descent
- Optimizers
- Training Loop

**Technologies**

- PyTorch
- Matplotlib

---

## ✅ Week 1 — LLM Foundations

**Topics**

- Tokenization
- Embeddings
- Transformer Architecture
- Self-Attention
- PyTorch Fundamentals
- Open Models
- Context Window
- Generation Parameters

**Technologies**

- Hugging Face Transformers
- GPT-2
- PyTorch

**Project**

- LLM Explainer Dashboard

---

## ✅ Week 2 — Fine-Tuning Fundamentals

**Topics**

- Prompt Engineering vs Fine-Tuning
- Instruction Tuning
- Chat Templates
- Dataset Preparation
- Supervised Fine-Tuning (SFT)
- Model Evaluation

**Technologies**

- Hugging Face Transformers
- TRL
- Datasets
- PEFT

**Project**

- Fine-Tuned Custom Assistant

---

## ⏳ Week 3 — Efficient Fine-Tuning & Quantization

**Topics**

- LoRA
- QLoRA
- DoRA
- PEFT
- 4-bit & 8-bit Quantization
- Unsloth
- Flash Attention 2
- Gradient Checkpointing
- Memory Optimization
- Single GPU Fine-Tuning
- Model Merging

**Technologies**

- PEFT
- BitsAndBytes
- Unsloth

---

## ⬜ Week 4 — Reinforcement Learning from Human Feedback (RLHF)

**Topics**

- Reward Modeling
- PPO
- Full RLHF Pipeline
- Human Preference Data
- Challenges in RLHF

---

## ⬜ Week 5 — Modern Preference Optimization

**Topics**

- DPO
- ORPO
- KTO
- SimPO
- IPO
- RLHF vs DPO Comparison

**Technologies**

- TRL
- Unsloth

---

## ⬜ Week 6 — Retrieval-Augmented Generation (RAG)

**Topics**

- Embeddings
- Vector Databases
- Advanced RAG (HyDE, Self-RAG, CRAG)
- RAG Evaluation (RAGAS)
- Agentic RAG
- Multi-Modal RAG

**Technologies**

- LangChain
- LlamaIndex
- FAISS
- ChromaDB

---

## ⬜ Week 7 — Local LLMs & Production Deployment

**Topics**

- Ollama
- llama.cpp
- GGUF Quantization
- vLLM
- TGI
- LangChain Integration
- FastAPI Serving
- Monitoring & Cost Optimization

---

## ⬜ Week 8 — AI Agents & Tool Use

**Topics**

- ReAct
- Function Calling
- LangGraph
- CrewAI
- Multi-Agent Systems
- Agent Memory & Evaluation

---

## ⬜ Week 9 — Model Evaluation, Safety & Alignment

**Topics**

- LLM Evaluation Frameworks
- Bias & Hallucination Detection
- Safety Fine-Tuning
- Guardrails
- Constitutional AI
- Red Teaming

---

## ⬜ Week 10 — AI Engineering Capstone

**Goal**

Build a complete production-grade AI system combining:

- Fine-Tuning
- RAG
- AI Agents
- Local Deployment

---

# 🎨 Generative AI Track

A parallel track to the weekly curriculum. The weeks go **deep** on one topic at a time; this track goes **wide** across Generative AI, with interview preparation and mini projects in every module.

➡️ **[Open the Generative AI Track](generative-ai/README.md)**

| Module | Topic | Level | Status |
| :----- | :---- | :---- | :----: |
| 01 | The Generative AI Landscape | Beginner | ✅ Complete |
| 02 | Prompt Engineering | Beginner | ✅ Complete |
| 03 | LLM APIs: Tokens, Streaming, Cost | Beginner | ✅ Complete |
| 04 | Embeddings & Vector Search | Beginner | ✅ Complete |
| 05 | Retrieval-Augmented Generation (RAG) | Intermediate | ✅ Complete |
| 06 | Evaluation & LLM-as-Judge | Intermediate | ⬜ Planned |
| 07 | Tool Use & Agents | Intermediate | ⬜ Planned |
| 08 | Guardrails, Safety & Hallucination | Intermediate | ⬜ Planned |
| 09 | Prompting vs RAG vs Fine-Tuning | Intermediate | ⬜ Planned |
| 10 | Multimodal & Diffusion Models | Intermediate | ⬜ Planned |
| 11 | Deployment & LLMOps | Intermediate | ⬜ Planned |

**Projects**

- ✅ [Support Ticket Assistant](generative-ai/projects/01-support-ticket-assistant) (beginner)
- Chat with Your Documents (RAG with citations)
- RAG Evaluation Harness
- Tool-Using Agent
- Production LLM API

---

# 🛠️ Technology Stack

### Core

- Python
- PyTorch

### LLM Ecosystem

- Hugging Face Transformers
- TRL
- PEFT
- BitsAndBytes
- Unsloth

### Frameworks

- LangChain
- LangGraph
- LlamaIndex

### Local LLMs

- Ollama
- llama.cpp
- vLLM

### Retrieval

- FAISS
- ChromaDB

### Deployment

- FastAPI
- Streamlit

---

# 📖 Learning Philosophy

> **Learn → Experiment → Build → Document → Share**

Every module includes:

- Clear explanations
- Hands-on coding
- Interactive notebooks
- Real-world projects

---

⭐ **This repository is continuously evolving as I explore modern AI Engineering.**
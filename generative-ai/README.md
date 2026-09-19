# 🎨 Generative AI Track

A hands-on track that takes you from **beginner to intermediate Generative AI**, built for two goals: knowing the concepts well enough to explain them in an interview, and having built the things you're asked about.

It is part of the [AI Engineering Journey](../README.md). The weekly curriculum in [`weeks/`](../weeks) goes deep on one topic at a time. This track goes **wide**, covering the full GenAI landscape with interview preparation built into every module.

---

## 🎯 How This Track Works

Every module follows the same pattern:

```text
Learn the concept
        ↓
Run and modify the code
        ↓
Practice the interview questions
        ↓
Build a mini project
```

Each module contains:

- 📖 A concept README explaining the idea in plain language
- 💻 A runnable `.py` script and a matching `.ipynb` notebook
- 🎤 An **Interview Q&A** section: short answer, deeper answer, and the follow-ups interviewers ask

---

## 📈 Progress

### 🟢 Beginner

| Module | Topic | Status |
| :----- | :---- | :----: |
| [01](01-beginner/01-genai-landscape) | The Generative AI Landscape | ✅ Complete |
| [02](01-beginner/02-prompt-engineering) | Prompt Engineering | ✅ Complete |
| [03](01-beginner/03-llm-apis) | LLM APIs: Tokens, Streaming, Cost | ✅ Complete |
| [04](01-beginner/04-embeddings-vector-search) | Embeddings & Vector Search | ✅ Complete |

### 🟡 Intermediate

| Module | Topic | Status |
| :----- | :---- | :----: |
| [05](02-intermediate/05-rag) | Retrieval-Augmented Generation (RAG) | ✅ Complete |
| [06](02-intermediate/06-evaluation) | Evaluation & LLM-as-Judge | ✅ Complete |
| 07 | Tool Use & Agents | ⬜ Planned |
| 08 | Guardrails, Safety & Hallucination | ⬜ Planned |
| 09 | Prompting vs RAG vs Fine-Tuning | ⬜ Planned |
| 10 | Multimodal & Diffusion Models | ⬜ Planned |
| 11 | Deployment & LLMOps | ⬜ Planned |

### 🏆 Projects

| Project | Builds on | Status |
| :------ | :-------- | :----: |
| [Support Ticket Assistant](projects/01-support-ticket-assistant) (beginner) | Modules 02, 03, 04 | ✅ Complete |
| Chat with Your Documents (RAG with citations) | Modules 04, 05 | ⬜ Planned |
| RAG Evaluation Harness | Module 06 | ⬜ Planned |
| Tool-Using Agent | Module 07 | ⬜ Planned |
| Production LLM API (FastAPI, streaming, caching, cost tracking) | Module 11 | ⬜ Planned |

------ | :-------- | :----: |
| Chat with Your PDF (RAG with citations) | Modules 04, 05 | ⬜ Planned |
| Structured Extraction Pipeline (documents to validated JSON) | Modules 02, 03 | ⬜ Planned |
| Prompt & RAG Evaluation Harness | Module 06 | ⬜ Planned |
| Tool-Using Agent | Module 07 | ⬜ Planned |
| Text-to-Image Demo | Module 10 | ⬜ Planned |
| Production LLM API (FastAPI, streaming, caching, cost tracking) | Module 11 | ⬜ Planned |

---

## 📂 Structure

```text
generative-ai/
├── 01-beginner/
│   ├── 01-genai-landscape/
│   │   ├── assets/
│   │   ├── genai_landscape.py
│   │   ├── genai_landscape.ipynb
│   │   └── README.md
│   ├── 02-prompt-engineering/
│   │   ├── prompt_engineering.py
│   │   ├── prompt_engineering.ipynb
│   │   └── README.md
│   ├── 03-llm-apis/
│   │   ├── assets/
│   │   ├── llm_apis.py
│   │   ├── llm_apis.ipynb
│   │   └── README.md
│   └── 04-embeddings-vector-search/
│       ├── assets/
│       ├── embeddings_vector_search.py
│       ├── embeddings_vector_search.ipynb
│       └── README.md
├── 02-intermediate/
│   ├── 05-rag/
│   │   ├── assets/
│   │   ├── rag.py
│   │   ├── rag.ipynb
│   │   └── README.md
│   └── 06-evaluation/
│       ├── assets/
│       ├── evaluation.py
│       ├── evaluation.ipynb
│       └── README.md
├── projects/
│   └── 01-support-ticket-assistant/
│       ├── assistant/
│       ├── data/
│       ├── tests/
│       ├── main.py
│       └── README.md
├── interview-prep/
│   ├── README.md
│   └── cheatsheet.md
├── tools/
│   └── py_to_notebook.py   # builds each notebook from its .py script
├── README.md
└── requirements.txt
```

The `02-intermediate/` modules and further projects are added as each one is completed.

---

## 🎤 Interview Prep

Interview material lives inside each module, so you learn a topic and practice its questions together. [`interview-prep/`](interview-prep) collects it in one place:

- **[Cheatsheet](interview-prep/cheatsheet.md)**: one-line answers for quick revision
- **[Question index](interview-prep/README.md)**: every question, linked to the module that answers it

---

## ⚙️ Getting Started

The track shares the repository's virtual environment.

```powershell
# From repository root
venv\Scripts\activate

pip install -r generative-ai\requirements.txt
```

Modules run locally on open models by default, so no API key is needed. Modules that need a hosted API (such as function calling) say so at the top of their README. Keep keys in a `.env` file, which is already gitignored.

---

## 🔗 How It Connects to the Weekly Curriculum

| This track | Goes deeper in |
|---|---|
| Module 01 — GenAI Landscape | [Week 1 — LLM Foundations](../weeks/week-01-llm-foundations) |
| Module 02 — Prompt Engineering | [Week 2, Day 1](../weeks/week-02-fine-tuning-fundamentals/day-01-prompt-engineering) |
| Module 03 — LLM APIs | [Week 1, Day 5](../weeks/week-01-llm-foundations/day-05-context-window) |
| Module 04 — Embeddings | [Week 1, Day 2](../weeks/week-01-llm-foundations/day-02-transformers) |
| Module 05 — RAG | Week 6 — RAG *(planned)*, and [Week 2](../weeks/week-02-fine-tuning-fundamentals) for the fine-tuning alternative |
| Module 06 — Evaluation | [Week 2, Day 5](../weeks/week-02-fine-tuning-fundamentals/day-05-model-evaluation), Week 9 *(planned)* |
| Module 07 — Agents | Week 8 — AI Agents & Tool Use *(planned)* |
| Module 09 — Prompting vs RAG vs Fine-Tuning | [Weeks 2–3](../weeks) |
| Module 11 — Deployment | Week 7 — Local LLMs & Deployment *(planned)* |

---

⭐ This track grows as I learn. Feedback and suggestions are welcome.

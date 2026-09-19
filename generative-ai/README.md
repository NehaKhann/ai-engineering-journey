# 🎨 Generative AI Track

A hands-on track that takes you from **beginner to intermediate Generative AI**, built for two goals: knowing the concepts well enough to explain them in an interview, and having built the things you're asked about.

It is part of the [AI Engineering Journey](../README.md). The weekly curriculum in [`weeks/`](../weeks) goes deep on one topic at a time. This track goes **wide**, covering the full GenAI landscape with interview preparation built into every module.

---

## 👋 New to AI? Start Here

**What is generative AI, in one minute?** Ordinary software follows rules you wrote. Older machine learning learns to *label* things (spam or not spam). **Generative AI** learns the patterns in huge amounts of data so well that it can *produce* new text, code, or images that follow those patterns.

Chat assistants are built on **large language models (LLMs)**: they read text and predict what comes next, one small piece at a time. That single idea explains both why they are so capable and why they sometimes make things up confidently. Module 01 lets you build a tiny one yourself so it stops feeling like magic.

**What you need**

- Basic Python: functions, lists, dictionaries, and running a script
- An ordinary laptop. **No GPU and no paid account**: everything runs on a CPU with small open models
- Patience for first-run downloads: models come from Hugging Face, and each module says roughly how large
- *Optional:* an API key for a hosted model. The sections that use one are skipped without it

**How to work through it**

1. Skim the **[Glossary](GLOSSARY.md)** once, and come back to it whenever a word is unfamiliar. You don't need to memorize it.
2. Go **in order, 01 to 11**. Each module builds on the ones before it.
3. In each module: read the README, run the script (or open the notebook), then **answer the interview questions out loud before opening the answers**. Explaining an idea is the real test.
4. Build the projects when you reach them: project 01 after Module 04, projects 02 and 03 after Modules 05 to 08, and project 04 after Module 11.

**Two things to know before you start**

- The numbers in these modules come from a **small 0.5-billion-parameter model on a CPU** and from small test sets. They demonstrate *methods*. Larger models score higher, so don't quote them as benchmarks.
- Some modules are heavier than others. **Module 09 trains a model** (roughly 10 to 20 minutes on a laptop), and Modules 10 and 11 download and run larger models.

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
| [07](02-intermediate/07-tool-use-agents) | Tool Use & Agents | ✅ Complete |
| [08](02-intermediate/08-guardrails-safety) | Guardrails, Safety & Hallucination | ✅ Complete |
| [09](02-intermediate/09-prompting-vs-rag-vs-finetuning) | Prompting vs RAG vs Fine-Tuning | ✅ Complete |
| [10](02-intermediate/10-multimodal-diffusion) | Multimodal & Diffusion Models | ✅ Complete |
| [11](02-intermediate/11-deployment-llmops) | Deployment & LLMOps | ✅ Complete |

### 🏆 Projects

| Project | Level | Builds on | Status |
| :------ | :---- | :-------- | :----: |
| [01 · Support Ticket Assistant](projects/01-support-ticket-assistant) | 🟢 Beginner | Modules 02, 03, 04 | ✅ Complete |
| [02 · Ask Your Documents](projects/02-ask-your-documents) (RAG with citations, PDF support, evaluation harness) | 🟡 Intermediate | Modules 04, 05, 06 | ✅ Complete |
| [03 · Tool-Using Agent](projects/03-tool-using-agent) (sandboxed files, human approval, red-teaming) | 🟡 Intermediate | Modules 07, 08 | ✅ Complete |
| [04 · Production LLM API](projects/04-production-llm-api) (auth, rate limits, caching, streaming, metrics) | 🟡 Intermediate | Modules 03, 11 | ✅ Complete |

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
│   ├── 06-evaluation/
│   │   ├── assets/
│   │   ├── evaluation.py
│   │   ├── evaluation.ipynb
│   │   └── README.md
│   ├── 07-tool-use-agents/
│   │   ├── tool_use_agents.py
│   │   ├── tool_use_agents.ipynb
│   │   └── README.md
│   ├── 08-guardrails-safety/
│   │   ├── guardrails.py
│   │   ├── guardrails.ipynb
│   │   └── README.md
│   ├── 09-prompting-vs-rag-vs-finetuning/
│   │   ├── prompting_rag_finetuning.py
│   │   ├── prompting_rag_finetuning.ipynb
│   │   └── README.md
│   ├── 10-multimodal-diffusion/
│   │   ├── assets/
│   │   ├── multimodal_diffusion.py
│   │   ├── multimodal_diffusion.ipynb
│   │   └── README.md
│   └── 11-deployment-llmops/
│       ├── deployment_llmops.py
│       ├── deployment_llmops.ipynb
│       └── README.md
├── projects/
│   ├── 01-support-ticket-assistant/
│   ├── 02-ask-your-documents/
│   ├── 03-tool-using-agent/
│   └── 04-production-llm-api/
├── interview-prep/
│   ├── README.md
│   └── cheatsheet.md
├── tools/
│   ├── py_to_notebook.py   # builds each notebook from its .py script
│   └── check_notebooks.py  # checks notebooks still match their scripts
├── GLOSSARY.md             # plain-English definitions of every term
├── README.md
├── requirements.txt        # everything the track needs
└── requirements-ci.txt     # the small set the automated checks use
```

The `02-intermediate/` modules and further projects are added as each one is completed.

---

## ✅ Automated Checks

A GitHub Actions workflow (`.github/workflows/generative-ai-tests.yml`) runs on every change to this folder. It downloads no models, so it is fast. It checks that:

- every script compiles
- every notebook is valid **and still matches the script it was generated from** (edit a script, then regenerate its notebook with `python generative-ai/tools/py_to_notebook.py <script>`)
- all four projects' tests pass

To run the same checks yourself, from the repository root: `python generative-ai/tools/check_notebooks.py`, and `python -m unittest discover -s tests` inside each project folder.

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
| Module 08 — Guardrails | [Module 02](01-beginner/02-prompt-engineering), Week 9 *(planned)* |
| Module 09 — Prompting vs RAG vs Fine-Tuning | [Weeks 2–3](../weeks) |
| Module 11 — Deployment | Week 7 — Local LLMs & Deployment *(planned)* |

---

⭐ This track grows as I learn. Feedback and suggestions are welcome.

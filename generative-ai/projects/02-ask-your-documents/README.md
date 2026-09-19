# 📚 Project 02 — Ask Your Documents

**Generative AI Track • Intermediate Project** (uses Modules 04, 05, and 06)

Point it at a folder of your own files (`.txt`, `.md`, or `.pdf`), ask questions in plain English, and get answers **with citations** that name the file (and PDF page) they came from. It runs on two small local models, so no API key is needed and your documents never leave your machine.

The project also includes something most RAG demos leave out: an **evaluation harness** that scores retrieval and answers on a golden set and tells you *why* each wrong answer went wrong.

> **The idea to take away:** building the RAG pipeline is the easy part. Knowing whether it works, and which part to fix when it doesn't, is the real skill. That's why measuring is built in.

---

## 🎯 What You'll Practice

| Skill | Where |
|---|---|
| Loading text, Markdown, and **PDF** documents | `docs_qa/loaders.py` |
| **Chunking** with overlap, keeping title and source with every chunk | `docs_qa/chunking.py` |
| A **persistent vector store** with metadata, and re-ingesting to update | `docs_qa/store.py` (ChromaDB) |
| Retrieval, prompting, and **citations** | `docs_qa/rag.py` |
| Refusing when nothing relevant was found (a **similarity threshold**) | `docs_qa/rag.py` |
| **Evaluating** retrieval and generation separately | `docs_qa/evaluate.py` |
| Diagnosing failures: **retrieval vs generation** | the evaluation report |

---

## 📂 Project Structure

```text
02-ask-your-documents/
├── main.py                  # command line: ingest, ask, evaluate, sources
├── docs_qa/
│   ├── models.py            # the embedding model and the chat model
│   ├── loaders.py           # read .txt, .md, .pdf
│   ├── chunking.py          # overlapping chunks with title and source
│   ├── store.py             # ChromaDB vector store, saved on disk
│   ├── rag.py               # retrieve, prompt, cite, refuse
│   └── evaluate.py          # golden-set scoring and failure diagnosis
├── sample_docs/             # a fictional company handbook (6 files)
├── data/golden.json         # 24 answerable and 2 unanswerable questions
└── tests/test_docs_qa.py    # 14 tests, no model needed
```

---

## ⚙️ Setup

```powershell
# From repository root
venv\Scripts\activate

cd generative-ai\projects\02-ask-your-documents

pip install "transformers>=4.56" torch chromadb pypdf
```

## ▶️ Usage

```powershell
# 1. Index a folder (do this once; the index is saved in .index/)
python main.py ingest sample_docs

# 2. Ask questions
python main.py ask "How many vacation days do new employees get?"
python main.py ask "What's the nightly cap on hotels?" --show-retrieved

# 3. See what is indexed
python main.py sources

# 4. Score the whole system on the golden set
python main.py evaluate

# Run the unit tests (no models needed)
python -m unittest discover -s tests -v
```

To use your own documents, run `python main.py ingest path\to\your\folder`. **Re-ingesting a file replaces its old chunks**, so editing a document and re-running `ingest` updates the answers without rebuilding everything. That is the freshness advantage RAG has over fine-tuning (Module 09).

---

## 🧠 Design

### The pipeline

```text
 ingest:  files ─► load ─► chunk (60 words, 15 overlap) ─► embed ─► ChromaDB (with source, page, title)

 ask:     question ─► embed ─► top 3 chunks
                          │
                          ├─ best similarity below 0.2?  ─► "I couldn't find that in the documents."  (the model is never called)
                          │
                          └─ otherwise ─► numbered sources + question ─► model ─► answer with [1] [2]
                                                                                     │
                                                          "I don't know."? ─► refuse │
                                                          otherwise ─► show only the CITED sources
```

### Design choices, and why

- **Every chunk carries its source, page, and title.** That is what makes citations possible, and prepending the title to each chunk before embedding is meant to help a chunk that starts mid-document still say what it's about (we did not measure it against leaving the title out).
- **A similarity threshold refuses before generating.** If nothing retrieved is close enough (best score below 0.2), the system says so without calling the model. That saves a call and prevents a confident answer built on irrelevant text. The value 0.2 was chosen from Module 04's observation that off-topic questions scored under about 0.15 and real ones above about 0.3. It is a starting point, not a tuned value.
- **Only cited sources are shown.** The answer prints the files the model actually cited. If it cited nothing, the best match is shown instead.
- **The prompt was chosen by measurement.** Module 09 showed the same retrieval scoring 1, 1, 6, and 5 out of 12 across four prompt wordings on this small model. The prompt used here is the one that worked best there: no example answer (the model copied it), and an instruction to say "I don't know" *only* if no source mentions the answer. **A caution:** Module 09 picked that prompt by scoring it on 12 test questions, and those same 12 appear in this golden set. That is a small leak (choosing something on the data you then report on flatters the result), so the golden set marks those 12 and the report scores them **separately** from 12 questions that were never used to choose anything.
- **Re-ingesting replaces.** Adding a file deletes that file's old chunks first, so nothing stale lingers.

### The evaluation harness

For every golden question it records the **rank of the first retrieved chunk that contains the evidence** and whether the answer was correct, then reports:

| Metric | Meaning |
|---|---|
| **Hit@1 / Hit@3 / MRR** | Was the right chunk retrieved, and how high (Module 06) |
| **Answer accuracy** | Did the answer contain the expected fact |
| **Wrong because retrieval** | The evidence was never retrieved. Fix chunking, embeddings, or `k`. |
| **Wrong because generation** | The evidence *was* retrieved but the answer was still wrong. Fix the prompt or the model. |
| **Correctly declined** | Unanswerable questions the system refused rather than guessed |

That retrieval-vs-generation split is what tells you where to spend your time.

---

## 📊 Results on the Golden Set

`python main.py evaluate` on the sample handbook (24 answerable questions phrased the way a real employee would ask, plus 2 the documents cannot answer. Twelve of the 24 are the questions Module 09 used to choose the prompt; the other 12 were never used for anything):

| | Result |
|---|---|
| Retrieval Hit@1 | **83%** |
| Retrieval Hit@3 | **100%** |
| Retrieval MRR | 0.90 |
| Answers correct (24 answerable) | **50%** (12 of 24) |
| ...on the 12 questions the prompt was chosen on | 5 of 12 (optimistic in principle) |
| ...on the 12 questions it never saw | **7 of 12** (the honest number) |
| Wrong because **retrieval** failed | **0** |
| Wrong because **generation** failed | **12** |
| Unanswerable questions correctly declined | 2 of 2 |

Example of the working path, and of a refusal:

```text
Q: What's the nightly cap on accommodation when I travel?
A: $180 per night
   source: expenses.md  (similarity 0.49)

Q: how do I bake a chocolate cake
A: I couldn't find that in the documents.
   (no relevant chunk (similarity below threshold))
```

### What the numbers say

**Retrieval is not the problem. The generator is.** The right chunk was in the top 3 for every single question (Hit@3 100%), so all 12 wrong answers were **generation failures**: the evidence was sitting in the prompt and the small model still got it wrong. The harness's retrieval-versus-generation split is what makes this visible. Without it you might spend days tuning chunk sizes and embeddings, which would not have helped.

The 12 failures fall into recognizable groups:

| Failure | Count | Example |
|---|---|---|
| Answered only with a citation | 2 | *"When does my initial salary payment land?"* → `[1]` |
| Wrongly said "I don't know" | 5 | *"Is plugging in a flash drive allowed?"* → `I don't know.` |
| Grabbed a neighboring number | 2 | *"What money do I get to set up my work-from-home space?"* → `$50 per month toward internet costs` (the answer is the **$500** stipend, one sentence earlier) |
| Repeated the source text instead of answering | 3 | *"How long is the trial period for new hires?"* → the start of the onboarding chunk |

This matches what Modules 05 and 09 found with the same small model: it is unreliable at reading a passage and pulling out one fact. The fixes are on the generation side: a stronger model (Module 03), a prompt tuned on your own data, or an extractive step that pulls the answer span out of the chunk.

### Read these numbers with care

- **The prompt-selection overlap did not visibly inflate the result.** The 12 questions the prompt was chosen on scored 5 of 12, and the 12 it never saw scored 7 of 12. With 12 questions each, that gap is noise, so the honest reading is "about half, with no sign of leakage", not "the fresh questions are easier". The split is there so you can check.
- **24 questions is still small.** One question is about 4 points, so treat the exact percentages as approximate.
- **The scorer is crude.** It checks whether an expected string appears in the answer (Module 06 covers better scorers, and their own limits).
- **The corpus is tiny and clean:** six short documents. Retrieval will not stay this good on thousands of similar documents.
- **The model is small.** A larger model would likely score much higher on the same retrieved chunks, and would change the picture from "generation is the bottleneck" to something less lopsided.
- Results are **repeatable**, because decoding is greedy.

---

## 🎤 How to Talk About This Project in an Interview

**The 30-second pitch:**
> "I built a document Q&A tool with citations: it loads text, Markdown, and PDFs, chunks them with overlap, stores them in ChromaDB, and answers from the top chunks, refusing when nothing relevant is found. Then I built an evaluation harness with a golden set that separates retrieval failures from generation failures, so I could see that the bottleneck was the small generator, not retrieval."

| They ask | You can say |
|---|---|
| "How would you build RAG over company documents?" | Load, chunk with overlap and metadata, embed, store, retrieve top-k, prompt with numbered sources, cite, and refuse below a similarity threshold. |
| "How do you know it works?" | A golden set of realistic questions, measured for retrieval (Hit@k, MRR) and answers separately, including unanswerable questions. |
| "The answer is wrong. How do you debug it?" | Check whether the evidence was retrieved. If not, it's a retrieval problem. If yes, it's generation. The harness reports both counts. |
| "How do you handle document updates?" | Re-ingest the changed file: its old chunks are deleted and new ones added, no full rebuild. |
| "How do you prevent hallucination?" | Ground the answer in retrieved text, require citations, refuse below a threshold, and measure the refusal behavior. |

---

## 🚀 Extend It

- Add **hybrid search** (keyword plus semantic) and measure it against semantic alone, as in Module 06
- Add a **reranker** that rescores the top 10 chunks before generating
- Chunk **by heading** instead of by word count
- Add **metadata filters** (for example only search one department's documents)
- Swap in a hosted model (Module 03) and compare the generation failures

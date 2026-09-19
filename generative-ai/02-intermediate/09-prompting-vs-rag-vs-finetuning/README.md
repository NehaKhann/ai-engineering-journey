# 📘 Module 09 — Prompting vs RAG vs Fine-Tuning

**Generative AI Track • Intermediate**

You now have three ways to change what a model does:

| Approach | What you change | Cost to try |
|---|---|---|
| **Prompting** | The instructions and examples in the input | Minutes |
| **RAG** | The knowledge pasted into the input | Hours to days |
| **Fine-tuning** | The model's own weights | Days to weeks |

*"Which one should I use?"* is probably the most common intermediate GenAI interview question. The usual answers are slogans like "RAG for knowledge, fine-tuning for style". Slogans are a fine start, but an interviewer who pushes ("why?", "what breaks?") wants evidence.

So this module **runs all three on the same task** and measures what each can and cannot do, including the thing that matters most in real companies: **what happens when the facts change.**

---

## 🎯 Objective

In this module, you'll:

- **Fine-tune a model yourself** with LoRA on a CPU, in about two minutes
- Compare prompting, RAG, fine-tuning, and fine-tuning plus RAG on the same 12 questions
- Run a **fair policy-update test** and see which approach keeps up
- Check fine-tuning for **side effects**: general ability and "I don't know"
- Measure the **cost per query** of RAG versus a fine-tuned model
- Use a **decision framework** that you can defend in an interview

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `prompting_rag_finetuning.py` | The whole experiment: data, RAG, LoRA training, update test, side effects, costs, decision rules. |
| `prompting_rag_finetuning.ipynb` | Interactive notebook version, generated from the script. |
| `README.md` | Concepts, real results, and interview Q&A. |

---

## ⚙️ Setup

```powershell
# From repository root
venv\Scripts\activate

cd generative-ai\02-intermediate\09-prompting-vs-rag-vs-finetuning

pip install "transformers>=4.56" torch peft
```

## ▶️ Run

```powershell
python prompting_rag_finetuning.py
```

This one is heavier than earlier modules: it trains a LoRA adapter (about two minutes on a laptop CPU) and runs a few hundred short generations, so the whole script takes roughly 10 to 20 minutes. No GPU or API key needed. Decoding is greedy, so results repeat.

---

## 🧠 Key Concepts

### 1. The Experiment

We teach a small model the fictional handbook from Module 05 in different ways, using 12 facts (vacation days, hotel limit, password length, and so on).

- **Fine-tuning data:** 36 question-and-answer pairs (3 phrasings per fact)
- **Test:** 12 questions, each **worded differently from all training questions**, so we measure whether the fact was learned rather than the sentences memorized
- **Fine-tuning method:** **LoRA** (Week 3). It trains small adapter matrices, 8.8 million of the model's 503 million parameters (1.7%), instead of every weight.

---

### 2. Results on Reworded Questions

| Approach | Correct (of 12) |
|---|---|
| Prompting only (no knowledge) | 1 |
| RAG (best of four prompts) | 6 |
| **Fine-tuned** | **12** |
| Fine-tuned + RAG | 9 |

On its own, the fine-tuned model wins clearly. Fine-tuning on a handful of facts is easy to memorize, and it generalized to new phrasings.

**Two notes on fairness:**

- RAG scored **1, 1, 6, and 5** out of 12 across four prompt wordings, with *identical retrieval*. A small model is very sensitive to the exact prompt: it copied an example answer from the prompt, or said "I don't know" when the answer was right in front of it. The table shows the best of four attempts, and all four are in the script's output. A stronger generator would narrow this gap.
- A single LoRA configuration was chosen in advance and not tuned on the test set.

---

### 3. The Test That Matters: Change a Policy

Real facts change. We updated **five facts that both approaches had answered correctly beforehand**, so any failure afterwards is about the update itself and not about never knowing the fact:

| Change | |
|---|---|
| Hotel limit | $180 → $220 |
| Remote days per week | 3 → 4 |
| Home-office stipend | $500 → $600 |
| Password length | 14 → 16 characters |
| Referral bonus | $2,000 → $2,500 |

| Setup | Picked up the change |
|---|---|
| Original model + RAG (edit the passage, re-embed) | **4 / 5** |
| Fine-tuned model, no context | **0 / 5** |
| Fine-tuned model + the updated passage | **3 / 5** |

| Cost to apply the update | |
|---|---|
| RAG: re-embed the passages | **0.04 seconds** |
| Fine-tuning: retrain | **~123 seconds** here (hours or days for a large model), plus re-testing everything |

Read this carefully:

- **The fine-tuned model gave the new number in none of the 5 cases** (the one we printed stated the old $180 with full confidence), and it had no way to know it was out of date. This is the solid result: **0 of 5 versus 4 of 5**.
- **Fine-tuned + the new passage got 3 of 5.** That is a one-question gap versus RAG alone, which is far too small to conclude anything from five questions. It is a *hint* that a fine-tuned model's memorized answer can compete with fresh context, not a finding. Testing it properly needs many more questions.
- We did not investigate RAG's one miss.

This is the strongest single argument for **RAG as the default for facts that change.**

---

### 4. Side Effects of Fine-Tuning

**Can it say "I don't know"?** We asked two questions the handbook cannot answer.

| Question | Original + RAG | Fine-tuned (with or without context) |
|---|---|---|
| Who is the CEO? | "I don't know." ✅ | "The CEO of Northwind Labs is Scott McNealy." ❌ |
| Policy on pets? | "I don't know." ✅ | "Pets are allowed in the office." ❌ |

The fine-tuned model **invented an answer to both**. A likely reason is that it was trained only on questions that have answers, so it never saw "I don't know" as an option (we did not test that explanation directly). For anything user-facing, that is a serious risk.

**Did it forget how to do other things?** On three general questions, the fine-tuned model answered the color-mixing question with *"The color of your hair is determined by genetics."* (The original was also wrong on that one, with "magenta".) Three questions is an anecdote, not a benchmark, but it is the kind of drift you must test for with a real evaluation set (Module 06), because fine-tuning can quietly damage abilities you never trained.

---

### 5. Cost per Query

RAG pastes context into **every** prompt. A fine-tuned model has a short prompt but a training bill and a maintenance burden.

| | Prompt tokens | Input cost per 100,000 queries* |
|---|---|---|
| Fine-tuned (short prompt) | 40 | $8.00 |
| RAG (3 passages) | 156 | $31.20 |

*Illustrative, at $2 per million input tokens (Module 03). Check current prices.

RAG costs about 4x more per query in this small example. Fine-tuning costs more up front and again **every time the knowledge changes**. Which is cheaper overall depends on traffic and on how often the facts change.

---

### 6. The Decision Framework

**Always start with prompting**, because it is the cheapest to try and to change. Add RAG when the problem is *knowledge*. Add fine-tuning when the problem is *behavior* that prompts can't hold.

```text
  Does a good prompt (clear instructions + a few examples) solve it?  ──yes──► STOP. Use prompting.
         │no
         ▼
  Is the gap missing, private, or changing KNOWLEDGE, or do you need citations?  ──yes──► add RAG
         │
         ▼
  Is the gap BEHAVIOR (a style, a strict format, a specialized skill) that prompts
  can't hold reliably, and do you have hundreds of good examples?  ──yes──► fine-tune
         │
         ▼
  Still not enough? Combine them: RAG for facts + fine-tuning for behavior.
```

| Situation | Approach |
|---|---|
| Support bot over help articles that change weekly | Prompting + **RAG** |
| Extract fields from invoices into JSON | **Prompting** (with a schema and validation) |
| Strict brand voice; prompts drift; 2,000 examples | Prompting + **fine-tuning** |
| Q&A over 50,000 private documents, with citations | Prompting + **RAG** |
| Cite the documents *and* write in a strict firm style | Prompting + RAG + **fine-tuning** |
| Strict format wanted, prompts fail, only 40 examples | **Collect more examples first** |

The script encodes these rules as a small function and asserts the table above.

---

## 🎤 Interview Q&A

Try answering each question out loud before opening the answer.

<details>
<summary><b>Q1. When would you use prompting, RAG, or fine-tuning?</b></summary>

**Short answer:** Start with prompting. Use RAG when the gap is knowledge (private, fresh, or large facts, or a need for citations). Use fine-tuning when the gap is behavior (style, strict format, a specialized skill) that prompting can't hold reliably.

**Deeper answer:** They form an escalation ladder ordered by cost and reversibility. Prompting changes in minutes and is easy to undo. RAG changes what the model sees, so it updates instantly and can cite sources and enforce permissions. Fine-tuning changes the weights, so it is slow to update, hard to inspect, and can cause side effects. They also combine: RAG for the facts, fine-tuning for the tone or format. Decide with measurements on your own task, not slogans.

**Follow-ups to expect:**
- What if prompting almost works? *Improve the prompt, add examples, and test it on an eval set before escalating.*
- Can you fine-tune *and* use RAG? *Yes, and often should: RAG supplies facts while the fine-tuned behavior handles style or format.*

</details>

<details>
<summary><b>Q2. Why is fine-tuning a poor way to add knowledge?</b></summary>

**Short answer:** Facts get baked into the weights, so they can't be updated cheaply, can't be cited or deleted, and the model can become overconfident.

**Deeper answer:** In our experiment, after five facts changed, the fine-tuned model got 0 of 5 right and kept confidently stating the old values, while RAG got 4 of 5. (Handing the fine-tuned model the updated passage recovered 3 of 5, a small gap on five questions, but consistent with the model's memory competing with the context.) It also can't point to a source, can't apply per-user permissions, and learned to answer *every* question, inventing answers to two the documents didn't cover. RAG handles all of those by keeping knowledge outside the model.

**Follow-ups to expect:**
- Is fine-tuning ever right for facts? *For stable, general domain knowledge at scale (for example specialized vocabulary), sometimes, but it is usually combined with retrieval.*

</details>

<details>
<summary><b>Q3. When is fine-tuning the right choice?</b></summary>

**Short answer:** When you need consistent behavior that prompts can't hold: a house style, a strict output format, a specialized skill, or lower latency and cost by using a smaller specialized model.

**Deeper answer:** Good signs: you have hundreds or thousands of high-quality examples, the desired behavior is stable, prompts are long or brittle, and you've already measured that prompting falls short. Fine-tuning can also shorten prompts (our fine-tuned prompts were 40 tokens versus 156 for RAG), reducing per-query cost at scale, and can let a small model match a large one on a narrow task. Watch for side effects: test general ability and whether it can still refuse or say "I don't know".

**Follow-ups to expect:**
- How many examples do you need? *It depends on the task. Dozens can shift a style, but reliable behavior usually needs hundreds or more, and quality beats quantity.*
- What is LoRA? *A method that trains small adapter matrices instead of all weights, so it needs far less memory and time. Here it trained 1.7% of the parameters.*

</details>

<details>
<summary><b>Q4. A policy changes tomorrow. How does each approach cope?</b></summary>

**Short answer:** RAG updates immediately by re-indexing the document. A fine-tuned model must be retrained, and stays wrong until then.

**Deeper answer:** Measured: re-embedding took 0.04 seconds and the RAG setup picked up 4 of 5 changes. The fine-tuned model picked up 0 of 5 and needed about two minutes to retrain even on a tiny model, plus re-testing to make sure nothing else broke. For a large model the retraining is hours or days and real money. A fine-tuned model that is also given the new passage may still lean on its memory, so don't assume you can fix staleness just by adding context; test it.

**Follow-ups to expect:**
- What would you do if you must fine-tune facts? *Keep the facts in retrieval anyway, fine-tune only stable behavior, and version and re-evaluate models on every change.*

</details>

<details>
<summary><b>Q5. What are the cost trade-offs between RAG and fine-tuning?</b></summary>

**Short answer:** RAG pays more per query (longer prompts, plus retrieval infrastructure). Fine-tuning pays up front (data, training, evaluation) and again whenever the knowledge or behavior changes.

**Deeper answer:** Per query, RAG added about 116 prompt tokens in our example (156 versus 40), roughly 4x the input cost. Fine-tuning shifts the cost to training, dataset curation, hosting a custom model, and re-doing all of that on every update. High volume with a stable task favors fine-tuning. Low or moderate volume, or frequently changing knowledge, favors RAG. Also count engineering time and risk, not just tokens.

**Follow-ups to expect:**
- How can RAG be cheaper? *Retrieve fewer, better chunks, cache repeated queries (Module 11), and rerank so fewer passages are needed.*

</details>

<details>
<summary><b>Q6. How do you decide with data instead of opinion?</b></summary>

**Short answer:** Build an evaluation set of real questions, run each approach on it, and compare accuracy, refusals, cost, and how each handles change.

**Deeper answer:** Measure more than accuracy on the happy path. Include reworded questions (to test generalization), unanswerable questions (to test "I don't know"), and a change-the-facts test (to test freshness). Vary the prompt for the RAG and prompting options, because small changes swung our RAG result from 1 to 6 out of 12 with identical retrieval. Compare on the same questions with the same scoring, report uncertainty on small sets (Module 06), and keep the eval set for re-testing after every change.

**Follow-ups to expect:**
- Why test unanswerable questions? *A system that always answers will confidently invent things, which our fine-tuned model did.*

</details>

---

## 🎯 Key Takeaways

After completing this module, you'll understand:

- The order to try things: prompting, then RAG for knowledge, then fine-tuning for behavior
- Why fine-tuning is a poor way to store facts: it picked up 0 of 5 updates, versus 4 of 5 for RAG
- That fine-tuned models can lose the ability to say "I don't know"
- How LoRA fine-tuning works and how cheap it is to try on a small model
- That RAG quality depends heavily on the generator and the prompt, not just retrieval
- How to compare the approaches with a fair, data-driven test

---

## 🔗 Go Deeper in the Weekly Curriculum

| Topic | Where |
|---|---|
| Fine-tuning fundamentals and SFT | [Week 2](../../../weeks/week-02-fine-tuning-fundamentals/README.md) |
| LoRA, QLoRA, and DoRA | [Week 3](../../../weeks/week-03-efficient-fine-tuning/README.md) |
| RAG in this track | [Module 05](../05-rag/README.md) |
| Evaluating the alternatives | [Module 06](../06-evaluation/README.md) |

---

## 🚀 What's Next?

**Module 10 • Multimodal & Diffusion Models**

Time to leave text behind: how models *understand* images with CLIP, and how they *generate* them by learning to remove noise.

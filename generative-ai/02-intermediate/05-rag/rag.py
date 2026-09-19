# pip: transformers>=4.56 torch matplotlib
# %% [markdown]
# # 📘 Module 05 — Retrieval-Augmented Generation (RAG)
#
# **Generative AI Track • Intermediate**
#
# A language model only knows what was in its training data. It has never seen **your** company's
# handbook, yesterday's tickets, or a document you wrote this morning. Ask it anyway and it will
# confidently make something up.
#
# **RAG** fixes this in three steps:
#
# 1. **Retrieve** the pieces of your documents that are relevant to the question (Module 04)
# 2. **Augment** the prompt by pasting those pieces in (Module 02)
# 3. **Generate** an answer grounded in them, with citations
#
# We test it on a fictional company handbook. The model cannot possibly know these facts, so any
# correct answer must come from retrieval, and we can **measure** exactly what RAG adds.

# %%
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

OUT = Path(__file__).parent / "assets" if "__file__" in globals() else Path("assets")
OUT.mkdir(exist_ok=True)

# Embedding model (Module 04) and chat model (Module 02), both small enough for a laptop CPU.
EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_ID = "Qwen/Qwen2.5-0.5B-Instruct"

embed_tok = AutoTokenizer.from_pretrained(EMBED_ID)
embed_model = AutoModel.from_pretrained(EMBED_ID).eval()
chat_tok = AutoTokenizer.from_pretrained(CHAT_ID)
chat_model = AutoModelForCausalLM.from_pretrained(CHAT_ID, dtype=torch.float32)


def embed(texts):
    """Unit-length embedding vectors, one row per text (mean pooling, as in Module 04)."""
    batch = embed_tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        hidden = embed_model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1).numpy()


def chat(messages, max_new_tokens=80):
    """One reply from the chat model. Greedy decoding, so results are repeatable."""
    text = chat_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = chat_tok(text, return_tensors="pt")
    out = chat_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=chat_tok.eos_token_id)
    return chat_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# %% [markdown]
# ## 1. The documents
#
# A fictional company handbook, in six short documents. Every fact in it is made up, so the model
# has no way to know it from training.

# %%
HANDBOOK = {
    "vacation.md": """Vacation and leave policy.
New employees receive 20 days of paid vacation per year, accrued monthly. Up to 5 unused days carry over into the next year; anything beyond that is lost on January 1st. Vacation requests go through the TimeOff portal and must be submitted at least 2 weeks in advance for absences longer than 3 days.
Public holidays are separate from vacation. Northwind Labs observes 10 public holidays per year.
Sick leave is 10 days per year. A doctor's note is required after 3 consecutive sick days.
Parental leave is 16 weeks fully paid for primary caregivers and 6 weeks fully paid for secondary caregivers.""",
    "expenses.md": """Expense policy.
Expenses under $75 do not need pre-approval. Anything over $75 needs written approval from your manager before you spend it. Submit all expenses within 30 days through the Expensify app, or they will not be reimbursed.
When traveling, meals are reimbursed up to $60 per day. Hotels are reimbursed up to $180 per night. Flights must be economy class for trips shorter than 6 hours.
Personal alcohol is never reimbursable. Laptops and other hardware must be purchased through the IT department, not expensed.""",
    "remote_work.md": """Remote work policy.
Employees may work remotely up to 3 days per week. Everyone must be reachable during core hours, 10:00 to 15:00 in their local time.
New remote workers receive a one-time home office stipend of $500, plus $50 per month toward internet costs.
Employees must live in their country of employment. Working from abroad is allowed for up to 20 days per year with manager approval. A VPN is required whenever you use public wifi.""",
    "security.md": """Information security policy.
Passwords must be at least 14 characters long, and multi-factor authentication is required on every work account. Laptops use full-disk encryption.
If you lose a device, report it to security within 1 hour by emailing security@northwind.example. Suspected phishing emails go to phishing@northwind.example.
Customer data must never be stored on personal devices, and USB storage drives are not permitted. Every employee must complete security training by March 31 each year.""",
    "onboarding.md": """New employee onboarding.
On your first day, pick up your laptop from the IT desk at 9:30. You are assigned an onboarding buddy who will answer questions during your first month. Managers request GitHub and Slack access at least 5 days before your start date.
Complete the security training and the code of conduct course within your first 2 weeks. The probation period lasts 90 days, with 30, 60, and 90 day check-ins.
Your first paycheck arrives on the 25th of the month after you start.""",
    "benefits.md": """Benefits summary.
Health insurance covers 100% of the employee's premium and 70% of dependents' premiums. The company matches 401(k) contributions up to 4% of salary.
Each employee gets a learning budget of $1,200 per year for courses, books, and conferences, and a gym stipend of $30 per month.
Stock options vest over 4 years with a 1 year cliff. The employee referral bonus is $2,000, paid after the referred person completes 90 days.""",
}
print(f"{len(HANDBOOK)} documents, {sum(len(t.split()) for t in HANDBOOK.values())} words in total")

# %% [markdown]
# ## 2. The questions
#
# Twelve questions whose answers are in the handbook, plus two that are **not**. Each answerable
# question lists the strings that count as a correct answer, so we can score answers automatically.
# (String matching is a crude scorer. Module 06 covers better ones.)

# %%
QUESTIONS = [
    ("How many vacation days do new employees get?", ["20 days", "20 paid", "20 vacation", "twenty"]),
    ("How many unused vacation days can I carry over to next year?", ["5 days", "5 unused", "up to 5", "five"]),
    ("Above what amount do I need my manager's approval to spend money?", ["$75"]),
    ("What is the maximum hotel reimbursement per night?", ["$180"]),
    ("How many days a week can I work remotely?", ["3 days", "three days", "up to 3"]),
    ("How much is the one-time home office stipend?", ["$500"]),
    ("How quickly must I report a lost device?", ["1 hour", "one hour", "an hour", "within 1"]),
    ("What is the minimum password length?", ["14"]),
    ("How much is the yearly learning budget?", ["$1,200", "1,200", "1200"]),
    ("How big is the employee referral bonus?", ["$2,000", "2,000", "2000"]),
    ("On which day of the month is the first paycheck paid?", ["25th", "25"]),
    ("How long is parental leave for primary caregivers?", ["16 weeks", "16-week", "sixteen"]),
]
UNANSWERABLE = [
    "What is the company's policy on bringing pets to the office?",
    "Who is the CEO of Northwind Labs?",
]
NO_ANSWER = "i don't know"


def is_correct(answer, accepted):
    answer = answer.lower()
    return any(a.lower() in answer for a in accepted)


# %% [markdown]
# ## 3. Without retrieval: the model guesses
#
# First, the baseline. Ask the model the questions directly.

# %%
BASELINE_SYSTEM = "You are a helpful assistant for employees of Northwind Labs. Answer briefly."

baseline_answers = []
for question, accepted in QUESTIONS:
    answer = chat([{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": question}], max_new_tokens=60)
    baseline_answers.append(answer)

baseline_score = sum(is_correct(a, acc) for a, (_, acc) in zip(baseline_answers, QUESTIONS))
print(f"Without retrieval: {baseline_score}/{len(QUESTIONS)} correct\n")
for (question, accepted), answer in list(zip(QUESTIONS, baseline_answers))[:4]:
    print(f"Q: {question}\nA: {answer[:140]}\n")

# %% [markdown]
# It has no way to know, so any right answer is luck, and the wrong ones sound confident. That is
# the **hallucination** problem RAG is designed to remove.

# %% [markdown]
# ## 4. Step 1: Chunking
#
# We cannot paste whole documents into every prompt: it is wasteful, and long prompts dilute the
# answer. So we split documents into **chunks** and retrieve only the relevant ones.
#
# Chunk size is a real trade-off:
#
# - **Too small:** a fact gets cut in half, or loses the context that makes it meaningful
# - **Too big:** a chunk covers many topics, so it matches weakly and pads the prompt with noise
#
# We split by words with some **overlap**, so a fact near a boundary appears whole in at least one chunk.

# %%
def chunk_document(name, text, size, overlap):
    """Split text into chunks of about `size` words, each overlapping the previous by `overlap` words."""
    words = text.split()
    step = max(size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append({"source": name, "text": " ".join(piece)})
        if start + size >= len(words):
            break
    return chunks


def build_index(size, overlap):
    chunks = [c for name, text in HANDBOOK.items() for c in chunk_document(name, text, size, overlap)]
    return chunks, embed([c["text"] for c in chunks])


def retrieve(question, chunks, vectors, k=3):
    scores = vectors @ embed([question])[0]
    best = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in best]


chunks, vectors = build_index(size=60, overlap=15)
print(f"{len(chunks)} chunks of about 60 words. Example chunk:\n")
print(chunks[3]["source"], "->", chunks[3]["text"])

# %% [markdown]
# ### Which chunk size retrieves best?
#
# We can measure retrieval **without any language model**: for each question, does the right fact
# appear in the top-k retrieved chunks? (**Hit rate**: the share of questions where it does.)

# %%
def hit_rate(size, overlap, k):
    """Returns (hit rate, number of chunks, average words of retrieved context per question)."""
    chunks, vectors = build_index(size, overlap)
    hits, context_words = 0, 0
    for question, accepted in QUESTIONS:
        retrieved = retrieve(question, chunks, vectors, k)
        hits += any(is_correct(c["text"], accepted) for c, _ in retrieved)
        context_words += sum(len(c["text"].split()) for c, _ in retrieved)
    return hits / len(QUESTIONS), len(chunks), context_words / len(QUESTIONS)


sizes = [20, 40, 60, 100, 200]
print(f"{'Chunk size (words)':<20}{'Chunks':<9}{'Hit@1':<9}{'Hit@3':<9}{'Words pasted into prompt (k=3)'}")
print("-" * 78)
results = []
for size in sizes:
    hit1, n_chunks, _ = hit_rate(size, size // 4, 1)
    hit3, _, prompt_words = hit_rate(size, size // 4, 3)
    results.append((size, n_chunks, hit1, hit3))
    print(f"{size:<20}{n_chunks:<9}{hit1:<9.0%}{hit3:<9.0%}{prompt_words:.0f}")

plt.figure(figsize=(7, 4.5))
plt.plot(sizes, [r[2] * 100 for r in results], marker="o", label="Hit@1")
plt.plot(sizes, [r[3] * 100 for r in results], marker="s", label="Hit@3")
plt.xlabel("Chunk size (words)")
plt.ylabel("Questions where the answer was retrieved (%)")
plt.ylim(0, 105)
plt.title("Retrieval quality vs chunk size")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "chunk_size.png", dpi=150)
plt.show()

# %% [markdown]
# On a corpus this small (six documents), every size finds the answer almost every time, so hit
# rate alone barely separates them. The cost shows up in the last column: bigger chunks paste far
# more text into every prompt, which costs more tokens (Module 03) and gives the model more to
# get lost in. On a large corpus with many similar documents, oversized chunks also match more
# weakly. **Pick the smallest chunks that keep each fact intact.**

# %% [markdown]
# ## 5. Steps 2 and 3: Augment and Generate
#
# Now paste the retrieved chunks into the prompt as **numbered sources** and tell the model to
# answer only from them and to cite. The instruction "say you don't know" matters: it gives the
# model a way out other than guessing.

# %%
# Version 1: the obvious first attempt.
RAG_SYSTEM_V1 = (
    "Answer the question using ONLY the numbered sources below. "
    "Cite the source number in square brackets, like [1]. "
    f"If the answer is not in the sources, reply exactly: {NO_ANSWER.capitalize()}."
)

# Version 2: says what the answer should look like, and shows one example.
RAG_SYSTEM_V2 = (
    "You answer questions about company policy using ONLY the numbered sources provided.\n"
    "Write one short sentence that states the answer, then the source number in square brackets.\n"
    "Example: Employees get 30 days of leave. [2]\n"
    f"If the sources do not contain the answer, write exactly: {NO_ANSWER.capitalize()}."
)


def build_rag_messages(question, retrieved, system=RAG_SYSTEM_V2):
    sources = "\n".join(f"[{i}] ({c['source']}) {c['text']}" for i, (c, _) in enumerate(retrieved, 1))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
    ]


def rag_answer(question, chunks, vectors, k=3, system=RAG_SYSTEM_V2):
    retrieved = retrieve(question, chunks, vectors, k)
    return chat(build_rag_messages(question, retrieved, system), max_new_tokens=80), retrieved


chunks, vectors = build_index(size=60, overlap=15)
question = QUESTIONS[2][0]
answer, retrieved = rag_answer(question, chunks, vectors)

print("Question:", question, "\n")
print("The prompt the model sees:\n")
print(build_rag_messages(question, retrieved)[1]["content"], "\n")
print("Answer:", answer)
print("\nCited sources:", [c["source"] for c, _ in retrieved])

# %% [markdown]
# ## 6. Did RAG help? Measure it
#
# The same 12 questions, now with retrieval, against the no-retrieval baseline.

# %%
def score(answers):
    correct = sum(is_correct(a, acc) for a, (_, acc) in zip(answers, QUESTIONS))
    refusals = sum(NO_ANSWER in a.lower() for a in answers)
    return correct, refusals


answers_v1 = [rag_answer(q, chunks, vectors, system=RAG_SYSTEM_V1)[0] for q, _ in QUESTIONS]
answers_v2 = [rag_answer(q, chunks, vectors, system=RAG_SYSTEM_V2)[0] for q, _ in QUESTIONS]

print(f"{'':<34}{'Correct':<10}{'Refused to answer'}")
print("-" * 64)
print(f"{'No retrieval':<34}{baseline_score}/{len(QUESTIONS)}{'':<6}n/a")
for label, answers in [("RAG, prompt v1 (obvious)", answers_v1), ("RAG, prompt v2 (with example)", answers_v2)]:
    correct, refusals = score(answers)
    print(f"{label:<34}{correct}/{len(QUESTIONS)}{'':<6}{refusals}")

print("\nWhere prompt v1 failed even though retrieval had the fact:")
for (question, accepted), answer in zip(QUESTIONS, answers_v1):
    if not is_correct(answer, accepted):
        found = any(is_correct(c["text"], accepted) for c, _ in retrieve(question, chunks, vectors, 3))
        print(f"  retrieval found the fact: {found} | model said: {answer[:60]!r} | Q: {question}")

rag_answers = answers_v2
print("\nWhere prompt v2 is still wrong:")
for (question, accepted), answer in zip(QUESTIONS, answers_v2):
    if not is_correct(answer, accepted):
        found = any(is_correct(c["text"], accepted) for c, _ in retrieve(question, chunks, vectors, 3))
        print(f"  retrieval found the fact: {found} | model said: {answer[:80]!r} | Q: {question}")

# %% [markdown]
# When RAG gets a question wrong, there are exactly **two possible causes**, and you must know
# which one you have:
#
# - **Retrieval failure:** the right chunk was never retrieved. Fix chunking, embeddings, or `k`.
# - **Generation failure:** the right chunk was in the prompt but the model still answered badly.
#   Fix the prompt or use a stronger model.
#
# Printing "retrieval found the fact" is how you tell them apart. Here retrieval found the fact for
# every question, so all of prompt v1's failures were **generation** failures. The model was told
# to "cite like [1]" and some of the time answered with only the citation, or refused. A prompt
# that describes the shape of a good answer, with one example, lifted RAG from 5 to 8 correct out
# of 12 and removed the citation-only answers.
#
# It did **not** fix everything: the model still refused 4 questions whose answer was sitting in
# its prompt. That is the price of the "say I don't know" escape hatch on a small model, which we
# look at next.

# %% [markdown]
# ## 7. When the answer is not in the documents
#
# A good RAG system says **"I don't know"** instead of guessing. We ask two questions the handbook
# cannot answer.

# %%
print("Unanswerable questions\n")
for question in UNANSWERABLE:
    with_rag, _ = rag_answer(question, chunks, vectors)
    without = chat([{"role": "system", "content": BASELINE_SYSTEM}, {"role": "user", "content": question}], max_new_tokens=60)
    declined = NO_ANSWER in with_rag.lower()
    print(f"Q: {question}")
    print(f"  No retrieval: {without[:110]!r}")
    print(f"  RAG:          {with_rag[:110]!r}  -> {'declined correctly' if declined else 'DID NOT decline'}\n")

# %% [markdown]
# Both unanswerable questions were declined, where the no-retrieval model invented a policy and a
# CEO named "John Smith". But recall the previous section: the same instruction made the model
# **refuse 4 answerable questions**. Every "I don't know" option trades one error for another:
#
# | | Without the escape hatch | With the escape hatch |
# |---|---|---|
# | Answer exists | Usually answered | Sometimes wrongly refused |
# | Answer missing | Invents an answer (hallucination) | Correctly declines |
#
# You choose the balance by measuring both on your own questions. Larger models handle this far
# better, which is one reason RAG quality depends so much on the generator.
#
# ## 8. What can go wrong
#
# | Problem | Symptom | Typical fix |
# |---|---|---|
# | Bad chunking | The answer is split across two chunks | Tune size and overlap, chunk by section |
# | Retrieval miss | Right chunk not in the top k | Better embeddings, hybrid search, higher `k`, reranking |
# | Too much context | Answer buried in noise | Smaller chunks, fewer results, rerank |
# | Model ignores context | Answers from memory | Stricter prompt, stronger model |
# | No-answer case | Confident guess | "Say you don't know" instruction, similarity threshold |
# | Stale data | Answers from old documents | Re-index when documents change |
#
# ## 🎯 Key Takeaways
#
# - RAG = **retrieve** relevant chunks, **augment** the prompt with them, **generate** a grounded answer.
# - It is the standard way to give a model knowledge it was never trained on, without fine-tuning.
# - Chunk size is a trade-off. Measure **retrieval** separately from **generation**.
# - When RAG fails, find out whether **retrieval** or **generation** was at fault.
# - Cite sources, and give the model permission to say "I don't know".
#
# ## 🚀 What's Next?
#
# **Module 06 — Evaluation & LLM-as-Judge**: string matching only got us so far. Next you will
# build proper measurements for retrieval and answer quality.

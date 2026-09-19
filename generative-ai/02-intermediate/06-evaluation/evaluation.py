# pip: transformers>=4.56 torch matplotlib
# %% [markdown]
# # 📘 Module 06 — Evaluation & LLM-as-Judge
#
# **Generative AI Track • Intermediate**
#
# "It seems to work" is not a result. Every change you make to a GenAI system (a new prompt, a
# different chunk size, a bigger model) might help, might hurt, or might do nothing, and you cannot
# tell by reading a few outputs. **Evaluation** is how you find out.
#
# In this notebook you will:
#
# - Measure **retrieval** with hit rate, recall, and MRR
# - Compare three retrievers (keyword, semantic, hybrid) on the same questions
# - Ask **"is this difference real, or noise?"** with a bootstrap confidence interval
# - Score **answers** with string match, token F1, and an **LLM judge**
# - **Test the judge against human labels** before you trust it

# %%
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

OUT = Path(__file__).parent / "assets" if "__file__" in globals() else Path("assets")
OUT.mkdir(exist_ok=True)

EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_ID = "Qwen/Qwen2.5-0.5B-Instruct"

embed_tok = AutoTokenizer.from_pretrained(EMBED_ID)
embed_model = AutoModel.from_pretrained(EMBED_ID).eval()
chat_tok = AutoTokenizer.from_pretrained(CHAT_ID)
chat_model = AutoModelForCausalLM.from_pretrained(CHAT_ID, dtype=torch.float32).eval()


def embed(texts):
    batch = embed_tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        hidden = embed_model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1).numpy()


# %% [markdown]
# ## 1. The test bed: a golden set
#
# Evaluation needs a **golden set**: questions paired with known-correct answers. We reuse the
# fictional company handbook from Module 05 and chunk it the same way (60 words, 15 overlap).
#
# This time the questions are written as a real employee would phrase them, **not** copied from the
# handbook's wording, because that is where retrieval gets hard.

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


def chunk_document(name, text, size=60, overlap=15):
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append({"source": name, "text": " ".join(words[start:start + size])})
        if start + size >= len(words):
            break
    return chunks


CHUNKS = [c for name, text in HANDBOOK.items() for c in chunk_document(name, text)]
CHUNK_VECTORS = embed([c["text"] for c in CHUNKS])

# (question, text that a relevant chunk must contain)
GOLDEN = [
    ("What's the time-off allowance for someone who just joined?", "20 days of paid vacation"),
    ("If I don't use all my holidays, how many roll over?", "Up to 5 unused days"),
    ("Do I need sign-off before buying something costing more than seventy-five dollars?", "Anything over $75"),
    ("What's the nightly cap on accommodation when I travel?", "up to $180 per night"),
    ("How many office-free days per week are allowed?", "up to 3 days per week"),
    ("What money do I get to set up my work-from-home space?", "stipend of $500"),
    ("Who do I tell if my laptop goes missing, and how fast?", "within 1 hour"),
    ("How many characters must a password contain at minimum?", "at least 14 characters"),
    ("What's the annual amount for courses and books?", "learning budget of $1,200"),
    ("How much do I earn for recommending a friend who gets hired?", "referral bonus is $2,000"),
    ("When does my initial salary payment land?", "first paycheck arrives on the 25th"),
    ("How many weeks off does a new mother get?", "16 weeks fully paid"),
    ("Is plugging in a flash drive allowed?", "USB storage drives are not permitted"),
    ("What is the deadline for the yearly security course?", "March 31"),
    ("How long is the trial period for new hires?", "probation period lasts 90 days"),
    ("What share of my family's health premiums does the company cover?", "70% of dependents"),
]
print(f"{len(CHUNKS)} chunks, {len(GOLDEN)} golden questions")


def relevant(chunk, needle):
    return needle.lower() in chunk["text"].lower()


# %% [markdown]
# ## 2. Three retrievers
#
# - **Keyword (BM25):** ranks by shared words, weighting rare words more. Written from scratch below.
# - **Semantic:** embeddings and cosine similarity, as in Module 04.
# - **Hybrid:** merges both rankings with **reciprocal rank fusion**: each retriever gives a chunk
#   `1 / (60 + rank)` points, and the points are added up.

# %%
STOPWORDS = set("a an and are as at be by can do does for from how i if in is it me my of on or the to we what when who you your".split())


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9$%]+", text.lower()) if w not in STOPWORDS]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [tokenize(t) for t in texts]
        self.k1, self.b = k1, b
        self.avg_len = sum(len(d) for d in self.docs) / len(self.docs)
        n = len(self.docs)
        df = {}
        for d in self.docs:
            for w in set(d):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query):
        result = np.zeros(len(self.docs))
        for i, doc in enumerate(self.docs):
            for w in tokenize(query):
                tf = doc.count(w)
                if tf:
                    result[i] += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_len))
        return result


bm25 = BM25([c["text"] for c in CHUNKS])


def rank_bm25(question):
    return list(np.argsort(bm25.scores(question))[::-1])


def rank_dense(question):
    return list(np.argsort(CHUNK_VECTORS @ embed([question])[0])[::-1])


def rank_hybrid(question, k=60):
    fused = np.zeros(len(CHUNKS))
    for ranking in (rank_bm25(question), rank_dense(question)):
        for position, index in enumerate(ranking):
            fused[index] += 1 / (k + position + 1)
    return list(np.argsort(fused)[::-1])


RETRIEVERS = {"keyword (BM25)": rank_bm25, "semantic": rank_dense, "hybrid (RRF)": rank_hybrid}

# %% [markdown]
# ## 3. Retrieval metrics
#
# For each question we find the **rank** of the first relevant chunk (1 = best), then compute:
#
# | Metric | Meaning |
# |---|---|
# | **Hit@k** | Share of questions where a relevant chunk is in the top k |
# | **MRR** | Mean of `1 / rank` of the first relevant chunk. Rewards ranking it *higher*. |
#
# Hit@k says "was it found?". MRR says "how near the top?".

# %%
def first_relevant_rank(ranking, needle):
    for position, index in enumerate(ranking, 1):
        if relevant(CHUNKS[index], needle):
            return position
    return len(ranking) + 1


ranks = {name: [first_relevant_rank(fn(q), needle) for q, needle in GOLDEN] for name, fn in RETRIEVERS.items()}


def metrics(rank_list):
    n = len(rank_list)
    return {
        "hit@1": sum(r <= 1 for r in rank_list) / n,
        "hit@3": sum(r <= 3 for r in rank_list) / n,
        "MRR": sum(1 / r for r in rank_list) / n,
    }


print(f"{'Retriever':<18}{'Hit@1':<9}{'Hit@3':<9}{'MRR'}")
print("-" * 42)
for name, rank_list in ranks.items():
    m = metrics(rank_list)
    print(f"{name:<18}{m['hit@1']:<9.0%}{m['hit@3']:<9.0%}{m['MRR']:.2f}")

print("\nQuestions where keyword search ranked the right chunk worse than 3rd:")
for (question, _), rank in zip(GOLDEN, ranks["keyword (BM25)"]):
    if rank > 3:
        print(f"  rank {rank:>2}: {question}")

# %% [markdown]
# ## 4. Is the difference real, or noise?
#
# With only 16 questions, one question flipping changes a score by 6 points. Before you claim
# "A beats B", check whether the gap could just be luck. A **bootstrap** re-draws the 16 questions
# *with replacement* thousands of times and sees how much the gap moves. If the range of plausible
# gaps includes 0, you have not shown a difference.

# %%
def bootstrap_gap(scores_a, scores_b, trials=5000, seed=0):
    """95% interval for mean(a) - mean(b), resampling questions together (a paired bootstrap)."""
    rng = np.random.default_rng(seed)
    a, b = np.array(scores_a, float), np.array(scores_b, float)
    n = len(a)
    gaps = [(a[i] - b[i]).mean() for i in (rng.integers(0, n, n) for _ in range(trials))]
    return np.percentile(gaps, [2.5, 97.5])


hit1 = {name: [r <= 1 for r in rank_list] for name, rank_list in ranks.items()}
print("Difference in Hit@1, with a 95% bootstrap interval\n")
for a, b in [("semantic", "keyword (BM25)"), ("hybrid (RRF)", "semantic"), ("hybrid (RRF)", "keyword (BM25)")]:
    gap = np.mean(hit1[a]) - np.mean(hit1[b])
    low, high = bootstrap_gap(hit1[a], hit1[b])
    verdict = "clear difference" if low > 0 or high < 0 else "could be noise"
    print(f"{a:<16} minus {b:<16} {gap:+.0%}   [{low:+.0%}, {high:+.0%}]   {verdict}")

# %%
names = list(RETRIEVERS)
plt.figure(figsize=(7.5, 4.5))
x = np.arange(len(names))
plt.bar(x - 0.27, [metrics(ranks[n])["hit@1"] * 100 for n in names], 0.27, label="Hit@1")
plt.bar(x, [metrics(ranks[n])["hit@3"] * 100 for n in names], 0.27, label="Hit@3")
plt.bar(x + 0.27, [metrics(ranks[n])["MRR"] * 100 for n in names], 0.27, label="MRR x100")
plt.xticks(x, names)
plt.ylim(0, 105)
plt.ylabel("Score (%)")
plt.title(f"Retrievers on {len(GOLDEN)} paraphrased questions")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "retrievers.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Evaluating answers
#
# Retrieval is easy to score because there is one right chunk. **Answers are free text**, and the
# same correct answer can be worded a hundred ways. We built 16 answers to handbook questions and
# **labeled each one by hand** as correct or incorrect. Half are correct but phrased differently
# from the handbook, and half are plausible but wrong. Now we can test how well each *scorer*
# agrees with a human.

# %%
# (question, reference answer, candidate answer, is_correct according to a human)
ANSWERS = [
    ("How many vacation days do new employees get?", "20 days", "Twenty days of paid vacation each year.", True),
    ("How many vacation days do new employees get?", "20 days", "New hires get 10 days of vacation.", False),
    ("What is the maximum hotel reimbursement per night?", "$180", "Hotels are covered up to 180 dollars a night.", True),
    ("What is the maximum hotel reimbursement per night?", "$180", "You can claim up to $250 per night for a hotel.", False),
    ("How many days a week can I work remotely?", "3 days", "Up to three days each week.", True),
    ("How many days a week can I work remotely?", "3 days", "Remote work is allowed 5 days per week.", False),
    ("How quickly must I report a lost device?", "within 1 hour", "Tell security within an hour of losing it.", True),
    ("How quickly must I report a lost device?", "within 1 hour", "Report it within 24 hours.", False),
    ("What is the minimum password length?", "14 characters", "Passwords need at least 14 characters.", True),
    ("What is the minimum password length?", "14 characters", "The minimum is 8 characters.", False),
    ("How big is the employee referral bonus?", "$2,000", "You receive two thousand dollars after 90 days.", True),
    ("How big is the employee referral bonus?", "$2,000", "The referral bonus is $500.", False),
    ("When is the first paycheck paid?", "the 25th of the month after you start", "On the 25th, the month after your start date.", True),
    ("When is the first paycheck paid?", "the 25th of the month after you start", "It arrives on the 1st of the month you start.", False),
    ("How long is parental leave for primary caregivers?", "16 weeks", "Sixteen weeks, fully paid.", True),
    ("How long is parental leave for primary caregivers?", "16 weeks", "Primary caregivers get 6 weeks.", False),
]
labels = [a[3] for a in ANSWERS]


def scorer_exact(question, reference, candidate):
    """Correct only if the reference text appears verbatim in the answer."""
    return reference.lower() in candidate.lower()


def token_f1(reference, candidate):
    ref, cand = tokenize(reference), tokenize(candidate)
    common = sum(min(ref.count(w), cand.count(w)) for w in set(ref))
    if not common:
        return 0.0
    precision, recall = common / len(cand), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def scorer_f1(question, reference, candidate):
    """Correct if the word overlap (F1) with the reference is at least 0.5."""
    return token_f1(reference, candidate) >= 0.5


def accuracy(scorer):
    predictions = [scorer(q, ref, cand) for q, ref, cand, _ in ANSWERS]
    return sum(p == l for p, l in zip(predictions, labels)) / len(labels), predictions


for name, scorer in [("exact string match", scorer_exact), ("token F1 >= 0.5", scorer_f1)]:
    acc, predictions = accuracy(scorer)
    missed = sum(l and not p for p, l in zip(predictions, labels))
    print(f"{name:<22}agrees with the human on {acc:.0%} of answers  ({missed} correct answers wrongly marked wrong)")

# %% [markdown]
# Exact match punishes correct answers that use different words ("Twenty days" is not "20 days").
# That is the main weakness of automatic string metrics, and why teams reach for an **LLM judge**.

# %% [markdown]
# ## 6. LLM-as-judge, and why you must test the judge
#
# An **LLM judge** is a model asked to grade another model's answer. It handles paraphrase, which
# string metrics cannot. But a judge is just another model: it can be biased, inconsistent, or wrong.
# So we never assume it works. We **measure its agreement with our human labels**, exactly like any
# other component.
#
# We ask the judge a yes/no question and read the model's probability of answering "Yes", which is
# more stable than parsing free text.

# %%
YES_ID = chat_tok.encode("Yes", add_special_tokens=False)[0]
NO_ID = chat_tok.encode("No", add_special_tokens=False)[0]


def judge_probability(question, reference, candidate):
    """Probability (0 to 1) that the judge says the candidate agrees with the reference."""
    prompt = (
        "You are grading an answer against a reference answer.\n"
        f"Question: {question}\nReference answer: {reference}\nCandidate answer: {candidate}\n"
        "Does the candidate give the same fact as the reference? Reply with only Yes or No."
    )
    text = chat_tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    with torch.no_grad():
        logits = chat_model(**chat_tok(text, return_tensors="pt")).logits[0, -1]
    return torch.softmax(logits[[YES_ID, NO_ID]], dim=0)[0].item()


probabilities = [judge_probability(q, ref, cand) for q, ref, cand, _ in ANSWERS]
judge_predictions = [p >= 0.5 for p in probabilities]
judge_accuracy = sum(p == l for p, l in zip(judge_predictions, labels)) / len(labels)

print(f"LLM judge (0.5B model) agrees with the human on {judge_accuracy:.0%} of answers\n")
print(f"{'Human label':<14}{'P(Yes)':<9}Candidate answer")
print("-" * 70)
for (q, ref, cand, label), p in zip(ANSWERS, probabilities):
    print(f"{'correct' if label else 'wrong':<14}{p:<9.2f}{cand}")

avg_correct = np.mean([p for p, l in zip(probabilities, labels) if l])
avg_wrong = np.mean([p for p, l in zip(probabilities, labels) if not l])
print(f"\nAverage P(Yes) on correct answers: {avg_correct:.2f}, on wrong answers: {avg_wrong:.2f}")

# %% [markdown]
# ### Can the judge at least *rank* better than chance?
#
# Even when a judge's yes/no threshold is off, its **scores** may still separate good from bad. The
# fraction of (correct, wrong) pairs where the correct answer gets the higher score is called the
# **AUC**: 1.0 is perfect separation and 0.5 is a coin flip.

# %%
correct_scores = [p for p, l in zip(probabilities, labels) if l]
wrong_scores = [p for p, l in zip(probabilities, labels) if not l]
auc = np.mean([(c > w) + 0.5 * (c == w) for c in correct_scores for w in wrong_scores])
print(f"Judge AUC: {auc:.2f}  (1.0 = perfect, 0.5 = coin flip)")

best_threshold = max(sorted(set(probabilities)), key=lambda t: sum((p >= t) == l for p, l in zip(probabilities, labels)))
tuned = sum((p >= best_threshold) == l for p, l in zip(probabilities, labels)) / len(labels)
print(f"With the best possible threshold ({best_threshold:.2f}) it would agree on {tuned:.0%}, but that number is")
print("optimistic: we picked the threshold using the same labels we are scoring against.")

# %% [markdown]
# ## 7. Judge biases to check for
#
# Research and practice have found consistent problems with LLM judges. You should test for them:
#
# | Bias | What it is | How to test |
# |---|---|---|
# | **Position bias** | Prefers the first (or last) option in a comparison | Swap the order and see if the verdict flips |
# | **Verbosity bias** | Prefers longer answers | Add padding to an answer and see if its score rises |
# | **Self-preference** | Favors text written by itself or its own model family | Judge with a different model family |
# | **Inconsistency** | Different verdicts on the same input | Re-run and compare |
#
# We test verbosity here: same wrong answers, padded with confident-sounding filler.

# %%
FILLER = " To elaborate, this reflects the standard policy that applies across the whole company, as confirmed in the handbook."
padded = [judge_probability(q, ref, cand + FILLER) for q, ref, cand, label in ANSWERS if not label]
plain = [p for p, l in zip(probabilities, labels) if not l]
print(f"Wrong answers, average P(Yes) as written:  {np.mean(plain):.2f}")
print(f"Wrong answers, average P(Yes) with filler: {np.mean(padded):.2f}")
flipped = sum((a >= 0.5) != (b >= 0.5) for a, b in zip(plain, padded))
print(f"Adding filler flipped the judge's verdict on {flipped} of {len(plain)} wrong answers.")

# %% [markdown]
# ## 8. Which evaluation should you use?
#
# | Method | Good for | Weakness |
# |---|---|---|
# | Exact or string match | Short factual answers, IDs, numbers | Fails on paraphrase |
# | Token F1 / ROUGE | Cheap overlap check | Rewards words, not meaning |
# | Retrieval metrics (Hit@k, MRR) | Retrieval quality, no LLM needed | Needs relevance labels |
# | LLM judge | Open-ended answers, tone, groundedness | Biased. Must be validated against humans. |
# | Human review | Ground truth | Slow and costly, so use it to check the others |
#
# ## 🎯 Key Takeaways
#
# - You cannot improve what you do not measure. Build a **golden set** first.
# - Score **retrieval and generation separately**. Retrieval metrics need no LLM.
# - With small test sets, check whether a difference is **real or noise** (bootstrap).
# - String metrics miss paraphrase. LLM judges handle it but are **biased and fallible**.
# - **Test your judge against human labels** before trusting its scores. A weak judge gives confident nonsense.
#
# ## 🚀 What's Next?
#
# **Module 07 — Tool Use & Agents**: instead of only answering, let the model take actions.

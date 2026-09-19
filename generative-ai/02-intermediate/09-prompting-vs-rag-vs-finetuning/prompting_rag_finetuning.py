# pip: transformers>=4.56 torch peft
# %% [markdown]
# # 📘 Module 09 — Prompting vs RAG vs Fine-Tuning
#
# **Generative AI Track • Intermediate**
#
# You now have three ways to change what a model does:
#
# | Approach | What you change | Cost to try |
# |---|---|---|
# | **Prompting** | The instructions and examples in the input | Minutes |
# | **RAG** | The knowledge pasted into the input | Hours to days |
# | **Fine-tuning** | The model's own weights | Days to weeks |
#
# "Which one should I use?" is probably the single most common intermediate GenAI interview
# question. The usual answers are slogans ("RAG for knowledge, fine-tuning for style"). Here we go
# further: we **run all of them on the same task** and measure what each one can and cannot do.
#
# **The experiment.** We teach a small model the fictional company handbook from Module 05 in three
# ways. Then we do what real companies do constantly: **change a policy**, and see who notices.

# %%
import re
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_ID = "Qwen/Qwen2.5-0.5B-Instruct"

embed_tok = AutoTokenizer.from_pretrained(EMBED_ID)
embed_model = AutoModel.from_pretrained(EMBED_ID).eval()
tok = AutoTokenizer.from_pretrained(CHAT_ID)
model = AutoModelForCausalLM.from_pretrained(CHAT_ID, dtype=torch.float32)


def embed(texts):
    batch = embed_tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        hidden = embed_model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1).numpy()


def generate(messages, max_new_tokens=60):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def prompt_tokens(messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return len(tok(text)["input_ids"])


# %% [markdown]
# ## 1. The knowledge and the questions
#
# Twelve facts from the handbook. Each has **three training questions** (used to fine-tune) and one
# **test question worded differently**, so we measure whether the model learned the *fact* and not
# just the sentences it saw.

# %%
FACTS = [
    dict(train=["How many vacation days do new employees get?", "What is the vacation allowance for new hires?", "How much paid vacation does a new employee receive per year?"],
         test="What's the time-off allowance for someone who just joined?",
         answer="New employees get 20 days of paid vacation per year.", accepted=["20 days", "20 paid", "twenty"]),
    dict(train=["How many unused vacation days can I carry over?", "What is the vacation carry-over limit?", "How many vacation days roll into next year?"],
         test="If I don't use all my holidays, how many roll over?",
         answer="Up to 5 unused vacation days carry over into the next year.", accepted=["5 unused", "5 vacation", "up to 5", "five"]),
    dict(train=["When do I need my manager's approval to spend money?", "What is the expense amount that requires manager approval?", "Do expenses need pre-approval?"],
         test="Do I need sign-off before buying something costing more than seventy-five dollars?",
         answer="Expenses over $75 need written approval from your manager.", accepted=["$75"]),
    dict(train=["What is the hotel reimbursement limit?", "How much will the company pay per night for a hotel?", "What is the maximum hotel cost while traveling?"],
         test="What's the nightly cap on accommodation when I travel?",
         answer="Hotels are reimbursed up to $180 per night.", accepted=["$180"]),
    dict(train=["How many days a week can I work remotely?", "What is the remote work limit?", "How often am I allowed to work from home?"],
         test="How many office-free days per week are allowed?",
         answer="Employees may work remotely up to 3 days per week.", accepted=["3 days", "three days", "up to 3"]),
    dict(train=["How much is the home office stipend?", "What do remote workers get to set up their office?", "Is there a stipend for a home office?"],
         test="What money do I get to set up my work-from-home space?",
         answer="New remote workers receive a one-time home office stipend of $500.", accepted=["$500"]),
    dict(train=["How quickly must I report a lost device?", "What should I do if I lose my laptop?", "What is the deadline to report a lost device?"],
         test="Who do I tell if my laptop goes missing, and how fast?",
         answer="Report a lost device to security within 1 hour.", accepted=["1 hour", "one hour", "an hour"]),
    dict(train=["What is the minimum password length?", "How long must passwords be?", "What are the password rules?"],
         test="How many characters must a password contain at minimum?",
         answer="Passwords must be at least 14 characters long.", accepted=["14"]),
    dict(train=["How much is the yearly learning budget?", "What is the annual training allowance?", "How much can I spend on courses each year?"],
         test="What's the annual amount for courses and books?",
         answer="Each employee gets a learning budget of $1,200 per year.", accepted=["$1,200", "1,200", "1200"]),
    dict(train=["How big is the employee referral bonus?", "What do I get for referring someone?", "What is the referral reward?"],
         test="How much do I earn for recommending a friend who gets hired?",
         answer="The employee referral bonus is $2,000.", accepted=["$2,000", "2,000", "2000"]),
    dict(train=["When is the first paycheck paid?", "On what day does my first salary arrive?", "What day of the month is the first paycheck?"],
         test="When does my initial salary payment land?",
         answer="Your first paycheck arrives on the 25th of the month after you start.", accepted=["25th", "25"]),
    dict(train=["How long is parental leave for primary caregivers?", "How many weeks of parental leave do primary caregivers get?", "What is the parental leave policy?"],
         test="How many weeks off does a new mother get?",
         answer="Parental leave is 16 weeks fully paid for primary caregivers.", accepted=["16 weeks", "16-week", "sixteen"]),
]
TRAIN = [(q, f["answer"]) for f in FACTS for q in f["train"]]
TEST = [(f["test"], f["accepted"]) for f in FACTS]
UNANSWERABLE = ["Who is the CEO of Northwind Labs?", "What is the company policy on bringing pets to the office?"]
GENERAL = [("What is the capital of France?", ["paris"]), ("What is 12 times 12?", ["144"]), ("What color do you get by mixing blue and yellow?", ["green"])]
print(f"{len(TRAIN)} training examples, {len(TEST)} test questions (worded differently from training)")


def is_correct(answer, accepted):
    return any(a.lower() in answer.lower() for a in accepted)


# %% [markdown]
# ## 2. Approach 1 and 2: prompting alone, and RAG
#
# For RAG we keep the handbook as short passages, embed them, and paste the best two or three into the
# prompt (the pipeline from Module 05). Small models are sensitive to the exact prompt wording, so we
# try four prompts and report all of them.

# %%
PASSAGES = [
    "New employees receive 20 days of paid vacation per year. Up to 5 unused vacation days carry over into the next year.",
    "Parental leave is 16 weeks fully paid for primary caregivers and 6 weeks fully paid for secondary caregivers.",
    "Expenses under $75 do not need pre-approval. Anything over $75 needs written approval from your manager.",
    "When traveling, meals are reimbursed up to $60 per day. Hotels are reimbursed up to $180 per night.",
    "Employees may work remotely up to 3 days per week. New remote workers receive a one-time home office stipend of $500.",
    "Passwords must be at least 14 characters long. Report a lost device to security within 1 hour.",
    "The company matches 401(k) contributions up to 4% of salary. Each employee gets a learning budget of $1,200 per year. The employee referral bonus is $2,000.",
    "The probation period lasts 90 days. Your first paycheck arrives on the 25th of the month after you start.",
]
passage_vectors = embed(PASSAGES)

NO_ANSWER = "I don't know."
PLAIN_SYSTEM = "You are a helpful assistant for employees of Northwind Labs. Answer briefly."

# RAG quality depends heavily on the prompt when the generator is a small model, so we try four.
RAG_PROMPTS = {
    "v1: example answer, 2 passages": (
        "You answer questions about company policy using ONLY the numbered sources provided.\n"
        "Write one short sentence that states the answer, then the source number in square brackets.\n"
        f"Example: Employees get 30 days of leave. [2]\nIf the sources do not contain the answer, write exactly: {NO_ANSWER}", 2),
    "v2: unrelated example, 3 passages": (
        "You answer questions about company policy using ONLY the numbered sources provided.\n"
        "Write one short sentence that states the answer, then the source number in square brackets.\n"
        f"Example: The office opens at 8:00. [1]\nIf the sources do not contain the answer, write exactly: {NO_ANSWER}", 3),
    "v3: no example, refuse only if absent, 3": (
        "Answer the question in one short sentence using only the numbered sources. "
        "Then add the source number like [1]. "
        f"Only write \"{NO_ANSWER}\" if none of the sources mention the answer.", 3),
    "v4: same as v3 but 2 passages": (
        "Answer the question in one short sentence using only the numbered sources. "
        "Then add the source number like [1]. "
        f"Only write \"{NO_ANSWER}\" if none of the sources mention the answer.", 2),
}
CHOSEN_PROMPT = "v3: no example, refuse only if absent, 3"


def plain_messages(question):
    return [{"role": "system", "content": PLAIN_SYSTEM}, {"role": "user", "content": question}]


def rag_messages(question, passages=None, vectors=None, prompt=None):
    passages = PASSAGES if passages is None else passages
    vectors = passage_vectors if vectors is None else vectors
    system, k = RAG_PROMPTS[prompt or CHOSEN_PROMPT]
    best = np.argsort(vectors @ embed([question])[0])[::-1][:k]
    sources = "\n".join(f"[{i}] {passages[j]}" for i, j in enumerate(best, 1))
    return [{"role": "system", "content": system}, {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"}]


def score(make_messages):
    answers = [generate(make_messages(q)) for q, _ in TEST]
    return sum(is_correct(a, acc) for a, (_, acc) in zip(answers, TEST)), answers


base_correct, _ = score(plain_messages)
print(f"Prompting only (no knowledge): {base_correct}/{len(TEST)} correct\n")

print(f"{'RAG prompt attempt':<44}{'Correct':<10}{'Said I don\'t know'}")
print("-" * 72)
rag_by_prompt, rag_answers = {}, {}
for name in RAG_PROMPTS:
    correct, answers = score(lambda q, name=name: rag_messages(q, prompt=name))
    rag_by_prompt[name] = correct
    rag_answers[name] = answers
    print(f"{name:<44}{correct}/{len(TEST):<8}{sum('don' in a.lower() and 'know' in a.lower() for a in answers)}")
rag_correct = rag_by_prompt[CHOSEN_PROMPT]
rag_flags = [is_correct(a, acc) for a, (_, acc) in zip(rag_answers[CHOSEN_PROMPT], TEST)]
print(f"\nWe use {CHOSEN_PROMPT!r} below. Note the spread: the retrieval was identical in every attempt.")

# %% [markdown]
# ## 3. Approach 3: fine-tuning
#
# Now we change the model's **weights**. We use **LoRA** (Week 3): instead of updating all 500
# million parameters, we train small adapter matrices, about 1.7% of the model. The training data is
# our 36 question-and-answer pairs. Loss is computed only on the **answer** tokens.

# %%
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
))
model.print_trainable_parameters()


def encode(question, answer):
    """Tokenize one training example, hiding the prompt from the loss so only the answer is learned."""
    prompt = tok.apply_chat_template(plain_messages(question), tokenize=False, add_generation_prompt=True)
    prompt_ids = tok(prompt)["input_ids"]
    answer_ids = tok(answer + "<|im_end|>")["input_ids"]
    return prompt_ids + answer_ids, [-100] * len(prompt_ids) + answer_ids


def train(pairs, epochs=8, batch_size=4, lr=3e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    rng = np.random.default_rng(0)
    losses = []
    model.train()
    start = time.time()
    for epoch in range(epochs):
        order = rng.permutation(len(pairs))
        for i in range(0, len(order), batch_size):
            batch = [encode(*pairs[j]) for j in order[i:i + batch_size]]
            width = max(len(ids) for ids, _ in batch)
            input_ids = torch.tensor([ids + [tok.pad_token_id] * (width - len(ids)) for ids, _ in batch])
            labels = torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in batch])
            attention = torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in batch])
            loss = model(input_ids=input_ids, attention_mask=attention, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        print(f"  epoch {epoch + 1}/{epochs}  loss {np.mean(losses[-(len(pairs) // batch_size):]):.3f}")
    model.eval()
    return time.time() - start


print("Fine-tuning:")
training_seconds = train(TRAIN)
print(f"Training took {training_seconds:.0f} seconds on this CPU.")

# %%
ft_correct, ft_answers = score(plain_messages)
ft_flags = [is_correct(a, acc) for a, (_, acc) in zip(ft_answers, TEST)]
ftrag_correct, _ = score(rag_messages)

print(f"{'Approach':<34}{'Correct on 12 reworded questions'}")
print("-" * 60)
print(f"{'Prompting only':<34}{base_correct}/{len(TEST)}")
print(f"{'RAG':<34}{rag_correct}/{len(TEST)}")
print(f"{'Fine-tuned':<34}{ft_correct}/{len(TEST)}")
print(f"{'Fine-tuned + RAG':<34}{ftrag_correct}/{len(TEST)}")

print("\nFine-tuned model, no context provided:")
for (question, accepted), answer in list(zip(TEST, ft_answers))[:4]:
    print(f"  Q: {question}\n  A: {answer}  {'(correct)' if is_correct(answer, accepted) else '(WRONG)'}")

# %% [markdown]
# ## 4. The test that matters: change a policy
#
# Real policies change. HR announces new numbers and we must update the system. To make the test
# fair, we change **only facts that both approaches answered correctly before the change** (RAG in
# Section 2, the fine-tuned model in Section 3). That way, any failure afterwards is about *the
# update*, not about the model never knowing the fact.
#
# We test three setups:
#
# - **RAG on the original model:** edit the passage and re-embed it. No training.
# - **Fine-tuned model, no context:** its knowledge is baked into the weights.
# - **Fine-tuned model plus RAG:** we hand it the *updated* passage. Does it believe the context or its memory?

# %%
# For each fact: the text to replace in the handbook passages, and what a correct answer looks like afterwards.
UPDATES = {
    0: ([("20 days", "25 days")], ["25 days", "25 paid", "twenty-five"]),
    1: ([("Up to 5 unused", "Up to 10 unused")], ["10 unused", "10 vacation", "up to 10", "ten"]),
    2: ([("$75", "$100")], ["$100"]),
    3: ([("$180", "$220")], ["$220"]),
    4: ([("up to 3 days", "up to 4 days")], ["4 days", "four days", "up to 4"]),
    5: ([("$500", "$600")], ["$600"]),
    7: ([("14 characters", "16 characters")], ["16"]),
    8: ([("$1,200", "$1,500")], ["$1,500", "1,500", "1500"]),
    9: ([("$2,000", "$2,500")], ["$2,500", "2,500", "2500"]),
}
eligible = [i for i in UPDATES if rag_flags[i] and ft_flags[i]][:5]
print(f"Facts both approaches got right before the change: {[i for i in range(len(FACTS)) if rag_flags[i] and ft_flags[i]]}")
print(f"We update {len(eligible)} of them:")

UPDATED = list(PASSAGES)
for i in eligible:
    for old, new in UPDATES[i][0]:
        UPDATED = [p.replace(old, new) for p in UPDATED]
    print(f"  {UPDATES[i][0][0][0]!r} -> {UPDATES[i][0][0][1]!r}   ({FACTS[i]['test']})")

start = time.time()
updated_vectors = embed(UPDATED)
reindex_seconds = time.time() - start

update_questions = [(FACTS[i]["test"], UPDATES[i][1]) for i in eligible]
with model.disable_adapter():  # temporarily switch fine-tuning off: this is the ORIGINAL model
    rag_updated = [generate(rag_messages(q, UPDATED, updated_vectors)) for q, _ in update_questions]
ft_updated = [generate(plain_messages(q)) for q, _ in update_questions]
ftrag_updated = [generate(rag_messages(q, UPDATED, updated_vectors)) for q, _ in update_questions]


def update_score(answers):
    return sum(is_correct(a, acc) for a, (_, acc) in zip(answers, update_questions))


print(f"\n{'Setup':<34}{'Picked up the change'}")
print("-" * 58)
for label, answers in [("Original model + RAG (updated)", rag_updated), ("Fine-tuned, no context", ft_updated), ("Fine-tuned + RAG (updated)", ftrag_updated)]:
    print(f"{label:<34}{update_score(answers)}/{len(update_questions)}")

print("\nWhat each setup said to the first updated question:")
print(f"  Q: {update_questions[0][0]}")
for label, answers in [("RAG (original model)", rag_updated), ("Fine-tuned", ft_updated), ("Fine-tuned + RAG", ftrag_updated)]:
    print(f"  {label:<22}{answers[0][:85]!r}")
print(f"\nCost to apply the update:  RAG {reindex_seconds:.2f}s (re-embed)  vs  fine-tuning ~{training_seconds:.0f}s (retrain), plus re-testing everything.")

# %% [markdown]
# **How to read this** (numbers from our run; yours should match, since decoding is greedy):
#
# - **RAG on the original model picked up 4 of 5 changes.** Re-embedding took a few hundredths of a
#   second. We did not investigate the one miss.
# - **The fine-tuned model picked up 0 of 5.** It kept confidently giving the old numbers, and it has
#   no way to know it is out of date. Learning the change means training again (about two minutes
#   here, hours or days for a big model), then re-testing everything.
# - **Fine-tuned plus the updated passage got 3 of 5.** That is one question fewer than RAG on the
#   original model, a gap far too small to conclude anything from five questions. It is a hint, not a
#   finding: a fine-tuned model holds a memorized answer that can compete with the context, so handing
#   it the new passage may not fix it the way it fixes a plain model. Testing that properly would
#   take many more questions.
#
# This is the core reason RAG, not fine-tuning, is the default for facts that change.

# %% [markdown]
# ## 5. Side effects
#
# Fine-tuning changes the whole model, so check what else moved. We look at two things: whether it
# still answers **general** questions, and whether it can admit **ignorance**.

# %%
def general_answers():
    return [generate([{"role": "user", "content": q}], 60) for q, _ in GENERAL]


tuned_general = general_answers()
with model.disable_adapter():
    original_general = general_answers()
    original_unknown = [generate(rag_messages(q), 40) for q in UNANSWERABLE]
tuned_unknown_plain = [generate(plain_messages(q), 40) for q in UNANSWERABLE]
tuned_unknown_rag = [generate(rag_messages(q), 40) for q in UNANSWERABLE]

print("General questions (3 only, so this is an anecdote, not a benchmark):")
for (question, accepted), a_original, a_tuned in zip(GENERAL, original_general, tuned_general):
    print(f"  Q: {question}")
    print(f"     original:   {'OK ' if is_correct(a_original, accepted) else 'BAD'} {a_original[:80]!r}")
    print(f"     fine-tuned: {'OK ' if is_correct(a_tuned, accepted) else 'BAD'} {a_tuned[:80]!r}")

print("\nQuestions the handbook cannot answer (the right behavior is to say it doesn't know):")
for question, a_orig, a_ft, a_ftrag in zip(UNANSWERABLE, original_unknown, tuned_unknown_plain, tuned_unknown_rag):
    print(f"  Q: {question}")
    print(f"     original + RAG:        {a_orig[:80]!r}")
    print(f"     fine-tuned, no context:{a_ft[:80]!r}")
    print(f"     fine-tuned + RAG:      {a_ftrag[:80]!r}")

# %% [markdown]
# **How to read this.**
#
# - **General ability:** with only three questions this is an anecdote, not a benchmark. Both models
#   got the mixing-colors question wrong, but the fine-tuned one answered with an unrelated sentence
#   about hair color. A proper check needs a real test set (Module 06), because fine-tuning can quietly
#   damage abilities you didn't train.
# - **Saying "I don't know":** the original model with RAG declined both unanswerable questions. The
#   fine-tuned model **invented an answer to both**, with or without the context. A likely reason is
#   that it was trained only on question-and-answer pairs where an answer always exists, so it never
#   saw "I don't know" as an option (we did not test that explanation directly). That is a serious
#   risk for anything user-facing.

# %% [markdown]
# ## 6. The cost of each approach
#
# RAG pastes context into **every** prompt, so it costs more input tokens per query. A fine-tuned
# model has a short prompt but a training bill and a maintenance burden. We measure the prompt sizes
# and turn them into an illustrative cost using an input price of **$2 per million tokens** (the
# order of magnitude from Module 03; check current prices).

# %%
question = TEST[0][0]
plain_tokens = prompt_tokens(plain_messages(question))
rag_tokens = prompt_tokens(rag_messages(question))
PRICE_PER_TOKEN = 2.00 / 1_000_000

print(f"{'':<26}{'Prompt tokens':<16}{'Input cost per 100,000 queries'}")
print("-" * 68)
print(f"{'Fine-tuned (short prompt)':<26}{plain_tokens:<16}${plain_tokens * 100_000 * PRICE_PER_TOKEN:,.2f}")
print(f"{'RAG (3 passages)':<26}{rag_tokens:<16}${rag_tokens * 100_000 * PRICE_PER_TOKEN:,.2f}")
print("\nRAG costs more per query. Fine-tuning costs more up front and every time the knowledge changes.")

# %% [markdown]
# ## 7. A decision framework
#
# Order matters: **always start with prompting**, because it is the cheapest to try and to change.
# Add RAG when the problem is *knowledge*. Add fine-tuning when the problem is *behavior* and
# prompting can't hold it.
#
# ```text
#   Does a good prompt (clear instructions + a few examples) solve it?  ──yes──► STOP. Use prompting.
#          │no
#          ▼
#   Is the gap missing, private, or changing KNOWLEDGE, or do you need citations?  ──yes──► add RAG
#          │
#          ▼
#   Is the gap BEHAVIOR (a style, a strict format, a specialized skill) that prompts
#   can't hold reliably, and do you have hundreds of good examples?  ──yes──► fine-tune
#          │
#          ▼
#   Still not enough? Combine them: RAG for facts + fine-tuning for behavior.
# ```

# %%
def recommend(knowledge_changes=False, needs_citations=False, private_or_large_knowledge=False,
              needs_consistent_style_or_format=False, prompting_holds_it=True, labeled_examples=0):
    """A teaching aid that encodes the decision flow above. Returns a list of approaches, in order."""
    plan = ["prompting"]
    if knowledge_changes or needs_citations or private_or_large_knowledge:
        plan.append("RAG")
    if needs_consistent_style_or_format and not prompting_holds_it:
        plan.append("fine-tuning" if labeled_examples >= 300 else "collect more examples, then fine-tune")
    return plan


SCENARIOS = [
    ("Support bot over help-center articles that change weekly", dict(knowledge_changes=True, needs_citations=True), ["prompting", "RAG"]),
    ("Extract fields from invoices into JSON", dict(needs_consistent_style_or_format=True, prompting_holds_it=True), ["prompting"]),
    ("Write every reply in a strict brand voice; prompts drift, we have 2,000 examples",
     dict(needs_consistent_style_or_format=True, prompting_holds_it=False, labeled_examples=2000), ["prompting", "fine-tuning"]),
    ("Q&A over 50,000 private legal documents with citations", dict(private_or_large_knowledge=True, needs_citations=True), ["prompting", "RAG"]),
    ("Legal assistant: cite the documents AND write in the firm's exact style (1,500 examples)",
     dict(needs_citations=True, needs_consistent_style_or_format=True, prompting_holds_it=False, labeled_examples=1500), ["prompting", "RAG", "fine-tuning"]),
    ("Strict format wanted, prompts fail, but only 40 examples",
     dict(needs_consistent_style_or_format=True, prompting_holds_it=False, labeled_examples=40), ["prompting", "collect more examples, then fine-tune"]),
]
for description, kwargs, expected in SCENARIOS:
    plan = recommend(**kwargs)
    assert plan == expected, (description, plan)
    print(f"{' -> '.join(plan):<62} | {description}")

# %% [markdown]
# ## 🎯 Key Takeaways
#
# - **Start with prompting.** It's the cheapest to try and the easiest to change.
# - **RAG fixes knowledge:** fresh, private, or large facts, with citations, updated in seconds.
# - **Fine-tuning fixes behavior:** style, strict formats, specialized skills. It is a poor way to store facts.
# - A fine-tuned model **keeps giving outdated answers** after a policy change (0 of 5 here), and adding the fresh passage recovered only 3 of 5 (a one-question gap versus RAG alone, so a hint, not proof). RAG picked up 4 of 5 in seconds.
# - Fine-tuning on facts taught the model to answer everything, so it **invented answers** to questions the documents can't answer. Test for side effects.
# - RAG quality depends on the **generator and the prompt** as much as on retrieval: the same retrieval scored 1, 1, 6 and 5 out of 12 across four prompts.
# - The approaches **combine**: RAG for the facts, fine-tuning for the behavior.
# - Decide with **measurements on your task**, not slogans.
#
# ## 🚀 What's Next?
#
# **Module 10 — Multimodal & Diffusion Models**: leaving text behind to see how models generate and understand images.

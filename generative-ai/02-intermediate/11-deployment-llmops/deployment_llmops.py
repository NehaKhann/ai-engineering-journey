# pip: transformers>=4.56 torch matplotlib
# %% [markdown]
# # 📘 Module 11 — Deployment & LLMOps
#
# **Generative AI Track • Intermediate**
#
# A notebook that works is not a product. Before real users depend on a model you must answer:
# **how fast is it, how many users can it serve at once, what does it cost, and how will I know when
# it breaks?** Doing this well is often called **LLMOps**.
#
# This module is a set of **measured experiments** on the real local model, each answering one of
# those questions:
#
# 1. **Latency:** where does the time actually go?
# 2. **Batching:** why do servers group requests?
# 3. **Caching:** how much can you save, and when does a cache serve a *wrong* answer?
# 4. **Quantization:** how much smaller and faster is a compressed model, and what does it cost in quality?
# 5. **Observability:** what should you log, and what do the numbers tell you?
#
# The numbers below come from a laptop CPU, so the **absolute values will differ on your machine or on
# a GPU**. What matters is the shape of each result. The companion project, the
# [Production LLM API](../../projects/04-production-llm-api/README.md), turns these ideas into a
# working service.

# %%
import copy
import time

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
CHAT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"

tok = AutoTokenizer.from_pretrained(CHAT_ID, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(CHAT_ID, dtype=torch.float32).eval()
embed_tok = AutoTokenizer.from_pretrained(EMBED_ID)
embed_model = AutoModel.from_pretrained(EMBED_ID).eval()


def embed(texts):
    batch = embed_tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        hidden = embed_model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1).numpy()


def chat_prompt(user_text):
    return tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)


def timed_generate(m, prompt_texts, new_tokens):
    """Run generation and return (seconds, generated token count per prompt)."""
    batch = tok(prompt_texts, return_tensors="pt", padding=True)
    start = time.perf_counter()
    with torch.no_grad():
        m.generate(**batch, min_new_tokens=new_tokens, max_new_tokens=new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return time.perf_counter() - start


timed_generate(model, [chat_prompt("Warm up.")], 4)  # the first call is slower, so warm up once

# %% [markdown]
# ## 1. Where does the time go?
#
# Generating a reply has two phases:
#
# - **Prefill:** the model reads the whole prompt in one pass. This decides the **time to first token**.
# - **Decode:** it then produces the reply **one token at a time**. Each token needs a full pass.
#
# We measure both, for prompts of different lengths (Module 05's RAG prompts are long).

# %%
FILLER = "The quarterly report covers sales, support costs, hiring plans, and the product roadmap. "


def prompt_of_length(n_tokens):
    text = FILLER * (n_tokens // 15 + 1)
    ids = tok(text)["input_ids"][:n_tokens]
    return chat_prompt(tok.decode(ids))


NEW = 32
print(f"{'Prompt tokens':<16}{'Prefill (time to 1st token)':<30}{'Per generated token':<22}{'Reply speed'}")
print("-" * 82)
latency_rows = []
for n in (50, 200, 800):
    prompt = prompt_of_length(n)
    prefill = np.median([timed_generate(model, [prompt], 1) for _ in range(3)])
    total = np.median([timed_generate(model, [prompt], NEW) for _ in range(3)])
    per_token = (total - prefill) / (NEW - 1)
    latency_rows.append((n, prefill, per_token))
    print(f"{n:<16}{prefill * 1000:>8.0f} ms{'':<18}{per_token * 1000:>6.0f} ms{'':<13}{1 / per_token:.1f} tokens/s")

# %% [markdown]
# **What to notice.** In our run, prefill grew from about 0.2 seconds (50 tokens) to about 1.9 seconds
# (800 tokens): a long prompt makes the user wait roughly nine times longer for the *first* word. The
# per-token generation speed barely moved (about 81 to 86 ms per token).
# This is why long RAG prompts and long tool descriptions (Modules 05 and 07) hurt **responsiveness**
# as well as cost, and why streaming (Module 03) helps: it hides the decode time but not the prefill.

# %% [markdown]
# ## 2. Batching: serving many users at once
#
# A GPU or CPU is underused when it processes one request at a time. **Batching** groups several
# requests into one pass. We generate the same 32 tokens for batches of 1, 2, 4, and 8 prompts.

# %%
prompt = chat_prompt("Explain in a few sentences why the sky is blue.")
print(f"{'Batch size':<14}{'Time for the batch':<22}{'Throughput':<20}{'Speedup vs batch of 1'}")
print("-" * 76)
base_throughput = None
batch_rows = []
for size in (1, 2, 4, 8):
    seconds = np.median([timed_generate(model, [prompt] * size, NEW) for _ in range(2)])
    throughput = size * NEW / seconds
    base_throughput = base_throughput or throughput
    batch_rows.append((size, seconds, throughput))
    print(f"{size:<14}{seconds:<22.2f}{throughput:<20.1f}{throughput / base_throughput:.1f}x")

# %% [markdown]
# **What to notice.** Throughput (total tokens per second across all users) rose from about 12 to 68
# as the batch grew from 1 to 8, a 5.7x gain, because the hardware does more useful work per pass.
# The time for the *batch* rose too, from 2.7 to 3.8 seconds, so each user waited about 1.4x longer
# than they would alone. That is the central serving trade-off: **throughput versus per-request
# latency**. Here the trade was very favorable, and it gets less so as batches grow larger. Production servers such as vLLM use *continuous
# batching*, adding new requests to a running batch as others finish, to get most of the throughput
# without making everyone wait.

# %% [markdown]
# ## 3. Caching
#
# The cheapest request is the one you never send to the model. Two kinds of cache:
#
# - **Exact cache:** same input text, same answer
# - **Semantic cache:** a *similar* question reuses an earlier answer (using embeddings, Module 04)
#
# ### Exact cache
#
# We simulate 300 requests over 40 distinct questions where a few are asked much more than others
# (real traffic is like this), served through a small least-recently-used (LRU) cache.

# %%
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.capacity, self.items = capacity, OrderedDict()

    def get(self, key):
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        return None

    def put(self, key, value):
        self.items[key] = value
        self.items.move_to_end(key)
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)


rng = np.random.default_rng(0)
weights = 1 / np.arange(1, 41)  # question #1 is asked most, #40 least
weights /= weights.sum()
traffic = rng.choice(40, size=300, p=weights)

seconds_per_uncached_request = latency_rows[0][1] + NEW * latency_rows[0][2]  # measured above
print(f"Cost of one uncached request (from Section 1): {seconds_per_uncached_request:.1f} s\n")
print(f"{'Cache size':<14}{'Hit rate':<12}{'Model time saved'}")
print("-" * 46)
for capacity in (0, 5, 10, 20, 40):
    cache, hits = LRUCache(max(capacity, 1)), 0
    for question in traffic:
        if capacity and cache.get(int(question)) is not None:
            hits += 1
        else:
            cache.put(int(question), "answer")
    print(f"{capacity:<14}{hits / len(traffic):<12.0%}{hits * seconds_per_uncached_request / 60:.1f} minutes of {len(traffic) * seconds_per_uncached_request / 60:.1f}")

# %% [markdown]
# Because a few popular questions dominate, a cache of 5 entries served 29% of requests, 10 entries
# served 48%, and 40 entries served 88%. **Caching works best when questions repeat.** It works badly, or should not be used,
# when each request is unique, personalized, or needs fresh data.
#
# ### Semantic cache, and how it can serve the wrong answer
#
# A semantic cache also catches *rephrased* questions, which is attractive. But it decides by
# similarity, and Module 04 showed similarity finds *related* text, not *identical meaning*. We test
# it: paraphrases **should** hit the cache, while different questions on the same topic **must not**,
# because they need a different answer.

# %%
CACHED_QUESTIONS = [
    "How many vacation days do new employees get?",
    "What is the maximum hotel reimbursement per night?",
    "How long must a password be?",
    "How many days a week can I work remotely?",
    "How big is the employee referral bonus?",
    "When is the first paycheck paid?",
]
SHOULD_HIT = [  # paraphrases of the cached questions: reusing the answer is correct
    "What's the time-off allowance for a new hire?",
    "What is the nightly cap on hotels when traveling?",
    "What is the minimum number of characters in a password?",
    "How many office-free days per week are allowed?",
    "How much do I get for referring a friend?",
    "When does my initial salary payment arrive?",
]
MUST_MISS = [  # same topic, DIFFERENT question: reusing the cached answer would be wrong
    "How many sick days do new employees get?",
    "What is the maximum meal reimbursement per day?",
    "How often must a password be changed?",
    "How many days a week must I be in the office?",
    "How long is the referral bonus paid after hiring?",
    "When is the final paycheck paid?",
]
cached_vectors = embed(CACHED_QUESTIONS)


def best_similarity(question):
    return float((cached_vectors @ embed([question])[0]).max())


hit_scores = [best_similarity(q) for q in SHOULD_HIT]
miss_scores = [best_similarity(q) for q in MUST_MISS]

print(f"{'Similarity threshold':<24}{'Paraphrases served from cache':<32}{'WRONG answers served'}")
print("-" * 82)
for threshold in (0.6, 0.7, 0.8, 0.9):
    good = sum(s >= threshold for s in hit_scores)
    bad = sum(s >= threshold for s in miss_scores)
    print(f"{threshold:<24}{good}/{len(SHOULD_HIT):<30}{bad}/{len(MUST_MISS)}")

print("\nHighest-scoring near-misses (different questions the cache would wrongly treat as the same):")
for score, q in sorted(zip(miss_scores, MUST_MISS), reverse=True)[:3]:
    print(f"   {score:.2f}  {q}")

# %% [markdown]
# **This is the trap, and here it was severe.** At every threshold where the cache served anything,
# it served **more wrong answers than right ones**: at 0.6 it answered 4 of 6 paraphrases but also
# gave a wrong cached answer to all 6 near-miss questions, and by the time the threshold was strict
# enough to stop the wrong hits (0.9), it served no paraphrases either. "How many **sick** days?" is
# genuinely close to "How many **vacation** days?" in embedding space. (This is a small test with 6
# examples per group and a small embedding model, so treat it as a demonstration, not a benchmark.)
# A stricter threshold is safer but saves less. Use a semantic cache only where a slightly wrong reused answer
# is acceptable, keep the threshold high, and consider verifying a hit. For facts that matter, an
# exact cache is the safe choice.

# %% [markdown]
# ## 4. Quantization: a smaller, faster model
#
# **Quantization** (Week 3) stores the model's weights with fewer bits. We apply **dynamic int8
# quantization** to the model's linear layers and measure size, speed, and quality.

# %%
def model_megabytes(m):
    return sum(p.numel() * p.element_size() for p in m.parameters()) / 1e6


def text_loss(m, text):
    ids = tok(text, return_tensors="pt")
    with torch.no_grad():
        return m(**ids, labels=ids["input_ids"]).loss.item()


SAMPLE_TEXT = "Passwords must be at least fourteen characters long, and multi-factor authentication is required on every account."

try:
    quantized = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8)
    quantized_ok = True
except Exception as error:  # quantization backends differ by platform
    quantized_ok = False
    print("Dynamic quantization is not available on this machine:", type(error).__name__)

if quantized_ok:
    plain_prompt = chat_prompt("Explain in a few sentences why the sky is blue.")
    fp32_time = np.median([timed_generate(model, [plain_prompt], NEW) for _ in range(2)])
    int8_time = np.median([timed_generate(quantized, [plain_prompt], NEW) for _ in range(2)])
    fp32_size = model_megabytes(model)
    int8_size = sum(b.numel() * b.element_size() for b in quantized.state_dict().values() if isinstance(b, torch.Tensor)) / 1e6

    print(f"{'':<26}{'Size':<14}{'Time for 32 tokens':<22}{'Loss on a sample sentence (lower is better)'}")
    print("-" * 92)
    print(f"{'float32 (original)':<26}{fp32_size:<8.0f}MB{'':<4}{fp32_time:<22.2f}{text_loss(model, SAMPLE_TEXT):.3f}")
    print(f"{'int8 (dynamic quantization)':<26}{int8_size:<8.0f}MB{'':<4}{int8_time:<22.2f}{text_loss(quantized, SAMPLE_TEXT):.3f}")

    print("\nSame question, both models:")
    for label, m in [("float32", model), ("int8   ", quantized)]:
        batch = tok([plain_prompt], return_tensors="pt")
        with torch.no_grad():
            out = m.generate(**batch, max_new_tokens=25, do_sample=False, pad_token_id=tok.eos_token_id)
        print(f"  {label}: {tok.decode(out[0][batch['input_ids'].shape[1]:], skip_special_tokens=True)!r}")

# %% [markdown]
# Read the result honestly. The model became about 3.6x smaller (1,976 MB to 545 MB) and about 1.7x
# faster, but **quality dropped a lot**: the loss on a sample sentence rose from 2.8 to 4.7, and the
# reply drifted off-topic. This crude method, applied to a small 0.5-billion-parameter model, was too
# lossy for real use. Small models tend to suffer more from quantization than large ones, and more
# careful methods (Week 3) lose much less, so this is *not* a verdict on quantization in general. The
# lesson is to **always measure quality on your own task** after quantizing, as in Module 06, and
# never assume a compressed model is "the same".
#
# (PyTorch prints a notice that `torch.ao.quantization.quantize_dynamic` is deprecated in favor of
# the `torchao` library. It still works here, and the idea is unchanged.)

# %% [markdown]
# ## 5. Observability: log every request
#
# You cannot fix what you cannot see. Every request should leave a structured record. We wrap the
# model in a small function that logs one line per request, run a mix of repeated and new questions,
# and compute the numbers a dashboard would show.

# %%
import uuid

PRICE_IN, PRICE_OUT = 2.00 / 1e6, 10.00 / 1e6  # illustrative $ per token; use your real prices
LOG, response_cache = [], {}


def serve(question, max_new_tokens=20):
    start = time.perf_counter()
    cache_hit = question in response_cache
    if cache_hit:
        answer, prompt_tokens, completion_tokens = response_cache[question]
    else:
        batch = tok([chat_prompt(question)], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        generated = out[0][batch["input_ids"].shape[1]:]
        answer = tok.decode(generated, skip_special_tokens=True)
        prompt_tokens, completion_tokens = batch["input_ids"].shape[1], len(generated)
        response_cache[question] = (answer, prompt_tokens, completion_tokens)
    LOG.append({
        "request_id": uuid.uuid4().hex[:8], "latency_s": time.perf_counter() - start, "cache_hit": cache_hit,
        "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "cost_usd": 0.0 if cache_hit else prompt_tokens * PRICE_IN + completion_tokens * PRICE_OUT,
    })
    return answer


questions = ["What is a token?", "What is an embedding?", "What is RAG?", "What is a token?", "What is RAG?",
             "What is prompt injection?", "What is a token?", "What is an embedding?", "What is streaming?", "What is RAG?"]
for q in questions:
    serve(q)

latencies = np.array([r["latency_s"] for r in LOG])
print(f"Requests:            {len(LOG)}")
print(f"Cache hit rate:      {np.mean([r['cache_hit'] for r in LOG]):.0%}")
print(f"Latency p50 / p95:   {np.percentile(latencies, 50) * 1000:.0f} ms / {np.percentile(latencies, 95) * 1000:.0f} ms")
print(f"Cache hit latency:   {np.mean([r['latency_s'] for r in LOG if r['cache_hit']]) * 1000:.2f} ms   "
      f"vs miss {np.mean([r['latency_s'] for r in LOG if not r['cache_hit']]) * 1000:.0f} ms")
print(f"Tokens billed:       {sum(r['prompt_tokens'] + r['completion_tokens'] for r in LOG if not r['cache_hit'])}")
print(f"Estimated cost:      ${sum(r['cost_usd'] for r in LOG):.5f}")
print("\nOne log record:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in LOG[0].items()})

# %% [markdown]
# **Why percentiles?** The **average** hides problems. Here the p95 latency (what the slowest 1 in 20
# users experience) is far above the median, because a cache hit takes microseconds and a miss takes
# seconds. Alert on **p95 or p99**, not the mean.
#
# ### What to monitor in production
#
# | Signal | Why | Example alert |
# |---|---|---|
# | Latency p50 / p95 / p99 | User experience | p95 above 5 s |
# | Error and timeout rate | Reliability | Errors above 1% |
# | Tokens and cost per request | Budget, and spotting prompt bloat | Cost per day above budget |
# | Cache hit rate | Whether caching helps | Sudden drop |
# | Refusal and fallback rate | Model or guardrail behavior changed | Spike after a deploy |
# | Quality score on a fixed test set (Module 06) | Silent regressions | Drop after a prompt or model change |
# | User feedback (thumbs up or down) | Real-world quality | Falling trend |
#
# ## 🎯 Key Takeaways
#
# - Latency has two parts: **prefill** (grows with prompt length, sets time to first token) and **decode** (per token).
# - **Batching** raises total throughput but makes each request wait longer. That trade-off shapes every serving stack.
# - **Exact caching** is safe and effective when questions repeat. **Semantic caching** can serve wrong answers, so keep the threshold high and use it where errors are tolerable.
# - **Quantization** shrinks and speeds up a model at some quality cost. Measure the quality on your task.
# - Log **every request** with latency, tokens, cost, and cache status, and watch **percentiles**, not averages.
# - Everything here is measured on one CPU. Re-measure on your own hardware before making capacity decisions.
#
# ## 🚀 What's Next?
#
# **Project: Production LLM API**: put these ideas into a real FastAPI service with streaming, caching, rate limiting, and logging.

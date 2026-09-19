# 📘 Module 11 — Deployment & LLMOps

**Generative AI Track • Intermediate**

A notebook that works is not a product. Before real users depend on a model you have to answer four questions: **how fast is it, how many users can it serve at once, what does it cost, and how will I know when it breaks?** Doing this well is often called **LLMOps**.

This module is a set of **measured experiments** on the real local model, each answering one of those questions. The companion project, the [Production LLM API](../../projects/04-production-llm-api/README.md), turns the ideas into a working service.

> **A caution on the numbers:** they come from one laptop CPU, so the **absolute values will differ on your machine and on a GPU**. What matters is the *shape* of each result. Re-measure on your own hardware before making capacity decisions.

---

## 🎯 Objective

In this module, you'll:

- Split latency into **prefill** and **decode** and see which one long prompts hurt
- Measure how **batching** trades throughput against per-request latency
- Simulate **caching** on realistic traffic, and see how a **semantic cache serves wrong answers**
- Apply **quantization** and measure size, speed, and quality together
- Build request **logging** and read the numbers a dashboard would show

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `deployment_llmops.py` | All five experiments. |
| `deployment_llmops.ipynb` | Interactive notebook version, generated from the script. |
| `README.md` | Concepts, real results, and interview Q&A. |

---

## ⚙️ Setup

```powershell
# From repository root
venv\Scripts\activate

cd generative-ai\02-intermediate\11-deployment-llmops

pip install "transformers>=4.56" torch
```

## ▶️ Run

```powershell
python deployment_llmops.py
```

Runs on a laptop CPU with the local `Qwen2.5-0.5B-Instruct` model. No API key needed. It generates a few hundred short replies, so it takes several minutes.

---

## 🧠 Key Concepts

### 1. Where Does the Time Go?

Generating a reply has two phases:

- **Prefill:** the model reads the whole prompt in one pass. This sets the **time to first token**.
- **Decode:** it then produces the reply **one token at a time**.

| Prompt length | Prefill (time to first token) | Time per generated token |
|---|---|---|
| 50 tokens | 217 ms | 81 ms |
| 200 tokens | 490 ms | 82 ms |
| 800 tokens | **1,898 ms** | 86 ms |

A long prompt made the user wait about nine times longer for the **first word**, while the speed of the words after that barely moved. This is why long RAG prompts (Module 05) and long tool descriptions (Module 07) hurt **responsiveness** as well as cost, and why streaming (Module 03) helps: it hides the decode time but not the prefill.

---

### 2. Batching: Serving Many Users at Once

A processor is underused when it handles one request at a time. **Batching** groups requests into one pass. We generated 32 tokens for batches of 1, 2, 4, and 8 identical prompts:

| Batch size | Time for the batch | Throughput (tokens/s, all users) | Speedup |
|---|---|---|---|
| 1 | 2.65 s | 12.1 | 1.0x |
| 2 | 3.00 s | 21.3 | 1.8x |
| 4 | 3.34 s | 38.3 | 3.2x |
| 8 | 3.75 s | 68.3 | **5.7x** |

Total throughput rose 5.7x, but each user waited about **1.4x longer** than they would alone (2.65 s to 3.75 s). That is the central serving trade-off: **throughput versus per-request latency**. Here the trade was very favorable, and it becomes less so as batches grow. Production servers such as vLLM use *continuous batching*, adding new requests to a running batch as others finish, to get most of the gain without making everyone wait for the slowest.

---

### 3. Caching

The cheapest request is the one you never send to the model.

**Exact cache.** We simulated 300 requests over 40 distinct questions, where a few questions are asked far more than others (real traffic looks like this), through a least-recently-used (LRU) cache. One uncached request costs about 2.8 seconds of model time:

| Cache size | Hit rate | Model time saved |
|---|---|---|
| 0 | 0% | 0 of 14.1 minutes |
| 5 | 29% | 4.1 minutes |
| 10 | 48% | 6.7 minutes |
| 20 | 68% | 9.5 minutes |
| 40 | 88% | 12.3 minutes |

Caching works best when questions **repeat**. It works badly, and should be skipped, when requests are unique, personalized, or need fresh data.

**Semantic cache: the trap.** A semantic cache also catches *rephrased* questions by comparing embeddings (Module 04). But similarity finds *related* text, not *identical meaning*. We tested 6 paraphrases (which **should** hit the cache) and 6 different questions on the same topics (which **must not**, because they need a different answer):

| Similarity threshold | Paraphrases served from cache (good) | **Wrong answers served (bad)** |
|---|---|---|
| 0.6 | 4 / 6 | **6 / 6** |
| 0.7 | 2 / 6 | **4 / 6** |
| 0.8 | 0 / 6 | **3 / 6** |
| 0.9 | 0 / 6 | 0 / 6 |

At every threshold where the cache served anything, it served **more wrong answers than right ones**. The near-misses were dangerous: *"When is the final paycheck paid?"* scored 0.85 against the cached *"When is the first paycheck paid?"*, and *"How many sick days do new employees get?"* scored 0.81 against the vacation question. By the time the threshold was strict enough to stop the wrong hits, it also served no paraphrases.

Six examples per group and a small embedding model make this a demonstration, not a benchmark. But the lesson is solid: use a semantic cache only where a slightly wrong reused answer is tolerable, keep the threshold high, and prefer an **exact cache** for anything that matters.

---

### 4. Quantization: Smaller and Faster, at What Cost?

**Quantization** (Week 3) stores weights with fewer bits. We applied dynamic int8 quantization to the linear layers:

| | Size | Time for 32 tokens | Loss on a sample sentence (lower is better) |
|---|---|---|---|
| float32 (original) | 1,976 MB | 2.71 s | 2.81 |
| int8 (dynamic quantization) | 545 MB | 1.62 s | **4.68** |

The model became **3.6x smaller and 1.7x faster**, but **quality dropped a lot**: the loss rose from 2.8 to 4.7. Asked *"why is the sky blue?"*, the original gave a sensible answer while the quantized model drifted off-topic (*"To answer this question, we need to understand the context of Qwen..."*).

Read this carefully. This crude method, applied to a small 0.5-billion-parameter model, was too lossy for real use. Small models tend to suffer more from quantization than large ones, and more careful methods (Week 3) lose much less. So this is **not a verdict on quantization**. It is a demonstration that you must **measure quality on your own task** after quantizing, and never assume a compressed model is "the same". (PyTorch also notes that this particular function is deprecated in favor of the `torchao` library; it still works, and the idea is unchanged.)

---

### 5. Observability: Log Every Request

You cannot fix what you cannot see. Every request should leave a structured record. We ran 10 requests (some repeated) through a small logging wrapper:

| Signal | Value |
|---|---|
| Requests | 10 |
| Cache hit rate | 50% |
| Latency p50 / p95 | **814 ms / 1,662 ms** |
| Cache-hit latency | 0.00 ms |
| Cache-miss latency | 1,646 ms |
| Tokens billed (misses only) | 269 |
| Estimated cost | $0.00134 |

**Why percentiles?** The average would hide the split: a cache hit takes microseconds and a miss takes 1.6 seconds, so the p95 (what the slowest 1 in 20 users experience) is double the median. **Alert on p95 or p99, not the mean.**

**What to monitor in production:**

| Signal | Why | Example alert |
|---|---|---|
| Latency p50 / p95 / p99 | User experience | p95 above 5 s |
| Error and timeout rate | Reliability | Errors above 1% |
| Tokens and cost per request | Budget, and spotting prompt bloat | Daily cost above budget |
| Cache hit rate | Whether caching helps | Sudden drop |
| Refusal and fallback rate | Model or guardrail behavior changed | Spike after a deploy |
| Quality score on a fixed test set (Module 06) | Silent regressions | Drop after a prompt or model change |
| User feedback (thumbs up or down) | Real-world quality | Falling trend |

---

## 🎤 Interview Q&A

Try answering each question out loud before opening the answer.

<details>
<summary><b>Q1. What are prefill and decode, and why does it matter?</b></summary>

**Short answer:** Prefill processes the whole prompt in one pass and determines the time to first token. Decode generates the reply one token at a time and determines how fast text appears afterwards.

**Deeper answer:** Prefill cost grows with prompt length, decode cost with reply length. In our measurements an 800-token prompt took about 1.9 s to first token versus 0.2 s for 50 tokens, while per-token speed stayed near 80 ms. So long prompts (RAG context, many tool descriptions, long history) mostly hurt time to first token, while long replies hurt total time. Streaming hides decode time but not prefill. Prompt caching (Module 03) targets prefill by reusing work on a repeated prefix.

**Follow-ups to expect:**
- Which would you optimize for a chat app? *Time to first token, because users judge responsiveness by when text starts appearing.*

</details>

<details>
<summary><b>Q2. What is batching, and what is the trade-off?</b></summary>

**Short answer:** Batching processes several requests in one model pass. It raises total throughput but makes each individual request wait longer.

**Deeper answer:** In our test, throughput rose 5.7x at batch 8 while each request took about 1.4x longer. Naive static batching makes every request wait for the slowest one in its batch. **Continuous batching** (used by servers like vLLM) adds new requests to a running batch and removes finished ones, keeping the hardware busy without that waiting. The right batch size depends on your latency target: interactive apps favor smaller batches, offline jobs favor large ones.

**Follow-ups to expect:**
- What limits batch size? *Memory: each request needs its own key-value cache, so long prompts and long replies reduce how many fit.*

</details>

<details>
<summary><b>Q3. How would you cache LLM responses, and what can go wrong?</b></summary>

**Short answer:** Use an exact-match cache with a time limit for repeated requests. Be careful with semantic caching, which can return the wrong answer for a similar-but-different question.

**Deeper answer:** An exact cache is safe and effective when traffic repeats: in our simulation, 10 entries served 48% of requests and 40 entries served 88%. Include the user or tenant in the key when responses could depend on private data, add a TTL so answers don't go stale, and skip caching for personalized or time-sensitive requests. A semantic cache matches by embedding similarity, and in our test it served more wrong answers than right ones at every threshold that let anything through, because "sick days" and "vacation days" are close in meaning space. If you use one, keep the threshold high, restrict it to low-stakes queries, and measure wrong hits.

**Follow-ups to expect:**
- What about caching the model's internal state? *Prompt caching reuses the computed prefix of a repeated prompt, which cuts cost and prefill latency without changing the answer.*

</details>

<details>
<summary><b>Q4. What is quantization, and what are its trade-offs?</b></summary>

**Short answer:** Storing model weights in fewer bits (for example int8 or 4-bit instead of float32) so the model is smaller and often faster, at some cost in quality.

**Deeper answer:** We measured a 3.6x size reduction (1,976 MB to 545 MB) and a 1.7x speedup, but a large quality loss on a small model with a simple method (loss 2.8 to 4.7, and off-topic output). Better methods (GPTQ, AWQ, and 4-bit schemes like NF4 used in QLoRA) preserve much more quality, and larger models tolerate it better. Always evaluate the quantized model on your own task, not just perplexity, and check memory and speed on your target hardware, since gains depend on kernel support.

**Follow-ups to expect:**
- Quantization vs distillation? *Quantization compresses the same model. Distillation trains a smaller model to imitate a bigger one.*

</details>

<details>
<summary><b>Q5. What would you monitor for an LLM service in production?</b></summary>

**Short answer:** Latency percentiles, error rate, cost per request, cache hit rate, and a quality signal, with alerts on the tail rather than the average.

**Deeper answer:** Track p95 and p99 latency (the average hid a 1.6-second miss behind 0-millisecond hits in our log), errors and timeouts, tokens and dollars per request to catch prompt bloat, cache hit rate, and refusal or fallback rates that shift after a change. For quality, keep a fixed evaluation set (Module 06) and re-run it after every prompt or model change, and collect user feedback. Log a request ID, and never log secrets or raw sensitive data.

**Follow-ups to expect:**
- How do you catch a silent quality regression? *A scheduled evaluation on a fixed test set, plus sampling and reviewing real outputs.*

</details>

<details>
<summary><b>Q6. How would you decide between an API and hosting a model yourself?</b></summary>

**Short answer:** Compare the total cost at your volume, plus quality, latency, privacy, and how much operations work you can take on. APIs win at low or spiky volume and for top quality. Self-hosting wins for privacy needs or steady high volume.

**Deeper answer:** Self-hosting turns a per-token bill into a fixed hardware bill, which pays off only if the hardware stays busy. Batching, caching, and quantization (all measured above) raise how much traffic one machine handles, but you take on serving, scaling, monitoring, and updates. Start with an API, measure real usage, and only then compute a break-even with your own numbers. Include engineering time, since it often dominates.

**Follow-ups to expect:**
- What is a sensible first step toward self-hosting? *Serve a small open model behind the same interface as the API (as the Production LLM API project does), so you can compare quality and cost before committing.*

</details>

---

## 🎯 Key Takeaways

After completing this module, you'll understand:

- That latency has two parts, and long prompts hurt **time to first token** (prefill)
- How batching trades throughput against per-request latency, and why servers use continuous batching
- That exact caching is safe and effective, while **semantic caching can serve wrong answers**
- That quantization shrinks and speeds up a model at a quality cost you must measure
- Why to log every request and watch **percentiles**, not averages
- That all of these numbers are hardware-specific and must be re-measured

---

## 🔗 Go Deeper in the Weekly Curriculum

| Topic | Where |
|---|---|
| Quantization in detail | [Week 3, Day 1](../../../weeks/week-03-efficient-fine-tuning/day-01-quantization-basics/README.md) |
| Tokens, streaming, and cost | [Module 03](../../01-beginner/03-llm-apis/README.md) |
| Local LLMs, vLLM, and serving | Week 7 — Local LLMs & Production Deployment *(planned)* |

---

## 🚀 What's Next?

**Project 04 • Production LLM API**

Put these ideas into a real FastAPI service with authentication, rate limiting, caching, streaming, and metrics.

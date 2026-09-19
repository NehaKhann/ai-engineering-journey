# 🚀 Project 04 — Production LLM API

**Generative AI Track • Intermediate Project** (uses Modules 03 and 11)

A small but realistic web service that puts a language model behind an HTTP API, with the things a demo leaves out: **authentication, input validation, rate limiting, caching, streaming, metrics, and safe error handling**.

It runs the same local model used throughout this track, so no API key or GPU is needed. Every behavior is covered by tests that use a fake model and a fake clock, so the whole suite runs in under a second.

> **The point of this project:** the model is the easy part. Most of what makes an LLM service *production-ready* is the ordinary engineering around it, and each piece here maps to something you will be asked about in an interview.

---

## 🎯 What You'll Practice

| Skill | Where |
|---|---|
| Serving a model over HTTP with FastAPI, including **streaming** | `app/server.py`, `app/backends.py` |
| **Authentication** that doesn't leak timing or log secrets | `authenticate()` in `app/server.py` |
| **Rate limiting** with a token bucket | `app/ratelimit.py` |
| **Caching** with LRU eviction, expiry, and per-customer isolation | `app/cache.py` |
| **Input validation** and cost control (prompt and output caps) | `ChatRequest`, `prepare()` |
| **Error handling** that never exposes internals | the 502 path |
| **Observability**: request IDs, structured logs, latency percentiles, cost | `app/metrics.py`, the middleware |
| **Testing** time-dependent code without sleeping | injected clocks in `tests/test_api.py` |

---

## 📂 Project Structure

```text
04-production-llm-api/
├── run.py                # starts the server (refuses to start without API keys)
├── app/
│   ├── config.py         # settings from environment variables
│   ├── backends.py       # the model: LocalBackend (real) and FakeBackend (tests)
│   ├── cache.py          # TTL + LRU cache
│   ├── ratelimit.py      # token-bucket limiter, one bucket per API key
│   ├── metrics.py        # counters, latency percentiles, estimated cost
│   └── server.py         # the FastAPI app: routes, auth, logging middleware
├── tests/test_api.py     # 19 tests, no model needed
└── smoke_test.py         # measures the real model through the real API
```

---

## ⚙️ Setup

```powershell
# From repository root
venv\Scripts\activate

cd generative-ai\projects\04-production-llm-api

pip install "transformers>=4.56" torch fastapi uvicorn httpx
```

## ▶️ Run the Server

The server **refuses to start without API keys**. A service that quietly starts open to everyone because a setting was forgotten is a classic incident, so it fails closed.

```powershell
$env:LLM_API_KEYS = "my-secret-key"
python run.py
```

Then, in another terminal:

```powershell
# Health check (no key needed)
curl http://127.0.0.1:8000/health

# A chat request
curl -X POST http://127.0.0.1:8000/v1/chat `
  -H "X-API-Key: my-secret-key" -H "Content-Type: application/json" `
  -d '{"messages": [{"role": "user", "content": "What is a token?"}]}'

# Streaming: tokens arrive as they are generated
curl -N -X POST http://127.0.0.1:8000/v1/chat/stream `
  -H "X-API-Key: my-secret-key" -H "Content-Type: application/json" `
  -d '{"messages": [{"role": "user", "content": "Explain what an API is."}]}'

# Metrics
curl http://127.0.0.1:8000/metrics -H "X-API-Key: my-secret-key"
```

## 🧪 Run the Tests

```powershell
python -m unittest discover -s tests -v
```

---

## 📖 API Reference

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Is the service up? |
| `POST /v1/chat` | `X-API-Key` | Full reply as JSON. Cached. |
| `POST /v1/chat/stream` | `X-API-Key` | Reply as server-sent events, token by token |
| `GET /metrics` | `X-API-Key` | Traffic, cache hit rate, latency percentiles, tokens, estimated cost |

**Request body** for both chat endpoints:

```json
{"messages": [{"role": "user", "content": "What is a token?"}], "max_new_tokens": 64}
```

**Response** from `/v1/chat`:

```json
{"id": "a1b2c3d4e5f6", "text": "...", "usage": {"prompt_tokens": 21, "completion_tokens": 22}, "cached": false}
```

**Status codes:**

| Code | Meaning |
|---|---|
| 200 | Success |
| 401 | Missing or invalid API key |
| 413 | Prompt longer than the character limit |
| 422 | Invalid request (empty messages, unknown role, `max_new_tokens` too large) |
| 429 | Rate limit exceeded. The `Retry-After` header says how many seconds to wait. |
| 502 | The model backend failed. The message is generic on purpose. |

Every response carries an `X-Request-ID` header, and every request writes one structured JSON line to the log.

**Settings** (environment variables): `LLM_API_KEYS`, `LLM_RATE_LIMIT_PER_MINUTE` (30), `LLM_CACHE_SIZE` (256), `LLM_CACHE_TTL_SECONDS` (300), `LLM_MAX_NEW_TOKENS_CAP` (256).

---

## 🧠 Design Decisions

Each decision below is a question you could be asked, with the answer already written down.

### Authentication

- **Keys come from the environment, never from source code.** A key in a repository is a leaked key.
- **`hmac.compare_digest`** compares keys in constant time, so an attacker can't learn a key one character at a time by measuring how fast the server says no.
- **Keys are never logged.** Logs show an 8-character hash of the key (`client`), which identifies a caller without revealing the secret. A test asserts the key never appears in the log output.
- **Authentication runs before everything expensive.** A request with a bad key is rejected before it reaches the rate limiter, the cache, or the model. A test asserts the model is never called.

### Rate limiting: the token bucket

Each key has a bucket that holds up to `N` tokens and refills steadily. A request costs one token.

```text
30 requests per minute  ->  bucket holds 30, refills 1 token every 2 seconds
```

This allows a short burst (a full bucket) but caps the long-run rate, which is how provider rate limits usually behave. When the bucket is empty, the server returns **429 with a `Retry-After` header** computed from the refill rate, so well-behaved clients (Module 03's backoff logic) know exactly how long to wait. Each key has its own bucket, so one noisy customer can't starve the others.

### Caching

- **LRU eviction** keeps memory bounded. **TTL expiry** stops answers going stale.
- **The cache key includes who is asking.** One customer's cached answer is never served to another. Without this, a prompt containing private data could be answered from someone else's cache entry. It costs some hit rate and buys tenant isolation. A test deliberately breaks this and confirms it is caught.
- **Only the exact same request hits the cache** (same messages, same `max_new_tokens`). Module 11 showed that *semantic* caches can serve wrong answers, so this one is exact.
- Streaming responses aren't cached.

### Validation and cost control

Bad input is rejected *before* the model runs, because every model call costs money and time. The schema rejects empty messages and unknown roles. Code enforces a character limit (413) and an output-token cap (422), so a single request can't demand 100,000 tokens.

### Errors never leak internals

If the model backend throws, the caller gets a generic `502 "The model backend failed. Please try again."`. The full stack trace goes to the **server log only**. A test makes the fake backend raise an exception with a distinctive message and asserts that it never appears in the response. Error messages that expose file paths, library versions, or prompts help attackers.

### Observability

- Every response has an **`X-Request-ID`**, and the same ID is in the log line, so a user's bug report can be matched to a log entry.
- Logs are **structured JSON** (one object per line), which is what log tools can search.
- `/metrics` reports **latency p50 and p95**, not just an average, because the average hides the slow tail (Module 11). It also reports cache hit rate, error and rate-limit counts, token totals, and an estimated cost. **Cache hits add no tokens**, since they cost nothing, and failed calls bill nothing.

### Testing time without sleeping

The cache and the rate limiter take a `clock` function. Tests pass a fake clock and advance it by hand, so "the cache entry expires after 60 seconds" and "a token refills after 20 seconds" are tested instantly and exactly. Tests that call `sleep()` are slow and flaky.

---

## 🔬 Testing the Tests

19 passing tests only prove something if they would fail when the code is wrong. So the security-critical behaviors were **mutation-tested**: each was deliberately broken to confirm a test catches it.

| Deliberate break | Test that caught it |
|---|---|
| Cache key no longer includes the caller | `test_one_customers_cache_is_never_served_to_another` |
| Backend error message leaked to the caller | `test_backend_failure_returns_a_generic_502_and_leaks_nothing` |

---

## 📊 Measured on the Real Model

`python smoke_test.py` sends real requests through the real API to the local Qwen model. These numbers come from one laptop CPU, so treat them as illustrations of the *shape*, not as benchmarks.

| Check | Result |
|---|---|
| Same request, first time | **3,302 ms** (43 tokens generated) |
| Same request, second time (cached) | **4.8 ms**, about **686x faster**, and `cached: true` |
| Streaming: time to the **first token** | **312 ms** |
| Streaming: time until the **whole reply** is complete | **3,657 ms** (44 chunks) |
| Request with no API key | rejected with **401** in about 4 ms, model never called |
| Request with a wrong API key | rejected with **401** in about 3 ms, model never called |

What this shows:

- **Streaming changes how the service feels.** The reply took about 3.7 seconds to finish, but the first word appeared after 0.3 seconds, roughly a tenth of the wait (Module 03's time-to-first-token idea, measured through the real API).
- **The cache turns seconds into milliseconds**, but only for *repeated* requests. The cached figure includes the in-process test client's overhead, so a real network hop would add to it.
- **Rejected requests are almost free**, because authentication runs before anything expensive.

After the session, `/metrics` reported 3 requests, 0 errors, and a cache hit rate of 33%. Only cache *misses* add tokens and cost.

> **A measurement mistake worth knowing about.** The first version of the smoke test measured streaming with FastAPI's `TestClient` and reported "first token after 3,483 ms, complete after 3,483 ms". That is wrong: the test client buffers a streamed response and hands it back whole, so it cannot show when the first token arrived. The smoke test now starts a real local server and reads the stream over a real connection. If a measurement comes out suspiciously perfect, check that the tool can measure it.

---

## 🎤 How to Talk About This Project in an Interview

**The 30-second pitch:**
> "I built an LLM API service with FastAPI: key authentication, a token-bucket rate limiter, an LRU cache with expiry and per-customer isolation, streaming responses, and a metrics endpoint with latency percentiles and cost. The model sits behind a small interface, so all 19 tests run against a fake and a fake clock in under a second. I also mutation-tested the security behaviors to make sure the tests actually catch regressions."

| They ask | You can say |
|---|---|
| "How would you rate-limit an API?" | Token bucket per key: allows bursts, caps the sustained rate, and returns 429 with `Retry-After`. |
| "How do you cache LLM responses safely?" | Exact-match keys, TTL and LRU, and the caller in the key so tenants never share entries. Semantic caching risks wrong answers. |
| "How do you handle errors?" | Log the detail on the server, return a generic message, and count it in metrics. |
| "How would you scale it?" | Move the cache and limiter to Redis, run several workers behind a load balancer, and batch requests on a GPU server such as vLLM. |
| "What do you monitor?" | Latency p95 and p99, error and rate-limit rates, cache hit rate, token cost per request. |

**Honest limits:** the cache and rate limiter live in one process's memory, so running several server copies would give each its own state (use Redis for shared state). The local backend serializes generation with a lock, so it handles one request at a time. Streaming token counts are approximate. And there is no request timeout. All of these are deliberate simplifications for a learning project.

---

## 🚀 Extend It

- Store the cache and rate-limit buckets in **Redis** so several workers share them
- Add a **request timeout** and cancel generation when the client disconnects
- Put a **guardrail** (Module 08) in front of the model: PII redaction and injection detection
- Swap `LocalBackend` for a hosted model (Module 03) without touching the server
- Add **API-key scopes** (read-only vs admin) and per-key usage quotas

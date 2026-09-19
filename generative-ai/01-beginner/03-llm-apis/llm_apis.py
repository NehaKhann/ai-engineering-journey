# pip: transformers>=4.56 torch matplotlib anthropic
# %% [markdown]
# # 📘 Module 03 — LLM APIs: Tokens, Streaming, Retries, Cost
#
# **Generative AI Track • Beginner**
#
# In Module 02 the model ran on your own machine. Most real products instead **call a hosted model
# over an API**: you send text over the internet, the provider runs the model on their GPUs, and you
# pay per use. That changes what you have to think about:
#
# | Question | Why it matters |
# |---|---|
# | How much does a call cost? | You are billed per **token**, not per request |
# | Why does the reply take so long? | Long answers take seconds, so you **stream** them |
# | What if the call fails? | Networks and rate limits fail, so you **retry** safely |
#
# Everything here runs offline **except** Part 6, which calls a real hosted model and needs an API key.

# %%
import os
import random
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

OUT = Path(__file__).parent / "assets" if "__file__" in globals() else Path("assets")
OUT.mkdir(exist_ok=True)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# %% [markdown]
# ## 1. What an API call looks like
#
# A hosted-model call is an HTTP request with JSON in and JSON out. The provider's SDK
# (a Python library) wraps it, but this is what travels over the wire. The shapes below are
# **illustrative**, taken from the Claude Messages API:

# %%
request = {
    "model": "claude-opus-5",
    "max_tokens": 1024,
    "system": "You are a concise assistant.",
    "messages": [{"role": "user", "content": "What is a token?"}],
}

response = {
    "content": [{"type": "text", "text": "A token is a small chunk of text a model reads, such as a word or part of one."}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 21, "output_tokens": 22},
}

print("REQUEST fields")
for key, value in request.items():
    print(f"  {key:<11} {value}")
print("\nRESPONSE fields")
for key, value in response.items():
    print(f"  {key:<12} {value}")

# %% [markdown]
# The fields you will use every day:
#
# - **`model`**: which model to run. Larger models cost more and are slower.
# - **`max_tokens`**: a hard cap on the reply length. If the reply hits it, `stop_reason` is `max_tokens` and the text is cut off.
# - **`messages`**: the whole conversation. The API is **stateless**: it remembers nothing between calls.
# - **`usage`**: how many tokens you were billed for. This is how you track cost.
# - **`stop_reason`**: why generation stopped (`end_turn` is normal; `max_tokens` means truncated).

# %% [markdown]
# ## 2. Tokens: the unit of everything
#
# Models do not read letters or words. They read **tokens**, chunks of text. Providers bill by
# tokens, and the context window is measured in tokens, so you need a feel for them.

# %%
samples = {
    "English sentence": "The quick brown fox jumps over the lazy dog.",
    "Python code": "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    "Numbers": "3.14159265358979 1234567890",
    "Hindi": "नमस्ते दुनिया, आप कैसे हैं?",
    "Emoji": "Great job 🎉🎉🎉",
}

print(f"{'Text':<18}{'Characters':<12}{'Tokens':<8}{'Chars/token'}")
print("-" * 50)
for name, text in samples.items():
    n_tokens = len(tokenizer.encode(text))
    print(f"{name:<18}{len(text):<12}{n_tokens:<8}{len(text) / n_tokens:.1f}")

print("\nHow the English sentence is split:")
print(tokenizer.convert_ids_to_tokens(tokenizer.encode(samples["English sentence"])))

# %% [markdown]
# Roughly 4 characters per token for English, but **code, numbers, and non-English text use more
# tokens for the same length**, so they cost more.
#
# **Important:** every model family has its own tokenizer. The counts above come from the Qwen
# tokenizer, so they are only an estimate for another provider's model. For exact numbers, ask the
# provider: the Claude API has a `count_tokens` endpoint (shown in Part 6). Do not use another
# vendor's tokenizer to estimate Claude costs.

# %% [markdown]
# ## 3. Cost
#
# Cost = `input tokens × input price + output tokens × output price`. Output tokens cost more than
# input tokens. Prices are quoted **per million tokens**.
#
# The table is copied from Anthropic's published rates as of **June 2026**. Prices change, so check
# the provider's pricing page before relying on them.

# %%
PRICES = {  # USD per 1 million tokens: (input, output)
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def estimate_cost(model, input_tokens, output_tokens):
    input_price, output_price = PRICES[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


# A support bot: 10,000 requests per day, ~800 tokens in, ~200 tokens out per request.
requests_per_day, tokens_in, tokens_out = 10_000, 800, 200

print(f"Support bot: {requests_per_day:,} requests/day, {tokens_in} tokens in, {tokens_out} tokens out\n")
print(f"{'Model':<20}{'Per request':<14}{'Per month (30 days)'}")
print("-" * 52)
for model in PRICES:
    per_request = estimate_cost(model, tokens_in, tokens_out)
    print(f"{model:<20}${per_request:<13.5f}${per_request * requests_per_day * 30:,.0f}")

# %% [markdown]
# ### The hidden cost: history is re-sent every turn
#
# Because the API is stateless, a chatbot must send the **entire conversation** on every turn.
# Input tokens therefore grow with every message, and total cost grows much faster than the number
# of turns.

# %%
user_message = "Can you explain that in a bit more detail, with a small example?"
assistant_reply = (
    "Sure. Think of a token as a piece of a word. The model splits your text into pieces, "
    "turns each piece into a number, and predicts the next piece one at a time. For example, "
    "the word 'unbelievable' might be split into 'un', 'believ', and 'able'."
)
user_tokens = len(tokenizer.encode(user_message))
reply_tokens = len(tokenizer.encode(assistant_reply))

turns = range(1, 21)
sent_per_turn, history = [], 0
for _ in turns:
    history += user_tokens  # the new user message joins the history
    sent_per_turn.append(history)  # the whole history is sent as input
    history += reply_tokens  # the reply is appended for next time

cumulative = [sum(sent_per_turn[:i]) for i in range(1, len(sent_per_turn) + 1)]
new_tokens_only = [(user_tokens + reply_tokens) * i for i in turns]

print(f"Each turn adds about {user_tokens + reply_tokens} new tokens.")
print(f"Turn 1 sends {sent_per_turn[0]} input tokens; turn 20 sends {sent_per_turn[-1]}.")
print(f"After 20 turns you have been billed for {cumulative[-1]:,} input tokens in total,")
print(f"which is {cumulative[-1] / new_tokens_only[-1]:.1f}x the {new_tokens_only[-1]:,} tokens of actual conversation.")

plt.figure(figsize=(8, 4.5))
plt.plot(list(turns), cumulative, marker="o", label="Input tokens billed (history re-sent)")
plt.plot(list(turns), new_tokens_only, marker="s", label="Tokens of actual conversation")
plt.xticks(range(2, 21, 2))
plt.xlabel("Conversation turn")
plt.ylabel("Cumulative tokens")
plt.title("Why long chats get expensive")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "history_growth.png", dpi=150)
plt.show()

# %% [markdown]
# This is why real apps **trim or summarize old messages** and use **prompt caching**, which
# discounts the repeated part of the input. Caching is covered in Module 11.

# %% [markdown]
# ## 4. Streaming
#
# A long answer can take many seconds to finish. Without streaming the user stares at a blank
# screen until the whole reply is ready. With streaming, tokens are sent **as they are generated**,
# so text appears almost immediately. The total time is the same, but the *felt* speed is very
# different.
#
# Two numbers matter:
#
# - **Time to first token (TTFT)**: how long until anything appears
# - **Total time**: how long until the reply is complete
#
# We measure both with the local model, which supports streaming the same way.

# %%
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)


def stream_reply(prompt, max_new_tokens=80):
    """Generate a reply and print it as it arrives. Returns (time_to_first_token, total_time)."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id),
    )
    start = time.time()
    thread.start()

    first_token_time = None
    for chunk in streamer:
        if first_token_time is None:
            first_token_time = time.time() - start
        print(chunk, end="", flush=True)
    thread.join()
    return first_token_time, time.time() - start


print("Streaming reply:\n")
ttft, total = stream_reply("Explain what an API is in three sentences.")
print(f"\n\nTime to first token: {ttft:.2f}s")
print(f"Total time:          {total:.2f}s")
print(f"Without streaming, the user would wait the full {total:.2f}s before seeing anything.")

# %% [markdown]
# ## 5. Retries with exponential backoff
#
# Calls to a hosted API fail sometimes. What you do next depends on the kind of failure:
#
# | Failure | Cause | Retry? |
# |---|---|---|
# | `429` rate limit | Too many requests | ✅ Yes, after waiting |
# | `5xx` server error / overloaded | Provider problem | ✅ Yes |
# | Timeout / connection error | Network | ✅ Yes |
# | `400` bad request | Your request is wrong | ❌ No, fix the request |
# | `401` / `403` | Bad key or no permission | ❌ No |
# | `404` | Wrong model name or URL | ❌ No |
#
# When you retry, **wait longer each time** (exponential backoff) and add **random jitter**, so
# that many clients failing together do not all retry at the same instant.

# %%
class RetryableError(Exception):
    """A failure worth retrying, such as a 429 or a 5xx."""


class FatalError(Exception):
    """A failure that retrying cannot fix, such as a 400 or 401."""


def call_with_backoff(fn, max_attempts=5, base_delay=1.0, max_delay=30.0):
    """Call fn(); on RetryableError wait base_delay * 2^attempt (with jitter) and try again."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except RetryableError as error:
            if attempt == max_attempts - 1:
                raise
            delay = min(base_delay * 2**attempt, max_delay)
            delay = random.uniform(0, delay)  # "full jitter"
            print(f"  attempt {attempt + 1} failed ({error}); waiting {delay:.2f}s")
            time.sleep(delay)
        # FatalError is not caught here, so it propagates immediately.


def make_flaky_service(failures):
    """A fake API that fails `failures` times and then succeeds."""
    state = {"calls": 0}

    def service():
        state["calls"] += 1
        if state["calls"] <= failures:
            raise RetryableError("429 rate limited")
        return "OK"

    return service, state


random.seed(1)
print("Service that fails twice, then works:")
service, state = make_flaky_service(failures=2)
print("  result:", call_with_backoff(service, base_delay=0.1), f"(after {state['calls']} calls)")

print("\nService that always fails:")
service, state = make_flaky_service(failures=99)
try:
    call_with_backoff(service, max_attempts=3, base_delay=0.1)
except RetryableError as error:
    print(f"  gave up after {state['calls']} calls: {error}")

print("\nA fatal error is NOT retried:")


def bad_request():
    raise FatalError("400 invalid model name")


try:
    call_with_backoff(bad_request)
except FatalError as error:
    print(f"  raised immediately: {error}")

# %% [markdown]
# Good news: the official SDKs already do this for you. The Anthropic SDK retries connection errors,
# 408, 409, 429, and 5xx responses twice by default, and you can change that with `max_retries`.
# Write your own retry logic only when you need behavior beyond that.

# %% [markdown]
# ## 6. Calling a real hosted model
#
# This part needs an **API key** from https://console.anthropic.com and costs a small amount of
# money. Set the key as an environment variable, never in your code:
#
# ```powershell
# $env:ANTHROPIC_API_KEY = "sk-ant-..."
# ```
#
# or put `ANTHROPIC_API_KEY=...` in a `.env` file (already gitignored in this repo).
#
# If no key is set, this section is skipped and explains why.

# %%
HOSTED_MODEL = "claude-opus-5"
HAS_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))

if not HAS_KEY:
    print("ANTHROPIC_API_KEY is not set, so the hosted-model examples below are skipped.")
    print("Parts 1 to 5 already ran fully offline.")
else:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

# %%
def ask_claude(prompt, system="You are a concise assistant."):
    """One complete (non-streaming) call. Returns (text, usage)."""
    try:
        response = client.beta.messages.create(
            model=HOSTED_MODEL,
            max_tokens=1024,  # deliberately short: we expect a brief answer
            system=system,
            messages=[{"role": "user", "content": prompt}],
            # Optional safety net: if the model declines a request, re-run it on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.RateLimitError as error:
        retry_after = error.response.headers.get("retry-after", "unknown")
        raise RuntimeError(f"Rate limited. Retry after {retry_after}s.") from error
    except anthropic.AuthenticationError as error:
        raise RuntimeError("Invalid API key. Check ANTHROPIC_API_KEY.") from error
    except anthropic.BadRequestError as error:
        raise RuntimeError(f"Bad request: {error.message}") from error
    except anthropic.APIConnectionError as error:
        raise RuntimeError("Network error. Check your connection.") from error
    except anthropic.APIStatusError as error:
        raise RuntimeError(f"API error {error.status_code}: {error.message}") from error

    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined this request.")
    if response.stop_reason == "max_tokens":
        print("Warning: reply was cut off by max_tokens.")

    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.usage


if HAS_KEY:
    text, usage = ask_claude("In two sentences, what is a token in an LLM?")
    print(text)
    print(f"\nTokens billed: {usage.input_tokens} in, {usage.output_tokens} out")
    print(f"Estimated cost: ${estimate_cost(HOSTED_MODEL, usage.input_tokens, usage.output_tokens):.5f}")

# %% [markdown]
# Notice how the errors are handled **most specific first**. A `RateLimitError` is retryable, a
# `BadRequestError` is not, so a single `except Exception` would hide that difference.
#
# ### Streaming and exact token counts

# %%
if HAS_KEY:
    start, first = time.time(), None
    with client.messages.stream(
        model=HOSTED_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Explain what an API is in three sentences."}],
    ) as stream:
        for piece in stream.text_stream:
            if first is None:
                first = time.time() - start
            print(piece, end="", flush=True)
        final = stream.get_final_message()

    print(f"\n\nTime to first token: {first:.2f}s, total: {time.time() - start:.2f}s")
    print(f"Tokens billed: {final.usage.input_tokens} in, {final.usage.output_tokens} out")

    counted = client.messages.count_tokens(
        model=HOSTED_MODEL,
        messages=[{"role": "user", "content": "Explain what an API is in three sentences."}],
    )
    print(f"count_tokens says this prompt is {counted.input_tokens} tokens (exact, before paying for a call)")

# %% [markdown]
# ## 🎯 Key Takeaways
#
# - A hosted model is an HTTP call. The API is **stateless**: you send the whole conversation each time.
# - You are billed per **token**, and output tokens cost more than input tokens.
# - Long chats get expensive because **history is re-sent every turn**.
# - Tokenizers differ between models. Use the provider's own counter for exact numbers.
# - **Stream** long replies. Total time is unchanged, but users see text immediately.
# - Retry `429`, `5xx`, and network errors with **exponential backoff and jitter**. Never retry `400`, `401`, or `404`.
# - Keep keys in environment variables, cap `max_tokens`, check `stop_reason`, and log `usage`.
#
# ## 🚀 What's Next?
#
# **Module 04 — Embeddings & Vector Search**: how to turn text into numbers so you can search by meaning.

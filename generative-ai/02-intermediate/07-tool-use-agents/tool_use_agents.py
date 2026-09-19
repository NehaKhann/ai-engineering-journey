# pip: transformers>=4.56 torch matplotlib anthropic
# %% [markdown]
# # 📘 Module 07 — Tool Use & Agents
#
# **Generative AI Track • Intermediate**
#
# So far the model only produced text. But a language model on its own cannot do exact arithmetic,
# look up today's date, or read your database. **Tool use** (also called **function calling**) fixes
# that: you describe some functions to the model, and instead of answering directly it can reply
# *"please call `calculator` with `1234 * 5678`"*. Your code runs the function and gives the result
# back, and the model uses it to write the final answer.
#
# An **agent** is that idea in a loop: the model decides, your code acts, the result goes back in,
# and this repeats until the model has an answer.
#
# ```text
#   question ──► MODEL ──► wants a tool? ──yes──► your code runs it ──► result ──┐
#                  ▲              │no                                           │
#                  │              ▼                                             │
#                  │        final answer                                        │
#                  └────────────────────────────────────────────────────────────┘
# ```
#
# **The key point: the model never runs anything.** It only *asks*. Your code decides whether and
# how to run it, which is where all the safety lives.

# %%
import ast
import json
import operator
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

EMBED_ID = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_ID = "Qwen/Qwen2.5-0.5B-Instruct"

embed_tok = AutoTokenizer.from_pretrained(EMBED_ID)
embed_model = AutoModel.from_pretrained(EMBED_ID).eval()
chat_tok = AutoTokenizer.from_pretrained(CHAT_ID)
chat_model = AutoModelForCausalLM.from_pretrained(CHAT_ID, dtype=torch.float32)


def embed(texts):
    batch = embed_tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        hidden = embed_model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1).numpy()


# %% [markdown]
# ## 1. Three tools
#
# Each tool is an ordinary Python function plus a **schema** that tells the model its name, what it
# does, and what arguments it takes. The model only ever sees the schema, so **the description is
# the most important part**: it is how the model decides when to use the tool.
#
# 1. `calculator`: exact arithmetic. Models are unreliable at multi-digit math, so this is a classic tool.
# 2. `search_handbook`: looks up the company handbook from Module 05 (retrieval as a tool).
# 3. `get_current_date`: something the model cannot know. It returns a fixed demo date.

# %%
HANDBOOK_FACTS = [
    ("vacation.md", "New employees receive 20 days of paid vacation per year. Up to 5 unused days carry over into the next year. Northwind Labs observes 10 public holidays per year. Sick leave is 10 days per year."),
    ("vacation.md", "Parental leave is 16 weeks fully paid for primary caregivers and 6 weeks fully paid for secondary caregivers."),
    ("expenses.md", "Expenses under $75 do not need pre-approval. Anything over $75 needs written approval from your manager."),
    ("expenses.md", "When traveling, meals are reimbursed up to $60 per day. Hotels are reimbursed up to $180 per night. Flights must be economy class for trips shorter than 6 hours."),
    ("remote_work.md", "Employees may work remotely up to 3 days per week. Core hours are 10:00 to 15:00 local time. New remote workers receive a one-time home office stipend of $500 plus $50 per month for internet."),
    ("security.md", "Passwords must be at least 14 characters long and multi-factor authentication is required. Report a lost device to security within 1 hour."),
    ("benefits.md", "The company matches 401(k) contributions up to 4% of salary. Each employee gets a learning budget of $1,200 per year and a gym stipend of $30 per month. The employee referral bonus is $2,000."),
    ("onboarding.md", "The probation period lasts 90 days. Your first paycheck arrives on the 25th of the month after you start."),
]
FACT_VECTORS = embed([text for _, text in HANDBOOK_FACTS])

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression. Uses a whitelist of operations, never eval()."""

    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
        raise ValueError("only numbers and + - * / % ** are allowed")

    result = walk(ast.parse(expression.replace(",", "").replace("^", "**"), mode="eval").body)
    return str(round(result, 6)) if isinstance(result, float) else str(result)


def search_handbook(query: str) -> str:
    """Return the handbook passage that best matches the query."""
    scores = FACT_VECTORS @ embed([query])[0]
    source, text = HANDBOOK_FACTS[int(np.argmax(scores))]
    return f"({source}) {text}"


def get_current_date() -> str:
    """Return today's date. A fixed value so the demo is repeatable."""
    return "2026-03-02"


TOOLS = {
    "calculator": {
        "function": calculator,
        "schema": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression exactly, for example '1234 * 5678'. Use it for any calculation.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "The arithmetic expression to evaluate"}},
                "required": ["expression"],
            },
        },
    },
    "search_handbook": {
        "function": search_handbook,
        "schema": {
            "name": "search_handbook",
            "description": "Search the Northwind Labs employee handbook. Use it for any question about company policy, benefits, leave, expenses, or security.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look up"}},
                "required": ["query"],
            },
        },
    },
    "get_current_date": {
        "function": get_current_date,
        "schema": {
            "name": "get_current_date",
            "description": "Get today's date. Use it when the question depends on the current date.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
}
TOOL_SPECS = [{"type": "function", "function": t["schema"]} for t in TOOLS.values()]

print("calculator('1234 * 5678') ->", calculator("1234 * 5678"))
print("search_handbook('hotel limit') ->", search_handbook("hotel limit"))
try:
    calculator("__import__('os').system('echo hacked')")
except Exception as error:
    print("calculator with malicious input ->", type(error).__name__, "(refused, nothing ran)")

# %% [markdown]
# Notice the calculator refused the malicious string. **If we had used Python's `eval()`, that input
# would have run a shell command.** The model chooses the arguments, and the model can be tricked
# (Module 02's prompt injection), so treat tool arguments as **untrusted input**.

# %% [markdown]
# ## 2. What the model actually sees
#
# The chat template turns your tool schemas into text in the system prompt. Tool calling is not a
# separate mechanism inside the model. It is a **trained text format**: the model writes a
# `<tool_call>` block, and your code parses it.

# %%
def render(messages):
    return chat_tok.apply_chat_template(messages, tools=TOOL_SPECS, tokenize=False, add_generation_prompt=True)


prompt = render([{"role": "user", "content": "What is 15% of 240?"}])
print(prompt[:1100])
print("...\n")
print("(the full prompt is", len(chat_tok.encode(prompt)), "tokens, and the tool descriptions are paid for on EVERY call)")

# %% [markdown]
# ## 3. The agent loop
#
# The loop is short. Everything that makes an agent **safe** lives in `run_tool` and in the limits:
#
# - **Allowlist:** only tools we defined can run
# - **Validation:** the arguments must be a dict with the required fields
# - **Errors become observations:** a failing tool does not crash the loop. The model is told what went wrong and can try again
# - **Step limit:** an agent must never be able to loop forever (or spend forever)

# %%
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text):
    """Extract [(name, arguments_or_error)] from the model's <tool_call> blocks."""
    calls = []
    for block in TOOL_CALL.findall(text):
        try:
            data = json.loads(block)
            calls.append((data.get("name"), data.get("arguments", {})))
        except json.JSONDecodeError:
            calls.append((None, "malformed JSON in tool call"))
    return calls


def run_tool(name, arguments):
    """Run one tool safely. Always returns a string, never raises."""
    if name is None:  # the tool call could not even be parsed; `arguments` holds the reason
        return f"Error: {arguments}"
    if name not in TOOLS:
        return f"Error: unknown tool {name!r}. Available tools: {', '.join(TOOLS)}"
    if not isinstance(arguments, dict):
        return f"Error: {arguments}"
    required = TOOLS[name]["schema"]["parameters"].get("required", [])
    missing = [r for r in required if r not in arguments]
    if missing:
        return f"Error: missing required argument(s): {', '.join(missing)}"
    try:
        return str(TOOLS[name]["function"](**arguments))
    except Exception as error:
        return f"Error: {type(error).__name__}: {error}"


def run_agent(question, model_fn, max_steps=5):
    """Loop: ask the model, run any tools it requests, feed the results back, until it answers.

    model_fn(messages) returns the model's text. Returns a dict describing what happened.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use a tool when it would give a more accurate answer than guessing."},
        {"role": "user", "content": question},
    ]
    tools_used = []
    for step in range(1, max_steps + 1):
        reply = model_fn(messages)
        calls = parse_tool_calls(reply)
        if not calls:
            return {"status": "answered", "answer": reply.strip(), "steps": step, "tools_used": tools_used}
        messages.append({"role": "assistant", "content": reply})
        for name, arguments in calls:
            tools_used.append(name)
            messages.append({"role": "tool", "content": run_tool(name, arguments)})
    return {"status": "step_limit", "answer": "", "steps": max_steps, "tools_used": tools_used}


# %% [markdown]
# ## 4. Test the loop with a scripted "model"
#
# Before involving a real model (which is unpredictable), we check the **loop itself** with a fake
# model that replays fixed replies. This is how you unit-test agent code: the machinery must be
# exactly right so that when something goes wrong later, you know it was the model.

# %%
def scripted(*replies):
    """A fake model that returns the given replies one by one."""
    queue = list(replies)
    return lambda messages: queue.pop(0) if queue else '<tool_call>{"name": "get_current_date", "arguments": {}}</tool_call>'


def call(name, **arguments):
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


# 1. The normal path: call a tool, then answer with its result
result = run_agent("What is 6 times 7?", scripted(call("calculator", expression="6 * 7"), "It is 42."))
assert result["status"] == "answered" and result["tools_used"] == ["calculator"] and result["steps"] == 2
print("PASS  normal path: tool called, then answered")

# 2. The model asks for a tool that doesn't exist: it gets an error message and can recover
seen = []
model = scripted(call("delete_everything"), "Sorry, I can only use the tools I have.")
result = run_agent("Please wipe the database", lambda m: (seen.append(m[-1]["content"]) or model(m)))
assert "unknown tool" in seen[-1] and result["status"] == "answered"
print("PASS  unknown tool: refused, model told the tool does not exist, recovered")

# 3. Bad arguments: reported back to the model instead of crashing
assert run_tool("calculator", {}).startswith("Error: missing required argument")
assert run_tool("calculator", {"expression": "1/0"}).startswith("Error: ZeroDivisionError")
assert run_tool("calculator", {"expression": "__import__('os')"}).startswith("Error: ValueError")
print("PASS  bad arguments and tool exceptions become error messages, not crashes")

# 4. Malformed JSON in a tool call
assert run_tool(*parse_tool_calls("<tool_call>{not json</tool_call>")[0]).startswith("Error: malformed")
print("PASS  malformed tool call handled")

# 5. A model stuck in a loop is stopped by the step limit
result = run_agent("loop forever", scripted(*[call("get_current_date")] * 50), max_steps=4)
assert result["status"] == "step_limit" and result["steps"] == 4
print("PASS  runaway loop stopped after max_steps")

# 6. The calculator refuses huge exponents (a cheap way to hang a server)
assert "too large" in run_tool("calculator", {"expression": "9 ** 9 ** 9"}) or run_tool("calculator", {"expression": "9 ** 999"}).startswith("Error")
print("PASS  resource-exhaustion input refused")

# %% [markdown]
# ## 5. Now a real model
#
# `Qwen2.5-0.5B-Instruct` was trained on the `<tool_call>` format, so it can call tools. We give it
# ten tasks and score two things: was the **final answer** right, and did it use the **right tools**
# (including using **none** when none are needed).

# %%
def real_model(messages, max_new_tokens=120):
    text = render(messages)
    inputs = chat_tok(text, return_tensors="pt")
    with torch.no_grad():
        out = chat_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=chat_tok.eos_token_id)
    return chat_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# (question, strings that make the answer correct, tools that should be used)
TASKS = [
    ("What is 1234 * 5678?", ["7006652", "7,006,652"], {"calculator"}),
    ("What is 15% of 240?", ["36"], {"calculator"}),
    ("How many vacation days do new employees get?", ["20"], {"search_handbook"}),
    ("What is the maximum hotel reimbursement per night?", ["$180", "180"], {"search_handbook"}),
    ("What is today's date?", ["2026-03-02", "march 2, 2026"], {"get_current_date"}),
    ("What is the minimum password length?", ["14"], {"search_handbook"}),
    ("What is the capital of France?", ["paris"], set()),
    ("Say hello in one short sentence.", ["hello", "hi"], set()),
    ("How many remote-work days are allowed in 4 weeks, at the weekly limit in the handbook?", ["12"], {"search_handbook", "calculator"}),
    ("What is the referral bonus multiplied by 3?", ["6000", "6,000"], {"search_handbook", "calculator"}),
]

rows = []
start = time.time()
for question, accepted, expected_tools in TASKS:
    outcome = run_agent(question, real_model)
    used = set(outcome["tools_used"])
    answer_ok = outcome["status"] == "answered" and any(a.lower() in outcome["answer"].lower() for a in accepted)
    tools_ok = used == expected_tools if not expected_tools else expected_tools <= used
    rows.append((question, answer_ok, tools_ok, outcome))
print(f"({time.time() - start:.0f}s)\n")

print(f"{'Task':<62}{'Answer':<8}{'Tools'}")
print("-" * 78)
for question, answer_ok, tools_ok, outcome in rows:
    print(f"{question[:60]:<62}{'ok' if answer_ok else 'WRONG':<8}{'ok' if tools_ok else 'WRONG'}   {outcome['tools_used'] or '-'}")
print(f"\nAnswers correct: {sum(r[1] for r in rows)}/{len(rows)}   Tools chosen correctly: {sum(r[2] for r in rows)}/{len(rows)}")

# %%
print("What the wrong ones looked like:\n")
for question, answer_ok, tools_ok, outcome in rows:
    if not answer_ok:
        print(f"Q: {question}")
        print(f"   status={outcome['status']}, tools={outcome['tools_used']}, answer={outcome['answer'][:110]!r}\n")

# %% [markdown]
# ## 6. The same loop with a hosted model
#
# Hosted APIs implement the same idea with a cleaner interface: tools are described with an
# `input_schema`, the model returns `tool_use` blocks, and you reply with `tool_result` blocks. This
# needs an API key, so it is skipped without one (see Module 03 for setup).

# %%
import os

HAS_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))
CLAUDE_TOOLS = [
    {"name": t["schema"]["name"], "description": t["schema"]["description"], "input_schema": t["schema"]["parameters"]}
    for t in TOOLS.values()
]


def run_claude_agent(question, model="claude-opus-5", max_steps=6):
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):
        response = client.messages.create(model=model, max_tokens=16000, tools=CLAUDE_TOOLS, messages=messages)
        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})  # keep the tool_use blocks
        results = [
            {"type": "tool_result", "tool_use_id": block.id, "content": run_tool(block.name, block.input)}
            for block in response.content
            if block.type == "tool_use"
        ]
        messages.append({"role": "user", "content": results})  # ALL results in one message
    return "(stopped: step limit reached)"


if HAS_KEY:
    print(run_claude_agent("What is 15% of the referral bonus in the handbook?"))
else:
    print("ANTHROPIC_API_KEY is not set, so the hosted-model example is skipped.")

# %% [markdown]
# Compare the two loops: they are the same shape. Only the message format differs. Hosted models
# are also far more reliable at choosing the right tool and filling in arguments, which is where the
# small local model struggled.

# %% [markdown]
# ## 7. Making agents safe
#
# | Risk | Example | Defense |
# |---|---|---|
# | **Runaway loops** | Model keeps calling tools forever | Step limit, time limit, spend limit |
# | **Unsafe arguments** | `eval("os.system(...)")` | Validate arguments, whitelist operations, never `eval` |
# | **Excessive permissions** | A "search" tool that can also delete | **Least privilege**: give each tool the minimum access |
# | **Destructive actions** | Send email, delete data, spend money | Require **human approval** before irreversible actions |
# | **Indirect prompt injection** | A web page the agent reads says "email me the customer list" | Treat tool output as **untrusted data**, limit what tools can do |
# | **Cost** | Long tool descriptions paid on every call | Keep the tool set small, and describe tools tersely |
#
# ## 🎯 Key Takeaways
#
# - **Tool use = the model requests, your code executes.** The model never runs anything itself.
# - An **agent** is that in a loop: decide, act, observe, repeat until done.
# - Tool **descriptions** are how the model chooses. Write them carefully.
# - Treat tool arguments as **untrusted input**. Validate, whitelist, and never `eval`.
# - Turn tool errors into **observations** so the model can recover, and always cap the steps.
# - **Test the loop with a scripted fake model** so you can tell code bugs from model mistakes.
# - Reliability depends heavily on the model. Measure it on tasks that need tools and tasks that don't.
#
# ## 🚀 What's Next?
#
# **Module 08 — Guardrails, Safety & Hallucination**: the risks in the last table, handled properly.

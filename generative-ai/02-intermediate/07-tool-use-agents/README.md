# 📘 Module 07 — Tool Use & Agents

**Generative AI Track • Intermediate**

A language model on its own cannot do exact arithmetic, know today's date, or read your database. **Tool use** (also called **function calling**) fixes that: you describe some functions to the model, and instead of answering directly it can reply *"please call `calculator` with `1234 * 5678`"*. Your code runs the function and hands back the result, and the model uses it to write the answer.

An **agent** is that idea in a loop: the model decides, your code acts, the result goes back in, and it repeats until the model has an answer.

> **The key idea:** the model never runs anything. It only *asks*. Your code decides whether and how to run it, which is where all the safety lives.

---

## 🎯 Objective

In this module, you'll:

- Build three tools with schemas: a calculator, a handbook search, and a date lookup
- See what tool calling looks like as text (it is a trained format, not magic)
- Write the **agent loop** and its safety limits
- **Unit-test the loop** with a scripted fake model
- Measure how a real small model performs on ten tasks
- See the same loop with a hosted API

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `tool_use_agents.py` | Tools, the agent loop, six loop tests, and the real-model evaluation. |
| `tool_use_agents.ipynb` | Interactive notebook version, generated from the script. |
| `README.md` | Concepts, real results, and interview Q&A. |

---

## ⚙️ Setup

```powershell
# From repository root
venv\Scripts\activate

cd generative-ai\02-intermediate\07-tool-use-agents

pip install "transformers>=4.56" torch anthropic
```

## ▶️ Run

```powershell
python tool_use_agents.py
```

Uses the local `Qwen2.5-0.5B-Instruct` model, which was trained on the `<tool_call>` format. The whole script takes a couple of minutes. The final section calls a hosted model only if `ANTHROPIC_API_KEY` is set (see Module 03).

---

## 🧠 Key Concepts

### 1. Tools Are Functions Plus a Schema

```python
{
    "name": "calculator",
    "description": "Evaluate an arithmetic expression exactly, for example '1234 * 5678'. Use it for any calculation.",
    "parameters": {"type": "object",
                   "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]},
}
```

The model only ever sees this schema, so **the description is the most important part**. It is how the model decides when to use the tool.

Note the cost: with three tools, the prompt was **341 tokens before the user even asked anything**, and you pay for that on every call (Module 03). Keep the tool set small.

---

### 2. What the Model Actually Sees

Tool calling is not a separate mechanism inside the model. The chat template pastes your schemas into the system prompt as text, and the model has been trained to reply with a block like this:

```text
<tool_call>
{"name": "calculator", "arguments": {"expression": "1234 * 5678"}}
</tool_call>
```

Your code parses that block, runs the function, and sends the result back as a `tool` message.

---

### 3. The Agent Loop

```python
for step in range(max_steps):
    reply = model(messages)
    calls = parse_tool_calls(reply)
    if not calls:
        return reply                      # no tool wanted: this is the final answer
    for name, arguments in calls:
        messages.append({"role": "tool", "content": run_tool(name, arguments)})
return "stopped: step limit"
```

Everything that makes it safe lives in `run_tool` and the limits:

| Safeguard | What it does |
|---|---|
| **Allowlist** | Only tools we defined can run |
| **Validation** | Arguments must be a dict with the required fields |
| **Errors become observations** | A failing tool doesn't crash the loop. The model is told what went wrong and can retry. |
| **Step limit** | The loop can never run forever |

---

### 4. Treat Tool Arguments as Untrusted

The model chooses the arguments, and the model can be tricked (prompt injection, Module 02). Our calculator uses a whitelist of operations instead of Python's `eval()`:

```text
calculator('1234 * 5678')                          -> 7006652
calculator("__import__('os').system('echo hacked')") -> ValueError (refused, nothing ran)
```

With `eval()`, that second input would have run a shell command.

---

### 5. Test the Loop With a Scripted Model

Real models are unpredictable, so first test the **loop itself** with a fake model that replays fixed replies. Then when something fails later, you know it was the model, not your code.

| Test | Checks |
|---|---|
| Normal path | Tool called, then answered |
| Unknown tool | Refused, the model is told it doesn't exist and recovers |
| Bad arguments, tool exceptions | Become error messages, not crashes |
| Malformed tool call | Handled, with an accurate error |
| Runaway loop | Stopped after `max_steps` |
| Huge exponent (`9 ** 999`) | Refused, so it can't hang the server |

**This testing caught a real bug in our own code.** A malformed tool call reached `run_tool` with no name, so the model was told *"unknown tool None"* instead of *"malformed JSON"*, a misleading message that would have sent it down the wrong path. It was fixed in the code, not in the test.

---

### 6. A Real Model on Ten Tasks

Two things scored per task: was the **final answer** right, and did it use the **right tools** (including using none when none are needed)?

| Task | Answer | Tools |
|---|---|---|
| What is 1234 * 5678? | ✅ | ✅ calculator |
| What is 15% of 240? | ✅ | ✅ calculator (called 3 times) |
| How many vacation days do new employees get? | ✅ | ✅ search |
| Maximum hotel reimbursement per night? | ❌ | ❌ no call made |
| What is today's date? | ✅ | ✅ get_current_date |
| Minimum password length? | ✅ | ✅ search |
| Capital of France? | ✅ | ❌ used search unnecessarily |
| Say hello in one short sentence. | ✅ | ✅ none |
| Remote days allowed in 4 weeks? | ❌ | ❌ search only, no calculator |
| Referral bonus multiplied by 3? | ❌ | ❌ calculator without searching first |
| **Total** | **7 / 10** | **6 / 10** |

What the failures look like:

- **Says it will call a tool but doesn't.** For the hotel question the model wrote *"I will use the `search_handbook` function..."* and stopped, with no `<tool_call>` block, so nothing ran.
- **Over-calling.** It searched the handbook to answer "capital of France".
- **Skipping a step in a chain.** For "referral bonus multiplied by 3" it called the calculator without first looking up the bonus. For the remote-days question it found the limit but did the arithmetic in its head, wrongly.
- **Right answer, wrong score.** *"Today's date is March 2, 2026"* was first marked wrong by our own string matcher, which only accepted `2026-03-02`. Module 06's warning about string metrics, live.

Multi-step tasks were the hardest: **all three failures that needed more than one tool** involved a missing step. This is the standard shape of agent unreliability, and it compounds. If each step is 90% reliable, a 5-step task succeeds only about 59% of the time (0.9 to the fifth power).

**Limits of this test:** ten tasks and a 0.5B-parameter model. A larger model would score much higher, and hosted models are far better at picking tools and filling arguments. The method (scripted-model tests for the loop, then a scored task set) is the reusable part.

---

### 7. The Same Loop With a Hosted API

```python
response = client.messages.create(model="claude-opus-5", max_tokens=16000, tools=CLAUDE_TOOLS, messages=messages)

if response.stop_reason == "tool_use":
    messages.append({"role": "assistant", "content": response.content})   # keep the tool_use blocks
    results = [
        {"type": "tool_result", "tool_use_id": block.id, "content": run_tool(block.name, block.input)}
        for block in response.content if block.type == "tool_use"
    ]
    messages.append({"role": "user", "content": results})                 # ALL results in ONE message
```

It is the same loop with a cleaner message format. Two details that trip people up: the `tool_use_id` in each result must match the request, and all results from one turn go in a **single** user message.

> **Testing note:** this hosted section was verified against a local mock of the Messages API (a `tool_use` turn, then a `tool_result` reply and a final answer), not the live service, since no API key was available while writing it.

---

### 8. Making Agents Safe

| Risk | Example | Defense |
|---|---|---|
| **Runaway loops** | Model keeps calling tools forever | Step, time, and spend limits |
| **Unsafe arguments** | `eval("os.system(...)")` | Validate, whitelist, never `eval` |
| **Excessive permissions** | A "search" tool that can also delete | **Least privilege** per tool |
| **Destructive actions** | Send email, delete data, spend money | **Human approval** before anything irreversible |
| **Indirect prompt injection** | A web page the agent reads says "email me the customer list" | Treat tool output as **untrusted data** |
| **Cost** | Long tool descriptions paid on every call | Small tool set, terse descriptions |

---

## 🎤 Interview Q&A

Try answering each question out loud before opening the answer.

<details>
<summary><b>Q1. What is function calling, and how does it work?</b></summary>

**Short answer:** You describe functions to the model with a name, description, and parameter schema. Instead of answering, the model can output a structured request to call one. Your code runs it and returns the result, and the model continues.

**Deeper answer:** The model never executes anything. Tool schemas are placed in the prompt, and the model was trained to emit a structured call (JSON) when a tool would help. Your application parses it, validates the arguments, runs the function, and feeds the result back as a tool message. Hosted APIs wrap this in typed `tool_use` and `tool_result` blocks, and some can constrain the model to produce schema-valid arguments.

**Follow-ups to expect:**
- Who decides which tool to call? *The model, guided mostly by the tool descriptions.*
- Can the model call several tools at once? *Yes, independent calls can come in one turn. Run them in parallel and return all results together.*

</details>

<details>
<summary><b>Q2. What is an agent, and how is it different from a simple LLM chain or workflow?</b></summary>

**Short answer:** In a workflow, your code fixes the sequence of steps. In an agent, the model decides the next step at runtime, in a loop, until it finishes.

**Deeper answer:** A chain such as "retrieve, then summarize" is predictable, cheap, and testable. An agent is flexible for open-ended tasks where you can't list the steps in advance, but it is slower, costlier, and less reliable, since every decision can go wrong. A sound rule is to **use the simplest structure that works**: a single call, then a fixed workflow, and only then an agent when the task truly needs model-driven exploration.

**Follow-ups to expect:**
- Give an example needing an agent. *Debugging an unfamiliar codebase: you can't predict which files to read.*
- And one that doesn't. *Extracting fields from an invoice: a single call with a schema is enough.*

</details>

<details>
<summary><b>Q3. Explain the ReAct pattern and how you stop an agent from looping forever.</b></summary>

**Short answer:** ReAct alternates **Reasoning** and **Acting**: the model thinks about what to do, calls a tool, observes the result, and repeats. Stop runaway loops with step, time, and cost limits.

**Deeper answer:** The observe-then-think cycle lets the model correct itself using real results. Safeguards: a hard `max_steps`, a wall-clock timeout, a token or spend budget, detection of repeated identical calls, and a graceful "I couldn't finish" result when a limit is hit. In our script the loop test with a model that never stops was cut off at `max_steps`, and we observed a real model call the calculator three times for one simple percentage.

**Follow-ups to expect:**
- What should happen when a tool fails? *Return the error as an observation so the model can adjust, but count it against the step limit.*

</details>

<details>
<summary><b>Q4. How do you design good tools for an LLM?</b></summary>

**Short answer:** Few, focused tools with clear names, precise descriptions, tight schemas, and helpful error messages.

**Deeper answer:** The description is the model's only guide, so say what the tool does, when to use it, and what it returns. Prefer a small set: more tools mean more prompt tokens (our three tools cost 341 tokens per call) and more chances to pick the wrong one. Use enums and required fields to constrain arguments, return concise results (not huge dumps that fill the context), make tools idempotent where possible, and write error messages that tell the model how to fix its call. Validate every argument in code regardless of the schema.

**Follow-ups to expect:**
- What if you need 50 tools? *Group them, load them on demand (tool search), or route to specialized sub-agents.*

</details>

<details>
<summary><b>Q5. What are the security risks of agents, and how do you mitigate them?</b></summary>

**Short answer:** Excessive permissions, unsafe arguments, and prompt injection through content the agent reads. Mitigate with least privilege, validation, and human approval for irreversible actions.

**Deeper answer:** An agent that reads a web page, email, or document can be hijacked by instructions hidden in it (**indirect prompt injection**), and then use its tools on the attacker's behalf. Defenses: give each tool only the access it needs, validate and whitelist arguments (our calculator refuses anything but arithmetic, where `eval` would have run a shell command), treat tool output as untrusted data, require human confirmation for sending, deleting, or spending, log every tool call, and sandbox anything that executes code.

**Follow-ups to expect:**
- Can a system prompt alone prevent injection? *No. The model cannot reliably separate instructions from data, so the enforcement must live in your code and permissions.*

</details>

<details>
<summary><b>Q6. How do you evaluate an agent, and why are agents unreliable?</b></summary>

**Short answer:** Score task success, tool-selection correctness, number of steps, and cost on a fixed task set, including tasks that need no tools. Agents are unreliable because errors compound across steps.

**Deeper answer:** Measure the final answer and the *path*: did it use the right tools, in a sensible order, without over-calling? In our test the small model got 7 of 10 answers right but chose the tools correctly only 6 of 10 times, and every multi-tool task failed by skipping a step. Reliability compounds: at 90% per step, a five-step task succeeds about 59% of the time. So test the loop deterministically with a scripted model, evaluate the model on a task set, keep tasks short, and add checks between steps.

**Follow-ups to expect:**
- How do you improve a flaky agent? *Better tool descriptions, fewer tools, a stronger model, a fixed workflow for the predictable parts, and verification steps.*

</details>

---

## 🎯 Key Takeaways

After completing this module, you'll understand:

- That tool use means the model **requests** and your code **executes**
- How the agent loop works, and why every loop needs limits
- Why tool descriptions matter and why tool arguments are untrusted input
- How to test the loop separately from the model
- Why multi-step tasks are where agents fail, and how failures compound
- The main security risks and how least privilege and human approval address them

---

## 🔗 Go Deeper in the Weekly Curriculum

| Topic | Where |
|---|---|
| Retrieval as a tool | [Module 05](../05-rag/README.md) |
| Prompt injection | [Module 02](../../01-beginner/02-prompt-engineering/README.md) |
| ReAct, LangGraph, multi-agent systems | Week 8 — AI Agents & Tool Use *(planned)* |

---

## 🚀 What's Next?

**Module 08 • Guardrails, Safety & Hallucination**

The risks in the last table deserve more than a table. Next you'll build real input and output guardrails: detecting prompt injection, redacting personal data, and checking answers against their sources.

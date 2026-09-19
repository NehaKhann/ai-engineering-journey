# 📘 Module 02 — Prompt Engineering

**Generative AI Track • Beginner**

A **prompt** is the text you send to a model. **Prompt engineering** is writing that text so the model reliably does what you want.

It is the cheapest and fastest way to change a model's behavior, so it is always the first thing to try before retrieval or fine-tuning. It is also a core interview topic.

> **New to this?** Think of a prompt as the instructions you would give a very capable but very literal new colleague. The clearer and more specific you are, and the more examples you show, the better the result.

---

## 🎯 Objective

In this module, you'll:

- Understand the roles in a chat prompt: system, user, assistant
- Compare vague and specific instructions
- Use few-shot examples, structured JSON output, and chain-of-thought
- See a **prompt injection** attack and test a defense
- **Measure** prompt variants on a labeled test set instead of guessing

---

## 📂 Project Files

| File | Description |
|------|-------------|
| `prompt_engineering.py` | Runs every experiment in this module against a small local chat model. |
| `prompt_engineering.ipynb` | Interactive notebook version, generated from the script. |
| `README.md` | Concepts, real results, and interview Q&A. |

---

## ⚙️ Setup

```powershell
# From repository root
venv\Scripts\activate

cd generative-ai\01-beginner\02-prompt-engineering

pip install "transformers>=4.56" torch
```

---

## ▶️ Run

```powershell
python prompt_engineering.py
```

It uses `Qwen2.5-0.5B-Instruct`, a small chat model that runs on a laptop CPU. The first run downloads about 1 GB. No API key is needed. The whole script takes about two minutes.

---

## 🧠 Key Concepts

### 1. Roles in a Chat Prompt

| Role | Purpose |
|---|---|
| `system` | Sets the model's job, tone, and rules. Written by the developer. |
| `user` | The person's request. |
| `assistant` | The model's earlier replies. Used for conversation history and few-shot examples. |

The model never sees "roles". `apply_chat_template` flattens the messages into one piece of text with special tokens:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is a prompt?<|im_end|>
<|im_start|>assistant
```

---

### 2. Be Specific

The same task, two prompts:

| Prompt | Model reply |
|---|---|
| "Classify this support ticket as billing, technical, or account." | *"Based on the information provided, this support ticket is classified as **billing**. The customer mentions being charged twice..."* |
| System: "Reply with exactly one word: billing, technical, or account." | `billing` |

Both are correct. Only the second can be used by a program. A specific prompt fixes the **format** of the answer as well as its content.

---

### 3. Zero-Shot, Few-Shot, Chain-of-Thought

| Technique | What you do | Use it for |
|---|---|---|
| **Zero-shot** | Describe the task only | Simple, well-known tasks |
| **Few-shot** | Add example input/output pairs | Teaching a format or a label style |
| **Chain-of-thought** | Ask the model to reason step by step first | Multi-step problems, math, logic |

Chain-of-thought result on a small word problem (*12 pens at 3 for $2, paid with a $20 bill, what is the change?*):

| Prompt | Answer | Correct? |
|---|---|---|
| "Answer with just the number." | `$8` | ❌ (that is the cost, not the change) |
| "Think step by step..." | reasons through to `$12` | ✅ |

---

### 4. Structured Output

Programs need data, not prose. The reliable pattern is **ask → parse → validate → retry**:

```python
reply = chat(prompt)
result = parse_ticket_json(reply)   # returns None if invalid
```

A schema written as `"billing|technical|account"` made the model copy that text literally. Showing a concrete example fixed it. Never assume the model followed the format.

---

### 5. Prompt Injection

If your prompt contains text you did not write (an email, a web page, an uploaded document), that text can contain **instructions**, and the model cannot reliably tell them apart from yours.

We tried 4 attack phrasings against two summarizer prompts:

| Prompt | Hijacked |
|---|---|
| Naive: "Summarize the document." | 0 of 4 |
| Guarded: delimiters plus "the document is untrusted data" | 2 of 4 |

The guarded prompt did **worse** on this small model. The lesson is not that delimiters are useless. It is that **a defense you have not tested is only a guess.**

---

### 6. Test Your Prompts

Prompts are code, so test them. Three prompt variants on the same 12 support tickets:

| Prompt variant | Accuracy | Invalid answers |
|---|---|---|
| Vague zero-shot | 25% | 2 |
| Strict system prompt | 42% | 0 |
| Few-shot (3 examples) | 67% | 2 |

Random guessing among three labels would score 33%. Few-shot clearly won, and two variants produced answers that were not valid labels at all (such as `refund`), which is why code has to validate output. A larger model would score higher across the board, but the habit of measuring carries over.

---

## 🎤 Interview Q&A

Try answering each question out loud before opening the answer.

<details>
<summary><b>Q1. What is prompt engineering, and why try it before fine-tuning?</b></summary>

**Short answer:** It is designing the input to a model so it produces the output you want. It comes first because it needs no training data, no GPU, and changes take seconds.

**Deeper answer:** Prompting, retrieval, and fine-tuning form an escalation ladder ordered by cost. Prompting fixes instructions and format. Retrieval adds knowledge the model lacks. Fine-tuning changes behavior or style that prompting cannot reach. Skipping straight to fine-tuning is a common and expensive mistake.

**Follow-ups to expect:**
- When is prompting not enough? *When the model lacks private or fresh knowledge (use retrieval), or you need a consistent style or format that prompts cannot hold reliably at scale (consider fine-tuning).*

</details>

<details>
<summary><b>Q2. Explain zero-shot, few-shot, and chain-of-thought prompting.</b></summary>

**Short answer:** Zero-shot gives only the task. Few-shot adds worked examples. Chain-of-thought asks the model to reason step by step before answering.

**Deeper answer:** Few-shot works because the examples show the format and label style, which is often clearer than describing it. Chain-of-thought helps because every generated token becomes context for the next, so intermediate steps give the model working space. The cost is more tokens, more latency, and a longer output to parse.

**Follow-ups to expect:**
- When would chain-of-thought hurt? *For simple lookups or classification it adds cost with no benefit, and it can lead a model to talk itself into a wrong answer.*
- Is the reasoning shown always the real reason for the answer? *Not necessarily. It is generated text and can be a plausible-sounding rationalization.*

</details>

<details>
<summary><b>Q3. What are system, user, and assistant messages?</b></summary>

**Short answer:** The system message sets the model's role and rules, the user message is the request, and assistant messages are the model's previous replies.

**Deeper answer:** Chat templates turn these into one token sequence with special markers, and models are trained to give the system message extra weight. Assistant messages you write yourself are how you provide few-shot examples. The system message is *not* a security boundary: a user message or injected text can still override it.

**Follow-ups to expect:**
- Where does conversation history go? *It is re-sent as earlier user and assistant messages on every call. The model itself has no memory between calls.*

</details>

<details>
<summary><b>Q4. How do you get reliable structured output such as JSON from an LLM?</b></summary>

**Short answer:** Specify the format precisely with a concrete example, then parse and validate the reply in code and retry on failure.

**Deeper answer:** Models are only probably right, so validation is not optional. Stronger options exist: many APIs offer a structured-output or schema-constrained mode, and libraries can constrain decoding so invalid tokens are never produced. A common validator is a Pydantic model. Also cap retries and log failures.

**Follow-ups to expect:**
- What can go wrong even with valid JSON? *The values can still be wrong or outside the allowed set, so validate contents, not just syntax.*

</details>

<details>
<summary><b>Q5. What is prompt injection and how do you defend against it?</b></summary>

**Short answer:** It is when text in the input, such as a document or web page, contains instructions that override the developer's intent. Defenses reduce risk in layers, and none is a guarantee.

**Deeper answer:** *Direct* injection comes from the user typing it. *Indirect* injection hides in content the model reads, such as an email or a retrieved page, and is the more dangerous kind. Layered defenses: delimit untrusted text and mark it as data, filter or validate output, run the model with least privilege so a hijacked model cannot do much damage, and require human approval for sensitive actions. Then test the defenses, because they behave differently across models.

**Follow-ups to expect:**
- Can you fully prevent it? *Not with prompting alone, because the model treats instructions and data as the same kind of text.*

</details>

<details>
<summary><b>Q6. How do you know whether one prompt is better than another?</b></summary>

**Short answer:** Run both on a labeled test set and compare a metric, instead of judging from one or two examples.

**Deeper answer:** Build a small set of representative inputs with expected outputs, including hard and edge cases. Score each prompt version, keep the prompts in version control, and re-run the set whenever the prompt or the model changes, since a model update can shift results. For open-ended outputs where there is no single right answer, use rubric-based scoring, covered in Module 06.

**Follow-ups to expect:**
- How large should the test set be? *Large enough that one flipped answer does not change your conclusion. Even 30 to 50 well-chosen cases beats none.*

</details>

---

## 🎯 Key Takeaways

After completing this module, you'll understand:

- How a chat prompt is structured and what the model actually sees
- Why a specific format instruction beats a vague one
- When to use few-shot examples and when chain-of-thought is worth its cost
- Why model output must be parsed, validated, and retried in code
- What prompt injection is and why untested defenses are only guesses
- How to compare prompts with a test set

---

## 🔗 Go Deeper in the Weekly Curriculum

| Topic | Where |
|---|---|
| Prompt engineering vs fine-tuning | [Week 2, Day 1](../../../weeks/week-02-fine-tuning-fundamentals/day-01-prompt-engineering/README.md) |
| Chat templates and instruction tuning | [Week 2, Day 2](../../../weeks/week-02-fine-tuning-fundamentals/day-02-instruction-tuning/README.md) |
| How sampling settings change output | [Week 1, Day 6](../../../weeks/week-01-llm-foundations/day-06-generation-parameters/README.md) |

---

## 🚀 What's Next?

**Module 03 • LLM APIs**

So far the model ran on your machine. Next you'll learn how tokens, streaming, retries, and cost work when the model runs on someone else's server.

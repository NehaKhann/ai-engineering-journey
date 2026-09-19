"""Decide a ticket's category and priority. Three strategies, so we can measure which is best.

knn     Vote among the most similar past tickets (embeddings only, no LLM).
llm     Ask the chat model, with the same 4 fixed examples every time (prompting, Module 02).
hybrid  Ask the chat model, but show it the most similar past tickets as its examples.
        If the model's answer is unusable, fall back to knn.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass

CATEGORIES = ("billing", "technical", "account", "feedback")
PRIORITIES = ("low", "medium", "high")

SYSTEM_PROMPT = (
    "You triage customer support tickets. Reply ONLY with a JSON object.\n"
    "category must be one of: billing, technical, account, feedback.\n"
    "priority must be one of: low, medium, high.\n"
    'Example: {"category": "technical", "priority": "low"}'
)

# Used by the plain "llm" strategy: one fixed example per category.
FIXED_EXAMPLES = [
    {"text": "I was billed for a plan I cancelled.", "category": "billing", "priority": "high"},
    {"text": "The app crashes when I upload a photo.", "category": "technical", "priority": "high"},
    {"text": "How do I change my profile picture?", "category": "account", "priority": "low"},
    {"text": "It would be nice to have an export button.", "category": "feedback", "priority": "low"},
]


@dataclass
class TriageResult:
    category: str
    priority: str
    strategy: str  # the strategy that actually produced the answer
    fell_back: bool = False  # True when the LLM failed and knn answered instead


def parse_triage_json(reply):
    """Find the first {...} in a reply and check both fields. Returns a dict, or None if unusable."""
    match = re.search(r"\{.*?\}", reply, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    if data.get("category") in CATEGORIES and data.get("priority") in PRIORITIES:
        return {"category": data["category"], "priority": data["priority"]}
    return None


def vote(neighbors, field):
    """Weighted vote: each neighbor votes for its label, weighted by how similar it is."""
    totals = defaultdict(float)
    for ticket, similarity in neighbors:
        totals[ticket[field]] += similarity
    return max(totals, key=totals.get)


def triage_knn(text, index, k=5):
    neighbors = index.search(text, k)
    return TriageResult(vote(neighbors, "category"), vote(neighbors, "priority"), "knn")


def _example_turns(examples):
    """Turn example tickets into alternating user/assistant chat messages."""
    turns = []
    for ex in examples:
        turns.append({"role": "user", "content": ex["text"]})
        turns.append({"role": "assistant", "content": json.dumps({"category": ex["category"], "priority": ex["priority"]})})
    return turns


def ask_model(text, examples, chat_fn, max_attempts=2):
    """Ask the model for JSON. If the reply is unusable, tell it what was wrong and ask again.

    With greedy decoding the same prompt gives the same reply, so a retry must change something.
    Here the retry adds the bad reply and a correction to the conversation. Returns a dict or None.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *_example_turns(examples), {"role": "user", "content": text}]
    for _ in range(max_attempts):
        reply = chat_fn(messages, max_new_tokens=40)
        parsed = parse_triage_json(reply)
        if parsed:
            return parsed
        messages += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": "That was not valid. Reply with only a JSON object using an allowed category and priority."},
        ]
    return None


def triage_llm(text, index, chat_fn):
    parsed = ask_model(text, FIXED_EXAMPLES, chat_fn)
    if parsed:
        return TriageResult(parsed["category"], parsed["priority"], "llm")
    return _fallback(text, index)


def triage_hybrid(text, index, chat_fn, n_examples=4):
    examples = [ticket for ticket, _ in index.search(text, n_examples)][::-1]  # most similar last, nearest to the question
    parsed = ask_model(text, examples, chat_fn)
    if parsed:
        return TriageResult(parsed["category"], parsed["priority"], "hybrid")
    return _fallback(text, index)


def _fallback(text, index):
    result = triage_knn(text, index)
    result.fell_back = True
    return result


def triage(text, index, chat_fn, strategy="hybrid"):
    if strategy == "knn":
        return triage_knn(text, index)
    if strategy == "llm":
        return triage_llm(text, index, chat_fn)
    if strategy == "hybrid":
        return triage_hybrid(text, index, chat_fn)
    raise ValueError(f"Unknown strategy: {strategy!r}")

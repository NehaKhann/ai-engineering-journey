"""Answer a question from the stored documents, with citations (Module 05)."""

import re
from dataclasses import dataclass, field

NOT_FOUND = "I couldn't find that in the documents."
MIN_SIMILARITY = 0.2  # below this, no chunk is a real match. Off-topic questions scored under 0.15 in testing.

SYSTEM_PROMPT = (
    "Answer the question in one short sentence using only the numbered sources. "
    "Then add the source number like [1]. "
    'Only write "I don\'t know." if none of the sources mention the answer.'
)


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)  # the chunks the answer cites
    retrieved: list = field(default_factory=list)  # everything that was retrieved
    refused: bool = False
    reason: str = ""


def build_messages(question, hits):
    sources = "\n".join(f"[{i}] ({hit['source']}) {hit['text']}" for i, hit in enumerate(hits, 1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
    ]


def cited_numbers(text, n_sources):
    """The source numbers the model cited, like [1] or [2], ignoring numbers that do not exist."""
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", text) if 1 <= int(m) <= n_sources})


def answer_question(question, store, chat_fn, k=3, min_similarity=MIN_SIMILARITY):
    """Retrieve, then generate. Refuses without calling the model when nothing relevant was found."""
    hits = store.search(question, k)
    if not hits or hits[0]["score"] < min_similarity:
        return Answer(NOT_FOUND, retrieved=hits, refused=True, reason="no relevant chunk (similarity below threshold)")

    reply = chat_fn(build_messages(question, hits))
    if "don't know" in reply.lower():
        return Answer(reply, retrieved=hits, refused=True, reason="the model said the sources do not answer it")

    numbers = cited_numbers(reply, len(hits))
    sources = [hits[n - 1] for n in numbers] or hits[:1]  # if the model cited nothing, show the best match
    return Answer(reply, sources=sources, retrieved=hits)

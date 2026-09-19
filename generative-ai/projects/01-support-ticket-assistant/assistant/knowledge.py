"""The knowledge base of past tickets, searchable by meaning (Module 04)."""

import json
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TicketIndex:
    """Past tickets plus their embeddings. `embed_fn` is passed in so tests can use a fake one."""

    def __init__(self, tickets, embed_fn):
        self.tickets = tickets
        self.embed_fn = embed_fn
        self.vectors = embed_fn([t["text"] for t in tickets])  # one unit vector per ticket

    @classmethod
    def from_file(cls, embed_fn, path=DATA_DIR / "past_tickets.jsonl"):
        return cls(load_jsonl(path), embed_fn)

    def search(self, text, k=3):
        """Return the k most similar past tickets as (ticket, similarity), best first."""
        scores = self.vectors @ self.embed_fn([text])[0]
        best = np.argsort(scores)[::-1][:k]
        return [(self.tickets[i], float(scores[i])) for i in best]

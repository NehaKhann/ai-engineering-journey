"""Unit tests that need no models: a fake embedder and a fake chat function stand in for them.

Run from the project folder:  python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.knowledge import TicketIndex  # noqa: E402
from assistant.reply import build_reply_messages, draft_reply, grounding_score, novelty_score  # noqa: E402
from assistant.triage import ask_model, parse_triage_json, triage, vote  # noqa: E402

# A fake embedder: each word maps to one axis, so texts sharing words are similar.
VOCAB = ["charged", "twice", "crash", "app", "login", "password", "dark", "mode"]


def fake_embed(texts):
    vectors = np.array([[float(word in text.lower()) for word in VOCAB] for text in texts])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


TICKETS = [
    {"text": "charged twice", "category": "billing", "priority": "high", "resolution": "Refunded."},
    {"text": "app crash", "category": "technical", "priority": "high", "resolution": "Update the app."},
    {"text": "login password", "category": "account", "priority": "low", "resolution": "Reset it."},
    {"text": "dark mode", "category": "feedback", "priority": "low", "resolution": "Logged."},
]


class ParseTests(unittest.TestCase):
    def test_accepts_valid_json_inside_prose(self):
        reply = 'Sure! {"category": "billing", "priority": "high"} Hope that helps.'
        self.assertEqual(parse_triage_json(reply), {"category": "billing", "priority": "high"})

    def test_rejects_copied_schema(self):
        self.assertIsNone(parse_triage_json('{"category": "billing|technical", "priority": "high"}'))

    def test_rejects_unknown_label_and_bad_json(self):
        self.assertIsNone(parse_triage_json('{"category": "refund", "priority": "high"}'))
        self.assertIsNone(parse_triage_json("{not json}"))
        self.assertIsNone(parse_triage_json("no json at all"))


class SearchAndVoteTests(unittest.TestCase):
    def setUp(self):
        self.index = TicketIndex(TICKETS, fake_embed)

    def test_search_returns_most_similar_first(self):
        results = self.index.search("I was charged twice", k=2)
        self.assertEqual(results[0][0]["category"], "billing")
        self.assertGreaterEqual(results[0][1], results[1][1])

    def test_vote_weights_by_similarity(self):
        neighbors = [({"category": "a"}, 0.9), ({"category": "b"}, 0.3), ({"category": "b"}, 0.3)]
        self.assertEqual(vote(neighbors, "category"), "a")  # one strong vote beats two weak ones


class TriageTests(unittest.TestCase):
    def setUp(self):
        self.index = TicketIndex(TICKETS, fake_embed)

    def test_knn_strategy_needs_no_llm(self):
        def no_llm(*args, **kwargs):
            raise AssertionError("knn must not call the chat model")

        result = triage("my app will crash", self.index, no_llm, "knn")
        self.assertEqual(result.category, "technical")

    def test_hybrid_uses_model_answer_when_valid(self):
        chat = lambda messages, max_new_tokens=40: '{"category": "feedback", "priority": "low"}'
        result = triage("charged twice", self.index, chat, "hybrid")
        self.assertEqual((result.category, result.strategy, result.fell_back), ("feedback", "hybrid", False))

    def test_hybrid_falls_back_to_knn_when_model_is_unusable(self):
        chat = lambda messages, max_new_tokens=40: "I am not sure."
        result = triage("charged twice", self.index, chat, "hybrid")
        self.assertTrue(result.fell_back)
        self.assertEqual(result.category, "billing")  # knn answered

    def test_retry_sends_a_correction_the_second_time(self):
        seen = []

        def chat(messages, max_new_tokens=40):
            seen.append(list(messages))
            return "nope" if len(seen) == 1 else '{"category": "billing", "priority": "low"}'

        self.assertEqual(ask_model("hi", [], chat)["category"], "billing")
        self.assertEqual(len(seen), 2)
        self.assertGreater(len(seen[1]), len(seen[0]))  # the retry prompt is different, not a repeat

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            triage("x", self.index, None, "magic")


class ReplyTests(unittest.TestCase):
    RESOLUTION = "The duplicate charge was reversed and the refund arrives in 5 to 7 business days."
    NEIGHBORS = [({"resolution": RESOLUTION}, 0.8)]

    def test_reply_prompt_contains_only_the_supplied_resolution(self):
        messages = build_reply_messages("charged twice", self.RESOLUTION)
        self.assertIn("duplicate charge", messages[1]["content"])

    def test_faithful_draft_is_accepted(self):
        good = "Sorry about that. The duplicate charge was reversed and the refund arrives in 5 to 7 business days."
        reply, source = draft_reply("I was charged twice", self.NEIGHBORS, lambda m, max_new_tokens=80: good)
        self.assertEqual((reply, source), (good, "model"))

    def test_draft_that_ignores_the_resolution_uses_template(self):
        ignores = "Could you tell me more about what happened?"
        reply, source = draft_reply("I was charged twice", self.NEIGHBORS, lambda m, max_new_tokens=80: ignores)
        self.assertEqual(source, "template")
        self.assertIn(self.RESOLUTION, reply)

    def test_draft_with_invented_content_uses_template(self):
        invents = "The duplicate charge was reversed. Our engineers celebrated this remarkable victory across galaxies."
        _, source = draft_reply("I was charged twice", self.NEIGHBORS, lambda m, max_new_tokens=80: invents)
        self.assertEqual(source, "template")

    def test_weak_match_escalates_without_calling_the_model(self):
        def no_llm(*args, **kwargs):
            raise AssertionError("must not call the model when nothing matches")

        _, source = draft_reply("moon", [({"resolution": self.RESOLUTION}, 0.1)], no_llm)
        self.assertEqual(source, "escalate")

    def test_scores(self):
        self.assertEqual(grounding_score(self.RESOLUTION, self.RESOLUTION), 1.0)
        self.assertEqual(novelty_score(self.RESOLUTION, self.RESOLUTION, ""), 0.0)


if __name__ == "__main__":
    unittest.main()

"""Unit tests that need no models: a fake embedder and a fake chat function stand in for them.

Run from the project folder:  python -m unittest discover -s tests -v
"""

import sys
import tempfile
import unittest
import uuid
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docs_qa.chunking import chunk_record  # noqa: E402
from docs_qa.evaluate import evaluate  # noqa: E402
from docs_qa.loaders import find_documents, load_document  # noqa: E402
from docs_qa.rag import NOT_FOUND, answer_question, cited_numbers  # noqa: E402
from docs_qa.store import DocumentStore  # noqa: E402

DIM = 64


def fake_embed(texts):
    """Bag of words hashed into 64 slots, so texts that share words are similar."""
    vectors = np.zeros((len(texts), DIM))
    for row, text in enumerate(texts):
        for word in text.lower().replace(".", " ").replace("?", " ").replace(":", " ").split():
            vectors[row, zlib.crc32(word.encode()) % DIM] += 1
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def new_store():
    return DocumentStore(None, fake_embed, collection=f"test_{uuid.uuid4().hex}")


VACATION = {"source": "vacation.md", "page": None, "text": "Vacation policy.\nNew employees receive twenty days of paid vacation each year and may carry over five unused days."}
EXPENSES = {"source": "expenses.md", "page": None, "text": "Expense policy.\nHotels are reimbursed up to one hundred eighty dollars per night while traveling on company business."}


def minimal_pdf(text):
    """Build a tiny one-page PDF containing `text`, so the PDF loader can be tested without any file."""
    stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref_at)
    return out


class ChunkingTests(unittest.TestCase):
    def test_chunks_respect_size_and_overlap(self):
        record = {"source": "a.txt", "page": None, "text": "Title.\n" + " ".join(f"w{i}" for i in range(100))}
        chunks = chunk_record(record, size=30, overlap=10)
        self.assertTrue(all(len(c["text"].split()) <= 30 for c in chunks))
        first, second = chunks[0]["text"].split(), chunks[1]["text"].split()
        self.assertEqual(first[-10:], second[:10])  # the overlap

    def test_every_chunk_knows_its_source_and_title(self):
        chunks = chunk_record(VACATION, size=10, overlap=2)
        self.assertTrue(all(c["source"] == "vacation.md" and c["title"] == "Vacation policy" for c in chunks))

    def test_overlap_must_be_smaller_than_size(self):
        with self.assertRaises(ValueError):
            chunk_record(VACATION, size=10, overlap=10)


class LoaderTests(unittest.TestCase):
    def test_loads_text_and_ignores_unsupported_files(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "a.md").write_text("Hello world", encoding="utf-8")
            (Path(folder) / "b.png").write_bytes(b"\x89PNG")
            (Path(folder) / "empty.txt").write_text("   ", encoding="utf-8")
            found = find_documents(folder)
            self.assertEqual([p.name for p in found], ["a.md", "empty.txt"])
            self.assertEqual(load_document(found[0])[0]["text"], "Hello world")
            self.assertEqual(load_document(found[1]), [])  # blank files produce nothing

    def test_reads_pdf_text_with_page_numbers(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            self.skipTest("pypdf is not installed")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "doc.pdf"
            path.write_bytes(minimal_pdf("Hello vacation"))
            records = load_document(path)
        self.assertEqual(records[0]["page"], 1)
        self.assertIn("Hello vacation", records[0]["text"])


class StoreTests(unittest.TestCase):
    def test_search_finds_the_relevant_chunk_first(self):
        store = new_store()
        store.add_records([VACATION, EXPENSES])
        hits = store.search("how many days of paid vacation", k=2)
        self.assertEqual(hits[0]["source"], "vacation.md")
        self.assertGreater(hits[0]["score"], hits[1]["score"])

    def test_reingesting_a_source_replaces_its_old_chunks(self):
        store = new_store()
        store.add_records([VACATION, EXPENSES])
        before = store.sources()
        updated = dict(VACATION, text="Vacation policy.\nNew employees receive twenty-five days of vacation.")
        store.add_records([updated])
        self.assertEqual(store.sources()["expenses.md"], before["expenses.md"])  # untouched
        texts = " ".join(h["text"] for h in store.search("vacation days", k=5) if h["source"] == "vacation.md")
        self.assertIn("twenty-five", texts)
        self.assertNotIn("carry over five", texts)  # the old text is gone

    def test_empty_store_returns_no_hits(self):
        self.assertEqual(new_store().search("anything"), [])


class RagTests(unittest.TestCase):
    def setUp(self):
        self.store = new_store()
        self.store.add_records([VACATION, EXPENSES])

    def test_cited_numbers_ignores_impossible_citations(self):
        self.assertEqual(cited_numbers("Twenty days. [1] Also [7] and [2]", 3), [1, 2])

    def test_refuses_without_calling_the_model_when_nothing_matches(self):
        def no_model(messages):
            raise AssertionError("the model must not be called")

        result = answer_question("zzz qqq xxx", self.store, no_model, min_similarity=0.5)
        self.assertEqual(result.text, NOT_FOUND)
        self.assertTrue(result.refused)

    def test_answer_lists_only_the_cited_sources(self):
        chat = lambda messages: "Twenty days per year. [1]"
        result = answer_question("how many vacation days", self.store, chat, k=2, min_similarity=0.0)
        self.assertEqual([s["source"] for s in result.sources], ["vacation.md"])

    def test_an_uncited_answer_still_shows_the_best_source(self):
        result = answer_question("how many vacation days", self.store, lambda m: "Twenty days.", k=2, min_similarity=0.0)
        self.assertEqual(result.sources[0]["source"], "vacation.md")

    def test_model_saying_i_dont_know_counts_as_a_refusal(self):
        result = answer_question("how many vacation days", self.store, lambda m: "I don't know.", min_similarity=0.0)
        self.assertTrue(result.refused)
        self.assertEqual(result.sources, [])


class EvaluateTests(unittest.TestCase):
    def test_separates_retrieval_failures_from_generation_failures(self):
        store = new_store()
        store.add_records([VACATION, EXPENSES])
        golden = [
            {"question": "how many days of paid vacation", "evidence": "twenty days", "expected": ["twenty"]},  # answered right
            {"question": "hotels reimbursed per night", "evidence": "one hundred eighty", "expected": ["180"]},  # retrieved, model wrong
            {"question": "vacation days", "evidence": "text that appears nowhere", "expected": ["x"]},  # retrieval miss
            {"question": "qqq zzz www", "evidence": None, "expected": None},  # unanswerable
        ]

        def chat(messages):
            return "Twenty days. [1]" if "vacation" in messages[1]["content"].split("Question:")[1] else "I am unsure. [1]"

        records, summary = evaluate(store, golden, chat, k=2)
        self.assertEqual(summary["questions"], 3)
        self.assertEqual(summary["wrong_because_retrieval"], 1)
        self.assertEqual(summary["wrong_because_generation"], 1)
        self.assertEqual(summary["unanswerable"], 1)
        self.assertEqual(summary["correctly_declined"], 1)
        self.assertAlmostEqual(summary["answer_accuracy"], 1 / 3)


if __name__ == "__main__":
    unittest.main()

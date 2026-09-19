"""Measure the system on a golden set (Module 06): retrieval and generation, separately."""

import json
from pathlib import Path

from .rag import answer_question


def load_golden(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contains_any(text, needles):
    text = text.lower()
    return any(n.lower() in text for n in needles)


def evaluate(store, golden, chat_fn, k=3):
    """Score every golden question. Answerable ones have `evidence` and `expected`; unanswerable ones have neither."""
    records = []
    for item in golden:
        answer = answer_question(item["question"], store, chat_fn, k=k)
        record = {"question": item["question"], "answer": answer.text, "refused": answer.refused}

        if item["expected"] is None:  # a question the documents cannot answer
            record["answerable"] = False
            record["correct"] = answer.refused
        else:
            texts = [hit["text"] for hit in answer.retrieved]
            rank = next((i for i, t in enumerate(texts, 1) if item["evidence"].lower() in t.lower()), None)
            record["answerable"] = True
            record["retrieval_rank"] = rank
            record["correct"] = (not answer.refused) and contains_any(answer.text, item["expected"])
        records.append(record)
    return records, summarize(records, k)


def summarize(records, k):
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    wrong = [r for r in answerable if not r["correct"]]
    return {
        "questions": len(answerable),
        "hit@1": sum(r["retrieval_rank"] == 1 for r in answerable) / len(answerable),
        f"hit@{k}": sum(r["retrieval_rank"] is not None for r in answerable) / len(answerable),
        "mrr": sum(1 / r["retrieval_rank"] for r in answerable if r["retrieval_rank"]) / len(answerable),
        "answer_accuracy": sum(r["correct"] for r in answerable) / len(answerable),
        "wrong_because_retrieval": sum(r["retrieval_rank"] is None for r in wrong),
        "wrong_because_generation": sum(r["retrieval_rank"] is not None for r in wrong),
        "unanswerable": len(unanswerable),
        "correctly_declined": sum(r["correct"] for r in unanswerable),
    }


def print_report(records, summary, k):
    print(f"\nRetrieval   Hit@1 {summary['hit@1']:.0%}   Hit@{k} {summary[f'hit@{k}']:.0%}   MRR {summary['mrr']:.2f}")
    print(f"Answers     {summary['answer_accuracy']:.0%} correct on {summary['questions']} answerable questions")
    print(f"            of the wrong ones, {summary['wrong_because_retrieval']} were retrieval failures "
          f"and {summary['wrong_because_generation']} were generation failures")
    print(f"Unanswerable {summary['correctly_declined']}/{summary['unanswerable']} correctly declined\n")
    for r in records:
        if not r["correct"]:
            why = "declined an unanswerable question wrongly answered" if not r["answerable"] else (
                "retrieval missed it" if r["retrieval_rank"] is None else "retrieved, but the answer was wrong")
            print(f"  WRONG ({why})\n    Q: {r['question']}\n    A: {r['answer'][:110]}")

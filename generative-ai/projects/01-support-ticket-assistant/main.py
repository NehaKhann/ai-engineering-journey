"""Support Ticket Assistant: triage tickets, find similar past tickets, draft replies.

Examples (run from this folder):
    python main.py triage "I was charged twice this month"
    python main.py triage "I was charged twice this month" --draft
    python main.py similar "my login stopped working"
    python main.py evaluate
"""

import argparse
import json
from pathlib import Path

from assistant.evaluate import evaluate, print_summary, save_chart
from assistant.knowledge import DATA_DIR, TicketIndex, load_jsonl
from assistant.models import chat, embed
from assistant.reply import draft_reply
from assistant.triage import triage

HERE = Path(__file__).resolve().parent


def show_similar(neighbors):
    print("\nSimilar past tickets:")
    for ticket, similarity in neighbors:
        print(f"  [{similarity:.2f}] ({ticket['category']}/{ticket['priority']}) {ticket['text']}")
        print(f"         Resolution: {ticket['resolution']}")


def command_triage(args, index):
    result = triage(args.text, index, chat, args.strategy)
    print(f"\nTicket:   {args.text}")
    print(f"Category: {result.category}")
    print(f"Priority: {result.priority}")
    note = " (LLM answer was unusable, used knn instead)" if result.fell_back else ""
    print(f"Strategy: {result.strategy}{note}")

    neighbors = index.search(args.text, args.k)
    show_similar(neighbors)
    if args.draft:
        reply, source = draft_reply(args.text, neighbors, chat)
        print(f"\nDraft reply (source: {source}):")
        print(reply)


def command_similar(args, index):
    print(f"\nTicket: {args.text}")
    show_similar(index.search(args.text, args.k))


def command_evaluate(args, index):
    eval_tickets = load_jsonl(DATA_DIR / "eval_tickets.jsonl")
    summaries = evaluate(index, eval_tickets, chat)
    print_summary(summaries)

    for name, s in summaries.items():
        if s["category_mistakes"]:
            print(f"\nCategory mistakes for '{name}':")
            for text, expected, got in s["category_mistakes"]:
                print(f"  expected {expected:<10} got {got:<10} | {text}")

    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "evaluation.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (HERE / "assets").mkdir(exist_ok=True)
    save_chart(summaries, HERE / "assets" / "strategy_comparison.png")
    print(f"\nSaved {results_dir / 'evaluation.json'} and assets/strategy_comparison.png")


def build_parser():
    parser = argparse.ArgumentParser(description="Support Ticket Assistant")
    commands = parser.add_subparsers(dest="command", required=True)

    triage_parser = commands.add_parser("triage", help="classify a ticket and show similar past tickets")
    triage_parser.add_argument("text")
    triage_parser.add_argument("--strategy", choices=["knn", "llm", "hybrid"], default="knn")
    triage_parser.add_argument("--draft", action="store_true", help="also draft a reply")
    triage_parser.add_argument("-k", type=int, default=3, help="number of similar tickets to show")
    triage_parser.set_defaults(handler=command_triage)

    similar_parser = commands.add_parser("similar", help="find similar past tickets")
    similar_parser.add_argument("text")
    similar_parser.add_argument("-k", type=int, default=3)
    similar_parser.set_defaults(handler=command_similar)

    evaluate_parser = commands.add_parser("evaluate", help="compare strategies on held-out tickets")
    evaluate_parser.set_defaults(handler=command_evaluate)
    return parser


def main():
    args = build_parser().parse_args()
    print("Loading models and indexing past tickets...")
    index = TicketIndex.from_file(embed)
    args.handler(args, index)


if __name__ == "__main__":
    main()

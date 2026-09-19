"""Ask Your Documents: ask questions about your own files and get answers with citations.

Examples (run from this folder):
    python main.py ingest sample_docs
    python main.py ask "How many vacation days do new employees get?"
    python main.py evaluate
    python main.py sources
"""

import argparse
import json
from pathlib import Path

from docs_qa.evaluate import evaluate, load_golden, print_report
from docs_qa.loaders import find_documents, load_document
from docs_qa.models import chat, embed
from docs_qa.rag import answer_question
from docs_qa.store import DocumentStore

HERE = Path(__file__).resolve().parent
INDEX_DIR = HERE / ".index"


def open_store():
    return DocumentStore(INDEX_DIR, embed)


def command_ingest(args):
    store = open_store()
    files = find_documents(args.folder)
    if not files:
        raise SystemExit(f"No .txt, .md or .pdf files found in {args.folder}")
    total = 0
    for path in files:
        added = store.add_records(load_document(path), size=args.size, overlap=args.overlap)
        total += added
        print(f"  {path.name}: {added} chunks")
    print(f"Indexed {total} chunks from {len(files)} files into {INDEX_DIR.name}/")


def command_ask(args):
    store = open_store()
    result = answer_question(args.question, store, chat, k=args.k)
    print(f"\nQ: {args.question}")
    print(f"A: {result.text}")
    if result.refused:
        print(f"   ({result.reason})")
    for hit in result.sources:
        page = f", page {hit['page']}" if hit["page"] else ""
        print(f"   source: {hit['source']}{page}  (similarity {hit['score']:.2f})")
    if args.show_retrieved:
        print("\nRetrieved chunks:")
        for hit in result.retrieved:
            print(f"   [{hit['score']:.2f}] {hit['source']}: {hit['text'][:90]}...")


def command_evaluate(args):
    records, summary = evaluate(open_store(), load_golden(args.golden), chat, k=args.k)
    print_report(records, summary, args.k)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    print(f"\nSaved {out / 'evaluation.json'}")


def command_sources(args):
    sources = open_store().sources()
    if not sources:
        print("Nothing indexed yet. Run: python main.py ingest <folder>")
    for name, count in sources.items():
        print(f"  {name}: {count} chunks")


def build_parser():
    parser = argparse.ArgumentParser(description="Ask Your Documents")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="index every .txt, .md and .pdf file in a folder")
    ingest.add_argument("folder")
    ingest.add_argument("--size", type=int, default=60, help="chunk size in words")
    ingest.add_argument("--overlap", type=int, default=15, help="words shared between neighboring chunks")
    ingest.set_defaults(handler=command_ingest)

    ask = commands.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=3, help="chunks to retrieve")
    ask.add_argument("--show-retrieved", action="store_true", help="also print every retrieved chunk")
    ask.set_defaults(handler=command_ask)

    evaluate_parser = commands.add_parser("evaluate", help="score the system on the golden set")
    evaluate_parser.add_argument("--golden", default=str(HERE / "data" / "golden.json"))
    evaluate_parser.add_argument("-k", type=int, default=3)
    evaluate_parser.set_defaults(handler=command_evaluate)

    commands.add_parser("sources", help="list indexed files").set_defaults(handler=command_sources)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.handler(arguments)

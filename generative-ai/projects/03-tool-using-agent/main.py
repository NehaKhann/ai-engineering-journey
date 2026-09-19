"""A file-assistant agent with sandboxed tools and human approval for anything that changes files.

Examples (run from this folder):
    python main.py ask "Who attended the kickoff meeting?"
    python main.py ask "Save a note titled ideas with the text buy more tests"
    python main.py ask "Summarize vendor_email.txt"
    python main.py evaluate
"""

import argparse
import json
from pathlib import Path

from agent.evaluate import evaluate, print_report
from agent.loop import run_agent
from agent.models import local_model
from agent.tools import Workspace

HERE = Path(__file__).resolve().parent


def ask_the_user(name, arguments):
    print(f"\n  The agent wants to run: {name}({json.dumps(arguments)})")
    return input("  Allow this? [y/N] ").strip().lower() in {"y", "yes"}


def command_ask(args):
    workspace = Workspace(args.workspace)
    approve = (lambda name, arguments: True) if args.auto_approve else ask_the_user
    result = run_agent(args.question, local_model, workspace.tools(), approve_fn=approve, max_steps=args.max_steps)

    print(f"\nQ: {args.question}")
    for step in result.steps:
        gate = "" if step.approved is None else (" [approved]" if step.approved else " [DECLINED]")
        print(f"  step {step.index}: {step.tool}({json.dumps(step.arguments)}){gate}\n      -> {step.result[:100]!r}")
    print(f"A: {result.answer or '(no answer: stopped because ' + result.status + ')'}")


def command_evaluate(args):
    report = evaluate(args.workspace, local_model)
    print_report(report)
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out / 'evaluation.json'}")


def build_parser():
    parser = argparse.ArgumentParser(description="File-assistant agent")
    parser.add_argument("--workspace", default=str(HERE / "workspace"), help="the folder the agent may see")
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="ask the agent a question")
    ask.add_argument("question")
    ask.add_argument("--auto-approve", action="store_true", help="approve every risky action WITHOUT asking (unsafe; for demos)")
    ask.add_argument("--max-steps", type=int, default=6)
    ask.set_defaults(handler=command_ask)

    commands.add_parser("evaluate", help="score the agent on a fixed task set").set_defaults(handler=command_evaluate)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.handler(arguments)

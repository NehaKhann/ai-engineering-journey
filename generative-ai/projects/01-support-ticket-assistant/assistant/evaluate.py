"""Measure each triage strategy on tickets it has never seen (Module 02: test your prompts)."""

import time

from .triage import triage


def evaluate(index, eval_tickets, chat_fn, strategies=("knn", "llm", "hybrid")):
    """Run every strategy over every eval ticket. Returns one summary dict per strategy."""
    summaries = {}
    for strategy in strategies:
        category_correct = priority_correct = fallbacks = 0
        mistakes = []
        start = time.time()
        for ticket in eval_tickets:
            result = triage(ticket["text"], index, chat_fn, strategy)
            category_correct += result.category == ticket["category"]
            priority_correct += result.priority == ticket["priority"]
            fallbacks += result.fell_back
            if result.category != ticket["category"]:
                mistakes.append((ticket["text"], ticket["category"], result.category))
        n = len(eval_tickets)
        summaries[strategy] = {
            "category_accuracy": category_correct / n,
            "priority_accuracy": priority_correct / n,
            "fallbacks": fallbacks,
            "seconds": round(time.time() - start, 1),
            "category_mistakes": mistakes,
            "n": n,
        }
    return summaries


def save_chart(summaries, path):
    """Bar chart of category and priority accuracy for each strategy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summaries)
    x = range(len(names))
    plt.figure(figsize=(7, 4.5))
    plt.bar([i - 0.2 for i in x], [summaries[n]["category_accuracy"] * 100 for n in names], 0.4, label="Category")
    plt.bar([i + 0.2 for i in x], [summaries[n]["priority_accuracy"] * 100 for n in names], 0.4, label="Priority")
    plt.xticks(list(x), names)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title(f"Triage strategies on {summaries[names[0]]['n']} held-out tickets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def print_summary(summaries):
    n = next(iter(summaries.values()))["n"]
    print(f"\nResults on {n} held-out tickets\n")
    print(f"{'Strategy':<10}{'Category':<11}{'Priority':<11}{'LLM fallbacks':<15}Time")
    print("-" * 55)
    for name, s in summaries.items():
        print(f"{name:<10}{s['category_accuracy']:<11.0%}{s['priority_accuracy']:<11.0%}{s['fallbacks']:<15}{s['seconds']}s")

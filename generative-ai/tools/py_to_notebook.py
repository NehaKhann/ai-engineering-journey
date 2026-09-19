"""Convert a percent-format Python script into a Jupyter / Colab notebook.

Every module keeps ONE source of truth: a runnable .py file. This tool builds the
matching .ipynb from it so the script and the notebook never drift apart.

Percent format:

    # pip: transformers torch        <- optional first line, becomes a Colab install cell
    # %% [markdown]                   <- start of a markdown cell (lines start with "# ")
    # # Title
    # %%                              <- start of a code cell
    print("hello")

Usage (from repository root):

    python generative-ai/tools/py_to_notebook.py generative-ai/01-beginner/02-prompt-engineering/prompt_engineering.py
"""

import json
import sys
from pathlib import Path

REPO = "NehaKhann/ai-engineering-journey"
COLAB = "https://colab.research.google.com/github"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"


def make_cell(kind, lines):
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return None
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def parse(text):
    pip_packages = None
    cells, kind, buffer = [], None, []

    def flush():
        nonlocal buffer
        if kind is not None:
            cell = make_cell(kind, buffer)
            if cell:
                cells.append(cell)
        buffer = []

    for line in text.splitlines():
        if line.startswith("# pip:") and pip_packages is None and not cells and kind is None:
            pip_packages = line[len("# pip:"):].strip()
        elif line.startswith("# %% [markdown]"):
            flush()
            kind = "markdown"
        elif line.startswith("# %%"):
            flush()
            kind = "code"
        elif kind == "markdown":
            buffer.append(line[2:] if line.startswith("# ") else "")
        elif kind == "code":
            buffer.append(line)
    flush()
    return pip_packages, cells


def is_percent_format(script):
    """True if the script uses the '# %%' cell markers this tool understands."""
    return any(line.startswith("# %%") for line in Path(script).read_text(encoding="utf-8").splitlines())


def build_notebook(script):
    """Return (notebook_text, cell_count, output_path) for a script, without writing anything."""
    script = Path(script).resolve()
    repo_root = next(p for p in script.parents if (p / ".git").exists())
    relative = script.with_suffix(".ipynb").relative_to(repo_root).as_posix()

    pip_packages, cells = parse(script.read_text(encoding="utf-8"))

    badge = f"[![Open In Colab]({BADGE})]({COLAB}/{REPO}/blob/main/{relative})"
    if cells and cells[0]["cell_type"] == "markdown":
        cells[0]["source"][-1] += "\n"
        cells[0]["source"] += ["\n", badge]

    if pip_packages:
        # Quote version specifiers such as transformers>=4.56, which a shell would read as a redirect.
        quoted = " ".join(f'"{p}"' if any(c in p for c in "<>=") else p for p in pip_packages.split())
        install = make_cell("code", [f"!pip install -q {quoted}"])
        cells.insert(1 if cells and cells[0]["cell_type"] == "markdown" else 0, install)

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", len(cells), script.with_suffix(".ipynb")


def main(script):
    text, count, out = build_notebook(script)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out.name} ({count} cells)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        main(path)

"""Check that every notebook is valid and still matches the script it was built from.

Each module keeps ONE source of truth (its .py script) and generates the .ipynb from it. If someone
edits the script and forgets to regenerate the notebook, the two drift apart. This catches that.

Usage (from the repository root):   python generative-ai/tools/check_notebooks.py
Exit code 0 means everything is fine. To fix a mismatch, run:
    python generative-ai/tools/py_to_notebook.py <path to the script>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py_to_notebook import build_notebook, is_percent_format  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # the generative-ai folder


def main():
    problems, checked, skipped = [], 0, []
    for notebook in sorted(ROOT.rglob("*.ipynb")):
        relative = notebook.relative_to(ROOT).as_posix()
        try:
            data = json.loads(notebook.read_text(encoding="utf-8"))
            assert data["nbformat"] == 4 and isinstance(data["cells"], list) and data["cells"]
        except Exception as error:
            problems.append(f"{relative}: not a valid notebook ({type(error).__name__}: {error})")
            continue

        script = notebook.with_suffix(".py")
        if not script.exists() or not is_percent_format(script):
            skipped.append(relative)  # written by hand, not generated from a script
            continue

        expected, _, _ = build_notebook(script)
        if notebook.read_text(encoding="utf-8").replace("\r\n", "\n") != expected:
            problems.append(f"{relative}: out of date. Regenerate it with py_to_notebook.py")
        else:
            checked += 1

    print(f"{checked} notebooks match their scripts; {len(skipped)} skipped (not generated): {skipped}")
    for problem in problems:
        print("PROBLEM:", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

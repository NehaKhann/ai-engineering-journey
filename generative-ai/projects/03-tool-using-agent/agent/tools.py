"""The agent's tools. Each one is a plain function bound to a sandboxed workspace."""

import ast
import operator
import re
from dataclasses import dataclass
from pathlib import Path

from .sandbox import SandboxError, check_readable, resolve_inside

MAX_READ_CHARS = 3000
MAX_SEARCH_RESULTS = 8

_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}


def safe_calculate(expression):
    """Arithmetic only, evaluated by walking the syntax tree. Never eval()."""

    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("exponent too large")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return -walk(node.operand) if isinstance(node.op, ast.USub) else walk(node.operand)
        raise ValueError("only numbers and + - * / % ** are allowed")

    result = walk(ast.parse(expression.replace(",", "").replace("^", "**"), mode="eval").body)
    return str(round(result, 6)) if isinstance(result, float) else str(result)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments
    function: callable
    requires_approval: bool = False  # tools with side effects must be approved by a human

    def spec(self):
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def _schema(properties, required):
    return {"type": "object", "properties": {k: {"type": "string", "description": v} for k, v in properties.items()}, "required": required}


class Workspace:
    """A folder the agent may read from, with a `notes/` subfolder it may write to (after approval)."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.notes_dir = self.root / "notes"

    # ---- read-only tools ----
    def list_files(self, folder="."):
        base = resolve_inside(self.root, folder)
        if not base.is_dir():
            raise SandboxError("not a folder")
        lines = [f"{p.relative_to(self.root).as_posix()} ({p.stat().st_size} bytes)" for p in sorted(base.rglob("*")) if p.is_file()]
        return "\n".join(lines) or "(no files)"

    def read_file(self, path):
        text = check_readable(resolve_inside(self.root, path)).read_text(encoding="utf-8", errors="replace")
        return text[:MAX_READ_CHARS] + ("\n[truncated]" if len(text) > MAX_READ_CHARS else "")

    def search_files(self, query):
        if not query or not query.strip():
            raise SandboxError("query must not be empty")
        pattern, matches = re.compile(re.escape(query.strip()), re.IGNORECASE), []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".csv", ".json"}:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line):
                    matches.append(f"{path.relative_to(self.root).as_posix()}:{number}: {line.strip()[:120]}")
                    if len(matches) >= MAX_SEARCH_RESULTS:
                        return "\n".join(matches)
        return "\n".join(matches) or "no matches"

    # ---- a tool with a side effect ----
    def write_note(self, title, content):
        slug = re.sub(r"[^a-z0-9_-]+", "-", str(title).lower()).strip("-")[:50]
        if not slug:
            raise SandboxError("the title must contain letters or digits")
        self.notes_dir.mkdir(exist_ok=True)
        target = resolve_inside(self.root, f"notes/{slug}.md")  # the file name is built by us from the sanitized slug
        if target.exists():
            raise SandboxError(f"a note named {slug!r} already exists; choose a different title")
        target.write_text(str(content)[:5000], encoding="utf-8")
        return f"saved notes/{slug}.md"

    def tools(self):
        return {t.name: t for t in [
            Tool("list_files", "List the files in the workspace, or in one folder of it.",
                 _schema({"folder": "Folder to list. Leave out for the whole workspace."}, []), self.list_files),
            Tool("read_file", "Read a text file from the workspace, for example 'meeting_notes.md'.",
                 _schema({"path": "File path relative to the workspace"}, ["path"]), self.read_file),
            Tool("search_files", "Search every workspace file for a word or phrase. Returns file:line matches. Use it to find which file mentions something.",
                 _schema({"query": "Text to search for"}, ["query"]), self.search_files),
            Tool("calculator", "Evaluate an arithmetic expression exactly, for example '1250 * 12'.",
                 _schema({"expression": "The expression"}, ["expression"]), safe_calculate),
            Tool("write_note", "Save a short note to the notes folder. The user must approve this.",
                 _schema({"title": "Short title", "content": "Text of the note"}, ["title", "content"]), self.write_note, requires_approval=True),
        ]}

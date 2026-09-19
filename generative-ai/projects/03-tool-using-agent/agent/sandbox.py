"""Keep the agent inside one folder. This is the most important safety code in the project."""

from pathlib import Path

ALLOWED_SUFFIXES = {".txt", ".md", ".csv", ".json"}
MAX_FILE_BYTES = 200_000


class SandboxError(Exception):
    """The requested path is not allowed. The message is safe to show to the model."""


def resolve_inside(root, user_path):
    """Turn a path from the model into a real path, refusing anything outside `root`.

    The model chooses this string, and the model can be tricked, so we treat it as hostile input:
    `..\\..\\secrets`, absolute paths like `C:\\Windows`, and symlinks pointing outside are all refused.
    """
    if not isinstance(user_path, str) or not user_path.strip() or "\x00" in user_path:
        raise SandboxError("path must be a non-empty string")
    root = Path(root).resolve()
    candidate = (root / user_path).resolve()  # resolves .., absolute paths, and symlinks
    if candidate != root and root not in candidate.parents:
        raise SandboxError("path is outside the workspace")
    return candidate


def check_readable(path):
    """Only small text-like files may be read."""
    if not path.is_file():
        raise SandboxError("no such file")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise SandboxError(f"only these file types can be read: {', '.join(sorted(ALLOWED_SUFFIXES))}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise SandboxError("file is too large")
    return path

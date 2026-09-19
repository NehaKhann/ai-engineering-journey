"""Read documents from disk: plain text, Markdown, and PDF."""

from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


def read_pdf(path):
    """Extract text from a PDF, page by page. Returns a list of (page_number, text)."""
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ImportError("Reading PDFs needs the 'pypdf' package: pip install pypdf") from error
    reader = PdfReader(str(path))
    return [(number, (page.extract_text() or "").strip()) for number, page in enumerate(reader.pages, 1)]


def load_document(path):
    """Return a list of {"source", "page", "text"} records for one file (one per PDF page)."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return [{"source": path.name, "page": n, "text": t} for n, t in read_pdf(path) if t]
    text = path.read_text(encoding="utf-8").strip()
    return [{"source": path.name, "page": None, "text": text}] if text else []


def find_documents(folder):
    """All supported files directly inside a folder, in a stable order."""
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)

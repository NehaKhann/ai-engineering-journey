"""Split documents into overlapping chunks that each remember where they came from (Module 05)."""


def chunk_record(record, size=60, overlap=15):
    """Split one loaded document into chunks of about `size` words.

    Every chunk carries the document title (its first line), so a chunk that starts mid-document
    still says what it is about. That helps retrieval, and it lets the answer cite its source.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = record["text"].split()
    title = record["text"].splitlines()[0].strip().rstrip(".") if record["text"] else record["source"]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if piece:
            chunks.append({
                "source": record["source"],
                "page": record["page"],
                "title": title,
                "text": " ".join(piece),
            })
        if start + size >= len(words):
            break
    return chunks


def embedding_text(chunk):
    """The text we embed: the title plus the chunk body."""
    return f"{chunk['title']}: {chunk['text']}"

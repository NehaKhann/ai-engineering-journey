"""A persistent vector store for document chunks, built on ChromaDB (Module 04)."""

import chromadb

from .chunking import chunk_record, embedding_text


class DocumentStore:
    """Chunks and their embeddings, saved on disk so you only ingest once.

    `embed_fn` is passed in (rather than imported) so tests can use a fake embedder.
    """

    def __init__(self, path, embed_fn, collection="documents"):
        self.embed_fn = embed_fn
        self.client = chromadb.PersistentClient(path=str(path)) if path else chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add_records(self, records, size=60, overlap=15):
        """Chunk and store loaded documents. Re-adding a source REPLACES its old chunks."""
        sources = {r["source"] for r in records}
        for source in sources:
            self.collection.delete(where={"source": source})

        chunks = [c for record in records for c in chunk_record(record, size, overlap)]
        if not chunks:
            return 0
        counters = {}
        ids = []
        for chunk in chunks:
            counters[chunk["source"]] = counters.get(chunk["source"], 0) + 1
            ids.append(f"{chunk['source']}:{counters[chunk['source']]}")
        self.collection.add(
            ids=ids,
            documents=[c["text"] for c in chunks],
            embeddings=self.embed_fn([embedding_text(c) for c in chunks]).tolist(),
            metadatas=[{"source": c["source"], "page": c["page"] or 0, "title": c["title"]} for c in chunks],
        )
        return len(chunks)

    def search(self, query, k=3):
        """The k most similar chunks as dicts with 'text', 'source', 'page', 'title' and 'score' (0 to 1)."""
        if self.collection.count() == 0:
            return []
        result = self.collection.query(query_embeddings=self.embed_fn([query]).tolist(), n_results=min(k, self.collection.count()))
        return [
            {"text": text, "source": meta["source"], "page": meta["page"] or None, "title": meta["title"], "score": 1 - distance}
            for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
        ]

    def sources(self):
        """Map each source file to how many chunks it has."""
        counts = {}
        for meta in self.collection.get()["metadatas"]:
            counts[meta["source"]] = counts.get(meta["source"], 0) + 1
        return dict(sorted(counts.items()))

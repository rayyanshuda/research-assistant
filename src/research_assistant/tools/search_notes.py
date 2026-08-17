"""search_notes(query) — citation-aware retrieval over the local vector store."""
from __future__ import annotations

from ..vectorstore import VectorStore

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def search_notes(query: str, top_k: int = 5) -> list[dict]:
    """Return the top_k most relevant chunks for `query`, each with its citation."""
    results = _get_store().query(query, n_results=top_k)
    return [
        {
            "text": r.text,
            "citation": r.citation,
            "doc_id": r.doc_id,
            "source": r.source,
            "relevance": round(1 - r.distance, 4) if r.distance is not None else None,
        }
        for r in results
    ]

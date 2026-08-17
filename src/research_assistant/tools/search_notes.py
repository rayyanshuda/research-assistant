"""search_notes(query) - citation-aware retrieval over a vector store.

Takes the VectorStore explicitly rather than reaching for a module-level
singleton, because each web session now has its own private, isolated
store (uploaded PDFs must never be searchable by a different visitor). The
CLI still passes the one shared persistent store, unchanged.
"""
from __future__ import annotations

from ..vectorstore import VectorStore


def search_notes(store: VectorStore, query: str, top_k: int = 5) -> list[dict]:
    """Return the top_k most relevant chunks for `query`, each with its citation."""
    if store.count() == 0:
        return [
            {
                "info": (
                    "No documents have been uploaded to this session yet. "
                    "Ask the user to upload a PDF first."
                )
            }
        ]
    results = store.query(query, n_results=top_k)
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
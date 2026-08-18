from __future__ import annotations

from ..vectorstore import VectorStore


def search_notes(store: VectorStore, query: str, top_k: int = 5) -> list[dict]:
    # return the top k most relevant chunks for `query`, with citation
    if store.count() == 0:
        return [
            {
                "info": (
                    "No documents have been uploaded to this session yet."
                    "Please upload a PDF first."
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
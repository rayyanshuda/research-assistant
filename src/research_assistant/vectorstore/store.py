"""Local Chroma vector store wrapper with citation-aware retrieval."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import chromadb

from ..config import settings
from ..embeddings import embedder as embedder_module
from ..ingestion.chunker import Chunk


@dataclass
class RetrievedChunk:
    text: str
    citation: str
    doc_id: str
    source: str
    distance: float
    page: int | None = None
    section: str | None = None


class VectorStore:
    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        ephemeral: bool = False,
    ):
        """ephemeral=True gives an in-memory Chroma collection, used for
        per-visitor sessions in the web app so one person's uploaded PDFs are
        never visible to another. persist_dir/collection_name are ignored in
        that mode.

        Important: chromadb.EphemeralClient() caches and shares its
        underlying System across every call in the same process (this is
        deliberate chromadb behavior, not a bug) - so two ephemeral
        VectorStores with the SAME collection name would silently share
        data. We give every ephemeral store a random, unique collection name
        so isolation holds regardless of that shared-System caching.
        """
        if ephemeral:
            self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(
                name=f"session-{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
            )
        else:
            self._client = chromadb.PersistentClient(path=persist_dir or settings.chroma_dir)
            self._collection = self._client.get_or_create_collection(
                name=collection_name or settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks. Safe to call repeatedly (upsert by chunk_id)."""
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = embedder_module.embed_texts(texts)
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=[c.to_chroma_metadata() for c in chunks],
        )

    def query(self, query_text: str, n_results: int = 5) -> list[RetrievedChunk]:
        query_vec = embedder_module.embed_query(query_text)
        result = self._collection.query(query_embeddings=[query_vec], n_results=n_results)

        retrieved: list[RetrievedChunk] = []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metas, dists):
            page = meta.get("page")
            section = meta.get("section")
            source = meta.get("source", "unknown")
            citation = f"{source}, p.{page}" if page is not None else (
                f"{source} → {section}" if section else source
            )
            retrieved.append(
                RetrievedChunk(
                    text=text,
                    citation=citation,
                    doc_id=meta.get("doc_id", ""),
                    source=source,
                    distance=dist,
                    page=page,
                    section=section,
                )
            )
        return retrieved

    def count(self) -> int:
        return self._collection.count()

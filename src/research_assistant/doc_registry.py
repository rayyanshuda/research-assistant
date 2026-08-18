"""In-memory registry mapping doc_id -> full document text + metadata.

Chroma stores chunks, not whole documents, so summarize_doc(doc_id)
needs somewhere to fetch the complete original text. Each ResearchAgent
(one per web session) owns its own registry dict - there's no shared
on-disk registry anymore now that the CLI and local ingestion pipeline
have been retired.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocRecord:
    doc_id: str
    source: str
    kind: str  # "pdf" | "markdown"
    full_text: str
    pages: int = 0  # display-only, for the web app's sidebar

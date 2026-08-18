# in-memory registry mapping doc_id -> full document text + metadata
# chroma stores chunks, not whole documents, so summarize_docs(doc_id)
# needs somewhere to fetch the complete original text.
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocRecord:
    doc_id: str
    source: str
    kind: str  # "pdf" | "markdown"
    full_text: str
    pages: int = 0  # display-only, for the web app's sidebar

"""Word-window chunking with citation metadata.

Each Chunk carries enough metadata to answer "where did this come from"
precisely: source file name, doc_id, and either a page number (PDF) or
section heading (markdown), plus a chunk index within that unit.

Chunking is done by word count rather than exact token count (via
tiktoken) on purpose: tiktoken's BPE file downloads from
openaipublic.blob.core.windows.net on first use, which fails in sandboxed
or restricted-network environments. Words-to-tokens for English text is
roughly 1 word ≈ 1.3 tokens, so `chunk_size` and `overlap` below are in
tokens and get converted to a word count internally — no network dependency,
no first-run download.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .markdown_parser import MarkdownSection
from .pdf_parser import PdfPage

_WORDS_PER_TOKEN = 0.75  # ≈ 1 / 1.3, rough English heuristic


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str  # file name, e.g. "LoRA.pdf"
    text: str
    # Citation locator: page number for PDFs, heading for markdown.
    page: int | None = None
    section: str | None = None
    metadata: dict = field(default_factory=dict)

    def citation(self) -> str:
        if self.page is not None:
            return f"{self.source}, p.{self.page}"
        if self.section:
            return f"{self.source} → {self.section}"
        return self.source

    def to_chroma_metadata(self) -> dict:
        meta = {"doc_id": self.doc_id, "source": self.source}
        if self.page is not None:
            meta["page"] = self.page
        if self.section is not None:
            meta["section"] = self.section
        meta.update(self.metadata)
        return meta


def _make_doc_id(source: str) -> str:
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def _split_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows, sized in approximate tokens."""
    words = text.split()
    if not words:
        return []

    window_words = max(round(chunk_size * _WORDS_PER_TOKEN), 1)
    overlap_words = max(round(overlap * _WORDS_PER_TOKEN), 0)

    if len(words) <= window_words:
        return [text]

    pieces = []
    step = max(window_words - overlap_words, 1)
    for start in range(0, len(words), step):
        window = words[start : start + window_words]
        pieces.append(" ".join(window))
        if start + window_words >= len(words):
            break
    return pieces


def chunk_pdf_pages(
    source: str,
    pages: list[PdfPage],
    chunk_size: int = 400,
    overlap: int = 60,
) -> list[Chunk]:
    doc_id = _make_doc_id(source)
    chunks: list[Chunk] = []
    for page in pages:
        pieces = _split_tokens(page.text, chunk_size, overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-p{page.page_number}-{i}",
                    doc_id=doc_id,
                    source=source,
                    text=piece,
                    page=page.page_number,
                )
            )
    return chunks


def chunk_markdown_sections(
    source: str,
    sections: list[MarkdownSection],
    chunk_size: int = 400,
    overlap: int = 60,
) -> list[Chunk]:
    doc_id = _make_doc_id(source)
    chunks: list[Chunk] = []
    for s_idx, section in enumerate(sections):
        pieces = _split_tokens(section.text, chunk_size, overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-s{s_idx}-{i}",
                    doc_id=doc_id,
                    source=source,
                    text=piece,
                    section=section.heading,
                )
            )
    return chunks
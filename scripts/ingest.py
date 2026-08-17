#!/usr/bin/env python3
"""Ingestion pipeline entrypoint.

Walks data/pdfs and data/notes, chunks everything, embeds the chunks into
the local Chroma store, and updates the doc registry (used by
summarize_doc) with each document's full text.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --pdfs-dir /path/to/pdfs --notes-dir /path/to/notes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_assistant.config import settings  # noqa: E402
from research_assistant.doc_registry import (  # noqa: E402
    DocRecord,
    load_registry,
    save_registry,
    upsert_doc,
)
from research_assistant.ingestion import (  # noqa: E402
    chunk_markdown_sections,
    chunk_pdf_pages,
    parse_markdown,
    parse_pdf,
)
from research_assistant.vectorstore import VectorStore  # noqa: E402


def ingest_pdfs(pdfs_dir: Path, store: VectorStore, registry: dict) -> int:
    count = 0
    for pdf_path in sorted(pdfs_dir.glob("*.pdf")):
        print(f"  parsing {pdf_path.name} ...")
        pages = parse_pdf(pdf_path)
        if not pages:
            print(f"    (no extractable text, skipping)")
            continue
        chunks = chunk_pdf_pages(
            pdf_path.name, pages, settings.chunk_size_tokens, settings.chunk_overlap_tokens
        )
        store.add_chunks(chunks)
        full_text = "\n\n".join(p.text for p in pages)
        upsert_doc(
            registry,
            DocRecord(doc_id=chunks[0].doc_id, source=pdf_path.name, kind="pdf", full_text=full_text),
        )
        print(f"    {len(pages)} pages -> {len(chunks)} chunks")
        count += len(chunks)
    return count


def ingest_markdown(notes_dir: Path, store: VectorStore, registry: dict) -> int:
    count = 0
    for md_path in sorted(notes_dir.glob("**/*.md")):
        print(f"  parsing {md_path.name} ...")
        sections = parse_markdown(md_path)
        if not sections:
            print(f"    (empty, skipping)")
            continue
        chunks = chunk_markdown_sections(
            md_path.name, sections, settings.chunk_size_tokens, settings.chunk_overlap_tokens
        )
        store.add_chunks(chunks)
        full_text = "\n\n".join(s.text for s in sections)
        upsert_doc(
            registry,
            DocRecord(doc_id=chunks[0].doc_id, source=md_path.name, kind="markdown", full_text=full_text),
        )
        print(f"    {len(sections)} sections -> {len(chunks)} chunks")
        count += len(chunks)
    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs and markdown notes into the vector store.")
    parser.add_argument("--pdfs-dir", type=Path, default=settings.pdfs_dir)
    parser.add_argument("--notes-dir", type=Path, default=settings.notes_dir)
    args = parser.parse_args()

    print(f"Chroma dir: {settings.chroma_dir}")
    store = VectorStore()
    registry = load_registry()

    total = 0
    if args.pdfs_dir.exists():
        print(f"Ingesting PDFs from {args.pdfs_dir} ...")
        total += ingest_pdfs(args.pdfs_dir, store, registry)
    else:
        print(f"(no pdfs dir at {args.pdfs_dir}, skipping)")

    if args.notes_dir.exists():
        print(f"Ingesting markdown notes from {args.notes_dir} ...")
        total += ingest_markdown(args.notes_dir, store, registry)
    else:
        print(f"(no notes dir at {args.notes_dir}, skipping)")

    save_registry(registry)
    print(f"\nDone. {total} chunks embedded and stored. Vector store now has {store.count()} chunks total.")


if __name__ == "__main__":
    main()

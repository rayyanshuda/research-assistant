"""Ingestion/chunking tests. These don't need API keys — pure parsing + chunking logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_assistant.ingestion import (
    chunk_markdown_sections,
    chunk_pdf_pages,
    parse_markdown,
    parse_pdf,
)

PDFS_DIR = Path(__file__).resolve().parents[1] / "data" / "pdfs"
NOTES_DIR = Path(__file__).resolve().parents[1] / "data" / "notes"


def test_parse_pdf_extracts_pages():
    for pdf_path in PDFS_DIR.glob("*.pdf"):
        pages = parse_pdf(pdf_path)
        assert len(pages) > 0, f"{pdf_path.name} produced no pages"
        assert all(p.text.strip() for p in pages)
        assert pages[0].page_number == 1


def test_chunk_pdf_pages_have_citations():
    for pdf_path in PDFS_DIR.glob("*.pdf"):
        pages = parse_pdf(pdf_path)
        chunks = chunk_pdf_pages(pdf_path.name, pages, chunk_size=400, overlap=60)
        assert len(chunks) > 0
        for c in chunks:
            assert c.source == pdf_path.name
            assert c.page is not None
            assert c.citation() == f"{pdf_path.name}, p.{c.page}"
            assert c.text.strip()
            # unique chunk_ids
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "duplicate chunk_ids"


def test_parse_markdown_sections():
    md_path = NOTES_DIR / "sample-note.md"
    sections = parse_markdown(md_path)
    assert len(sections) >= 2
    headings = [s.heading for s in sections]
    assert "Why citation-aware chunking matters" in headings
    assert "Chunk size tradeoffs" in headings


def test_chunk_markdown_sections_have_citations():
    md_path = NOTES_DIR / "sample-note.md"
    sections = parse_markdown(md_path)
    chunks = chunk_markdown_sections(md_path.name, sections, chunk_size=400, overlap=60)
    assert len(chunks) > 0
    for c in chunks:
        assert c.section is not None
        assert c.citation() == f"{md_path.name} → {c.section}"


if __name__ == "__main__":
    # Allow running as a plain script too (no pytest dependency required).
    tests = [
        test_parse_pdf_extracts_pages,
        test_chunk_pdf_pages_have_citations,
        test_parse_markdown_sections,
        test_chunk_markdown_sections_have_citations,
    ]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print("\nAll ingestion tests passed.")

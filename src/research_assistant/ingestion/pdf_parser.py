"""PDF -> per-page text extraction using PyMuPDF (fitz)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (new import name; `fitz` alias is deprecated)


@dataclass
class PdfPage:
    page_number: int  # 1-indexed, human-friendly for citations
    text: str


def parse_pdf(path: str | Path) -> list[PdfPage]:
    """Extract text from every page of a PDF.

    Returns one PdfPage per page, in order, with 1-indexed page numbers so
    citations can say "page 7" the way a human would.
    """
    path = Path(path)
    pages: list[PdfPage] = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(PdfPage(page_number=i + 1, text=text))
    return pages

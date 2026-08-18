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
    # extract text from every page of a PDF file
    # return one PdfPage per page for indexing page numbers citations
    path = Path(path)
    with fitz.open(path) as doc:
        return _extract_pages(doc)


def parse_pdf_bytes(data: bytes) -> list[PdfPage]:
    # same as parse_pdf but for a PDF in memory
    with fitz.open(stream=data, filetype="pdf") as doc:
        return _extract_pages(doc)


def _extract_pages(doc) -> list[PdfPage]:
    pages: list[PdfPage] = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append(PdfPage(page_number=i + 1, text=text))
    return pages

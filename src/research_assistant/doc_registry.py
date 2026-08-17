"""Registry mapping doc_id -> full document text + metadata.

Chroma stores chunks, not whole documents, so summarize_doc(doc_id) needs
somewhere to fetch the complete original text. This is a simple JSON file
keyed by doc_id (kept in sync by the ingestion script).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import settings


@dataclass
class DocRecord:
    doc_id: str
    source: str
    kind: str  # "pdf" | "markdown"
    full_text: str
    pages: int = 0  # display-only, for sidebars/listings; 0 for pre-existing records

def load_registry() -> dict[str, DocRecord]:
    path = settings.doc_registry_path
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: DocRecord(**v) for k, v in raw.items()}


def save_registry(registry: dict[str, DocRecord]) -> None:
    path = settings.doc_registry_path
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: asdict(v) for k, v in registry.items()}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def upsert_doc(registry: dict[str, DocRecord], record: DocRecord) -> None:
    registry[record.doc_id] = record


def get_doc(doc_id: str) -> DocRecord | None:
    return load_registry().get(doc_id)


def find_doc_by_source(source_substring: str) -> DocRecord | None:
    """Best-effort lookup by (partial, case-insensitive) file name, since the
    agent often knows the file name from a citation but not the raw doc_id."""
    needle = source_substring.lower()
    for record in load_registry().values():
        if needle in record.source.lower():
            return record
    return None

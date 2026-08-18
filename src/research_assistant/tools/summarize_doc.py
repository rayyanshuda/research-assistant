from __future__ import annotations

import anthropic

from ..config import settings
from ..doc_registry import DocRecord

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.require_anthropic_key())
    return _client


def _find_by_source(registry: dict[str, DocRecord], source_substring: str) -> DocRecord | None:
    needle = source_substring.lower()
    for record in registry.values():
        if needle in record.source.lower():
            return record
    return None


def summarize_doc(registry: dict[str, DocRecord], doc_id: str, focus: str | None = None) -> dict:
    record = registry.get(doc_id) or _find_by_source(registry, doc_id)
    if record is None:
        return {"error": f"No document matches '{doc_id}' in this session."}

    prompt = "Summarize the following document"
    if focus:
        prompt += f", focusing specifically on: {focus}"
    prompt += f".\n\nDocument: {record.source}\n\n{record.full_text[:120_000]}"

    response = _get_client().messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    summary_text = "".join(block.text for block in response.content if block.type == "text")
    return {"source": record.source, "doc_id": record.doc_id, "summary": summary_text}
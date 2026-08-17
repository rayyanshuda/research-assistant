"""summarize_doc(doc_id) — summarize a full ingested document via Claude."""
from __future__ import annotations

import anthropic

from ..config import settings
from ..doc_registry import find_doc_by_source, get_doc

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.require_anthropic_key())
    return _client


def summarize_doc(doc_id: str, focus: str | None = None) -> dict:
    """Summarize the full text of a document.

    `doc_id` may be an exact doc_id (as returned by search_notes) or a
    partial/full source file name (e.g. "LoRA.pdf") — we fall back to a
    fuzzy lookup by file name since that's what the model usually has on
    hand from a citation.
    """
    record = get_doc(doc_id) or find_doc_by_source(doc_id)
    if record is None:
        return {"error": f"No ingested document matches '{doc_id}'."}

    prompt = f"Summarize the following document"
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

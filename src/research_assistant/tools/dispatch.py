# Maps a claude tool_use block's name/input to the Python function call.
from __future__ import annotations

from ..doc_registry import DocRecord
from ..vectorstore import VectorStore
from .fetch_web import fetch_web
from .search_notes import search_notes
from .summarize_doc import summarize_doc


def dispatch_tool(name: str, tool_input: dict, *, store: VectorStore, registry: dict[str, DocRecord]):
    handlers = {
        "search_notes": lambda inp: search_notes(store, inp["query"], inp.get("top_k", 5)),
        "fetch_web": lambda inp: fetch_web(inp["query"], inp.get("max_results", 5)),
        "summarize_doc": lambda inp: summarize_doc(registry, inp["doc_id"], inp.get("focus")),
    }
    handler = handlers.get(name)
    if handler is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return handler(tool_input)
    except Exception as exc:  # keep the agent loop alive on tool errors
        return {"error": f"{name} failed: {exc}"}
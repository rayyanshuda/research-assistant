# mcp server: connecting this project's tools to claude desktop (or any other MCP client)

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from research_assistant.agent import ResearchAgent  # noqa: E402
from research_assistant.config import settings  # noqa: E402
from research_assistant.tools.dispatch import dispatch_tool  # noqa: E402
from research_assistant.vectorstore import VectorStore  # noqa: E402

mcp = FastMCP("research-assistant")

_STORE_DIR = str(settings.project_root / "data" / "mcp_chroma")

_agent = ResearchAgent(
    verbose=False,
    max_questions=0,  # no demo cap - this only ever runs locally, for you
    store=VectorStore(persist_dir=_STORE_DIR, collection_name="mcp"),
    doc_registry={},
)


@mcp.tool()
def index_document(file_path: str) -> dict:
    """Parse, chunk, embed, and index a local PDF so it becomes searchable
    via search_notes and summarize_doc. Run this once per document - there's
    no upload button here, just a path to a PDF already on this computer."""
    path = Path(file_path).expanduser()
    if not path.exists():
        return {"error": f"No file found at {path}"}
    if path.suffix.lower() != ".pdf":
        return {"error": f"Only PDF files are supported right now, got '{path.suffix}'"}
    result = _agent.add_document(path.read_bytes(), path.name)
    return {"status": "indexed", **result}


@mcp.tool()
def search_notes(query: str, top_k: int = 5) -> dict:
    """Search indexed documents for passages relevant to `query`. Always try
    this before fetch_web - indexed documents are the primary source of
    truth. Returns passages with citations (source file + page number)."""
    return dispatch_tool(
        "search_notes", {"query": query, "top_k": top_k}, store=_agent.store, registry=_agent.doc_registry
    )


@mcp.tool()
def fetch_web(query: str, max_results: int = 5) -> dict:
    """Search the public web. Use only to supplement indexed documents -
    e.g. when search_notes doesn't have enough, or the question is about
    something outside what's been indexed (recent events, general facts)."""
    return dispatch_tool(
        "fetch_web", {"query": query, "max_results": max_results}, store=_agent.store, registry=_agent.doc_registry
    )


@mcp.tool()
def summarize_doc(doc_id: str, focus: str | None = None) -> dict:
    """Summarize an entire indexed document, not just search_notes chunks.
    Pass either the doc_id (from a search_notes result) or the source file
    name, e.g. 'LoRA.pdf'."""
    return dispatch_tool(
        "summarize_doc", {"doc_id": doc_id, "focus": focus}, store=_agent.store, registry=_agent.doc_registry
    )


if __name__ == "__main__":
    mcp.run()

# Research Assistant

A research agent that answers questions about PDFs or notes you upload, with citations back to the source file and page. Can also fall back to the web when the documents don't have the answer.

## How it works

``` Markdown

PDF upload (web/index.html → webapp.py)
        │
        ▼
  ingestion/ (PyMuPDF)
        │  chunks carry {source, page, doc_id}
        ▼
  embeddings/ (pluggable: Gemini / local ONNX / OpenAI) - currently using Gemini AI Studio
        │
        ▼
  vectorstore/ (chroma, ephemeral per session)
        │
        ▼
  agent/ (claude tool-use loop: search_notes, fetch_web, summarize_doc)
        │
        ▼
  webapp.py + web/index.html
```

## How to Set Up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and fill in:
#   ANTHROPIC_API_KEY=... (used by the agent loop)
#   GEMINI_API_KEY=...    (used for embeddings, free tier, no card)
```

## Running the app

Each browser session starts with an empty, private, in-memory document store, and you upload PDFs yourself with the sidebar's "+ Upload PDF" button once the page is open. Every visitor's documents, conversation history, and question cap (5 Q's) are isolated from every other visitor's, nothing is shared, and nothing is written to disk.

Two independent cost controls protect a public demo from unbounded API spend: a per-session question cap (`DEMO_MAX_QUESTIONS`), and a
server-side cap tracked by IP address so refreshing the page (which resets the browser's session_id) doesn't hand a visitor a fresh quota.

## Limitations

- Chunking is token-window based (400 tokens). It's fine for a version 1 for me, but it should be optimized if retrieval quality is poor on specific notes.
- No re-ranking step: retrieval is a straight top-k cosine similarity query against Chroma. Re-ranking involves a cross-encoder model, that will produce more precise results, but it's expensive to run the re-ranking comparison for every query, which is why for now a single-stage retrieval is enough.
- `fetch_web` results aren't cited with the same accuracy as `search_notes` results (no page-level, just title/URL) since web pages don't have "page numbers."
- The web app's IP-based rate limit is a demo-cost control, visitors sharing a VPN/office network share a quota.

## MCP Server

The same three tools (`search_notes`, `fetch_web`, `summarize_doc`) can also be used directly from Claude Desktop instead of the web app, with a small MCP (Model Context Protocol) server at `scripts/mcp_server.py`. This is a separate, local-only path, it keeps its own persistent document store on disk (`data/mcp_chroma/`), completely separate from the ephemeral per-visitor stores the deployed web app uses.

### Setup

`requirements.txt` already includes the `mcp` package the server itself needs. Optionally, to test it locally first with an inspector UI before connecting it to Claude Desktop:

```bash
pip install "mcp[cli]"
mcp dev scripts/mcp_server.py
```

To connect it to Claude Desktop:

1. In Claude Desktop, go to the Claude menu  **Settings** → **Developer** tab → **Edit Config**. This opens `claude_desktop_config.json`, on macOS that's `~/Library/Application Support/Claude/claude_desktop_config.json`, on Windows `%APPDATA%\Claude\claude_desktop_config.json`.
2. Add this entry, replacing the paths with your own paths to this repo's virtual environment and `scripts/mcp_server.py`:

```json
{
  "mcpServers": {
    "research-assistant": {
      "command": "/absolute/path/to/research-assistant/.venv/bin/python",
      "args": ["/absolute/path/to/research-assistant/scripts/mcp_server.py"]
    }
  }
}
```

3. Quit and reopen Claude Desktop. Your `.env` file (the same one the web app uses) supplies the API keys, you don't need to duplicate them into the config.

After it's connected, Claude Desktop can call:

- `index_document(file_path)`: parse, chunk, embed, and index a local PDF
- `search_notes(query)`: search indexed documents, with citations
- `fetch_web(query)`: fall back to a web search
- `summarize_doc(doc_id)`: summarize a whole indexed document
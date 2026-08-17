To re-index after adding new files, just run `python scripts/ingest.py`
again — chunks are upserted by ID so re-running is safe and idempotent.

To point the CLI at a different notes vault entirely:

```bash
python scripts/ingest.py --pdfs-dir /path/to/your/pdfs --notes-dir /path/to/your/notes
```

## Running the web app

```bash
python scripts/serve.py [--port 8000] [--host 127.0.0.1]
```

Then open the printed URL. Unlike the CLI, this does **not** need
`ingest.py` to have been run first — each browser session starts with an
empty, private, in-memory document store, and you upload PDF(s) yourself
via the sidebar's "+ Upload PDF" button once the page is open. Every
visitor's documents, conversation history, and question cap are isolated
from every other visitor's — nothing is shared, and nothing is written to
disk.

Two independent cost controls protect a public demo from unbounded API
spend: a per-session question cap (`DEMO_MAX_QUESTIONS`), and a
server-side cap tracked by IP address so refreshing the page (which resets
the browser's session_id) doesn't hand a visitor a fresh quota.

## Deployment

The live demo runs on [Render](https://render.com) (free tier) from the
`render.yaml` blueprint at the repo root, with `research-assistant.rayyanhuda.com`
pointed at it via a CNAME record. A few things worth knowing if you're
redeploying or debugging this:

- **`render.yaml`** defines the build (`pip install -r requirements.txt`)
  and start (`python scripts/serve.py --host 0.0.0.0 --port $PORT`)
  commands, plus the non-secret env vars. `ANTHROPIC_API_KEY` and
  `GEMINI_API_KEY` are marked `sync: false` so they're entered directly in
  Render's dashboard rather than committed anywhere.
- **Cold starts**: Render's free tier spins the service down after ~15
  minutes idle. The first request after that takes 30-50 seconds to wake
  up, and since sessions live in memory, any in-progress session (including
  uploaded PDFs) is lost when it sleeps. Fine for a personal demo; the
  Starter plan ($7/mo) removes this if it becomes a problem before a
  specific interview.
- **DNS**: if your CNAME to Render is proxied through Cloudflare (orange
  cloud), keep an eye on certificate renewal — Render's automatic TLS
  renewal needs to reach the origin directly every ~90 days, which can
  occasionally need the record set to "DNS only" or Cloudflare's SSL mode
  set to Full/Full-strict.
- **Multi-worker note**: sessions and the IP rate-limit tracker are both
  plain in-memory Python dicts (see `webapp.py`), which only works because
  this runs as a single process. It would need real shared storage (Redis,
  etc.) before scaling to multiple server processes/instances.

## Testing

```bash
pip install pytest
pytest tests/
```

32 tests, fully offline — no API keys, network access, or cost required to
run them. Anthropic/embedding calls are mocked throughout; a few tests
(`test_document_isolation.py`) use the real chunking/retrieval/vectorstore
code with only the embedder and Anthropic client faked, so they exercise
the actual per-session isolation boundary rather than a stand-in for it.
That test suite is what originally caught a real bug — `chromadb`'s
`EphemeralClient()` silently shares its underlying storage across
same-named collections within one process, which would have let two "private"
sessions leak into each other before it was fixed.

## Design notes / where to extend

- **Citation tracking**: metadata is attached at chunk-creation time in
  `ingestion/chunker.py` (page number for PDFs, heading for markdown), not
  reconstructed after the fact — see `Chunk.citation()`.
- **fetch_web**: uses `duckduckgo-search` (no API key needed) as a
  zero-setup default. Swap `tools/fetch_web.py` for a paid provider (Brave,
  Tavily, Bing) or Claude's native server-side `web_search` tool if you
  want higher-quality results — nothing else in the agent loop needs to
  change.
- **summarize_doc**: reads full document text from the doc registry — a
  file on disk (`data/doc_registry.json`, written by `scripts/ingest.py`)
  for the CLI, an in-memory dict per session for the web app — since Chroma
  only stores chunks, not full documents.
- **Prompt caching**: the system prompt, tool definitions, and growing
  conversation history all carry `cache_control` breakpoints
  (`agent/agent_loop.py`), so repeated tokens within a session are cache
  reads instead of full-price input tokens on every turn.
- **MCP layer (optional)**: the three tools in `tools/` are already
  plain, side-effect-isolated functions with JSON-schema definitions in
  `tools/definitions.py` — wrapping them in an MCP server for Claude
  Desktop would mean constructing one shared `VectorStore()` + doc registry
  at startup and registering each tool as an MCP handler that calls
  `dispatch_tool` underneath. Not built yet.
- **Scheduled weekly digest (stretch goal)**: not built yet. Would reuse
  `ResearchAgent.ask()` on a list of saved queries and email the results.

## Known limitations

- Chunking is token-window based (400 tokens, 60 overlap), not semantic —
  fine for a v1, worth revisiting if retrieval quality is poor on your
  specific notes.
- No re-ranking step; retrieval is a straight top-k cosine similarity
  query against Chroma.
- `fetch_web` results aren't cited with the same rigor as `search_notes`
  results (no page-level anchor, just title/URL) since web pages don't
  have stable "page numbers."
- The web app's IP-based rate limit is a demo-cost control, not a security
  boundary — visitors sharing a NAT/VPN/office network share a quota, and
  it trusts `X-Forwarded-For` from the proxy in front of it as-is.
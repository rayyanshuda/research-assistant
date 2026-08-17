# Personal Research Assistant

An agent over your own notes and PDFs that answers questions with citations
back to the source file + page/section, and can fall back to the web when
your notes don't have the answer.

## How it works

```
data/pdfs/*.pdf, data/notes/**/*.md
        │
        ▼
  ingestion/  (PyMuPDF for PDFs, header-aware split for markdown)
        │  chunks carry {source, page|section, doc_id}
        ▼
  embeddings/  (pluggable: Gemini / local ONNX / OpenAI)
        │
        ▼
  vectorstore/  (local Chroma, persisted to data/chroma/)
        │
        ▼
  agent/  (Claude tool-use loop: search_notes, fetch_web, summarize_doc)
        │
        ▼
       cli.py
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and fill in:
#   ANTHROPIC_API_KEY=...   (required — used by the agent loop)
#   GEMINI_API_KEY=...      (required by default — used for embeddings, free tier, no card)
```

### Choosing an embedding backend

Embeddings are pluggable via `EMBEDDING_PROVIDER` in `.env`:

| Provider | Cost | Setup | Notes |
|---|---|---|---|
| `gemini` (default) | Free tier, no card | `GEMINI_API_KEY` from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | **Recommended if you're hosting this for other people to try.** Free tier does not require a credit card. Community-confirmed free-tier limits for `gemini-embedding-001` are roughly 100 requests/min, 30k tokens/min, 1,000 requests/day — Google doesn't publish these in one place, so treat them as approximate, but they're comfortable for a small public demo (embedding only happens at ingestion time and once per user query, not continuously). One thing worth knowing: Google's free tier terms allow using submitted content to improve their products (the paid tier opts out of this) — fine for your own notes, but if strangers will be pasting/uploading their own documents to try your hosted instance, that's worth disclosing to them or budgeting for the (very cheap, ~$0.15/1M tokens) paid tier instead. |
| `local` | Free, forever | None — no key, no account | Chroma's bundled ONNX MiniLM model, runs on the server's own CPU per request. Works fine for solo/CLI use on your own machine. Not a good fit once you're hosting for multiple concurrent users — every ingestion/query competes for the same local CPU instead of offloading to an API, so it doesn't scale the way a hosted demo needs to. |
| `openai` | Paid only | `OPENAI_API_KEY` + billing on the account | OpenAI removed free trial credits — a `429 insufficient_quota` error means no payment method is on file, not a bug. Best embedding quality of the three if cost isn't a concern. |

Switch providers any time by changing `EMBEDDING_PROVIDER` and re-running `ingest.py` — chunks are upserted by ID, so re-running against the same provider is safe. Don't mix providers within one collection though (their vector spaces aren't compatible); if you switch, wipe `data/chroma/` and re-ingest fully rather than assuming old + new chunks blend well.

Drop PDFs into `data/pdfs/` and markdown notes into `data/notes/` (a
`sample-note.md` is there as a placeholder — delete it once you add your
own). Three real arxiv PDFs are already in `data/pdfs/` as a starter set:
`Introduction to RL.pdf`, `LoRA.pdf`, `Rehearse.pdf`.

## Usage

```bash
# 1. Ingest: parse, chunk, embed, and store everything in data/pdfs and data/notes
python scripts/ingest.py

# 2. Chat with your notes
python src/research_assistant/cli.py
```

Example session:

```
you> what does the LoRA paper say about rank?
  [tool] search_notes({'query': 'LoRA rank hyperparameter'})

assistant> LoRA freezes the pretrained weights and injects trainable
low-rank matrices into each layer; the paper finds performance is
fairly insensitive to rank choice, with even rank=1 or 2 working
reasonably well for many tasks [LoRA.pdf, p.4].
```

To re-index after adding new files, just run `python scripts/ingest.py`
again — chunks are upserted by ID so re-running is safe and idempotent.

## Re-pointing at your real notes vault

```bash
python scripts/ingest.py --pdfs-dir /path/to/your/pdfs --notes-dir /path/to/your/notes
```

## Design notes / where to extend

- **Citation tracking**: metadata is attached at chunk-creation time in
  `ingestion/chunker.py` (page number for PDFs, heading for markdown), not
  reconstructed after the fact — see `Chunk.citation()`.
- **fetch_web**: uses `duckduckgo-search` (no API key needed) as a
  zero-setup default. Swap `tools/fetch_web.py` for a paid provider (Brave,
  Tavily, Bing) or Claude's native server-side `web_search` tool if you
  want higher-quality results — nothing else in the agent loop needs to
  change.
- **summarize_doc**: reads full document text from `data/doc_registry.json`
  (written by `scripts/ingest.py`), since Chroma only stores chunks.
- **MCP layer (optional)**: the three tools in `tools/` are already
  plain, side-effect-isolated functions with JSON-schema definitions in
  `tools/definitions.py` — wrapping them in an MCP server for Claude
  Desktop is mostly a matter of registering each as an MCP tool handler
  that calls `dispatch_tool` under the hood. Not built yet.
- **Scheduled weekly digest (stretch goal)**: not built yet. Would reuse
  `ResearchAgent.ask()` on a list of saved queries and email the results —
  natural next step once the core loop is solid.

## Known limitations

- Chunking is token-window based (400 tokens, 60 overlap), not semantic —
  fine for a v1, worth revisiting if retrieval quality is poor on your
  specific notes.
- No re-ranking step; retrieval is a straight top-k cosine similarity
  query against Chroma.
- `fetch_web` results aren't cited with the same rigor as `search_notes`
  results (no page-level anchor, just title/URL) since web pages don't
  have stable "page numbers."

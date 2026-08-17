# Sample Note — Retrieval-Augmented Agents

This is a placeholder note so the markdown ingestion path has something real
to chunk. Delete it once you point --notes-dir at your actual notes vault.

## Why citation-aware chunking matters

If a chunk doesn't carry its source file and page/section along with it,
the agent can still answer questions but can't back up its answers. Storing
that metadata at chunk-creation time (not reconstructed later) is what
makes citations reliable.

## Chunk size tradeoffs

Smaller chunks (haha, tinier chunks) improve retrieval precision but lose
surrounding context; larger chunks preserve context but dilute the
embedding with less-relevant text. 400 tokens with 60 of overlap is a
reasonable starting point — tune it against your own notes once you have
real recall/precision signal.

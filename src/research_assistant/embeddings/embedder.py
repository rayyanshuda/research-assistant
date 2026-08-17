"""Pluggable embedding backends, selected via EMBEDDING_PROVIDER in .env.

- "gemini" (default): Google's gemini-embedding-001. Has a genuine free tier
  (no credit card required) — as of writing, community-confirmed limits are
  roughly 100 requests/min, 30k tokens/min, 1,000 requests/day. Recommended
  when hosting this for multiple users, since it doesn't tie up local
  compute per request the way "local" does. Get a key at
  https://aistudio.google.com/apikey.
- "local": chromadb's bundled ONNX MiniLM model. Runs on-device, no API
  key, no account, no billing, no rate limits. Downloads a small (~80MB)
  model file from Chroma's CDN the first time it's used, then caches it —
  after that it works fully offline. Fine for solo/CLI use; not a good fit
  for a hosted multi-user instance since every request competes for the
  same local CPU.
- "openai": OpenAI's text-embedding-3-small. Requires OPENAI_API_KEY and
  billing/credits on the account (the free trial credits OpenAI used to
  hand out are gone — a 429 insufficient_quota error means no payment
  method is on file, not a bug in this code).

All three backends are called through embed_texts()/embed_query() below, so
nothing else in the codebase needs to know which one is active.
"""
from __future__ import annotations

import time

from ..config import settings

_openai_client = None
_gemini_client = None
_local_embedding_fn = None

_OPENAI_BATCH_SIZE = 96
_GEMINI_BATCH_SIZE = 32  # keep comfortably under the free-tier TPM ceiling


def _with_retries(call, max_retries: int = 5, base_delay: float = 2.0):
    """Retry on rate-limit/quota errors with exponential backoff.

    This exists because embedding calls are exactly where you first notice
    a free-tier rate limit (lots of chunks, tight per-minute caps) — a
    transient 429 shouldn't kill the whole ingestion run.
    """
    for attempt in range(max_retries):
        try:
            return call()
        except Exception as exc:
            message = str(exc).lower()
            is_rate_limit = "429" in str(exc) or "rate" in message or "quota" in message or "resource_exhausted" in message
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt)
            print(f"  (rate limited, retrying in {delay:.0f}s... [{attempt + 1}/{max_retries}])")
            time.sleep(delay)


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=settings.require_openai_key())
    return _openai_client


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client(api_key=settings.require_gemini_key())
    return _gemini_client


def _get_local_embedding_fn():
    global _local_embedding_fn
    if _local_embedding_fn is None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        _local_embedding_fn = DefaultEmbeddingFunction()
    return _local_embedding_fn


def _embed_openai(texts: list[str]) -> list[list[float]]:
    client = _get_openai_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _OPENAI_BATCH_SIZE):
        batch = texts[start : start + _OPENAI_BATCH_SIZE]
        resp = _with_retries(
            lambda batch=batch: client.embeddings.create(model=settings.openai_embedding_model, input=batch)
        )
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    client = _get_gemini_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _GEMINI_BATCH_SIZE):
        batch = texts[start : start + _GEMINI_BATCH_SIZE]
        result = _with_retries(
            lambda batch=batch: client.models.embed_content(
                model=settings.gemini_embedding_model, contents=batch
            )
        )
        vectors.extend([e.values for e in result.embeddings])
    return vectors


def _embed_local(texts: list[str]) -> list[list[float]]:
    fn = _get_local_embedding_fn()
    return [list(v) for v in fn(texts)]


_BACKENDS = {"local": _embed_local, "openai": _embed_openai, "gemini": _embed_gemini}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of chunk texts using whichever backend EMBEDDING_PROVIDER selects."""
    if not texts:
        return []
    backend = _BACKENDS.get(settings.embedding_provider)
    if backend is None:
        raise RuntimeError(
            f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
            f"Valid options: {', '.join(_BACKENDS)}."
        )
    return backend(texts)


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]
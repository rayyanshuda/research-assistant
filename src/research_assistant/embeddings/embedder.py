# pluggable embedding backends, selects via EMBEDDING_PROVIDER in .env
from __future__ import annotations

import time

from ..config import settings

_openai_client = None
_gemini_client = None
_local_embedding_fn = None

_OPENAI_BATCH_SIZE = 96
_GEMINI_BATCH_SIZE = 32  # keep under the TPM ceiling


def _with_retries(call, max_retries: int = 5, base_delay: float = 2.0):
    # retry on rate-limit / quato errors with exponential backoff
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
    # embed a batch of chunk texts using EMBEDDING_PROVIDOER
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
# tests for pluggable embedding backend dispatch/batching/retry logic
# mock the real network calls (OpenAI/Gemini/loacl ONNX model) and verify provider selection, batching, and retry-on-429
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import research_assistant.embeddings.embedder as embedder  # noqa: E402
from research_assistant.config import settings  # noqa: E402


def test_local_backend_dispatch():
    settings_backup = settings.embedding_provider
    object.__setattr__(settings, "embedding_provider", "local")
    try:
        embedder._local_embedding_fn = lambda texts: [[float(len(t))] for t in texts]
        vectors = embedder.embed_texts(["a", "bb", "ccc"])
        assert vectors == [[1.0], [2.0], [3.0]]
        print("OK  local backend dispatch")
    finally:
        object.__setattr__(settings, "embedding_provider", settings_backup)
        embedder._local_embedding_fn = None


def test_openai_backend_batches_correctly():
    settings_backup = settings.embedding_provider
    object.__setattr__(settings, "embedding_provider", "openai")
    calls = []

    class FakeEmbeddingData:
        def __init__(self, embedding):
            self.embedding = embedding

    class FakeResponse:
        def __init__(self, batch):
            self.data = [FakeEmbeddingData([float(len(t))]) for t in batch]

    class FakeEmbeddings:
        def create(self, model, input):
            calls.append(list(input))
            return FakeResponse(input)

    class FakeClient:
        embeddings = FakeEmbeddings()

    embedder._openai_client = FakeClient()
    embedder._OPENAI_BATCH_SIZE = 2  # force multiple batches with a small input
    try:
        texts = ["a", "bb", "ccc", "dddd", "e"]
        vectors = embedder.embed_texts(texts)
        assert vectors == [[1.0], [2.0], [3.0], [4.0], [1.0]]
        assert len(calls) == 3, f"expected 3 batches of <=2, got {len(calls)}: {calls}"
        print(f"OK  openai backend batches correctly ({len(calls)} batches for {len(texts)} texts)")
    finally:
        object.__setattr__(settings, "embedding_provider", settings_backup)
        embedder._openai_client = None
        embedder._OPENAI_BATCH_SIZE = 96


def test_gemini_backend_batches_correctly():
    settings_backup = settings.embedding_provider
    object.__setattr__(settings, "embedding_provider", "gemini")
    calls = []

    class FakeEmbedding:
        def __init__(self, values):
            self.values = values

    class FakeEmbedResult:
        def __init__(self, batch):
            self.embeddings = [FakeEmbedding([float(len(t))]) for t in batch]

    class FakeModels:
        def embed_content(self, model, contents):
            calls.append(list(contents))
            return FakeEmbedResult(contents)

    class FakeClient:
        models = FakeModels()

    embedder._gemini_client = FakeClient()
    embedder._GEMINI_BATCH_SIZE = 2  # force multiple batches with a small input
    try:
        texts = ["a", "bb", "ccc", "dddd", "e"]
        vectors = embedder.embed_texts(texts)
        assert vectors == [[1.0], [2.0], [3.0], [4.0], [1.0]]
        assert len(calls) == 3, f"expected 3 batches of <=2, got {len(calls)}: {calls}"
        print(f"OK  gemini backend batches correctly ({len(calls)} batches for {len(texts)} texts)")
    finally:
        object.__setattr__(settings, "embedding_provider", settings_backup)
        embedder._gemini_client = None
        embedder._GEMINI_BATCH_SIZE = 32


def test_retry_on_rate_limit_then_succeeds():
    attempts = {"count": 0}

    def flaky_call():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("429 rate limit exceeded")
        return "success"

    result = embedder._with_retries(flaky_call, max_retries=5, base_delay=0.01)
    assert result == "success"
    assert attempts["count"] == 3
    print("OK  retry-on-429 recovers after transient rate limiting")


def test_retry_gives_up_on_non_rate_limit_error():
    def always_fails():
        raise ValueError("some unrelated bug")

    try:
        embedder._with_retries(always_fails, max_retries=3, base_delay=0.01)
        raise AssertionError("expected ValueError to propagate immediately")
    except ValueError:
        print("OK  non-rate-limit errors are not retried")


def test_unknown_provider_raises_clear_error():
    settings_backup = settings.embedding_provider
    object.__setattr__(settings, "embedding_provider", "not-a-real-provider")
    try:
        try:
            embedder.embed_texts(["hi"])
            raise AssertionError("expected RuntimeError for unknown provider")
        except RuntimeError as exc:
            assert "not-a-real-provider" in str(exc)
            print("OK  unknown EMBEDDING_PROVIDER raises a clear error")
    finally:
        object.__setattr__(settings, "embedding_provider", settings_backup)


if __name__ == "__main__":
    test_local_backend_dispatch()
    test_openai_backend_batches_correctly()
    test_gemini_backend_batches_correctly()
    test_retry_on_rate_limit_then_succeeds()
    test_retry_gives_up_on_non_rate_limit_error()
    test_unknown_provider_raises_clear_error()
    print("\nAll embedder dispatch tests passed.")
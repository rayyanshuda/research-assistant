"""Proves the core guarantee: one session's uploaded PDF is never visible to
another session. Uses the REAL ResearchAgent, VectorStore(ephemeral=True),
and the real chunking/registry code - only the embedder and the Anthropic
client are mocked (no network, no API key, no cost), so this exercises the
actual isolation boundary, not a stand-in for it.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import research_assistant.embeddings.embedder as embedder_module  # noqa: E402

PDFS_DIR = Path(__file__).resolve().parents[1] / "data" / "pdfs"


@pytest.fixture(autouse=True)
def _restore_embedder():
    """Each test below monkeypatches embedder_module.embed_texts/embed_query
    directly (module-attribute reassignment, not unittest.mock.patch), so
    without this fixture the fake embedder leaks into whichever test file
    pytest collects next (e.g. test_embedder_dispatch.py), which calls the
    real embedder.embed_texts and gets these fakes instead. Only matters
    when running the full suite via `pytest tests/` - each file's own
    `__main__` block is a fresh process, so it never hit this."""
    original_embed_texts = embedder_module.embed_texts
    original_embed_query = embedder_module.embed_query
    yield
    embedder_module.embed_texts = original_embed_texts
    embedder_module.embed_query = original_embed_query


def _fake_embed_texts(texts):
    dim = 32
    vectors = []
    for text in texts:
        vec = [0.0] * dim
        for word in text.lower().split():
            vec[hash(word) % dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


def _make_session_agent():
    """A private per-session agent, exactly like webapp.py's _new_session_agent()."""
    from research_assistant.agent import ResearchAgent
    from research_assistant.config import Settings
    from research_assistant.vectorstore import VectorStore

    # settings is a frozen dataclass instance, so patch the class method
    # instead of setting an instance attribute (which would raise).
    with patch("research_assistant.agent.agent_loop.anthropic.Anthropic"), patch.object(
        Settings, "require_anthropic_key", return_value="fake-key-for-testing"
    ):
        return ResearchAgent(verbose=False, store=VectorStore(ephemeral=True), doc_registry={})


def test_two_sessions_have_independent_stores():
    embedder_module.embed_texts = _fake_embed_texts
    embedder_module.embed_query = lambda q: _fake_embed_texts([q])[0]

    agent_a = _make_session_agent()
    agent_b = _make_session_agent()

    lora_bytes = (PDFS_DIR / "LoRA.pdf").read_bytes()
    doc = agent_a.add_document(lora_bytes, "LoRA.pdf")

    assert agent_a.store.count() == doc["chunks"] > 0
    assert agent_b.store.count() == 0, "session B's store must stay empty after session A uploads"
    print(f"OK  session A has {agent_a.store.count()} chunks, session B has 0 (uploaded to A only)")


def test_doc_list_does_not_leak_across_sessions():
    embedder_module.embed_texts = _fake_embed_texts
    embedder_module.embed_query = lambda q: _fake_embed_texts([q])[0]

    agent_a = _make_session_agent()
    agent_b = _make_session_agent()

    rl_bytes = (PDFS_DIR / "Introduction to RL.pdf").read_bytes()
    agent_a.add_document(rl_bytes, "Introduction to RL.pdf")

    docs_a = agent_a.list_documents()
    docs_b = agent_b.list_documents()

    assert len(docs_a) == 1 and docs_a[0]["source"] == "Introduction to RL.pdf"
    assert docs_b == [], "session B must not see session A's uploaded document in its sidebar list"
    print("OK  session A's document list has 1 entry, session B's is empty")


def test_search_notes_is_scoped_to_the_calling_sessions_store():
    from research_assistant.tools.search_notes import search_notes

    embedder_module.embed_texts = _fake_embed_texts
    embedder_module.embed_query = lambda q: _fake_embed_texts([q])[0]

    agent_a = _make_session_agent()
    agent_b = _make_session_agent()

    rehearse_bytes = (PDFS_DIR / "Rehearse.pdf").read_bytes()
    agent_a.add_document(rehearse_bytes, "Rehearse.pdf")

    results_a = search_notes(agent_a.store, "what is this paper about")
    results_b = search_notes(agent_b.store, "what is this paper about")

    assert any("Rehearse.pdf" in r.get("source", "") for r in results_a), "session A should find its own upload"
    assert len(results_b) == 1 and "info" in results_b[0], "session B should get the 'no documents uploaded' message"
    assert "upload" in results_b[0]["info"].lower()
    print("OK  search_notes only ever searches the store it was explicitly given - no cross-session leakage")


def test_summarize_doc_cannot_summarize_another_sessions_document():
    from research_assistant.tools.summarize_doc import summarize_doc

    embedder_module.embed_texts = _fake_embed_texts
    embedder_module.embed_query = lambda q: _fake_embed_texts([q])[0]

    agent_a = _make_session_agent()
    agent_b = _make_session_agent()

    lora_bytes = (PDFS_DIR / "LoRA.pdf").read_bytes()
    doc = agent_a.add_document(lora_bytes, "LoRA.pdf")

    # Session B's registry is empty, so even knowing session A's doc_id
    # (which it shouldn't, but let's be thorough) must not resolve.
    result_b = summarize_doc(agent_b.doc_registry, doc["doc_id"])
    assert "error" in result_b, "session B must not be able to summarize a document it never uploaded"
    print("OK  summarize_doc cannot resolve a doc_id belonging to a different session's registry")


if __name__ == "__main__":
    test_two_sessions_have_independent_stores()
    test_doc_list_does_not_leak_across_sessions()
    test_search_notes_is_scoped_to_the_calling_sessions_store()
    test_summarize_doc_cannot_summarize_another_sessions_document()
    print("\nAll document isolation tests passed.")
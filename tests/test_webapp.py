"""Offline tests for the FastAPI web app - session isolation, cap enforcement,
upload handling, and routing. ResearchAgent is monkeypatched to a fake (no
API key/network needed), so this only proves the web layer's plumbing, not
answer quality or real embedding/retrieval (see test_document_isolation.py
for that, using the real ResearchAgent).
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import research_assistant.webapp as webapp  # noqa: E402


class FakeAgent:
    """Stands in for ResearchAgent: answers deterministically, enforces the
    same style of question cap and doc registry, without touching
    Anthropic/Gemini/Chroma at all."""

    _instances_created = 0

    def __init__(self, verbose=False, max_questions=3, store=None, doc_registry=None):
        FakeAgent._instances_created += 1
        self.id = FakeAgent._instances_created
        self._asked = 0
        self._max = max_questions
        self.store = store
        self.doc_registry = doc_registry if doc_registry is not None else {}

    def ask(self, message: str) -> str:
        if self._asked >= self._max:
            return "limit reached"
        self._asked += 1
        return f"agent#{self.id} answered '{message}' (turn {self._asked})"

    def questions_remaining(self):
        return max(self._max - self._asked, 0)

    def add_document(self, pdf_bytes: bytes, filename: str) -> dict:
        doc_id = f"doc-{len(self.doc_registry) + 1}"
        self.doc_registry[doc_id] = {"doc_id": doc_id, "source": filename, "pages": 1}
        return {"doc_id": doc_id, "source": filename, "pages": 1, "chunks": 1}

    def list_documents(self) -> list[dict]:
        return list(self.doc_registry.values())


def _make_client():
    from fastapi.testclient import TestClient

    webapp._SESSIONS.clear()
    webapp._IP_USAGE.clear()
    FakeAgent._instances_created = 0
    webapp.ResearchAgent = FakeAgent  # patch the class the module uses
    return TestClient(webapp.app)


def _fake_pdf_upload(name="test.pdf"):
    # Content doesn't need to be a real PDF - FakeAgent.add_document never parses it.
    return {"file": (name, io.BytesIO(b"%PDF-1.4 fake content"), "application/pdf")}


def test_new_session_created_when_none_provided():
    client = _make_client()
    resp = client.post("/api/chat", json={"message": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"]
    assert "agent#1" in body["answer"]
    print("OK  a fresh session_id is minted when none is supplied")


def test_same_session_id_reuses_the_same_agent():
    client = _make_client()
    first = client.post("/api/chat", json={"message": "q1"}).json()
    sid = first["session_id"]
    second = client.post("/api/chat", json={"message": "q2", "session_id": sid}).json()
    assert second["session_id"] == sid
    assert "turn 2" in second["answer"]  # same agent instance, second turn
    assert FakeAgent._instances_created == 1, "must not create a second agent for the same session"
    print("OK  reusing a session_id reuses the same agent instance (conversation continuity)")


def test_different_sessions_get_isolated_agents():
    client = _make_client()
    a = client.post("/api/chat", json={"message": "from user A"}).json()
    b = client.post("/api/chat", json={"message": "from user B"}).json()
    assert a["session_id"] != b["session_id"]
    assert FakeAgent._instances_created == 2, "two different sessions must get two separate agents"
    print("OK  two visitors with no session_id get fully isolated agents (no cross-contamination)")


def test_question_cap_surfaces_through_the_api():
    client = _make_client()
    resp = client.post("/api/chat", json={"message": "q1"})
    sid = resp.json()["session_id"]
    client.post("/api/chat", json={"message": "q2", "session_id": sid})
    client.post("/api/chat", json={"message": "q3", "session_id": sid})
    fourth = client.post("/api/chat", json={"message": "q4", "session_id": sid}).json()
    assert fourth["questions_remaining"] == 0
    assert "limit" in fourth["answer"].lower()
    print("OK  the per-session question cap is visible through the HTTP API")


def test_upload_creates_a_session_and_returns_doc_list():
    client = _make_client()
    resp = client.post("/api/upload", files=_fake_pdf_upload("paper.pdf"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"]
    assert body["doc"]["source"] == "paper.pdf"
    assert len(body["docs"]) == 1
    print("OK  uploading without a session_id creates one and returns the doc list")


def test_upload_rejects_non_pdf():
    client = _make_client()
    files = {"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    resp = client.post("/api/upload", files=files)
    assert resp.status_code == 400
    print("OK  non-PDF uploads are rejected with a 400")


def test_two_sessions_do_not_share_uploaded_doc_lists():
    client = _make_client()
    a = client.post("/api/upload", files=_fake_pdf_upload("a.pdf")).json()
    b = client.post("/api/upload", files=_fake_pdf_upload("b.pdf")).json()
    assert a["session_id"] != b["session_id"]
    assert [d["source"] for d in a["docs"]] == ["a.pdf"]
    assert [d["source"] for d in b["docs"]] == ["b.pdf"]

    # Fetching session A's docs by ID must never include session B's upload.
    docs_a = client.get("/api/docs", params={"session_id": a["session_id"]}).json()["docs"]
    assert [d["source"] for d in docs_a] == ["a.pdf"]
    print("OK  two sessions' uploaded documents never appear in each other's lists")


def test_upload_enforces_max_uploads_per_session():
    client = _make_client()
    from research_assistant.config import settings

    session_id = None
    resp = None
    for i in range(settings.max_uploads_per_session + 1):
        resp = client.post("/api/upload", data={"session_id": session_id} if session_id else {}, files=_fake_pdf_upload(f"doc{i}.pdf"))
        if resp.status_code == 200:
            session_id = resp.json()["session_id"]
    assert resp.status_code == 400, "uploading past the per-session cap must be rejected"
    print(f"OK  upload is capped at {settings.max_uploads_per_session} documents per session")


def test_index_html_is_served_at_root():
    client = _make_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Personal Research Assistant" in resp.text
    assert "text/html" in resp.headers["content-type"]
    print("OK  index.html is served at / (frontend reachable)")


# --- IP-level rate limiting: closes the "refresh the page for a new cap" loophole ---
# FakeAgent's own per-session cap defaults to 3, well under settings.demo_max_questions
# (5), so these tests exercise the IP-level check specifically, not the agent's.

def _chat(client, message, session_id=None, ip="1.2.3.4"):
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    return client.post("/api/chat", json=payload, headers={"X-Forwarded-For": ip})


def test_refreshing_the_page_does_not_reset_the_ip_cap():
    from research_assistant.config import settings

    client = _make_client()
    # Simulate a visitor who refreshes before every single question, so the
    # frontend's in-memory sessionId is discarded and no session_id is ever
    # sent - each request looks like the start of a brand new session.
    last = None
    for i in range(settings.demo_max_questions):
        last = _chat(client, f"q{i}").json()
        assert "agent#" in last["answer"], f"question {i} should have gone through, got: {last['answer']}"
    assert FakeAgent._instances_created == settings.demo_max_questions, (
        "each no-session_id request does get its own agent/session"
    )

    # One more "refresh" from the same IP: a brand new session_id would
    # normally mean a brand new agent with a full cap, but the IP-level
    # tracker must still catch it.
    over_limit = _chat(client, "one more").json()
    assert over_limit["questions_remaining"] == 0
    assert "ip address" in over_limit["answer"].lower()
    print("OK  refreshing the page (new session_id every time) does not reset the demo cap")


def test_ip_cap_applies_even_within_a_single_persisted_session():
    from research_assistant.config import settings

    client = _make_client()
    sid = None
    for i in range(settings.demo_max_questions):
        resp = _chat(client, f"q{i}", session_id=sid, ip="9.9.9.9").json()
        sid = resp["session_id"]
    over_limit = _chat(client, "one more", session_id=sid, ip="9.9.9.9").json()
    assert over_limit["questions_remaining"] == 0
    assert FakeAgent._instances_created == 1, "same session_id the whole time - only one agent"
    print("OK  the IP cap also kicks in for a visitor who keeps the same session_id")


def test_different_ips_get_independent_caps():
    from research_assistant.config import settings

    client = _make_client()
    for i in range(settings.demo_max_questions):
        resp_a = _chat(client, f"a{i}", ip="10.0.0.1").json()
        assert "agent#" in resp_a["answer"]
    # IP A is now exhausted; IP B must be unaffected.
    resp_b = _chat(client, "hello from B", ip="10.0.0.2").json()
    assert "agent#" in resp_b["answer"]
    assert resp_b["questions_remaining"] != 0
    print("OK  two different IPs get fully independent caps")


def test_ip_usage_not_double_counted_once_session_cap_already_hit():
    """An already-session-capped visitor (FakeAgent max=3) hammering /api/chat
    on the same session_id must not burn through the IP quota (5) for free
    rejection messages - only real questions should count against it."""
    from research_assistant.config import settings

    client = _make_client()
    sid = None
    for i in range(3):  # FakeAgent's own cap
        resp = _chat(client, f"q{i}", session_id=sid, ip="8.8.8.8").json()
        sid = resp["session_id"]

    # Session is now capped by FakeAgent itself. Hammer it well past the IP
    # cap's threshold - none of these should consume IP quota.
    for _ in range(settings.demo_max_questions + 2):
        _chat(client, "are we there yet", session_id=sid, ip="8.8.8.8")

    assert webapp._ip_questions_used("8.8.8.8") == 3, "only the 3 real questions should count"
    print("OK  requests rejected by the session's own cap don't also burn IP quota")


if __name__ == "__main__":
    test_new_session_created_when_none_provided()
    test_same_session_id_reuses_the_same_agent()
    test_different_sessions_get_isolated_agents()
    test_question_cap_surfaces_through_the_api()
    test_upload_creates_a_session_and_returns_doc_list()
    test_upload_rejects_non_pdf()
    test_two_sessions_do_not_share_uploaded_doc_lists()
    test_upload_enforces_max_uploads_per_session()
    test_index_html_is_served_at_root()
    test_refreshing_the_page_does_not_reset_the_ip_cap()
    test_ip_cap_applies_even_within_a_single_persisted_session()
    test_different_ips_get_independent_caps()
    test_ip_usage_not_double_counted_once_session_cap_already_hit()
    print("\nAll webapp tests passed.")
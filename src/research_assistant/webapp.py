"""Local web app: FastAPI backend + static chat frontend for the research assistant.

Wraps ResearchAgent with per-visitor session isolation - one agent instance
per browser session, tracked by a session_id the frontend generates and
resends. Each session gets its OWN VectorStore(ephemeral=True) (in-memory,
never touches disk, never shared) and its own empty doc registry, so
uploaded PDFs, conversation history, and the 5-question demo cap are all
private per visitor. There is no shared document corpus in this mode -
visitors must upload their own PDF(s) before they can ask anything about
one; nothing here reads from data/pdfs or the CLI's persistent store.

Run with:
    python scripts/serve.py
then open http://localhost:8000

This is intentionally simple for a low-traffic personal demo: sessions live
in an in-memory dict (lost on restart, not shared across multiple server
processes) and are swept on a plain TTL, no database. Fine for "show this
in an interview"; would need real session storage (Redis, etc.) if this
ever needed to survive restarts or run behind multiple worker processes.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import ResearchAgent
from .config import settings
from .vectorstore import VectorStore

app = FastAPI(title="Personal Research Assistant")

_STATIC_DIR = Path(__file__).resolve().parent / "web"

_SESSIONS: dict[str, tuple[ResearchAgent, float]] = {}
_SESSION_TTL_SECONDS = 2 * 60 * 60  # sweep sessions/IP usage idle longer than this

# ip -> list of timestamps of questions actually asked from that IP, within
# the last _SESSION_TTL_SECONDS. Separate from _SESSIONS on purpose: this
# survives a visitor discarding their session_id, since it's keyed by IP
# instead.
_IP_USAGE: dict[str, list[float]] = {}


def _cleanup_expired() -> None:
    now = time.time()
    cutoff = now - _SESSION_TTL_SECONDS

    expired = [sid for sid, (_, last_used) in _SESSIONS.items() if last_used < cutoff]
    for sid in expired:
        del _SESSIONS[sid]

    for ip in list(_IP_USAGE.keys()):
        recent = [t for t in _IP_USAGE[ip] if t > cutoff]
        if recent:
            _IP_USAGE[ip] = recent
        else:
            del _IP_USAGE[ip]


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _ip_questions_used(ip: str) -> int:
    return len(_IP_USAGE.get(ip, []))


def _record_ip_question(ip: str) -> None:
    _IP_USAGE.setdefault(ip, []).append(time.time())


def _new_session_agent() -> ResearchAgent:
    return ResearchAgent(verbose=False, store=VectorStore(ephemeral=True), doc_registry={})


def _get_or_create_agent(session_id: str | None) -> tuple[str, ResearchAgent]:
    _cleanup_expired()
    if session_id and session_id in _SESSIONS:
        agent, _ = _SESSIONS[session_id]
        _SESSIONS[session_id] = (agent, time.time())
        return session_id, agent

    new_id = session_id or str(uuid.uuid4())
    agent = _new_session_agent()
    _SESSIONS[new_id] = (agent, time.time())
    return new_id, agent


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    questions_remaining: int | None


class UploadResponse(BaseModel):
    session_id: str
    doc: dict
    docs: list[dict]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    _cleanup_expired()
    ip = _client_ip(request)

    if settings.demo_max_questions > 0 and _ip_questions_used(ip) >= settings.demo_max_questions:
        return ChatResponse(
            answer=(
                f"This demo is capped at {settings.demo_max_questions} questions per visitor. "
                "That's tracked server-side by IP address, not just this browser tab - "
                "refreshing the page or starting a new session won't reset it. Run your own "
                "copy locally (see the README) to remove the cap."
            ),
            session_id=req.session_id or str(uuid.uuid4()),
            questions_remaining=0,
        )

    session_id, agent = _get_or_create_agent(req.session_id)

    already_capped = agent.questions_remaining() == 0
    answer = agent.ask(req.message)
    if not already_capped:
        _record_ip_question(ip)

    remaining = agent.questions_remaining()
    if remaining is not None and settings.demo_max_questions > 0:
        ip_remaining = max(settings.demo_max_questions - _ip_questions_used(ip), 0)
        remaining = min(remaining, ip_remaining)
    return ChatResponse(answer=answer, session_id=session_id, questions_remaining=remaining)


@app.post("/api/upload", response_model=UploadResponse)
async def upload(session_id: str | None = Form(default=None), file: UploadFile = File(...)) -> UploadResponse:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id, agent = _get_or_create_agent(session_id)

    if len(agent.doc_registry) >= settings.max_uploads_per_session:
        raise HTTPException(
            status_code=400,
            detail=f"This session already has {settings.max_uploads_per_session} documents "
            "(the max for a demo session). Type 'reset' / refresh to start over.",
        )

    data = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File too large - max {settings.max_upload_mb}MB.")

    try:
        doc = agent.add_document(data, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return UploadResponse(session_id=session_id, doc=doc, docs=agent.list_documents())


@app.get("/api/docs")
def list_docs(session_id: str | None = None) -> dict:
    if not session_id or session_id not in _SESSIONS:
        return {"docs": []}
    _, agent = _get_or_create_agent(session_id)
    return {"docs": agent.list_documents()}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "active_sessions": len(_SESSIONS)}


# Registered after the /api routes above so they take priority; this mount
# is the fallback that serves index.html and any other static assets.
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
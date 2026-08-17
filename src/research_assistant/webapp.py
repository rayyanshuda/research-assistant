"""Local web app: FastAPI backend + static chat frontend for the research assistant.

Wraps ResearchAgent with per-visitor session isolation — one agent instance
per browser session, tracked by a session_id the frontend generates and
resends — so multiple people trying this don't share conversation history,
retrieved context, or each other's 5-question demo cap.

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

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import ResearchAgent
from .vectorstore import VectorStore

app = FastAPI(title="Personal Research Assistant")

_STATIC_DIR = Path(__file__).resolve().parent / "web"

_SESSIONS: dict[str, tuple[ResearchAgent, float]] = {}
_SESSION_TTL_SECONDS = 2 * 60 * 60  # sweep sessions idle longer than this


def _cleanup_sessions() -> None:
    now = time.time()
    expired = [sid for sid, (_, last_used) in _SESSIONS.items() if now - last_used > _SESSION_TTL_SECONDS]
    for sid in expired:
        del _SESSIONS[sid]


def _get_or_create_agent(session_id: str | None) -> tuple[str, ResearchAgent]:
    _cleanup_sessions()
    if session_id and session_id in _SESSIONS:
        agent, _ = _SESSIONS[session_id]
        _SESSIONS[session_id] = (agent, time.time())
        return session_id, agent

    new_id = session_id or str(uuid.uuid4())
    agent = ResearchAgent(verbose=False)
    _SESSIONS[new_id] = (agent, time.time())
    return new_id, agent


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    questions_remaining: int | None


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    session_id, agent = _get_or_create_agent(req.session_id)
    answer = agent.ask(req.message)
    return ChatResponse(answer=answer, session_id=session_id, questions_remaining=agent.questions_remaining())


@app.get("/api/health")
def health() -> dict:
    store = VectorStore()
    return {"status": "ok", "chunks_indexed": store.count(), "active_sessions": len(_SESSIONS)}


# Registered after the /api routes above so they take priority; this mount
# is the fallback that serves index.html and any other static assets.
app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
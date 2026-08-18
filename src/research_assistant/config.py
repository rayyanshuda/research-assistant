# config laoded from env variables
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# load .env from the project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")

    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "gemini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"
    pdfs_dir: Path = _PROJECT_ROOT / "data" / "pdfs"
    notes_dir: Path = _PROJECT_ROOT / "data" / "notes"
    doc_registry_path: Path = _PROJECT_ROOT / "data" / "doc_registry.json"

    chroma_dir: str = os.getenv("CHROMA_DIR", str(_PROJECT_ROOT / "data" / "chroma"))
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "notes")

    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 60

    # Caps questions per ResearchAgent session (agent/agent_loop.py), stops a
    # single demo visitor from running up unbounded Claude API cost. Set to 0
    # to disable the cap entirely (unlimited questions) for local use.
    demo_max_questions: int = int(os.getenv("DEMO_MAX_QUESTIONS", "5"))

    # Per-session upload limits for the web app (webapp.py), stops a single
    # visitor from uploading unbounded PDFs/pages and running up embedding cost.
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "20"))
    max_uploads_per_session: int = int(os.getenv("MAX_UPLOADS_PER_SESSION", "3"))

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.openai_api_key

    def require_anthropic_key(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.anthropic_api_key

    def require_gemini_key(self) -> str:
        if not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey and add it to .env"
            )
        return self.gemini_api_key


settings = Settings()

#!/usr/bin/env python3
"""Simple REPL for the research assistant."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_assistant.agent import ResearchAgent  # noqa: E402
from research_assistant.vectorstore import VectorStore  # noqa: E402


def main():
    store = VectorStore()
    n = store.count()
    if n == 0:
        print(
            "Vector store is empty. Run `python scripts/ingest.py` first to index your "
            "notes/PDFs, then come back."
        )
        return
    print(f"Research assistant ready — {n} chunks indexed. Type a question, or 'exit' to quit.\n")

    agent = ResearchAgent()
    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        if query.lower() == "reset":
            agent.reset()
            print("(conversation reset)")
            continue

        answer = agent.ask(query)
        remaining = agent.questions_remaining()
        print(f"\nassistant> {answer}\n")
        if remaining is not None:
            print(f"({remaining} question{'s' if remaining != 1 else ''} left this session)\n")


if __name__ == "__main__":
    main()

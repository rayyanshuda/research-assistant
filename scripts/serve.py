#!/usr/bin/env python3
"""Run the local web version of the research assistant.

    python scripts/serve.py [--port 8000] [--host 127.0.0.1]

Then open the printed URL in a browser. Unlike the CLI, this does NOT need
ingest.py to have been run first - each browser session starts with an
empty, private, in-memory document store, and you upload PDF(s) via the
sidebar's Upload button once the page is open. ingest.py only populates the
CLI's separate shared/persistent store (data/chroma/), which this web app
never reads from.

No --reload here: if you edit any .py file, stop this (Ctrl+C) and re-run
it to pick up the change - it won't hot-reload on its own.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn

    from research_assistant.webapp import app

    print(f"Starting research assistant web app at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
# uses duckduckgo-search to aid in providing the best response (No API)
from __future__ import annotations


def fetch_web(query: str, max_results: int = 5) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return [
            {
                "error": (
                    "duckduckgo-search is not installed. Run `pip install duckduckgo-search` "
                    "or wire up a different web search provider in tools/fetch_web.py."
                )
            }
        ]

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # network errors, rate limits, etc.
        return [{"error": f"web search failed: {exc}"}]

    return [
        {
            "title": h.get("title"),
            "url": h.get("href"),
            "snippet": h.get("body"),
        }
        for h in hits
    ]

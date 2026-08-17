"""Claude tool-use JSON schema definitions for the three agent tools."""

TOOL_DEFINITIONS = [
    {
        "name": "search_notes",
        "description": (
            "Search the user's ingested personal notes and PDFs for relevant passages. "
            "Always try this first before fetch_web — the user's own material is the "
            "primary source of truth. Returns passages with citations (source file + "
            "page/section) that MUST be used when answering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "top_k": {
                    "type": "integer",
                    "description": "Number of passages to retrieve (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_web",
        "description": (
            "Search the public web. Use this only to supplement the user's notes — "
            "e.g. when search_notes doesn't have enough information, or the question "
            "is about something outside the user's personal material (recent events, "
            "general facts not in their notes)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Web search query."},
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "summarize_doc",
        "description": (
            "Summarize an entire ingested document, not just the chunks returned by "
            "search_notes. Use this when the user asks for an overview of a whole "
            "paper/note rather than an answer to a specific question. Pass either the "
            "doc_id or the source file name from a prior search_notes citation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "The doc_id or source file name to summarize (e.g. 'LoRA.pdf').",
                },
                "focus": {
                    "type": "string",
                    "description": "Optional: what aspect of the document to focus the summary on.",
                },
            },
            "required": ["doc_id"],
        },
    },
]

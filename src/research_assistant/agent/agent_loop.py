"""Claude tool-use agent loop.

Keeps calling the model, executing whatever tools it asks for, and feeding
results back, until the model returns a plain text answer (stop_reason ==
"end_turn"). Conversation history is kept in-memory per ResearchAgent
instance so multi-turn CLI sessions have context.

Prompt caching: the system prompt and tool definitions never change, and
the conversation history only grows, so we mark cache breakpoints on all
three. That turns repeated tokens (system + tools + everything asked so
far) into 0.1x-priced cache reads on every call after the first, instead of
paying full input price for the same content over and over within a
session. See _apply_cache_breakpoint() for how the message-history
breakpoint is kept to a single, moving marker (Anthropic allows at most 4
cache breakpoints per request; we use 3: system, tools, messages).

Session question cap: this is meant to be usable as a public demo, so each
ResearchAgent enforces a max number of questions per session
(DEMO_MAX_QUESTIONS in .env, default 5) before it stops calling the API at
all — keeps a single visitor from running up unbounded cost.
"""
from __future__ import annotations

import anthropic

from ..config import settings
from ..tools import TOOL_DEFINITIONS, dispatch_tool

SYSTEM_PROMPT = """You are a personal research assistant with access to the user's own \
notes and PDFs, plus the ability to search the web.

Rules:
1. ALWAYS call search_notes first for any question that could be answered from the \
user's material. Prefer their notes over general knowledge.
2. Only call fetch_web if search_notes doesn't turn up enough, or the question is \
explicitly about something outside the user's notes (e.g. current events).
3. Use summarize_doc when the user wants an overview of an entire document rather \
than an answer to a narrow question.
4. EVERY factual claim drawn from search_notes or fetch_web must be followed by an \
inline citation in square brackets, e.g. [LoRA.pdf, p.3] or [DuckDuckGo: example.com]. \
Never state something from a source without citing it.
5. If the notes and the web disagree, say so explicitly rather than picking one silently.
6. If you don't have enough information after searching, say so plainly instead of \
guessing.
7. Format any mathematical notation as LaTeX: $...$ for inline math, $$...$$ for \
block/display equations. The web UI renders these properly — don't approximate math \
with plain unicode symbols when LaTeX is available.
"""
"""

_CACHED_SYSTEM = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

# Mark the last tool definition as a cache breakpoint — this caches the whole
# tools array (breakpoints cache everything up to and including themselves).
_CACHED_TOOLS = [dict(t) for t in TOOL_DEFINITIONS]
_CACHED_TOOLS[-1] = {**_CACHED_TOOLS[-1], "cache_control": {"type": "ephemeral"}}


def _apply_cache_breakpoint(history: list[dict]) -> None:
    """Move the message-history cache breakpoint to the end of `history`.

    Strips any cache_control we previously added to earlier messages (so we
    never exceed the 4-breakpoint-per-request limit as a session grows),
    then marks the last content block of the last message. Only touches
    "user" messages, since those are the plain dicts we construct ourselves
    (initial questions, tool_results) — assistant messages are SDK response
    objects we pass straight through and never need to mark.
    """
    for msg in history:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop("cache_control", None)

    if not history:
        return
    last = history[-1]
    if last.get("role") != "user":
        return
    if isinstance(last["content"], str):
        last["content"] = [{"type": "text", "text": last["content"]}]
    content = last["content"]
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1]["cache_control"] = {"type": "ephemeral"}


class ResearchAgent:
    def __init__(
        self,
        model: str | None = None,
        max_tool_iterations: int = 8,
        verbose: bool = True,
        max_questions: int | None = None,
    ):
        self._client = anthropic.Anthropic(api_key=settings.require_anthropic_key())
        self._model = model or settings.anthropic_model
        self._max_tool_iterations = max_tool_iterations
        self._verbose = verbose
        self._history: list[dict] = []
        self._max_questions = settings.demo_max_questions if max_questions is None else max_questions
        self._questions_asked = 0

    def reset(self):
        self._history = []
        self._questions_asked = 0

    def questions_remaining(self) -> int | None:
        """Returns None if the cap is disabled (max_questions <= 0)."""
        if self._max_questions <= 0:
            return None
        return max(self._max_questions - self._questions_asked, 0)

    def ask(self, user_message: str) -> str:
        if self._max_questions > 0 and self._questions_asked >= self._max_questions:
            return (
                f"This session has hit its {self._max_questions}-question demo limit. "
                "Type 'reset' to start a fresh session, or run your own copy without a "
                "cap by setting DEMO_MAX_QUESTIONS in .env."
            )
        self._questions_asked += 1

        self._history.append({"role": "user", "content": user_message})

        for _ in range(self._max_tool_iterations):
            _apply_cache_breakpoint(self._history)
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1536,
                system=_CACHED_SYSTEM,
                tools=_CACHED_TOOLS,
                messages=self._history,
            )
            self._history.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                return "".join(block.text for block in response.content if block.type == "text")

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if self._verbose:
                    print(f"  [tool] {block.name}({block.input})")
                result = dispatch_tool(block.name, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": _stringify(result),
                    }
                )
            self._history.append({"role": "user", "content": tool_results})

        return (
            "I hit the tool-call limit for this turn without reaching a final answer. "
            "Try narrowing your question."
        )


def _stringify(result) -> str:
    import json

    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)
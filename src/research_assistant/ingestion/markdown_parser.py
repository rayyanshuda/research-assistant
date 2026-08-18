# markdown to section extraction
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class MarkdownSection:
    heading: str
    level: int
    text: str


def parse_markdown(path: str | Path) -> list[MarkdownSection]:
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()

    sections: list[MarkdownSection] = []
    current_heading = "(untitled)"
    current_level = 0
    buffer: list[str] = []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            sections.append(MarkdownSection(heading=current_heading, level=current_level, text=text))

    for line in lines:
        match = _HEADER_RE.match(line)
        if match:
            flush()
            buffer = []
            current_level = len(match.group(1))
            current_heading = match.group(2).strip()
        else:
            buffer.append(line)
    flush()

    return sections

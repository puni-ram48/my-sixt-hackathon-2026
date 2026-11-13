"""Stage 1: split a contract into clause units that keep their source position.

CUAD stores each contract as one long paragraph, so clause units are made here:
blank-line blocks first, then sentence windows for blocks that are still too long.
Every unit keeps its character offsets, so an answer can always be traced back.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_CHARS = 700
_BLOCK = re.compile(r"\S(?:.*?)(?=\n\s*\n|\Z)", re.S)
_SENTENCE = re.compile(r"[^.;:!?\n]+(?:[.;:!?]+|\n|$)|\n", re.S)
_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z ,'()/&-]{2,80}\.?\s*$")


@dataclass
class Clause:
    clause_id: str
    doc_id: str
    index: int
    start: int
    end: int
    text: str
    heading: str | None = None
    head_start: int | None = None  # offset of the heading line, if one was attached

    @property
    def span_start(self) -> int:
        """Start of the clause including its heading (titles live in headings)."""
        return self.head_start if self.head_start is not None else self.start


def _windows(block_text: str, offset: int) -> list[tuple[int, int, str]]:
    """Group sentences of a long block into pieces of at most MAX_CHARS characters."""
    pieces: list[tuple[int, int, str]] = []
    cur_start, cur_end = None, None
    for m in _SENTENCE.finditer(block_text):
        if not m.group().strip():
            continue
        s, e = offset + m.start(), offset + m.end()
        if cur_start is None:
            cur_start, cur_end = s, e
        elif e - cur_start > MAX_CHARS:
            pieces.append((cur_start, cur_end, ""))
            cur_start, cur_end = s, e
        else:
            cur_end = e
    if cur_start is not None:
        pieces.append((cur_start, cur_end, ""))
    return pieces


def split_clauses(doc_id: str, text: str) -> list[Clause]:
    spans: list[tuple[int, int]] = []
    for m in _BLOCK.finditer(text):
        block = m.group()
        if len(block) <= MAX_CHARS:
            spans.append((m.start(), m.end()))
        else:
            spans.extend((s, e) for s, e, _ in _windows(block, m.start()))

    clauses: list[Clause] = []
    pending_heading: str | None = None
    pending_start: int | None = None
    for start, end in spans:
        chunk = text[start:end].strip()
        if not chunk:
            continue
        # A short title line ("8. Governing Law") is attached to the clause that follows it.
        if len(chunk) <= 80 and _HEADING.match(chunk) and not chunk.endswith("."):
            pending_heading, pending_start = chunk, start
            continue
        clauses.append(Clause(
            clause_id=f"{doc_id}#{len(clauses)}", doc_id=doc_id, index=len(clauses),
            start=start, end=end, text=chunk, heading=pending_heading, head_start=pending_start,
        ))
        pending_heading, pending_start = None, None
    return clauses

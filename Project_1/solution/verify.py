"""Grounding gate: a value is only trusted if it can be found in the document.

A wrong booked value costs 100 and a review costs 1, so a value that cannot be
traced back to the document is turned into null (a review) instead of being booked.
"""
from __future__ import annotations

import difflib
import re
from typing import Any

_FIX = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})
_TOKEN = re.compile(r"[a-z0-9]+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
MULTI_SEP = " | "  # CUAD joins several gold answers with this separator
BOOLEAN_WORDS = {"true", "false", "yes", "no"}


def tokens(s: Any) -> list[str]:
    """Lowercase alphanumeric tokens. Ignores case, punctuation and spacing."""
    s = _THOUSANDS.sub("", str(s).translate(_FIX).lower())  # 1,250 -> 1250
    return _TOKEN.findall(s)


_RAW_TOKEN = re.compile(r"[A-Za-z0-9]+")
_SENTENCE_MARKS = set(".;:!?\n")
_SKIPPABLE = set(" \t\"'“”‘’")
# "(A)", "(iv)", "(12)" at the end of the text before a value, or "5.6" / "2." before it.
_LIST_MARKER = re.compile(r"(?:^|[\s\n])(?:\([A-Za-z0-9]{1,5}\)|\d+(?:\.\d+)*\.)\s*$")
# Values shorter than this are names, dates or titles; the sentence rule does not apply.
MIN_TOKENS_FOR_SENTENCE_RULE = 8


class Grounder:
    """Checks values against one document. Build once per document."""

    def __init__(self, text: str):
        self.tokens = tokens(text)
        self.joined = " " + " ".join(self.tokens) + " "
        self._text = text
        self._raw: list[tuple[str, int, int]] | None = None

    def aligned(self, value: Any) -> bool | None:
        """True if the value covers whole sentences of the document.

        The value must match a run of words in the text that starts right after a
        sentence end (or a line break, or the start of the text) and stops right
        before a sentence end. A fragment such as "laws of the State of Illinois",
        which starts in the middle of a sentence, is not aligned.
        None means the value is not a string, so the check does not apply.
        """
        if not isinstance(value, str):
            return None
        target = [t.lower() for t in _RAW_TOKEN.findall(value)]
        if not target:
            return False
        if self._raw is None:
            self._raw = [(m.group().lower(), m.start(), m.end()) for m in _RAW_TOKEN.finditer(self._text)]
        n = len(target)
        for i in range(len(self._raw) - n + 1):
            if [w for w, _, _ in self._raw[i:i + n]] != target:
                continue
            start, end = self._raw[i][1], self._raw[i + n - 1][2]
            if self._sentence_start(start) and self._sentence_end(end):
                return True
        return False

    def _sentence_start(self, pos: int) -> bool:
        k = pos - 1
        while k >= 0 and self._text[k] in _SKIPPABLE:
            k -= 1
        if k < 0 or self._text[k] in _SENTENCE_MARKS:
            return True
        # A list marker ("(A)", "(iv)", "5.6", "2.") also starts a new clause.
        before = self._text[max(0, k - 12): k + 1]
        return bool(_LIST_MARKER.search(before))

    def _sentence_end(self, pos: int) -> bool:
        k = pos
        while k < len(self._text) and self._text[k] in _SKIPPABLE:
            k += 1
        return k >= len(self._text) or self._text[k] in _SENTENCE_MARKS

    def _score_tokens(self, value_tokens: list[str]) -> float:
        if not value_tokens:
            return 0.0
        if " " + " ".join(value_tokens) + " " in self.joined:
            return 1.0
        # Not an exact match (e.g. OCR noise or a small rewording): score the share
        # of value tokens that line up, in order, with the document tokens.
        matcher = difflib.SequenceMatcher(None, value_tokens, self.tokens, autojunk=False)
        matched = sum(block.size for block in matcher.get_matching_blocks())
        return matched / len(value_tokens)

    def score(self, value: Any) -> float | None:
        """1.0 = found verbatim, 0.0 = not found, None = cannot be checked as text."""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, str) and value.strip().lower() in BOOLEAN_WORDS:
            return None  # checkboxes are marks on a form, not words in the text layer
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if isinstance(value, (int, float)):
            return self._score_tokens(tokens(value))
        if isinstance(value, str):
            parts = [p for p in value.split(MULTI_SEP) if p.strip()]
            if not parts:
                return 0.0
            # Every answer in a multi-answer value must be grounded.
            return min(self._score_tokens(tokens(p)) for p in parts)
        return None  # lists and nested objects are not checked here


def gate(value: Any, confidence: float, grounding: float | None,
         min_grounding: float = 0.9) -> tuple[Any, float]:
    """Apply the gate. An unsupported value becomes null (sent to review)."""
    if value is None:
        return None, confidence
    if grounding is not None and grounding < min_grounding:
        return None, 0.0
    return value, confidence

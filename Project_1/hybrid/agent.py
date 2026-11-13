"""Experiment E: agentic RAG - the LLM searches and reads the contract itself before answering.

Why: one-shot retrieval uses the field definition as the query, and BM25 misses clauses that use different
wording ("aggregate liability shall not exceed" vs "Cap On Liability"). Here the model writes its own
queries, reads clauses it finds interesting (with neighbours), and only then answers with quotes.
Long contracts are no problem: nothing is truncated, the agent navigates.

Protocol (plain JSON, so it works on any OpenAI-compatible endpoint, no tool-calling support needed):
  model -> {"actions": [{"tool": "search", "query": "..."}, {"tool": "read", "clause_id": "C12"}]}
  us    -> the results
  ...      at most MAX_TURNS turns, then:
  model -> {"answers": {"<field>": {"value", "evidence", "explanation", "confidence", "supported"}}}

The answers then go through the SAME grounding, clause-type check, verifier and decision rules as the
hybrid pipeline. The clauses the agent actually saw play the role of the retrieved candidates.
Search is BM25 over clauses (no embedding model, so this runs on any machine).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .extract import Extraction, _to_extraction, cid
from .fields import FieldSpec
from .llm import LLMResult, chat, parse_json_object
from .retrieve import Candidate, DocIndex

log = logging.getLogger("hybrid.agent")
MAX_TURNS = 4            # search/read turns before the agent must answer
MAX_ACTIONS = 15         # actions per turn (41 fields, 4 turns)
SEARCH_K = 4             # clauses returned per search
SNIPPET = 600            # characters of a clause shown in search results
OUTLINE_CHARS = 70       # characters per clause in the outline
OUTLINE_MAX = 24_000     # cap on the outline size

SYSTEM = """You are a careful contract analyst. You extract fields from ONE contract that you cannot see in full.
You can explore it with tools, then you must answer. A wrong answer is much worse than null ("I don't know").

Tools (reply with ONE JSON object and nothing else):
{{"actions": [{{"tool": "search", "query": "<keywords a lawyer would expect in the clause>"}},
             {{"tool": "read", "clause_id": "C12"}}]}}
- search: keyword search over the contract's clauses; returns the best {k} clauses. Use the contract's own legal
  wording, synonyms and several queries per field. Up to {max_actions} actions per turn, batch them.
- read: the full clause plus its neighbours.
Send exactly ONE JSON object per reply, then stop and wait: the tool results come in the next message.
How to explore:
- Identity information (title, parties, dates) is usually in the first clauses: always read them.
- Search for EVERY field at least once (with the contract's likely wording) before deciding it is absent.
  Do not conclude "absent" from the outline alone.
You have {turns} exploration turns. Answer only after you have explored, with:
{{"answers": {{"<field name>": {{"value": "...", "evidence": [{{"clause_id": "C12", "quote": "<copied exactly>"}}],
  "explanation": "<at most 20 words>", "confidence": 0.0, "supported": true}}}}}}
For a field the contract does not contain: {{"<field name>": {{"value": null, "supported": false}}}}

Answer rules:
- Only use text you have seen through the tools. Quotes must be copied character-for-character.
- The clause must really be the kind of clause the definition describes; shared words are not enough.
- For a clause-type field, the value is the complete sentence(s) that state it (several joined with " | ");
  for a date, name or title, the value exactly as written.
- confidence = your probability that the value is correct and complete.
Include EVERY field in "answers".
"""

USER = """FIELDS:
{fields}

CONTRACT OUTLINE (clause id: start of text):
{outline}
"""


@dataclass
class AgentRun:
    extractions: dict[str, Extraction]
    calls: list[LLMResult]
    seen: list[int]                       # clause indices shown to the agent, in order
    log: list[dict] = field(default_factory=list)


def outline(idx: DocIndex) -> str:
    lines, size = [], 0
    for i, c in enumerate(idx.clauses):
        head = f"[{c.heading}] " if c.heading else ""
        line = f"{cid(i)}: {head}{' '.join(c.text.split())[:OUTLINE_CHARS]}"
        size += len(line) + 1
        if size > OUTLINE_MAX:
            lines.append(f"... ({len(idx.clauses) - i} more clauses; use search)")
            break
        lines.append(line)
    return "\n".join(lines)


def _field_lines(specs: list[FieldSpec]) -> str:
    return "\n".join(f'- "{f.name}": {f.description}' for f in specs)


class _Tools:
    def __init__(self, idx: DocIndex):
        self.idx, self.shown_full, self.seen = idx, set(), []

    def _mark(self, i: int) -> None:
        if i not in self.seen:
            self.seen.append(i)

    def search(self, query: str) -> str:
        scores = self.idx.bm25.scores(str(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:SEARCH_K]
        out = []
        for i in order:
            if scores[i] <= 0:
                continue
            self._mark(i)
            c = self.idx.clauses[i]
            if i in self.shown_full:
                out.append(f"[{cid(i)}] (already read)")
            else:
                text = " ".join(c.text.split())
                out.append(f"[{cid(i)}]{' (' + c.heading + ')' if c.heading else ''} {text[:SNIPPET]}"
                           + (" ..." if len(text) > SNIPPET else ""))
        return "\n".join(out) or "(no matching clauses)"

    def read(self, clause_id: str) -> str:
        try:
            i = int(str(clause_id).strip().lstrip("Cc"))
            assert 0 <= i < len(self.idx.clauses)
        except (ValueError, AssertionError):
            return f"(unknown clause id {clause_id!r})"
        out = []
        for j in range(max(0, i - 1), min(len(self.idx.clauses), i + 2)):
            self._mark(j)
            c = self.idx.clauses[j]
            if j in self.shown_full:
                out.append(f"[{cid(j)}] (already read)")
                continue
            self.shown_full.add(j)
            out.append(f"[{cid(j)}]{' (' + c.heading + ')' if c.heading else ''}{' <- requested' if j == i else ''}\n{c.text}")
        return "\n\n".join(out)


def run_agent(idx: DocIndex, specs: list[FieldSpec]) -> AgentRun:
    tools = _Tools(idx)
    messages = [{"role": "system", "content": SYSTEM.format(k=SEARCH_K, max_actions=MAX_ACTIONS, turns=MAX_TURNS)},
                {"role": "user", "content": USER.format(fields=_field_lines(specs), outline=outline(idx))}]
    calls, trace, answers = [], [], None
    for turn in range(MAX_TURNS + 1):
        if turn == MAX_TURNS:
            messages.append({"role": "user", "content": "No more tool turns. Reply now with {\"answers\": {...}} for every field."})
        try:
            res = chat(messages)
        except Exception as e:
            log.warning("%s: agent call failed: %s", idx.doc_id, e)
            break
        calls.append(res)
        reply = parse_json_object(res.text)
        messages.append({"role": "assistant", "content": res.text})
        if isinstance(reply.get("answers"), dict):
            answers = reply["answers"]
            trace.append({"turn": turn, "answered": len(answers)})
            break
        actions = reply.get("actions") if isinstance(reply.get("actions"), list) else []
        if not actions:
            messages.append({"role": "user", "content": "Reply with a JSON object with \"actions\" or \"answers\"."})
            trace.append({"turn": turn, "error": "no actions"})
            continue
        results = []
        for a in actions[:MAX_ACTIONS]:
            if not isinstance(a, dict):
                continue
            if a.get("tool") == "search":
                results.append(f"## search {json.dumps(str(a.get('query', '')))}\n{tools.search(a.get('query', ''))}")
            elif a.get("tool") == "read":
                results.append(f"## read {a.get('clause_id')}\n{tools.read(a.get('clause_id'))}")
        trace.append({"turn": turn, "actions": actions[:MAX_ACTIONS]})
        left = MAX_TURNS - turn - 1
        messages.append({"role": "user", "content": "\n\n".join(results) + f"\n\n({left} exploration turns left.)"})

    seen_ids = {cid(i) for i in tools.seen}
    extractions = {}
    for f in specs:
        if answers is None:
            extractions[f.name] = Extraction(f.name, error="agent gave no answers")
        else:
            extractions[f.name] = _to_extraction(f.name, answers.get(f.name), seen_ids)
    return AgentRun(extractions, calls, tools.seen, trace)


def seen_candidates(idx: DocIndex, run: AgentRun) -> list[Candidate]:
    """Clauses the agent saw, in the order it saw them: the agent's 'retrieved' set."""
    out = []
    for rank, i in enumerate(run.seen, start=1):
        c = Candidate(i, idx.clauses[i], rrf=1.0 / (60 + rank))
        c.ranks["agent"] = rank
        c.reasons = [f"seen by agent (#{rank})"]
        out.append(c)
    return out

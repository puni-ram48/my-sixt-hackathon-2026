"""Stage 4: evidence-based extraction from retrieved candidate clauses only.

Each field is shown with its definition and the IDs of ITS candidate clauses; the clause texts are
listed once below the fields. The model must answer from those clauses, quote its evidence verbatim,
name the source clause, explain, give a confidence and say whether the field is supported.
Fields are sent in groups (fields_per_call) so each prompt stays focused.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .fields import FieldSpec
from .llm import LLMResult, chat, parse_json_object
from .retrieve import Candidate, DocIndex

log = logging.getLogger("hybrid.extract")

PROMPT = """You extract fields from a contract. You only see clauses that a search system selected as candidates.
Reply with ONE JSON object and nothing else.

Rules:
- Answer each field ONLY from its own candidate clauses (listed with the field). Never use outside knowledge.
- Similar words are not enough: the clause must actually be the kind of clause the field definition describes.
  If no candidate clause states the field, return null. A wrong value is much worse than null.
- "evidence": up to 3 quotations copied character-for-character from the clauses, each with its clause id.
- "value": for a clause-type field, the complete sentence(s) that state it, copied exactly (several joined with " | ");
  for a date, name or title, the value exactly as written.
- "confidence": your probability (0 to 1) that the value is correct and complete.
- "supported": true only if the quoted evidence directly states the field.

Format per field:
{{"<field name>": {{"value": "...", "evidence": [{{"clause_id": "C12", "quote": "..."}}], "explanation": "<at most 20 words>", "confidence": 0.0, "supported": true}}}}
If a field has no answer: {{"<field name>": {{"value": null, "supported": false}}}}

FIELDS:
{fields}

CLAUSES:
{clauses}
"""


@dataclass
class Extraction:
    field: str
    value: Any = None
    evidence: list[dict] = field(default_factory=list)  # [{"clause_id": "C12", "quote": "..."}]
    explanation: str = ""
    confidence: float = 0.0
    supported: bool = False
    error: str | None = None

    def trace(self) -> dict:
        return {"value": self.value, "evidence": self.evidence, "explanation": self.explanation,
                "confidence": self.confidence, "supported": self.supported, "error": self.error}


def cid(index: int) -> str:
    return f"C{index}"


def _clause_block(idx: DocIndex, indices: list[int]) -> str:
    parts = []
    for i in sorted(indices):
        c = idx.clauses[i]
        head = f" (heading: {c.heading})" if c.heading else ""
        parts.append(f"[{cid(i)}]{head}\n{c.text}")
    return "\n\n".join(parts)


def build_prompt(idx: DocIndex, specs: list[FieldSpec], cands: dict[str, list[Candidate]]) -> str:
    lines, used = [], set()
    for f in specs:
        ids = [c.index for c in cands.get(f.name, [])]
        used.update(ids)
        shape = (" (answer: complete sentence(s) for a clause; exact text for a date, name or title)"
                 if f.is_span else "")
        lines.append(f'- "{f.name}": {f.description}{shape}\n  candidates: '
                     + (", ".join(cid(i) for i in ids) if ids else "none"))
    return PROMPT.format(fields="\n".join(lines), clauses=_clause_block(idx, list(used)))


def _to_extraction(name: str, item: Any, valid_ids: set[str]) -> Extraction:
    if not isinstance(item, dict):
        return Extraction(name, value=item if isinstance(item, (str, int, float)) else None)
    try:
        conf = min(max(float(item.get("confidence", 0.0) or 0.0), 0.0), 1.0)
    except (TypeError, ValueError):
        conf = 0.0
    evidence = []
    for ev in item.get("evidence") or []:
        if isinstance(ev, dict) and isinstance(ev.get("quote"), str) and ev["quote"].strip():
            evidence.append({"clause_id": str(ev.get("clause_id", "")), "quote": ev["quote"],
                             "in_candidates": str(ev.get("clause_id", "")) in valid_ids})
    value = item.get("value")
    if isinstance(value, str) and not value.strip():
        value = None
    return Extraction(name, value=value, evidence=evidence, explanation=str(item.get("explanation") or "")[:300],
                      confidence=conf, supported=bool(item.get("supported")) and value is not None)


def extract(idx: DocIndex, specs: list[FieldSpec], cands: dict[str, list[Candidate]],
            fields_per_call: int = 14, pool=None) -> tuple[dict[str, Extraction], list[LLMResult]]:
    """Returns one Extraction per field and the LLM call records (for cost/latency)."""
    groups = [specs[i:i + fields_per_call] for i in range(0, len(specs), fields_per_call)]

    def run(group: list[FieldSpec]):
        prompt = build_prompt(idx, group, cands)
        try:
            res = chat([{"role": "user", "content": prompt}])
        except Exception as e:
            log.warning("%s: extraction call failed: %s", idx.doc_id, e)
            return {f.name: Extraction(f.name, error=str(e)) for f in group}, None
        answer = parse_json_object(res.text)
        if not answer:
            log.warning("%s: malformed extraction reply: %r", idx.doc_id, res.text[:200])
        out = {}
        for f in group:
            valid = {cid(c.index) for c in cands.get(f.name, [])}
            out[f.name] = _to_extraction(f.name, answer.get(f.name), valid)
            if not answer:
                out[f.name].error = "malformed reply"
        return out, res

    results = list(pool.map(run, groups)) if pool is not None else [run(g) for g in groups]
    extractions, calls = {}, []
    for out, res in results:
        extractions.update(out)
        if res is not None:
            calls.append(res)
    return extractions, calls

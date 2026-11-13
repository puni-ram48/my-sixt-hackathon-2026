"""Stage 5: independent semantic verification of each extracted answer.

The verifier is a separate call with a different role (auditor, not extractor). It gets:
the field definition, the extracted value, the quoted evidence, the source clause WITH its
neighbouring clauses and heading, and the other clauses retrieval associated with the field
(to look for contradictions). It does NOT see the extractor's confidence or explanation,
so it cannot simply agree with the extractor's reasoning.

Independence caveat: with the same model this is a second opinion, not an independent one;
the two calls share the model's blind spots. Set LLM_VERIFY_MODEL to use a different model.
An off-the-shelf NLI cross-encoder was considered and not used: generic NLI models are trained on
short everyday sentences and do not know legal clause types, which is exactly the failure we target.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .extract import Extraction, cid
from .fields import FieldSpec
from .llm import LLMResult, chat, parse_json_object, verifier_model
from .retrieve import Candidate, DocIndex

log = logging.getLogger("hybrid.verify")
DECISIONS = ("SUPPORTED", "UNSUPPORTED", "AMBIGUOUS")

PROMPT = """You are an independent auditor checking answers that another system extracted from a contract.
Be strict: an answer is only SUPPORTED if every check passes. Judge only from the text shown.

For each item, check:
1. clause_type_match: is the source clause really the kind of clause the field definition describes
   (not just a clause that shares some words with it)?
2. quote_supports_value: does the quoted text actually state the extracted value?
3. complete: is the value complete and faithful to the source (not a fragment that changes the meaning)?
4. contradiction: do the other listed clauses contradict or materially change the answer?

decision: "SUPPORTED" (all checks pass), "UNSUPPORTED" (wrong clause type, or the quote does not support the value),
or "AMBIGUOUS" (partly supported, incomplete, or contradicted elsewhere).

Reply with ONE JSON object and nothing else:
{{"<item id>": {{"clause_type_match": true, "quote_supports_value": true, "complete": true, "contradiction": false, "decision": "SUPPORTED", "reason": "<at most 20 words>"}}}}

ITEMS:
{items}
"""


@dataclass
class Verification:
    decision: str = "AMBIGUOUS"
    clause_type_match: bool | None = None
    quote_supports_value: bool | None = None
    complete: bool | None = None
    contradiction: bool | None = None
    reason: str = ""
    error: str | None = None

    def trace(self) -> dict:
        return dict(self.__dict__)


def _context(idx: DocIndex, ex: Extraction) -> tuple[str, set[int]]:
    """Source clause(s) of the evidence plus one neighbour on each side."""
    src: list[int] = []
    for ev in ex.evidence:
        src.extend(idx.locate(ev["quote"]))
    if not src and isinstance(ex.value, str):
        src = idx.locate(ex.value.split(" | ")[0])
    shown: list[int] = []
    for i in dict.fromkeys(src):
        for j in range(max(0, i - 1), min(len(idx.clauses), i + 2)):
            if j not in shown:
                shown.append(j)
    blocks = []
    for j in sorted(shown)[:6]:
        c = idx.clauses[j]
        tag = "SOURCE" if j in src else "context"
        head = f" heading: {c.heading};" if c.heading else ""
        blocks.append(f"  [{cid(j)} {tag};{head}] {c.text}")
    return ("\n".join(blocks) if blocks else "  (evidence could not be located in the document)"), set(shown)


def build_prompt(idx: DocIndex, items: list[tuple[str, FieldSpec, Extraction, list[Candidate]]]) -> str:
    out = []
    for item_id, f, ex, cands in items:
        context, shown = _context(idx, ex)
        others = [c for c in cands if c.index not in shown][:2]
        quotes = " | ".join(ev["quote"] for ev in ex.evidence) or "(none given)"
        other_txt = "\n".join(f"  [{cid(c.index)}] {c.clause.text[:700]}" for c in others) or "  (none)"
        out.append(f"### {item_id}\nField: {f.name}\nDefinition: {f.description}\n"
                   f"Extracted value: {ex.value}\nQuoted evidence: {quotes}\n"
                   f"Source clause and surroundings:\n{context}\n"
                   f"Other clauses retrieved for this field:\n{other_txt}")
    return PROMPT.format(items="\n\n".join(out))


def _parse(item) -> Verification:
    if not isinstance(item, dict):
        return Verification(error="missing in reply")
    decision = str(item.get("decision", "")).strip().upper()
    if decision not in DECISIONS:
        decision = "AMBIGUOUS"
    b = lambda k: item.get(k) if isinstance(item.get(k), bool) else None  # noqa: E731
    return Verification(decision, b("clause_type_match"), b("quote_supports_value"), b("complete"),
                        b("contradiction"), str(item.get("reason") or "")[:300])


def verify(idx: DocIndex, specs: dict[str, FieldSpec], extractions: dict[str, Extraction],
           cands: dict[str, list[Candidate]], batch: int = 12, pool=None, model: str | None = None
           ) -> tuple[dict[str, Verification], list[LLMResult]]:
    todo = [(f"F{n}", specs[name], ex, cands.get(name, []))
            for n, (name, ex) in enumerate((k, v) for k, v in extractions.items() if v.value is not None)]
    groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]

    def run(group):
        try:
            res = chat([{"role": "user", "content": build_prompt(idx, group)}], model=model or verifier_model())
        except Exception as e:
            log.warning("%s: verification call failed: %s", idx.doc_id, e)
            return {f.name: Verification(error=str(e)) for _, f, _, _ in group}, None
        answer = parse_json_object(res.text)
        return {f.name: _parse(answer.get(item_id)) for item_id, f, _, _ in group}, res

    results = list(pool.map(run, groups)) if pool is not None else [run(g) for g in groups]
    out, calls = {}, []
    for o, res in results:
        out.update(o)
        if res is not None:
            calls.append(res)
    return out, calls

"""Stage 6: transparent decision policy. Signals are NOT averaged; rules are applied in order.

Actions (what the prediction line gets):
  BOOK     value kept, confidence >= threshold          -> booked automatically
  REVIEW   value kept as a suggestion, confidence capped below threshold -> a human checks it
  REJECT   value removed (null): evidence contradicts it (fabricated / wrong clause / verifier says no)
  ABSTAIN  null: the system found nothing it can support ("I don't know")

Order (first matching rule wins):
  1. extraction failed or returned null/unsupported         -> ABSTAIN
  2. value not found in the document (grounding < 0.9)      -> REJECT  (fabricated)
  3. evidence clause classified as another clause type      -> REJECT  (wrong clause)
  4. verifier UNSUPPORTED                                   -> REJECT
  5. verifier AMBIGUOUS, or contradiction reported          -> REVIEW
  6. classifier unsure about the evidence clause type       -> REVIEW
  7. other clauses look strongly like this field but are not in the answer -> REVIEW (probably incomplete)
  8. long clause answer that is not whole sentences         -> REVIEW  (fragment)
  9. evidence outside the field's retrieved candidates      -> REVIEW
 10. field not reliable enough on calibration data          -> REVIEW  (hybrid/field_policy.py)
 11. extractor confidence below threshold                   -> REVIEW
 12. otherwise                                              -> BOOK
The model's self-reported confidence is only rule 11: necessary, never sufficient.

Why rule 7: on dev-val the main booked error was not the wrong clause but an INCOMPLETE answer: the gold
for a clause field lists every passage that states it (2-5 passages), while the model quoted one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from solution.verify import MIN_TOKENS_FOR_SENTENCE_RULE, Grounder, tokens

from .extract import Extraction, cid
from .fields import FieldSpec
from .retrieve import Candidate, DocIndex
from .verify import Verification


@dataclass
class Policy:
    threshold: float = 0.9      # declared booking threshold (same one the scorer uses)
    min_grounding: float = 0.9  # baseline's grounding gate
    use_classifier: bool = True
    clf_reject: float = 0.02    # P(field type) below this for the evidence clause -> wrong clause
    clf_review: float = 0.10    # below this -> human review
    review_cap: float = 0.5     # confidence written for REVIEW items (always below threshold)
    clf_incomplete: float | None = 0.5  # another clause with P(field) >= this, not covered -> incomplete
    require_in_candidates: bool = True  # False when the extractor saw the whole document (config G)
    field_policy: dict | None = None    # {field: {"bookable": bool, ...}}; None = every field may be booked


@dataclass
class Decision:
    action: str
    value: Any
    confidence: float
    reasons: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)

    def output(self) -> dict:
        return {"value": self.value, "confidence": round(self.confidence, 3)}


def _evidence_clauses(idx: DocIndex, ex: Extraction) -> list[list[int]]:
    """Clause indices for each piece of evidence (falls back to the value's parts)."""
    pieces = [ev["quote"] for ev in ex.evidence] or (
        [p for p in ex.value.split(" | ") if p.strip()] if isinstance(ex.value, str) else [])
    return [idx.locate(p) for p in pieces]


def signals_for(idx: DocIndex, grounder: Grounder | None, f: FieldSpec, ex: Extraction,
                cands: list[Candidate]) -> dict:
    s: dict[str, Any] = {"confidence": ex.confidence, "supported_by_extractor": ex.supported}
    if ex.value is None:
        return s
    s["grounding"] = grounder.score(ex.value) if grounder else None
    if grounder and f.is_span and isinstance(ex.value, str) and len(tokens(ex.value)) >= MIN_TOKENS_FOR_SENTENCE_RULE:
        s["sentence_aligned"] = all(grounder.aligned(p) for p in ex.value.split(" | ") if p.strip())
    located = _evidence_clauses(idx, ex)
    s["evidence_clauses"] = [[cid(i) for i in loc] for loc in located]
    cand_rank = {c.index: r for r, c in enumerate(cands, start=1)}
    flat = [i for loc in located for i in loc]
    s["in_candidates"] = bool(flat) and any(i in cand_rank for i in flat)
    s["best_candidate_rank"] = min((cand_rank[i] for i in flat if i in cand_rank), default=None)
    hit = next((c for c in cands if c.index in flat), None)
    if hit is not None:
        s["bm25_rank"], s["embed_rank"] = hit.ranks.get("bm25"), hit.ranks.get("embed")
        s["embed_cosine"] = round(hit.scores["embed"], 3) if "embed" in hit.scores else None
    p = idx.class_proba(f.name)
    if p is not None and located and all(located):
        # Each piece of evidence must sit in at least one clause of the right type.
        s["clause_type_p"] = round(float(min(max(p[i] for i in loc) for loc in located)), 4)
    if p is not None:
        # Clauses that look like this field but are not part of the answer (completeness check).
        covered = set(flat)
        others = sorted(((float(p[i]), i) for i in range(len(p)) if i not in covered and p[i] >= 0.2), reverse=True)
        s["uncovered_same_type"] = [[cid(i), round(q, 3)] for q, i in others[:5]]
    return s


def decide(f: FieldSpec, ex: Extraction, sig: dict, ver: Verification | None, policy: Policy) -> Decision:
    def out(action: str, reason: str) -> Decision:
        if action == "BOOK":
            return Decision(action, ex.value, ex.confidence, [reason], sig)
        if action == "REVIEW":
            return Decision(action, ex.value, min(ex.confidence, policy.review_cap), [reason], sig)
        return Decision(action, None, 0.0, [reason], sig)

    if ex.error and ex.value is None:
        return out("ABSTAIN", f"extraction failed: {ex.error}")
    if ex.value is None:
        return out("ABSTAIN", "no supporting clause found")
    if not ex.supported:
        return out("ABSTAIN", "extractor did not mark the value as supported")
    g = sig.get("grounding")
    if g is not None and g < policy.min_grounding:
        return out("REJECT", f"value not found in document (grounding {g:.2f})")
    p = sig.get("clause_type_p")
    if policy.use_classifier and p is not None and p < policy.clf_reject:
        return out("REJECT", f"evidence clause is not a '{f.name}' clause (p={p:.3f})")
    if ver is not None:
        if ver.decision == "UNSUPPORTED":
            return out("REJECT", f"verifier: UNSUPPORTED ({ver.reason})")
        if ver.decision == "AMBIGUOUS" or ver.contradiction:
            return out("REVIEW", f"verifier: {ver.decision}{' + contradiction' if ver.contradiction else ''} ({ver.reason})")
    if policy.use_classifier and p is not None and p < policy.clf_review:
        return out("REVIEW", f"clause type uncertain (p={p:.3f})")
    if policy.use_classifier and policy.clf_incomplete is not None:
        missing = [c for c, q in sig.get("uncovered_same_type", []) if q >= policy.clf_incomplete]
        if missing:
            return out("REVIEW", f"answer probably incomplete: {', '.join(missing[:3])} also look like '{f.name}'")
    if sig.get("sentence_aligned") is False:
        return out("REVIEW", "clause answer is not whole sentences")
    if policy.require_in_candidates and not sig.get("in_candidates"):
        return out("REVIEW", "evidence not located in the field's retrieved clauses")
    if policy.field_policy is not None and not policy.field_policy.get(f.name, {}).get("bookable", False):
        fp = policy.field_policy.get(f.name, {})
        return out("REVIEW", f"field not reliable on calibration data ({fp.get('correct', 0)}/{fp.get('n', 0)} correct)")
    if ex.confidence < policy.threshold:
        return out("REVIEW", f"confidence {ex.confidence:.2f} below threshold {policy.threshold}")
    return out("BOOK", "all checks passed" + (" (verifier SUPPORTED)" if ver is not None else ""))

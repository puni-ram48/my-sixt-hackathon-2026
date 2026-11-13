"""Stage 2: hybrid candidate retrieval with an explanation for every candidate.

Three rankers over the clauses of one document:
  bm25        exact terminology (field name + definition as the query)
  embed       semantic similarity between the field query and the clause (bge-small)
  classifier  P(clause is of this field type), from the supervised clause classifier

Rankings are fused with weighted Reciprocal Rank Fusion: score = sum over rankers of w / (RRF_K + rank).
On held-out contracts the supervised classifier is a far better ranker than BM25 or embeddings, so it gets a
larger weight (chosen on calibration contracts with hybrid/eval_retrieval.py, not on the evaluation set).
RRF uses only ranks, so it needs no score calibration between BM25 and cosine similarity,
and one ranker that misses a clause cannot veto the others.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bm25 import BM25, words
from .classify import ClauseClassifier, clause_text, get_classifier
from .clauses import Clause, split_clauses
from .embed import Embedder, get_embedder
from .fields import FieldSpec

RRF_K = 60
DEPTH = 30  # how deep each ranker's list goes into the fusion
ALL_METHODS = ("bm25", "embed", "classifier")


@dataclass
class Candidate:
    index: int
    clause: Clause
    rrf: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def trace(self) -> dict:
        return {"clause_id": self.clause.clause_id, "heading": self.clause.heading, "rrf": round(self.rrf, 4),
                "ranks": self.ranks, "scores": {k: round(v, 4) for k, v in self.scores.items()},
                "reasons": self.reasons, "preview": self.clause.text[:120]}


class DocIndex:
    """All retrieval state for one document. Build once per document."""

    def __init__(self, doc_id: str, text: str, embedder: Embedder | None = None,
                 classifier: ClauseClassifier | None = None, use_models: bool = True):
        self.doc_id, self.text = doc_id, text
        self.clauses = split_clauses(doc_id, text)
        self.bm25 = BM25([clause_text(c) for c in self.clauses])
        self.embedder = embedder if embedder is not None else (get_embedder() if use_models else None)
        self.classifier = classifier if classifier is not None else (get_classifier() if use_models else None)
        self.emb = self.embedder.passages([clause_text(c) for c in self.clauses]) if self.embedder and self.clauses else None
        self.proba = self.classifier.proba(self.emb) if (self.classifier is not None and self.emb is not None) else None

    # -- per-ranker scores ----------------------------------------------------------
    def _bm25(self, f: FieldSpec) -> np.ndarray:
        return np.asarray(self.bm25.scores(f.query()), dtype=np.float32)

    def _embed(self, f: FieldSpec) -> np.ndarray | None:
        if self.emb is None:
            return None
        return self.emb @ self.embedder.queries([f.query()])[0]

    def class_proba(self, field_name: str) -> np.ndarray | None:
        if self.proba is None or not self.classifier.knows(field_name):
            return None
        return self.proba[:, self.classifier.index[field_name]]

    # -- fusion -----------------------------------------------------------------------
    def retrieve(self, f: FieldSpec, methods=ALL_METHODS, k: int = 6, weights: dict | None = None) -> list[Candidate]:
        if not self.clauses:
            return []
        q_words = set(words(f.query()))
        rankers: dict[str, np.ndarray] = {}
        if "bm25" in methods:
            rankers["bm25"] = self._bm25(f)
        if "embed" in methods and (s := self._embed(f)) is not None:
            rankers["embed"] = s
        if "classifier" in methods and (s := self.class_proba(f.name)) is not None:
            rankers["classifier"] = s
        if not rankers:  # nothing available: fall back to BM25 so retrieval never returns nothing
            rankers["bm25"] = self._bm25(f)

        cands: dict[int, Candidate] = {}
        for name, s in rankers.items():
            order = [i for i in np.argsort(-s)[:DEPTH] if not (name == "bm25" and s[i] <= 0)]
            for rank, i in enumerate(order, start=1):
                c = cands.setdefault(int(i), Candidate(int(i), self.clauses[int(i)]))
                c.rrf += (weights or {}).get(name, 1.0) / (RRF_K + rank)
                c.ranks[name], c.scores[name] = rank, float(s[i])
        for c in cands.values():
            c.reasons = self._explain(c, q_words)
        return sorted(cands.values(), key=lambda c: -c.rrf)[:k]

    def _explain(self, c: Candidate, q_words: set[str]) -> list[str]:
        out = []
        if "bm25" in c.ranks:
            hits = sorted(q_words & set(words(clause_text(c.clause))))[:6]
            out.append(f"bm25 rank {c.ranks['bm25']} (score {c.scores['bm25']:.1f}; terms: {', '.join(hits)})")
        if "embed" in c.ranks:
            out.append(f"embedding rank {c.ranks['embed']} (cosine {c.scores['embed']:.2f})")
        if "classifier" in c.ranks:
            out.append(f"classifier rank {c.ranks['classifier']} (p={c.scores['classifier']:.2f})")
        return out

    def neighbours(self, index: int, width: int = 1) -> list[Clause]:
        return self.clauses[max(0, index - width): index + width + 1]

    def locate(self, quote: str) -> list[int]:
        """Indices of clauses that overlap the first occurrence of `quote` in the document text."""
        span = find_span(self.text, quote)
        if span is None:
            return []
        s, e = span
        return [i for i, c in enumerate(self.clauses) if min(e, c.end) > max(s, c.span_start)]


def find_span(text: str, quote: str) -> tuple[int, int] | None:
    """Character span of `quote` in `text`, ignoring differences in whitespace. None if not found."""
    import re

    if not quote or not quote.strip():
        return None
    parts = [re.escape(p) for p in quote.split()]
    m = re.search(r"\s+".join(parts), text)
    if m is None:
        m = re.search(r"\s+".join(parts), text, re.I)
    return (m.start(), m.end()) if m else None

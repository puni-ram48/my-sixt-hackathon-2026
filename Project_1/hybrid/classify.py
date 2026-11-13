"""Stage 3: clause classification - does this clause belong to the requested field type?

Approach: one logistic-regression head per field over clause embeddings (one-vs-rest),
trained on the dev-TRAIN split only. Labels come from CUAD's gold evidence spans: a clause is a
positive example of a field if a gold span for that field covers a substantial part of it.
Clauses that match no field are negatives for every field, which teaches the model what
"the same words, but a different clause" looks like (e.g. a liability sentence that is
really an indemnity clause).

Why not zero-shot / NLI: CPU NLI over ~90 clauses x 41 fields per contract is slow, and generic
NLI models are not trained on legal clause types. We have ~300 labelled contracts, so a small
supervised head on frozen embeddings is the simplest reliable option.

At run time only numpy is needed (weights are stored in an .npz). Fields the classifier has never
seen (e.g. ExtractBench forms) get None, and the pipeline then relies on retrieval alone.

    python -m hybrid.classify          # train on dev-train, report on dev-val, save weights
"""
from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

import numpy as np

from .clauses import Clause

log = logging.getLogger("hybrid.classify")
WEIGHTS = Path(__file__).resolve().parent / "models" / "clause_classifier.npz"


def clause_text(c: Clause) -> str:
    return f"{c.heading}\n{c.text}" if c.heading else c.text


def clause_labels(clauses: list[Clause], gold_evidence: dict) -> list[set[str]]:
    """Fields whose gold spans cover at least half of the clause, or half of the span.

    The clause range includes its heading line (span_start): titles such as a document name live in headings,
    and the decision rules locate evidence over the same range.
    """
    labels: list[set[str]] = [set() for _ in clauses]
    for field, ev in (gold_evidence or {}).items():
        for span in (ev or {}).get("spans", []):
            s, e = span["start"], span["end"]
            for i, c in enumerate(clauses):
                ov = min(e, c.end) - max(s, c.span_start)
                if ov > 0 and ov >= 0.5 * min(e - s, c.end - c.span_start):
                    labels[i].add(field)
    return labels


class ClauseClassifier:
    def __init__(self, fields: list[str], coef: np.ndarray, intercept: np.ndarray):
        self.fields = list(fields)
        self.index = {f: i for i, f in enumerate(self.fields)}
        self.coef, self.intercept = coef, intercept

    def knows(self, field: str) -> bool:
        return field in self.index

    def proba(self, X: np.ndarray) -> np.ndarray:
        """n_clauses x n_fields matrix of P(clause is of field type)."""
        z = X @ self.coef.T + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path: Path = WEIGHTS) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, fields=np.array(self.fields), coef=self.coef, intercept=self.intercept)

    @classmethod
    def load(cls, path: Path = WEIGHTS) -> "ClauseClassifier | None":
        if not path.exists():
            log.warning("no clause classifier at %s (run: python -m hybrid.classify)", path)
            return None
        d = np.load(path, allow_pickle=False)
        return cls([str(f) for f in d["fields"]], d["coef"], d["intercept"])


_loaded: ClauseClassifier | None = None
_tried = False
_lock = threading.Lock()


def get_classifier() -> ClauseClassifier | None:
    global _loaded, _tried
    with _lock:  # worker threads must not see "tried" before loading has finished
        if not _tried:
            _loaded = ClauseClassifier.load()
            _tried = True
    return _loaded


# ---------------------------------------------------------------- training

def _dataset(records, embedder):
    from solution.data import cuad_text
    from .clauses import split_clauses

    X, Y, fields = [], [], sorted({f for r in records for f in r["gold"]})
    col = {f: i for i, f in enumerate(fields)}
    for r in records:
        clauses = split_clauses(r["doc_id"], cuad_text(r["doc_id"]))
        X.append(embedder.passages([clause_text(c) for c in clauses]))
        y = np.zeros((len(clauses), len(fields)), dtype=np.int8)
        for i, labs in enumerate(clause_labels(clauses, r.get("gold_evidence", {}))):
            for f in labs:
                if f in col:
                    y[i, col[f]] = 1
        Y.append(y)
    return np.vstack(X), np.vstack(Y), fields


def train(records, embedder, C: float = 4.0) -> ClauseClassifier:
    from sklearn.linear_model import LogisticRegression

    X, Y, fields = _dataset(records, embedder)
    coef = np.zeros((len(fields), X.shape[1]), dtype=np.float32)
    intercept = np.full(len(fields), -10.0, dtype=np.float32)  # never-seen-positive fields stay ~0
    for j, f in enumerate(fields):
        if Y[:, j].sum() < 3:
            continue
        m = LogisticRegression(C=C, max_iter=2000).fit(X, Y[:, j])
        coef[j], intercept[j] = m.coef_[0], m.intercept_[0]
    log.info("trained on %d clauses, %d fields", len(X), len(fields))
    return ClauseClassifier(fields, coef, intercept)


def report(clf: ClauseClassifier, records, embedder) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    X, Y, fields = _dataset(records, embedder)
    P = clf.proba(X)
    aps, aucs = [], []
    print(f"{'field':40s} {'pos':>4s} {'AP':>5s} {'AUC':>5s}")
    for j, f in enumerate(fields):
        if f not in clf.index or Y[:, j].sum() == 0:
            continue
        p = P[:, clf.index[f]]
        ap, auc = average_precision_score(Y[:, j], p), roc_auc_score(Y[:, j], p)
        aps.append(ap), aucs.append(auc)
        print(f"{f:40s} {int(Y[:, j].sum()):4d} {ap:5.2f} {auc:5.2f}")
    print(f"mean AP {np.mean(aps):.3f}   mean AUC {np.mean(aucs):.3f}   ({len(X)} val clauses)")


def main(argv=None) -> None:
    from .embed import get_embedder
    from .split import dev_records

    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=4.0)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--report-docs", type=int, default=20, help="dev-val contracts to report on")
    ap.add_argument("--max-train-docs", type=int, default=160,
                    help="CPU embedding is ~55 ms per clause; 160 contracts (~14k clauses) is enough for 41 heads")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    emb = get_embedder()
    if emb is None:
        raise SystemExit("fastembed is required to train the classifier")
    clf = train(dev_records("train")[: args.max_train_docs], emb, C=args.C)
    if not args.no_save:
        clf.save()
        print(f"saved {WEIGHTS}", flush=True)
    report(clf, dev_records("val")[: args.report_docs], emb)


if __name__ == "__main__":
    main()

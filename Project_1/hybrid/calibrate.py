"""Choose the clause-type thresholds (Policy.clf_reject / clf_review) without touching dev-val.

Calibration set: dev-TRAIN contracts that were NOT used to fit the classifier (index >= --skip).
For every gold evidence span we take the classifier probability of its best-matching clause
(this is what the decision engine computes for a correct answer). A threshold that rejects
x% of these would wrongly reject x% of correct, located answers.

    python -m hybrid.calibrate [--skip 160] [--limit-docs 60]
"""
from __future__ import annotations

import argparse

import numpy as np

from solution.data import cuad_text

from .fields import field_specs
from .retrieve import DocIndex
from .split import dev_records


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=160)
    ap.add_argument("--limit-docs", type=int, default=60)
    args = ap.parse_args(argv)
    records = dev_records("train")[args.skip: args.skip + args.limit_docs]
    pos, neg = [], []
    for rec in records:
        idx = DocIndex(rec["doc_id"], cuad_text(rec["doc_id"]))
        for f in field_specs(rec):
            p = idx.class_proba(f.name)
            if p is None:
                continue
            spans = (rec.get("gold_evidence", {}).get(f.name) or {}).get("spans", [])
            gold_idx = {i for s in spans for i, c in enumerate(idx.clauses) if min(s["end"], c.end) > max(s["start"], c.start)}
            for s in spans:
                hit = [i for i, c in enumerate(idx.clauses) if min(s["end"], c.end) > max(s["start"], c.start)]
                if hit:
                    pos.append(max(p[i] for i in hit))
            # Plausible wrong-clause candidates: the top BM25 clauses that hold no gold evidence.
            for c in idx.retrieve(f, methods=("bm25", "embed"), k=6):
                if c.index not in gold_idx:
                    neg.append(p[c.index])
    pos, neg = np.array(pos), np.array(neg)
    print(f"{len(records)} contracts: {len(pos)} gold evidence spans, {len(neg)} retrieved non-gold clauses")
    print(f"{'threshold':>9s} {'gold rejected':>14s} {'non-gold caught':>16s}")
    for t in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"{t:9.3f} {np.mean(pos < t):14.1%} {np.mean(neg < t):16.1%}")
    for q in (1, 2, 5, 10):
        print(f"gold p, {q}th percentile: {np.percentile(pos, q):.4f}")


if __name__ == "__main__":
    main()

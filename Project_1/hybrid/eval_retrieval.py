"""Stage 2 evaluation (no API calls): does retrieval put the gold evidence in the top-k clauses?

Uses CUAD's gold evidence character spans. For every field with a gold answer:
  hit@k       at least one top-k clause overlaps a gold span (the LLM *can* answer)
  cover@k     share of gold-evidence characters inside the top-k clauses (it can answer *completely*)
Evaluated on the dev-VAL split (the classifier was fitted on dev-train).

    python -m hybrid.eval_retrieval [--part val] [--limit-docs N]
"""
from __future__ import annotations

import argparse
import collections

import numpy as np

from solution.data import cuad_text

from .fields import field_specs
from .retrieve import DocIndex
from .split import dev_records

KS = (1, 3, 5, 8, 12)
CONFIGS = {
    "bm25": ("bm25",),
    "embed": ("embed",),
    "classifier": ("classifier",),
    "rrf(bm25+embed)": ("bm25", "embed"),
    "rrf(bm25+embed+clf)": ("bm25", "embed", "classifier"),
    "wrrf(clf x2)": ("bm25", "embed", "classifier"),
    "wrrf(clf x3)": ("bm25", "embed", "classifier"),
    "wrrf(clf x5)": ("bm25", "embed", "classifier"),
}
WEIGHTS = {"wrrf(clf x2)": {"classifier": 2.0}, "wrrf(clf x3)": {"classifier": 3.0}, "wrrf(clf x5)": {"classifier": 5.0}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="val", choices=["train", "val"])
    ap.add_argument("--limit-docs", type=int)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args(argv)

    records = dev_records(args.part)[args.offset:][: args.limit_docs]
    hits = {c: collections.Counter() for c in CONFIGS}
    cover = {c: collections.defaultdict(list) for c in CONFIGS}
    per_field = {c: collections.defaultdict(list) for c in ("bm25", "rrf(bm25+embed+clf)")}
    n_clauses, total = [], 0
    for rec in records:
        idx = DocIndex(rec["doc_id"], cuad_text(rec["doc_id"]))
        n_clauses.append(len(idx.clauses))
        for f in field_specs(rec):
            spans = (rec.get("gold_evidence", {}).get(f.name) or {}).get("spans", [])
            if rec["gold"].get(f.name) is None or not spans:
                continue
            total += 1
            gold_chars = sum(s["end"] - s["start"] for s in spans)
            for name, methods in CONFIGS.items():
                ranked = idx.retrieve(f, methods=methods, k=max(KS), weights=WEIGHTS.get(name))
                for k in KS:
                    top = [c.clause for c in ranked[:k]]
                    covered = sum(max(0, min(s["end"], c.end) - max(s["start"], c.span_start)) for s in spans for c in top)
                    hits[name][k] += covered > 0
                    cover[name][k].append(min(1.0, covered / gold_chars))
                if name in per_field:
                    top = [c.clause for c in ranked[:8]]
                    per_field[name][f.name].append(any(min(s["end"], c.end) > max(s["start"], c.span_start) for s in spans for c in top))

    print(f"{args.part}: {len(records)} contracts, median {int(np.median(n_clauses))} clauses each, "
          f"{total} gold-present fields")
    print(f"{'method':22s} " + " ".join(f"hit@{k:<3d}" for k in KS) + "  " + " ".join(f"cov@{k:<3d}" for k in KS))
    for name in CONFIGS:
        print(f"{name:22s} " + " ".join(f"{hits[name][k] / total:6.1%}" for k in KS) + "  "
              + " ".join(f"{np.mean(cover[name][k]):6.1%}" for k in KS))
    print("\nhit@8 per field (worst 10 for BM25), BM25 -> hybrid:")
    rows = sorted(per_field["bm25"], key=lambda f: np.mean(per_field["bm25"][f]))
    for f in rows[:10]:
        print(f"  {f:38s} {np.mean(per_field['bm25'][f]):5.0%} -> {np.mean(per_field['rrf(bm25+embed+clf)'][f]):5.0%}"
              f"  (n={len(per_field['bm25'][f])})")


if __name__ == "__main__":
    main()

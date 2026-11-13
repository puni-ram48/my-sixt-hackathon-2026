"""Deterministic train / validation split of the dev set, by document.

- train: used to fit the clause classifier and to choose retrieval depth.
- val:   held out for comparing baseline and hybrid configurations.
The 10 pilot contracts (data/pilot_cuad_dev.jsonl) were read by hand during prompt work,
so they are always put in train and never used for reporting.
test_public is never used for fitting anything.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

from solution.data import DATA, read_jsonl

VAL_BUCKETS = 5  # 1 in 5 dev documents goes to val


@lru_cache(maxsize=1)
def pilot_ids() -> frozenset[str]:
    path = DATA / "pilot_cuad_dev.jsonl"
    return frozenset(r["doc_id"] for r in read_jsonl(path)) if path.exists() else frozenset()


def split_of(doc_id: str) -> str:
    if doc_id in pilot_ids():
        return "train"
    bucket = int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest(), 16) % VAL_BUCKETS
    return "val" if bucket == 0 else "train"


def dev_records(part: str, source: str | None = "cuad") -> list[dict]:
    recs = read_jsonl(DATA / "unified/dev.jsonl")
    return [r for r in recs if (source is None or r.get("source_dataset") == source) and split_of(r["doc_id"]) == part]

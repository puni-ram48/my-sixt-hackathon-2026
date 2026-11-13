"""Run the hybrid pipeline on a JSONL file and write predictions in the submission format.

    python -m hybrid.run --data data/unified/dev.jsonl --out preds.jsonl [--config D_hybrid_verify]
                         [--trace trace.jsonl] [--limit N] [--resume] [--workers 4]

Predictions: {"doc_id": ..., "fields": {name: {"value": ..., "confidence": ...}}} (same as the baseline).
The trace file holds, per field: retrieved clauses and why, extracted evidence, the verifier's verdict,
and the decision with its reasons.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from solution.data import ROOT, read_jsonl
from solution.run import load_env

from .pipeline import CONFIGS, DEFAULT, process

log = logging.getLogger("hybrid.run")


def main(argv=None) -> int:
    load_env(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=DEFAULT, choices=sorted(CONFIGS))
    ap.add_argument("--trace", help="write per-field decision traces here (JSONL)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM calls")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if not all(os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")):
        print("LLM_BASE_URL, LLM_API_KEY and LLM_MODEL must be set (see .env.example).", file=sys.stderr)
        return 2

    cfg = CONFIGS[args.config]
    records = read_jsonl(args.data)[: args.limit]
    out = Path(args.out)
    done: set[str] = set()
    if args.resume and out.exists():
        done = {json.loads(line)["doc_id"] for line in out.open(encoding="utf-8") if line.strip()}
    mode = "a" if args.resume else "w"
    trace_f = open(args.trace, mode, encoding="utf-8") if args.trace else None
    with out.open(mode, encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as pool:
        for i, rec in enumerate(records):
            if rec["doc_id"] in done:
                continue
            try:
                line, trace = process(rec, cfg, pool)
            except Exception as e:  # last resort: never lose a document, send it to review
                log.exception("%s failed: %s", rec["doc_id"], e)
                props = (rec.get("schema") or {}).get("properties", {})
                line, trace = {"doc_id": rec["doc_id"], "fields": {n: {"value": None} for n in props}}, None
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
            if trace_f and trace:
                trace_f.write(json.dumps(trace, ensure_ascii=False) + "\n")
                trace_f.flush()
            booked = sum(1 for t in (trace or {}).get("fields", {}).values() if t["decision"] == "BOOK")
            print(f"[{i + 1}/{len(records)}] {rec['doc_id']}  booked={booked}", file=sys.stderr)
    if trace_f:
        trace_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

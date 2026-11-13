"""Pipeline: read records -> extract -> ground-check -> write predictions.

    python -m solution.run --data data/unified/dev.jsonl --out preds.jsonl [--limit N] [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

from .data import ROOT, doc_text, read_jsonl
from .extract import extract_fields
from .verify import MIN_TOKENS_FOR_SENTENCE_RULE, Grounder, gate, tokens

Answerer = Callable[[dict], dict]


def spec_is_span(spec: dict) -> bool:
    """True for fields the schema declares as text spans (CUAD clause fields)."""
    return spec.get("x-normalizer") == "span"


def process(rec: dict, answer: Answerer = extract_fields) -> dict:
    """One document in, one prediction line out. Never raises."""
    props = rec["schema"].get("properties", {})
    try:
        raw = answer(rec)
    except Exception as e:
        print(f"{rec['doc_id']}: extraction failed, all fields sent to review ({e})", file=sys.stderr)
        raw = {}

    text = doc_text(rec)
    grounder = Grounder(text) if text else None
    fields = {}
    for name in props:
        item = raw.get(name, {})
        value, confidence = item.get("value"), float(item.get("confidence", 0.0))
        grounding = grounder.score(value) if grounder else None
        if grounder is None and value is not None:
            # No text to check against (scan): the value is unverified, so lower its confidence.
            confidence *= 0.5
        if (grounder is not None and spec_is_span(props[name]) and isinstance(value, str)
                and len(tokens(value)) >= MIN_TOKENS_FOR_SENTENCE_RULE and grounder.aligned(value) is False):
            # Long clause answers should be whole sentences. A long fragment is probably incomplete.
            confidence *= 0.5
        value, confidence = gate(value, confidence, grounding)
        fields[name] = {"value": value, "confidence": round(confidence, 3)}
    return {"doc_id": rec["doc_id"], "fields": fields}


def load_env(path: Path) -> None:
    """Load KEY=VALUE lines from .env. Variables already set in the shell win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main(argv: list[str] | None = None) -> int:
    load_env(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="only the first N documents")
    ap.add_argument("--resume", action="store_true", help="skip documents already in --out")
    args = ap.parse_args(argv)

    if not all(os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")):
        print("LLM_BASE_URL, LLM_API_KEY and LLM_MODEL must be set (see .env.example).", file=sys.stderr)
        return 2

    records = read_jsonl(args.data)[: args.limit]
    out = Path(args.out)
    done: set[str] = set()
    if args.resume and out.exists():
        done = {json.loads(line)["doc_id"] for line in out.open(encoding="utf-8") if line.strip()}
    mode = "a" if args.resume else "w"
    with out.open(mode, encoding="utf-8") as f:
        for i, rec in enumerate(records):
            if rec["doc_id"] in done:
                continue
            f.write(json.dumps(process(rec), ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(records)}] {rec['doc_id']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

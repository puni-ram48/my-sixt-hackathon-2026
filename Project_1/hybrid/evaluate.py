"""Compare the baseline and hybrid configurations on the same contracts.

    python -m hybrid.evaluate --part val --limit-docs 20 [--configs baseline,A_bm25,B_embed,C_hybrid,D_hybrid_verify]
    python -m hybrid.evaluate --part val --limit-docs 20 --replay      # recompute metrics only, no API calls

Default data: the dev-VAL split (hybrid/split.py), never used to fit anything. Use --data for another file
(e.g. data/unified/test_public.jsonl for a final, one-off check).

Per config it writes data/runs/hybrid_eval/<tag>/<config>.preds.jsonl (submission format) and
<config>.trace.jsonl, then report.md and metrics.json with:
  raw extraction accuracy, booked precision / recall / F1, wrong-clause rate, false acceptance rate,
  false rejection rate, booking accuracy, coverage, human-review rate, net saving (scorer), latency, tokens, cost;
  for hybrid configs also a per-rule table: how often each decision rule fired and whether the value it
  acted on was actually right (to see if rules reject correct answers).
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from solution.data import DATA, ROOT, doc_text, read_jsonl
from solution.run import load_env

from .decide import Decision, Policy, decide
from .extract import Extraction
from .pipeline import CONFIGS, Config, process
from .retrieve import find_span
from .split import dev_records
from .verify import Verification

_spec = importlib.util.spec_from_file_location("scorer", ROOT / "5_score.py")
scorer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scorer)

OUT = DATA / "runs" / "hybrid_eval"
ALL = ["baseline", "A_bm25", "B_embed", "C_hybrid", "D_hybrid_verify", "E_agent", "F_agent_verify",
       "G_baseline_gated", "H_baseline_gated_verify"]


# ------------------------------------------------------------------ running

def run_baseline(rec: dict) -> tuple[dict, dict]:
    from solution.extract import extract_fields
    from solution.run import process as baseline_process

    from .llm import chat

    calls, raw = [], {}

    def chat_fn(messages):
        res = chat(messages)
        calls.append(res)
        return res.text

    def answer(r):
        out = extract_fields(r, chat_fn=chat_fn)
        raw.update(out)
        return out

    t0 = time.perf_counter()
    line = baseline_process(rec, answer=answer)
    usage = {"extract": {"calls": len(calls), "prompt_tokens": sum(c.prompt_tokens for c in calls),
                         "completion_tokens": sum(c.completion_tokens for c in calls),
                         "llm_latency_s": round(sum(c.latency_s for c in calls), 2)},
             "wall_s": round(time.perf_counter() - t0, 2)}
    return line, {"doc_id": rec["doc_id"], "config": "baseline", "usage": usage, "raw": raw}


def run_config(name: str, records: list[dict], out_dir: Path, workers: int) -> None:
    preds_path, trace_path = out_dir / f"{name}.preds.jsonl", out_dir / f"{name}.trace.jsonl"
    from .forms import FORMS_CONFIGS, process as forms_process

    if name == "baseline":
        fn = run_baseline
    elif name in FORMS_CONFIGS:
        fn = lambda r: forms_process(r, FORMS_CONFIGS[name])  # noqa: E731
    else:
        fn = lambda r: process(r, CONFIGS[name])  # noqa: E731
    results = {}

    def one(rec):
        try:
            return rec["doc_id"], fn(rec)
        except Exception as e:
            print(f"  {name} {rec['doc_id']} failed: {e}", file=sys.stderr)
            props = rec["schema"].get("properties", {})
            return rec["doc_id"], ({"doc_id": rec["doc_id"], "fields": {n: {"value": None} for n in props}},
                                   {"doc_id": rec["doc_id"], "config": name, "error": str(e), "usage": {}})

    with ThreadPoolExecutor(workers) as pool:
        for n, (doc_id, res) in enumerate(pool.map(one, records), start=1):
            results[doc_id] = res
            print(f"  {name} [{n}/{len(records)}] {doc_id[:70]}", file=sys.stderr)
    with preds_path.open("w", encoding="utf-8") as fp, trace_path.open("w", encoding="utf-8") as ft:
        for rec in records:
            line, trace = results[rec["doc_id"]]
            fp.write(json.dumps(line, ensure_ascii=False) + "\n")
            ft.write(json.dumps(trace, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ metrics

def classify_error(rec: dict, field: str, value) -> str:
    """correct / hallucinated / wrong_clause / boundary for a non-null predicted value."""
    gold = rec["gold"].get(field)
    kind = rec["schema"]["properties"].get(field, {}).get("x-normalizer", "string")
    if not scorer.is_present(gold):
        return "hallucinated"
    if scorer.values_match(value, gold, kind):
        return "correct"
    spans = (rec.get("gold_evidence", {}).get(field) or {}).get("spans", [])
    text = doc_text(rec) if rec.get("source_dataset") == "cuad" else None
    if text and spans and isinstance(value, str):
        located = [find_span(text, p) for p in value.split(" | ") if p.strip()]
        located = [s for s in located if s]
        if located:
            overlaps = any(min(e, g["end"]) > max(s, g["start"]) for s, e in located for g in spans)
            return "boundary" if overlaps else "wrong_clause"
    # Fall back to word overlap when the value cannot be located.
    p, g = scorer.norm_string(value).split(), scorer.norm_string(gold).split()
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    return "wrong_clause" if (common / max(len(p), 1) < 0.5 and common / max(len(g), 1) < 0.5) else "boundary"


def raw_values(trace: dict) -> dict:
    """The value the extractor proposed, before any gate."""
    if trace.get("config") == "baseline":
        return {k: (v or {}).get("value") for k, v in trace.get("raw", {}).items()}
    return {k: v["extraction"]["value"] for k, v in trace.get("fields", {}).items()}


def metrics(records, preds, traces, threshold: float, price_in: float, price_out: float) -> dict:
    gold_by = {r["doc_id"]: r for r in records}
    c = collections.Counter()
    for rec in records:
        doc_id = rec["doc_id"]
        fields = preds.get(doc_id, {})
        raw = raw_values(traces.get(doc_id, {}))
        for name, gold in rec["gold"].items():
            c["fields"] += 1
            present = scorer.is_present(gold)
            c["gold_present"] += present
            p = fields.get(name) or {}
            v = p.get("value")
            booked = scorer.is_present(v) and float(p.get("confidence", 1.0)) >= threshold
            rv = raw.get(name)
            raw_cat = classify_error(gold_by[doc_id], name, rv) if scorer.is_present(rv) else (
                "missed" if present else "true_null")
            c[f"raw_{raw_cat}"] += 1
            raw_ok = raw_cat in ("correct", "true_null")
            c["raw_accurate"] += raw_ok
            if scorer.is_present(rv):
                c["raw_correct" if raw_cat == "correct" else "raw_incorrect"] += 1
                if booked:
                    c["booked_of_raw_correct" if raw_cat == "correct" else "booked_of_raw_incorrect"] += 1
            if booked:
                cat = classify_error(gold_by[doc_id], name, v)
                c["booked"] += 1
                c[f"booked_{cat}"] += 1
    n, booked = c["fields"], c["booked"]
    tp = c["booked_correct"]
    prec = tp / booked if booked else 1.0
    rec_ = tp / c["gold_present"] if c["gold_present"] else 0.0
    m = {
        "docs": len(records), "fields": n, "gold_present": c["gold_present"],
        "raw_accuracy": c["raw_accurate"] / n,
        "raw_wrong_clause": c["raw_wrong_clause"], "raw_hallucinated": c["raw_hallucinated"],
        "raw_boundary": c["raw_boundary"], "raw_correct": c["raw_correct"],
        "booked": booked, "booked_correct": tp,
        "booked_wrong_clause": c["booked_wrong_clause"], "booked_boundary": c["booked_boundary"],
        "booked_hallucinated": c["booked_hallucinated"],
        "precision": prec, "recall": rec_, "f1": (2 * prec * rec_ / (prec + rec_)) if prec + rec_ else 0.0,
        "booking_accuracy": prec,
        "wrong_clause_rate_booked": c["booked_wrong_clause"] / booked if booked else 0.0,
        "false_acceptance_rate": c["booked_of_raw_incorrect"] / c["raw_incorrect"] if c["raw_incorrect"] else 0.0,
        "false_rejection_rate": 1 - c["booked_of_raw_correct"] / c["raw_correct"] if c["raw_correct"] else 0.0,
        "coverage": booked / n, "human_review_rate": 1 - booked / n,
    }
    rows = scorer.collect(records, preds)
    at = scorer.at_threshold(rows, threshold)
    m["net_saving_pct"] = scorer.net_saving(at, 1.0, 100.0)["saving_pct"]
    usage = [t.get("usage", {}) for t in traces.values()]
    tok_in = sum(u.get(s, {}).get("prompt_tokens", 0) for u in usage for s in ("extract", "verify"))
    tok_out = sum(u.get(s, {}).get("completion_tokens", 0) for u in usage for s in ("extract", "verify"))
    llm_s = sum(u.get(s, {}).get("llm_latency_s", 0) for u in usage for s in ("extract", "verify"))
    calls = sum(u.get(s, {}).get("calls", 0) for u in usage for s in ("extract", "verify"))
    d = max(len(records), 1)
    m.update({"llm_calls_per_doc": calls / d, "tokens_in_per_doc": tok_in / d, "tokens_out_per_doc": tok_out / d,
              "llm_latency_s_per_doc": llm_s / d,
              "cost_usd_per_doc": (tok_in * price_in + tok_out * price_out) / 1e6 / d})
    return m


def rule_table(records, traces) -> list[dict]:
    """For each decision reason: how often it fired and what the extracted value really was."""
    gold_by = {r["doc_id"]: r for r in records}
    table = collections.defaultdict(collections.Counter)
    for doc_id, t in traces.items():
        for name, ft in t.get("fields", {}).items():
            rule = ft["decision"] + ": " + _rule_key(ft["reasons"][0])
            v = ft["extraction"]["value"]
            cat = classify_error(gold_by[doc_id], name, v) if scorer.is_present(v) else (
                "missed" if scorer.is_present(gold_by[doc_id]["gold"].get(name)) else "true_null")
            table[rule][cat] += 1
    return [{"rule": k, **v} for k, v in sorted(table.items(), key=lambda kv: -sum(kv[1].values()))]


def _rule_key(reason: str) -> str:
    for key in ("extraction failed", "no supporting clause", "did not mark", "not found in document",
                "is not a", "verifier: UNSUPPORTED", "verifier: AMBIGUOUS", "contradiction", "clause type uncertain",
                "not whole sentences", "not located", "probably incomplete", "not reliable on calibration",
                "below threshold", "all checks passed"):
        if key in reason:
            return key
    return reason[:40]


def replay(trace: dict, cfg: Config, policy: Policy) -> dict:
    """Re-decide every field from saved signals with a different policy (no API calls)."""
    from .fields import FieldSpec

    fields = {}
    for name, ft in trace.get("fields", {}).items():
        e = ft["extraction"]
        ex = Extraction(name, e["value"], e["evidence"], e["explanation"], e["confidence"], e["supported"], e["error"])
        ver = Verification(**ft["verification"]) if ft.get("verification") else None
        spec = FieldSpec(name, "", "span", "string")
        d: Decision = decide(spec, ex, ft["signals"], ver, policy)
        fields[name] = d.output()
    return {"doc_id": trace["doc_id"], "fields": fields}


# ------------------------------------------------------------------ main

def fmt_table(results: dict) -> str:
    cols = [("precision", "booking precision", "{:.1%}"), ("recall", "recall", "{:.1%}"), ("f1", "F1", "{:.3f}"),
            ("raw_accuracy", "raw field accuracy", "{:.1%}"), ("coverage", "coverage (booked)", "{:.1%}"),
            ("human_review_rate", "human review", "{:.1%}"),
            ("booked_wrong_clause", "booked wrong-clause", "{}"), ("booked_hallucinated", "booked hallucinated", "{}"),
            ("booked_boundary", "booked boundary err", "{}"),
            ("wrong_clause_rate_booked", "wrong-clause rate (booked)", "{:.2%}"),
            ("false_acceptance_rate", "false acceptance", "{:.1%}"), ("false_rejection_rate", "false rejection", "{:.1%}"),
            ("raw_wrong_clause", "raw wrong-clause answers", "{}"), ("raw_hallucinated", "raw invented answers", "{}"),
            ("net_saving_pct", "net saving (1/100)", "{:.1%}"),
            ("llm_calls_per_doc", "LLM calls / doc", "{:.1f}"), ("tokens_in_per_doc", "input tokens / doc", "{:,.0f}"),
            ("tokens_out_per_doc", "output tokens / doc", "{:,.0f}"),
            ("llm_latency_s_per_doc", "LLM seconds / doc", "{:.1f}"), ("cost_usd_per_doc", "USD / doc (assumed price)", "${:.3f}")]
    names = list(results)
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for key, label, f in cols:
        lines.append(f"| {label} | " + " | ".join(f.format(results[n][key]) for n in names) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    load_env(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="JSONL with gold (default: dev-VAL CUAD split)")
    ap.add_argument("--part", default="val", choices=["val", "train"])
    ap.add_argument("--source", default="cuad", choices=["cuad", "extractbench"])
    ap.add_argument("--limit-docs", type=int, default=20)
    ap.add_argument("--offset", type=int, default=0, help="skip the first N documents of the split")
    ap.add_argument("--configs", default=",".join(ALL))
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--replay", action="store_true", help="only recompute metrics from saved files")
    ap.add_argument("--price-in", type=float, default=3.0, help="USD per 1M input tokens (assumed)")
    ap.add_argument("--price-out", type=float, default=15.0, help="USD per 1M output tokens (assumed)")
    args = ap.parse_args(argv)

    if args.data:
        records = [r for r in read_jsonl(args.data) if r.get("source_dataset") == args.source][: args.limit_docs]
        tag = Path(args.data).stem + f"_{len(records)}"
    else:
        records = dev_records(args.part, source=args.source)[args.offset: args.offset + args.limit_docs]
        tag = (f"dev_{args.part}_{len(records)}" + (f"_from{args.offset}" if args.offset else "")
               + ("" if args.source == "cuad" else f"_{args.source}"))
    out_dir = OUT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    print(f"{len(records)} contracts -> {out_dir}", file=sys.stderr)

    results, rules = {}, {}
    for name in names:
        if not args.replay:
            run_config(name, records, out_dir, args.workers)
        preds = {r["doc_id"]: r["fields"] for r in read_jsonl(out_dir / f"{name}.preds.jsonl")}
        traces = {t["doc_id"]: t for t in read_jsonl(out_dir / f"{name}.trace.jsonl")}
        results[name] = metrics(records, preds, traces, args.threshold, args.price_in, args.price_out)
        if name != "baseline":
            rules[name] = rule_table(records, traces)

    from .forms import FORMS_CONFIGS, replay as forms_replay

    # Threshold sweep from saved signals: how the booking policy trades coverage for precision.
    sweep = {}
    for name in names:
        traces = {t["doc_id"]: t for t in read_jsonl(out_dir / f"{name}.trace.jsonl")}
        for t in (0.8, 0.85, 0.9, 0.95):
            if name == "baseline":
                preds = {r["doc_id"]: r["fields"] for r in read_jsonl(out_dir / f"{name}.preds.jsonl")}
            elif name in FORMS_CONFIGS:
                by = {r["doc_id"]: r for r in records}
                preds = {d: forms_replay(tr, FORMS_CONFIGS[name], by[d], threshold=t)["fields"]
                         for d, tr in traces.items() if tr.get("fields")}
            else:
                cfg = CONFIGS[name]
                pol = Policy(**{**cfg.policy.__dict__, "threshold": t})
                if cfg.field_policy:
                    from .field_policy import load
                    pol.field_policy = load(cfg.field_policy) or {}
                preds = {d: replay(tr, cfg, pol)["fields"] for d, tr in traces.items()}
            m = metrics(records, preds, traces, t, args.price_in, args.price_out)
            sweep.setdefault(name, {})[t] = {k: m[k] for k in ("precision", "coverage", "booked", "booked_wrong_clause",
                                                               "booked_hallucinated", "net_saving_pct")}

    report = [f"# Baseline vs hybrid on {len(records)} contracts ({tag}), threshold {args.threshold}\n",
              fmt_table(results), "\n## Threshold sweep (replayed, no new API calls)\n",
              "| config | t | booked | precision | coverage | booked wrong-clause | booked hallucinated | net saving |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, by_t in sweep.items():
        for t, m in by_t.items():
            report.append(f"| {name} | {t} | {m['booked']} | {m['precision']:.1%} | {m['coverage']:.1%} | "
                          f"{m['booked_wrong_clause']} | {m['booked_hallucinated']} | {m['net_saving_pct']:.1%} |")
    for name, table in rules.items():
        report += [f"\n## Decision rules fired: {name}\n",
                   "| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |",
                   "|---|---:|---:|---:|---:|---:|---:|"]
        for row in table:
            report.append(f"| {row['rule']} | " + " | ".join(str(row.get(k, 0)) for k in
                          ("correct", "boundary", "wrong_clause", "hallucinated", "missed", "true_null")) + " |")
    text = "\n".join(report)
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps({"results": results, "sweep": sweep, "rules": rules}, indent=2),
                                          encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

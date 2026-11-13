"""Per-field reliability policy: which fields may be booked automatically at all.

Business rule: a booked error costs 100, a review costs 1, so automating a field only pays off if its
error rate among booked values is below ~1%. Self-reported confidence does not tell us that; history does.

Fitting (calibration data only, never dev-val or test):
  1. run a config on calibration contracts (dev-train contracts the classifier was not fitted on);
  2. replay its decisions WITHOUT the field policy -> the values every other rule would book;
  3. per field: n booked, n correct (scorer rules). A field is bookable if n >= MIN_N and
     errors / n <= MAX_ERROR_RATE. Everything else is sent to review (including unseen fields).
No field names are written in code; the table is learned and saved to hybrid/models/field_policy_<config>.json.

    python -m hybrid.field_policy --run-dir data/runs/hybrid_eval/dev_train_60_from160 --config G_baseline_gated
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

MODELS = Path(__file__).resolve().parent / "models"
MIN_N = 8
MAX_ERROR_RATE = 0.01


def path_for(config: str) -> Path:
    return MODELS / f"field_policy_{config}.json"


def load(config: str) -> dict | None:
    p = path_for(config)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fit(records: list[dict], traces: dict, cfg, min_n: int = MIN_N, max_error_rate: float = MAX_ERROR_RATE) -> dict:
    from .decide import Policy
    from .evaluate import classify_error, replay

    pol = Policy(**{**cfg.policy.__dict__, "field_policy": None})
    gold_by = {r["doc_id"]: r for r in records}
    stats = collections.defaultdict(collections.Counter)
    for doc_id, t in traces.items():
        if doc_id not in gold_by:
            continue
        for name, f in replay(t, cfg, pol)["fields"].items():
            if f.get("value") is not None and f.get("confidence", 0) >= pol.threshold:
                stats[name]["n"] += 1
                stats[name]["correct"] += classify_error(gold_by[doc_id], name, f["value"]) == "correct"
    fields = sorted({n for r in records for n in r["gold"]})
    table = {}
    for name in fields:
        n, c = stats[name]["n"], stats[name]["correct"]
        table[name] = {"n": n, "correct": c,
                       "bookable": n >= min_n and (n - c) / n <= max_error_rate}
    return table


def main(argv=None) -> None:
    from solution.data import read_jsonl

    from .pipeline import CONFIGS
    from .split import dev_records

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default="G_baseline_gated")
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--max-error-rate", type=float, default=MAX_ERROR_RATE)
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    traces = {t["doc_id"]: t for t in read_jsonl(run / f"{args.config}.trace.jsonl")}
    records = [r for r in dev_records("train") if r["doc_id"] in traces]  # calibration data must be dev-train
    if len(records) != len(traces):
        raise SystemExit("calibration traces must come from dev-train contracts only")
    table = fit(records, traces, CONFIGS[args.config], args.min_n, args.max_error_rate)
    MODELS.mkdir(parents=True, exist_ok=True)
    path_for(args.config).write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(f"{len(records)} calibration contracts -> {path_for(args.config)}")
    for name, s in sorted(table.items(), key=lambda kv: -kv[1]["n"]):
        if s["n"]:
            print(f"  {name:38s} {s['correct']:3d}/{s['n']:<3d} {'BOOKABLE' if s['bookable'] else 'review'}")
    print(f"bookable fields: {sum(s['bookable'] for s in table.values())} of {len(table)}")


if __name__ == "__main__":
    main()

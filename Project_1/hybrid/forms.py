"""Evidence-gated extraction for forms and scans (ExtractBench).

Why not the contract pipeline: these are 1-10 page forms (median 1) with printed labels and typed or handwritten
values, checkboxes and tables. The problem is "which value belongs to which label", not "which clause".
55% of dev PDFs have a text layer (words + coordinates); 45% are scans with none.

Pipeline per document:
  1. Render pages to images (pypdfium2) and read the text layer with word boxes (pdfplumber), if any.
  2. Reader A (LLM_MODEL) reads the page images (+ text layer as a hint) and returns, per field,
     value, confidence, page and the printed label next to the value.
  3. Reader B: an independent read by a model from ANOTHER family (FORMS_SECOND_MODEL, default gpt-5.5).
     For scans this is the only check available (no text to ground against).
  4. Gates (transparent rules, first match wins):
       both readers null                        -> ABSTAIN
       one reader null, or the readers disagree -> REVIEW
       text layer present, value not on the page (grounding < 0.9)            -> REJECT
       text layer present, value found but not next to its printed label      -> REVIEW (wrong association)
       field not reliable on calibration data (learned per field name)        -> REVIEW
       min(confidence A, confidence B) below threshold                        -> REVIEW
       otherwise                                                              -> BOOK
Heavy OCR/layout stacks (PaddleOCR, LayoutParser) were not used: slow to install and run on CPU and weak on
historical handwriting; two independent vision reads give a comparable "can we trust this?" signal.

    python -m hybrid.forms fit --run-dir <dir> --config forms_dual     # learn the field policy (dev-train only)
"""
from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from solution.data import source_path
from solution.verify import Grounder

from .fields import FieldSpec, field_specs
from .llm import chat, parse_json_object

log = logging.getLogger("hybrid.forms")
MODELS = Path(__file__).resolve().parent / "models"
MAX_PAGES = 8
DPI = 150
TEXT_HINT_CHARS = 20_000
_ALNUM = re.compile(r"[a-z0-9]+")
_PDF_LOCK = threading.Lock()  # pdfium is not thread-safe: concurrent renders crash with access violations

PROMPT = """You read a form or document from page images. Fill in the fields below.
Reply with ONE JSON object and nothing else:
{{"<field name>": {{"value": <value or null>, "confidence": <0 to 1>, "page": <page number>, "label": "<printed label next to the value, or null>"}}}}

Rules:
- Copy values exactly as printed or written. Do not guess, calculate or reformat.
- If a field is blank, missing, or you cannot read it, the value is null. A wrong value is much worse than null.
- Checkboxes / yes-no fields: "true" if the box is visibly marked, "false" if it is visibly empty; null if unclear.
- Fields of type object/array: give the JSON value.
- confidence = your probability that the value is exactly right.
{text_hint}
Fields:
{fields}
"""


@dataclass
class FormsConfig:
    name: str
    second_reader: str | None = field(default_factory=lambda: os.environ.get("FORMS_SECOND_MODEL", "gpt-5.5"))
    threshold: float = 0.9
    field_policy: str | None = None  # name of a fitted policy file (models/forms_policy_<name>.json)


FORMS_CONFIGS = {
    "forms_single": FormsConfig("forms_single", second_reader=None),
    "forms_dual": FormsConfig("forms_dual"),
    "forms_dual_policy": FormsConfig("forms_dual_policy", field_policy="forms_dual"),
    # Submitted: same as forms_dual_policy at the declared threshold 0.95. On 32 held-out dev-val forms it booked
    # 81 values, all correct (net saving +5.2%); at 0.9 101/101 (+6.5%). One threshold for both datasets.
    "forms_final": FormsConfig("forms_final", field_policy="forms_dual", threshold=0.95),
}


# ---------------------------------------------------------------- document access

def page_images(path: Path) -> list[str]:
    import pypdfium2 as pdfium

    out = []
    with _PDF_LOCK:
        pdf = pdfium.PdfDocument(str(path))
        try:
            for i in range(min(len(pdf), MAX_PAGES)):
                img = pdf[i].render(scale=DPI / 72).to_pil().convert("L")
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=80)
                out.append(base64.b64encode(buf.getvalue()).decode())
        finally:
            pdf.close()
    return out


@dataclass
class Word:
    text: str
    page: int
    x0: float
    x1: float
    top: float
    bottom: float


def text_layer(path: Path) -> tuple[str, list[Word]]:
    """Page text and words with boxes (normalised 0..1). Empty for scans."""
    import pdfplumber

    texts, words = [], []
    try:
        with _PDF_LOCK, pdfplumber.open(str(path)) as pdf:  # keep PDF parsing serial (pdfium underneath)
            for p, page in enumerate(pdf.pages[:MAX_PAGES], start=1):
                texts.append(page.extract_text() or "")
                w, h = float(page.width), float(page.height)
                for wd in page.extract_words():
                    for tok in _ALNUM.findall(wd["text"].lower()):
                        words.append(Word(tok, p, wd["x0"] / w, wd["x1"] / w, wd["top"] / h, wd["bottom"] / h))
    except Exception as e:
        log.warning("%s: no text layer (%s)", path.name, e)
    text = "\n".join(texts)
    return (text if len(words) > 20 else ""), (words if len(words) > 20 else [])


# ---------------------------------------------------------------- reading

def _norm_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str) and not v.strip():
        return None
    return v


def read(specs: list[FieldSpec], images: list[str], text: str, model: str | None) -> tuple[dict, object]:
    hint = (f"\nThe PDF text layer (may be incomplete or out of order) is:\n---\n{text[:TEXT_HINT_CHARS]}\n---\n"
            if text else "")
    fields = "\n".join(f"- {f.name} ({f.json_type})" + (f": {f.description}" if f.description else "") for f in specs)
    content = [{"type": "text", "text": PROMPT.format(text_hint=hint, fields=fields)}]
    content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}} for b in images]
    res = chat([{"role": "user", "content": content}], model=model)
    answer = parse_json_object(res.text)
    out = {}
    for f in specs:
        item = answer.get(f.name)
        if not isinstance(item, dict):
            item = {"value": item}
        try:
            conf = min(max(float(item.get("confidence", 0.0) or 0.0), 0.0), 1.0)
        except (TypeError, ValueError):
            conf = 0.0
        v = item.get("value")
        if f.kind == "json" and v is not None and not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)
        out[f.name] = {"value": _norm_value(v), "confidence": conf, "page": item.get("page"),
                       "label": item.get("label") if isinstance(item.get("label"), str) else None}
    return out, res


# ---------------------------------------------------------------- checks

def _find(words: list[Word], text: str, page=None) -> list[list[Word]]:
    toks = _ALNUM.findall(str(text).lower())
    if not toks:
        return []
    hits = []
    for i, w in enumerate(words):
        if w.text != toks[0] or (page and w.page != page):
            continue
        seq = words[i:i + len(toks)]
        if len(seq) == len(toks) and all(a.text == b for a, b in zip(seq, toks)) and len({s.page for s in seq}) == 1:
            hits.append(seq)
    return hits


def spatial(words: list[Word], value, label: str | None) -> str:
    """'near' if the value sits right of or just below its printed label, 'far' if found but elsewhere,
    'unknown' if either cannot be located (scan, checkbox, reformatted value)."""
    if not words or not label or not isinstance(value, str) or value.lower() in ("true", "false"):
        return "unknown"
    vals, labs = _find(words, value), _find(words, label)
    if not vals or not labs:
        return "unknown"
    for v in vals:
        vx0, vtop, vbot = min(w.x0 for w in v), min(w.top for w in v), max(w.bottom for w in v)
        for lab in labs:
            if lab[0].page != v[0].page:
                continue
            lx0, lx1 = min(w.x0 for w in lab), max(w.x1 for w in lab)
            ltop, lbot = min(w.top for w in lab), max(w.bottom for w in lab)
            line = max(lbot - ltop, 0.008)
            same_line = abs((vtop + vbot) / 2 - (ltop + lbot) / 2) < 1.2 * line and vx0 >= lx0 - 0.01
            below = 0 <= vtop - lbot < 4 * line and abs(vx0 - lx0) < 0.25
            inside = vx0 >= lx0 - 0.01 and vtop >= ltop - line and vtop - lbot < 1.5 * line and vx0 - lx1 < 0.4
            if same_line or below or inside:
                return "near"
    return "far"


def agree(a, b, kind: str) -> bool:
    from .evaluate import scorer

    if a is None or b is None:
        return False
    return scorer.values_match(a, b, kind) or scorer.normalize(a, "id") == scorer.normalize(b, "id")


def decide(f: FieldSpec, ra: dict, rb: dict | None, sig: dict, cfg: FormsConfig, policy: dict | None) -> tuple[str, dict, str]:
    va, ca = ra.get("value"), ra.get("confidence", 0.0)
    out_book = {"value": va, "confidence": round(min(ca, rb["confidence"]) if rb else ca, 3)}
    review = {"value": va if va is not None else (rb or {}).get("value"), "confidence": round(min(ca, 0.5), 3)}
    null = {"value": None, "confidence": 0.0}
    if rb is not None:
        vb = rb.get("value")
        if va is None and vb is None:
            return "ABSTAIN", null, "both readers: not present"
        if va is None or vb is None:
            return "REVIEW", review, "only one reader found a value"
        if not agree(va, vb, f.kind):
            return "REVIEW", review, f"readers disagree ({str(va)[:30]!r} vs {str(vb)[:30]!r})"
    elif va is None:
        return "ABSTAIN", null, "not present"
    g = sig.get("grounding")
    if g is not None and g < 0.9:
        return "REJECT", null, f"value not found in the text layer (grounding {g:.2f})"
    if sig.get("spatial") == "far":
        return "REVIEW", review, f"value found, but not next to its label {ra.get('label')!r}"
    if policy is not None and not policy.get(f.name, {}).get("bookable", False):
        p = policy.get(f.name, {})
        return "REVIEW", review, f"field not reliable on calibration data ({p.get('correct', 0)}/{p.get('n', 0)} correct)"
    if out_book["confidence"] < cfg.threshold:
        return "REVIEW", review, f"confidence {out_book['confidence']:.2f} below threshold {cfg.threshold}"
    return "BOOK", out_book, "all checks passed" + (" (two independent readers agree)" if rb else "")


def load_policy(name: str | None) -> dict | None:
    if not name:
        return None
    p = MODELS / f"forms_policy_{name}.json"
    if not p.exists():
        log.warning("forms policy %s not fitted: every field goes to review", name)
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- per document

def process(rec: dict, cfg: FormsConfig) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    specs = field_specs(rec)
    path = source_path(rec)
    null_line = {"doc_id": rec["doc_id"], "fields": {f.name: {"value": None, "confidence": 0.0} for f in specs}}
    if not path.exists():
        return null_line, {"doc_id": rec["doc_id"], "config": cfg.name, "error": "file missing", "usage": {}, "fields": {}}
    images = page_images(path)
    text, words = text_layer(path)
    calls, reads = [], {}
    readers = [("A", None)] + ([("B", cfg.second_reader)] if cfg.second_reader else [])
    for tag, model in readers:
        try:
            reads[tag], res = read(specs, images, text, model)
            calls.append(res)
        except Exception as e:
            log.warning("%s: reader %s failed: %s", rec["doc_id"], tag, e)
            return null_line, {"doc_id": rec["doc_id"], "config": cfg.name, "error": str(e), "usage": {}, "fields": {}}
    grounder = Grounder(text) if text else None
    policy = load_policy(cfg.field_policy)
    fields, trace = {}, {}
    for f in specs:
        ra, rb = reads["A"][f.name], reads.get("B", {}).get(f.name) if "B" in reads else None
        sig = {"text_layer": bool(text)}
        if ra["value"] is not None and grounder is not None and f.kind != "json":
            sig["grounding"] = grounder.score(ra["value"])
        sig["spatial"] = spatial(words, ra["value"], ra.get("label"))
        action, out, reason = decide(f, ra, rb, sig, cfg, policy)
        fields[f.name] = out
        trace[f.name] = {"decision": action, "reasons": [reason], "signals": sig,
                         "extraction": {"value": ra["value"], "confidence": ra["confidence"], "page": ra.get("page"),
                                        "label": ra.get("label")},
                         "second_read": rb, "verification": None}
    usage = {"extract": {"calls": len(calls), "prompt_tokens": sum(c.prompt_tokens for c in calls),
                         "completion_tokens": sum(c.completion_tokens for c in calls),
                         "llm_latency_s": round(sum(c.latency_s for c in calls), 2)},
             "wall_s": round(time.perf_counter() - t0, 2), "pages": len(images), "text_layer": bool(text)}
    return {"doc_id": rec["doc_id"], "fields": fields}, {"doc_id": rec["doc_id"], "config": cfg.name, "usage": usage,
                                                         "fields": trace}


def replay(trace: dict, cfg: FormsConfig, rec: dict, threshold: float | None = None, policy=False) -> dict:
    """Re-decide from a saved trace (no API calls)."""
    from dataclasses import replace

    c = replace(cfg, threshold=threshold if threshold is not None else cfg.threshold)
    pol = load_policy(cfg.field_policy) if policy is False else policy
    specs = {f.name: f for f in field_specs(rec)}
    fields = {}
    for name, t in trace.get("fields", {}).items():
        ra = {**t["extraction"]}
        _, out, _ = decide(specs[name], ra, t.get("second_read"), t["signals"], c, pol)
        fields[name] = out
    return {"doc_id": trace["doc_id"], "fields": fields}


def fit_policy(records, traces, cfg: FormsConfig, min_n: int = 5, max_error_rate: float = 0.01) -> dict:
    from .evaluate import classify_error

    stats = collections.defaultdict(collections.Counter)
    by = {r["doc_id"]: r for r in records}
    for d, t in traces.items():
        for name, f in replay(t, cfg, by[d], policy=None)["fields"].items():
            if f.get("value") is not None and f.get("confidence", 0) >= cfg.threshold:
                stats[name]["n"] += 1
                stats[name]["correct"] += classify_error(by[d], name, f["value"]) == "correct"
    return {n: {"n": s["n"], "correct": s["correct"],
                "bookable": s["n"] >= min_n and (s["n"] - s["correct"]) / s["n"] <= max_error_rate}
            for n, s in stats.items()}


def main(argv=None) -> None:
    from solution.data import read_jsonl

    from .split import dev_records

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", default="forms_dual")
    ap.add_argument("--min-n", type=int, default=5)
    args = ap.parse_args(argv)
    traces = {t["doc_id"]: t for t in read_jsonl(Path(args.run_dir) / f"{args.config}.trace.jsonl")}
    records = [r for r in dev_records("train", source="extractbench") if r["doc_id"] in traces]
    if len(records) != len(traces):
        raise SystemExit("calibration traces must come from dev-train documents only")
    table = fit_policy(records, traces, FORMS_CONFIGS[args.config], args.min_n)
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / f"forms_policy_{args.config}.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    ok = {n: s for n, s in table.items() if s["bookable"]}
    print(f"{len(records)} calibration documents; fields seen {len(table)}; bookable {len(ok)}")
    for n, s in sorted(table.items(), key=lambda kv: -kv[1]["n"])[:40]:
        print(f"  {n:40s} {s['correct']:3d}/{s['n']:<3d} {'BOOKABLE' if s['bookable'] else 'review'}")


if __name__ == "__main__":
    main()

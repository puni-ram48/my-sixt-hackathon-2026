"""Deterministic check: does a gold label match what the document actually prints?

ExtractBench gold labels come with gold_evidence = page + box (x, y, width, height as page fractions).
This script reads the PDF's own words inside that box (pdfplumber text layer, no LLM, no judgement)
and compares them with the gold value, and optionally with a predicted value:

  exact          the value's words appear in order in the printed words (case and punctuation ignored)
  spacing_only   same letters and digits, but spaces inserted or removed (e.g. "CASTLEHILL" vs "CASTLE HILL")
  different      anything else (text layer split or truncated differently, formatting, or another value)

Two uses:
  python -m hybrid.gold_check --doc <doc_id> --field <name> [--pred "..."] [--data FILE] [--crop out.png]
      one label: prints the printed text in the gold box and both comparisons; --crop saves the region as an image
  python -m hybrid.gold_check --survey [--data data/unified/dev.jsonl]
      every non-null text label of every text-layer ExtractBench document: how often gold matches the print
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata

from solution.data import DATA, read_jsonl, source_path

MARGIN = 0.01  # page-fraction margin around the gold box


_TOK = re.compile(r"[0-9a-z]+")


def _tokens(s: str) -> list[str]:
    """Lower-case letter/digit runs: punctuation and line breaks are ignored, spaces separate tokens."""
    return _TOK.findall(unicodedata.normalize("NFKC", str(s)).casefold())


def compare(value: str, printed: str) -> str:
    """How `value` relates to `printed` (the text inside the gold box, which may also hold label words).

    exact         the value's tokens appear, in order and contiguous, in the printed tokens
    spacing_only  not exact, but the value's letters and digits appear contiguous once spaces are removed
    different     anything else
    """
    v, p = _tokens(value), _tokens(printed)
    if not v:
        return "different"
    if any(p[i:i + len(v)] == v for i in range(len(p) - len(v) + 1)):
        return "exact"
    return "spacing_only" if "".join(v) in "".join(p) else "different"


def printed_text(rec: dict, field: str) -> tuple[str | None, dict | None]:
    """Text-layer words inside the field's gold box (None if no evidence box or no text layer)."""
    import pdfplumber

    ev = (rec.get("gold_evidence") or {}).get(field)
    if not ev or "bbox" not in ev:
        return None, None
    x, y, w, h = ev["bbox"]
    with pdfplumber.open(str(source_path(rec))) as pdf:
        page_no = int(ev.get("page", 1))
        if page_no < 1 or page_no > len(pdf.pages):
            return None, ev
        page = pdf.pages[page_no - 1]
        W, H = float(page.width), float(page.height)
        words = [wd for wd in page.extract_words()
                 if (x - MARGIN) * W <= wd["x0"] and wd["x1"] <= (x + w + MARGIN) * W
                 and (y - MARGIN) * H <= wd["top"] and wd["bottom"] <= (y + h + MARGIN) * H]
    words.sort(key=lambda wd: (round(wd["top"]), wd["x0"]))
    return (" ".join(wd["text"] for wd in words) or None), ev


def crop(rec: dict, field: str, out: str, scale: float = 4.0) -> None:
    import pypdfium2 as pdfium

    ev = rec["gold_evidence"][field]
    x, y, w, h = ev["bbox"]
    img = pdfium.PdfDocument(str(source_path(rec)))[int(ev.get("page", 1)) - 1].render(scale=scale).to_pil()
    W, H = img.size
    img.crop((int((x - 0.02) * W), int((y - 0.01) * H), int((x + w + 0.02) * W), int((y + h + 0.01) * H))).save(out)


def survey(records: list[dict]) -> None:
    counts, examples, docs = collections.Counter(), [], 0
    for rec in records:
        if rec.get("source_dataset") != "extractbench":
            continue
        docs_counted = False
        for field, gold in rec["gold"].items():
            if not isinstance(gold, str) or not gold.strip() or gold.strip().lower() in ("true", "false"):
                continue
            if rec["schema"]["properties"].get(field, {}).get("x-normalizer") == "json":
                continue
            try:
                text, _ = printed_text(rec, field)
            except Exception:
                continue
            if text is None:
                counts["no_text_in_box"] += 1
                continue
            docs_counted = True
            c = compare(gold, text)
            counts[c] += 1
            if c == "spacing_only" and len(examples) < 8:
                examples.append((rec["doc_id"], field, gold, text))
        docs += docs_counted
    checked = counts["exact"] + counts["spacing_only"] + counts["different"]
    print(f"documents with printed text in gold boxes: {docs}; text labels checked: {checked} "
          f"(+{counts['no_text_in_box']} with no printed text in the box: scans or image-only values)")
    for k in ("exact", "spacing_only", "different"):
        print(f"  {k:13s} {counts[k]:5d}  ({counts[k] / checked:.1%})" if checked else f"  {k}: 0")
    for d, f, g, t in examples:
        print(f"  spacing example: {d} | {f} | gold {g!r} | printed {t!r}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA / "unified/dev.jsonl"))
    ap.add_argument("--doc")
    ap.add_argument("--field")
    ap.add_argument("--pred")
    ap.add_argument("--crop", help="save the gold box region as a PNG")
    ap.add_argument("--survey", action="store_true")
    args = ap.parse_args(argv)
    records = read_jsonl(args.data)
    if args.survey:
        survey(records)
        return
    rec = next(r for r in records if r["doc_id"] == args.doc)
    gold = rec["gold"].get(args.field)
    text, ev = printed_text(rec, args.field)
    print(json.dumps({"doc_id": args.doc, "field": args.field, "gold_evidence": ev, "printed_in_box": text,
                      "gold": gold, "gold_vs_printed": compare(gold, text) if gold and text else None,
                      "pred": args.pred, "pred_vs_printed": compare(args.pred, text) if args.pred and text else None,
                      "pred_vs_gold": compare(args.pred, gold) if args.pred and gold else None}, indent=1))
    if args.crop:
        crop(rec, args.field, args.crop)
        print(f"saved {args.crop}")


if __name__ == "__main__":
    main()

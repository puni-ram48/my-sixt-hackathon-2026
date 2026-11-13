"""Reading input records and getting the text of each document."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MAX_PROMPT_CHARS = 60_000  # how much document text goes into one LLM prompt

_cuad_cache: dict[str, str] | None = None


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_path(rec: dict) -> Path:
    # Records were written on Windows, so they may contain backslashes.
    # Normalise them so the same path works on Linux (the hidden-test machine).
    return DATA / rec["pages"][0].replace("\\", "/")


_cuad_definitions: dict[str, str] | None = None


def cuad_definitions() -> dict[str, str]:
    """Definition of each CUAD clause type, from the question text in CUADv1.json.

    Each question reads: 'Highlight the parts ... related to "<Name>" ... Details: <definition>'.
    The schema records have no definitions, so this is the only place they are written down.
    """
    global _cuad_definitions
    if _cuad_definitions is None:
        import re

        raw = json.loads((DATA / "raw/cuad/CUADv1.json").read_text(encoding="utf-8"))["data"]
        found: dict[str, str] = {}
        for doc in raw:
            for para in doc["paragraphs"]:
                for qa in para["qas"]:
                    name = re.search(r'related to "(.+?)"', qa["question"])
                    details = re.search(r"Details:\s*(.*)$", qa["question"], re.S)
                    if name and details and name.group(1) not in found:
                        found[name.group(1)] = " ".join(details.group(1).split())
        _cuad_definitions = found
    return _cuad_definitions


def cuad_text(doc_id: str) -> str:
    global _cuad_cache
    if _cuad_cache is None:
        raw = json.loads((DATA / "raw/cuad/CUADv1.json").read_text(encoding="utf-8"))["data"]
        _cuad_cache = {d["title"]: "\n".join(p["context"] for p in d["paragraphs"]) for d in raw}
    return _cuad_cache[doc_id.split(":", 1)[1]]


def pdf_text(path: Path) -> str | None:
    """Text layer of a PDF. Returns None for scans, which have no text layer.

    pdfplumber reads form layouts better than pypdf, so it is tried first.
    """
    for extract in (_text_pdfplumber, _text_pypdf):
        try:
            text = extract(path)
        except Exception:
            continue
        if text and text.strip():
            return text
    return None


def _text_pdfplumber(path: Path) -> str:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _text_pypdf(path: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def doc_text(rec: dict) -> str | None:
    """Full text of the document, or None if we cannot read any text from it."""
    if rec.get("text"):  # supplied directly, e.g. a contract uploaded in demo_app.py
        return rec["text"]
    if rec["source_dataset"] == "cuad":
        return cuad_text(rec["doc_id"])
    path = source_path(rec)
    if path.suffix.lower() == ".pdf" and path.exists():
        return pdf_text(path)
    return None

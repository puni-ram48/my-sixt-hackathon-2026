"""Demo dashboard: upload a contract or a form, see what the system books, and where each value is in the document.

Runs the exact submitted system (contracts: FINAL, forms: forms_final, threshold 0.95).
Optional tool; needs streamlit (not in requirements.txt):
    pip install streamlit==1.67.0
    streamlit run demo_app.py          # from Project_1/, with .env filled in
"""
from __future__ import annotations

import html
import io
import json
import re
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from solution.data import DATA, read_jsonl, source_path  # noqa: E402
from solution.run import load_env  # noqa: E402

load_env(ROOT / ".env")

from demo_flow import flow_cards, flow_dot  # noqa: E402
from hybrid.evaluate import scorer  # noqa: E402
from hybrid.pipeline import CONFIGS, process  # noqa: E402
from hybrid.retrieve import find_span  # noqa: E402

THRESHOLD = 0.95
ICON = {"BOOK": "✅", "REVIEW": "🟡", "REJECT": "⛔", "ABSTAIN": "⚪"}
ORDER = {"BOOK": 0, "REVIEW": 1, "REJECT": 2, "ABSTAIN": 3}
UPLOADS = DATA / "uploads"

st.set_page_config(page_title="Teach a machine to say I don't know", layout="wide")


# ------------------------------------------------------------------ data helpers

@st.cache_data(show_spinner=False)
def dataset() -> dict[str, list[dict]]:
    out = {}
    for name in ("test_public", "dev"):
        p = DATA / "unified" / f"{name}.jsonl"
        if p.exists():
            out[name] = read_jsonl(p)
    return out


def cuad_schema() -> dict:
    for recs in dataset().values():
        for r in recs:
            if r["source_dataset"] == "cuad":
                return r["schema"]
    raise RuntimeError("no CUAD record found: run 1_download.sh and 2_prepare.py first")


def pdf_pages_text(data: bytes) -> list[str]:
    import pdfplumber

    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return [p.extract_text() or "" for p in pdf.pages]


def render_page(path: Path, page_no: int, boxes: list[tuple[list[float], str, str]], scale: float = 2.0):
    """Page image with labelled rectangles; boxes are (x, y, w, h as page fractions, colour, label)."""
    import pypdfium2 as pdfium
    from PIL import ImageDraw

    from hybrid.forms import _PDF_LOCK

    with _PDF_LOCK:
        pdf = pdfium.PdfDocument(str(path))
        page_no = max(1, min(page_no, len(pdf)))
        img = pdf[page_no - 1].render(scale=scale).to_pil().convert("RGB")
        pdf.close()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for (x, y, w, h), colour, label in boxes:
        rect = (x * W - 4, y * H - 4, (x + w) * W + 4, (y + h) * H + 4)
        draw.rectangle(rect, outline=colour, width=5)
        draw.text((rect[0], max(0, rect[1] - 14)), label, fill=colour)
    return img, page_no


def value_box(path: Path, value, page_hint) -> tuple[int, list[float]] | None:
    """Locate a value in the PDF text layer: (page, [x, y, w, h]) or None (scan / not printed verbatim)."""
    from hybrid.forms import _find, text_layer

    if not isinstance(value, str) or value.lower() in ("true", "false"):
        return None
    _, words = text_layer(path)
    if not words:
        return None
    hits = _find(words, value, page_hint if isinstance(page_hint, int) else None) or _find(words, value)
    if not hits:
        return None
    seq = hits[0]
    x0, x1 = min(w.x0 for w in seq), max(w.x1 for w in seq)
    t, b = min(w.top for w in seq), max(w.bottom for w in seq)
    return seq[0].page, [x0, t, x1 - x0, b - t]


def highlighted_context(text: str, value: str, page_starts: list[int] | None) -> str | None:
    parts = [p for p in str(value).split(" | ") if p.strip()]
    spans = [s for s in (find_span(text, p) for p in parts) if s]
    if not spans:
        return None
    blocks = []
    for s, e in spans[:3]:
        a, b = max(0, s - 350), min(len(text), e + 350)
        page = ""
        if page_starts:
            page = f"page {sum(1 for p in page_starts if p <= s)}, "
        snippet = (html.escape(text[a:s]) + "<mark style='background:#fff3a3'>" + html.escape(text[s:e]) + "</mark>"
                   + html.escape(text[e:b]))
        blocks.append(f"<div style='font-size:0.85em;color:#555'>{page}characters {s:,}–{e:,}</div>"
                      f"<div style='white-space:pre-wrap;border-left:3px solid #ccc;padding-left:8px;"
                      f"font-family:Georgia,serif;font-size:0.92em'>…{snippet}…</div>")
    return "<br>".join(blocks)


# ------------------------------------------------------------------ sidebar: input

st.sidebar.title("Document")
mode = st.sidebar.radio("Source", ["Pick from dataset (with gold)", "Upload a contract", "Upload a form"])
rec, page_starts = None, None

if mode == "Pick from dataset (with gold)":
    data = dataset()
    if not data:
        st.sidebar.error("No dataset found. Run 1_download.sh and 2_prepare.py first.")
    else:
        split = st.sidebar.selectbox("Split", list(data))
        kind = st.sidebar.selectbox("Type", ["contract (CUAD)", "form (ExtractBench)"])
        src = "cuad" if kind.startswith("contract") else "extractbench"
        recs = [r for r in data[split] if r["source_dataset"] == src]
        doc = st.sidebar.selectbox("Document", [r["doc_id"].split(":", 1)[1] for r in recs])
        rec = next(r for r in recs if r["doc_id"].split(":", 1)[1] == doc)

elif mode == "Upload a contract":
    up = st.sidebar.file_uploader("Contract (PDF or TXT)", type=["pdf", "txt"])
    if up is not None:
        raw = up.getvalue()
        if up.name.lower().endswith(".pdf"):
            pages = pdf_pages_text(raw)
            page_starts, pos = [], 0
            for p in pages:
                page_starts.append(pos)
                pos += len(p) + 1
            text = "\n".join(pages)
        else:
            text = raw.decode("utf-8", errors="replace")
        if not text.strip():
            st.sidebar.error("No text found in this file (a scanned contract has no text layer).")
        else:
            rec = {"doc_id": f"upload:{up.name}", "source_dataset": "cuad", "schema": cuad_schema(),
                   "pages": [], "text": text}
            st.sidebar.caption(f"{len(text):,} characters extracted; the 41 CUAD clause fields will be filled.")

else:
    up = st.sidebar.file_uploader("Form (PDF)", type=["pdf"])
    forms = [r for recs in dataset().values() for r in recs if r["source_dataset"] == "extractbench"]
    families = {}
    for r in forms:
        families.setdefault(re.sub(r"[-_ ]?p?\d+$", "", r["doc_id"].split("/")[-1])[:40], r)
    tmpl = st.sidebar.selectbox("Use the field schema of", sorted(families),
                                help="A form needs a schema (the fields to fill). Pick a known form type.")
    if up is not None and tmpl:
        UPLOADS.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", up.name)
        (UPLOADS / safe).write_bytes(up.getvalue())
        rec = {"doc_id": f"upload:{safe}", "source_dataset": "extractbench", "schema": families[tmpl]["schema"],
               "pages": [f"uploads/{safe}"]}
        st.sidebar.caption(f"{len(rec['schema'].get('properties', {}))} fields from the '{tmpl}' schema.")

run = st.sidebar.button("Run extraction", type="primary", disabled=rec is None)
st.sidebar.markdown("---")
st.sidebar.caption("Contracts: Claude Sonnet 5.5 + evidence gates. Forms: Claude + GPT-5.5 must agree. "
                   f"Booked = confidence ≥ {THRESHOLD}. Everything else goes to a human.")

# ------------------------------------------------------------------ run

st.title("Teach a machine to say \"I don't know\"")
st.caption("The model proposes, evidence decides: values are booked only when independent checks support them.")

if run and rec is not None:
    with st.spinner("Reading the document and checking the evidence (10–60 s)…"):
        try:
            line, trace = process(rec, CONFIGS["FINAL"])
            st.session_state["result"] = (rec, line, trace, page_starts)
        except Exception as e:  # show the problem instead of a stack trace
            st.error(f"Extraction failed: {e}")

if "result" not in st.session_state:
    st.info("Choose a document in the sidebar and press **Run extraction**. "
            "Below: what happens to a document. The router sends contracts and forms to different pipelines.")
    st.markdown("### 🔀 Router: is it a contract (text) or a form / scan (PDF)?")
    col_c, col_f = st.columns(2, gap="large")
    with col_c:
        st.markdown("#### 📄 Contract (CUAD)")
        st.markdown(flow_cards("contract"), unsafe_allow_html=True)
    with col_f:
        st.markdown("#### 🧾 Form / scan (ExtractBench)")
        st.markdown(flow_cards("form"), unsafe_allow_html=True)
    with st.expander("Full diagram (both pipelines on one page)"):
        st.graphviz_chart(flow_dot())
    st.stop()

rec, line, trace, page_starts = st.session_state["result"]
fields = trace.get("fields", {}) if trace else {}
gold = rec.get("gold") or {}
is_contract = rec["source_dataset"] == "cuad"
text = None
if is_contract:
    from solution.data import doc_text
    text = doc_text(rec)

st.subheader(rec["doc_id"])
with st.expander("🔀 How this document went through the system (live numbers)", expanded=True):
    st.markdown(f"**Router → {'📄 contract pipeline' if is_contract else '🧾 form / scan pipeline'}.** "
                "Each row is one step: the number is how many values are still in the running, "
                "the pill is how many stopped there.")
    left_col, _ = st.columns([3, 1])
    with left_col:
        st.markdown(flow_cards("contract" if is_contract else "form", fields, (trace or {}).get("usage", {})),
                    unsafe_allow_html=True)
if not fields:
    st.warning(f"No per-field decisions: {(trace or {}).get('error') or (trace or {}).get('reason') or 'see logs'}. "
               "All fields were sent to human review.")
    st.stop()

# ------------------------------------------------------------------ summary

counts = {k: sum(1 for f in fields.values() if f["decision"] == k) for k in ORDER}
c = st.columns(5)
c[0].metric("✅ Booked automatically", counts["BOOK"])
c[1].metric("🟡 Sent to review", counts["REVIEW"])
c[2].metric("⛔ Rejected", counts["REJECT"])
c[3].metric("⚪ Not present / unsure", counts["ABSTAIN"])
if gold:
    booked = [n for n, f in fields.items() if f["decision"] == "BOOK"]
    kinds = {n: s.get("x-normalizer", "string") for n, s in rec["schema"].get("properties", {}).items()}
    right = sum(1 for n in booked if gold.get(n) is not None
                and scorer.values_match(line["fields"][n]["value"], gold[n], kinds.get(n, "string")))
    c[4].metric("Booked values matching gold", f"{right} / {len(booked)}")

show = st.multiselect("Show decisions", list(ORDER), default=["BOOK", "REVIEW", "REJECT"])
pdf_path = source_path(rec) if not is_contract and rec.get("pages") else None

# ------------------------------------------------------------------ per field

for name, f in sorted(fields.items(), key=lambda kv: (ORDER[kv[1]["decision"]], kv[0])):
    if f["decision"] not in show:
        continue
    ex = f["extraction"]
    value = ex.get("value")
    out = line["fields"].get(name, {})
    head = f"{ICON[f['decision']]} **{f['decision']}** · {name}"
    if value is not None:
        head += f" · `{str(value)[:70]}`"
    with st.expander(head, expanded=f["decision"] == "BOOK"):
        left, right = st.columns([2, 3])
        with left:
            st.markdown(f"**Value:** {value if value is not None else '—'}")
            st.markdown(f"**Model confidence:** {ex.get('confidence', 0):.2f} · **output confidence:** "
                        f"{out.get('confidence', 0):.2f}")
            st.markdown(f"**Why:** {f['reasons'][0]}")
            if not is_contract and f.get("second_read") is not None:
                st.markdown(f"**Second reader (GPT-5.5):** {f['second_read'].get('value')} "
                            f"({f['second_read'].get('confidence', 0):.2f})")
            if gold:
                g = gold.get(name)
                kind = rec["schema"]["properties"].get(name, {}).get("x-normalizer", "string")
                ok = value is not None and g is not None and scorer.values_match(value, g, kind)
                st.markdown(f"**Gold:** {g if g is not None else '— (absent)'} "
                            + ("✔️ match" if ok else ("" if value is None else "✖️ no match")))
            sig = f.get("signals", {})
            with st.popover("Evidence signals"):
                st.json(sig)
        with right:
            if value is None:
                st.caption("No value proposed, so there is nothing to cite.")
            elif is_contract and text:
                ctx = highlighted_context(text, value, page_starts)
                clauses = ", ".join(c for loc in (sig.get("evidence_clauses") or []) for c in loc)
                st.markdown(f"**Citation from the contract**{' (clause ' + clauses + ')' if clauses else ''}:")
                st.markdown(ctx or "_The value could not be located verbatim in the text._", unsafe_allow_html=True)
            elif pdf_path is not None and pdf_path.exists():
                boxes, page = [], ex.get("page") if isinstance(ex.get("page"), int) else 1
                found = value_box(pdf_path, value, ex.get("page"))
                label_found = None
                if found:
                    page = found[0]
                    boxes.append((found[1], "red", "our value"))
                elif ex.get("label"):
                    # The value is not in the text layer (scan, or filled in as an image): box the printed label
                    # that the reader says the value sits next to.
                    label_found = value_box(pdf_path, ex["label"], ex.get("page"))
                    if label_found:
                        page = label_found[0]
                        boxes.append((label_found[1], "orange", f"label: {ex['label'][:40]}"))
                ev = (rec.get("gold_evidence") or {}).get(name)
                if ev and "bbox" in ev and int(ev.get("page", page)) == page:
                    boxes.append((ev["bbox"], "blue", "gold location"))
                img, page = render_page(pdf_path, page, boxes)
                if found:
                    how = "red box = where our value is printed"
                elif label_found:
                    how = (f"the value is not in the text layer (scan or filled-in image); orange box = the printed "
                           f"label '{ex['label']}' the reader read it next to")
                else:
                    how = "the value is not in the text layer (scan or handwriting); page reported by the reader"
                has_gold_box = any(c == "blue" for _, c, _ in boxes)
                st.markdown(f"**Citation from the PDF, page {page}:** {how}"
                            + (", blue box = gold annotation" if has_gold_box else ""))
                st.image(img)

st.markdown("---")
with st.expander("Raw output line (submission format)"):
    st.code(json.dumps(line, ensure_ascii=False, indent=1)[:20000], language="json")

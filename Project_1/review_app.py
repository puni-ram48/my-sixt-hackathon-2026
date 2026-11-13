"""Manual review of model answers, side by side with the document.

Put one prediction file per model run in data/runs/ (e.g. data/runs/sonnet_cuad.jsonl).
Verdicts are saved to data/reviews.csv (git-ignored).

Run from Project_1:
    .venv\\Scripts\\python.exe -m streamlit run review_app.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution.data import DATA, doc_text, read_jsonl, source_path  # noqa: E402
from solution.verify import Grounder  # noqa: E402

RUNS_DIR = DATA / "runs"
REVIEWS = DATA / "reviews.csv"
SPLITS = {"dev": DATA / "unified/dev.jsonl", "test_public": DATA / "unified/test_public.jsonl"}
VERDICTS = ["correct", "wrong", "gold_wrong", "unsure"]
COLUMNS = ["run", "doc_id", "field", "verdict", "note"]
GROUNDED_AT = 0.9


@st.cache_data
def load_records(path: str) -> list[dict]:
    return read_jsonl(path)


@st.cache_data
def load_predictions(path: str) -> dict[str, dict]:
    return {r["doc_id"]: r for r in read_jsonl(path)}


@st.cache_data
def load_text(rec_json: str) -> str | None:
    return doc_text(json.loads(rec_json))


def load_reviews() -> dict[tuple[str, str, str], dict]:
    if not REVIEWS.exists():
        return {}
    with REVIEWS.open(encoding="utf-8", newline="") as f:
        return {(r["run"], r["doc_id"], r["field"]): r for r in csv.DictReader(f)}


def save_reviews(reviews: dict) -> None:
    REVIEWS.parent.mkdir(parents=True, exist_ok=True)
    with REVIEWS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(reviews.values())


def set_verdict(reviews: dict, run: str, doc_id: str, field: str, verdict: str) -> None:
    key = (run, doc_id, field)
    if verdict:
        note = reviews.get(key, {}).get("note", "")
        reviews[key] = {"run": run, "doc_id": doc_id, "field": field, "verdict": verdict, "note": note}
    else:
        reviews.pop(key, None)
    save_reviews(reviews)


def answer(pred: dict | None, field: str) -> tuple[object, float]:
    """Value and confidence of one field. Missing confidence counts as 1.0, as in the scorer."""
    item = (pred or {}).get("fields", {}).get(field, {})
    conf = item.get("confidence")
    return item.get("value"), 1.0 if conf is None else float(conf)


def summary(reviews: dict, runs: list[str], preds: dict[str, dict], threshold: float) -> None:
    st.sidebar.markdown("**Results so far**")
    st.sidebar.caption("Only answers you have reviewed and that are not null.")
    for run in runs:
        counts = {v: 0 for v in VERDICTS}
        booked_wrong = 0
        for (r_run, doc_id, field), row in reviews.items():
            if r_run != run:
                continue
            value, conf = answer(preds[run].get(doc_id), field)
            if value is None:
                continue
            counts[row["verdict"]] += 1
            if row["verdict"] == "wrong" and conf >= threshold:
                booked_wrong += 1
        n = sum(counts.values())
        if n == 0:
            st.sidebar.write(f"`{run}`: no verdicts yet")
            continue
        st.sidebar.write(
            f"`{run}`: {n} reviewed, "
            f"correct {counts['correct'] / n:.0%}, wrong {counts['wrong'] / n:.0%}, "
            f"gold wrong {counts['gold_wrong'] / n:.0%}; "
            f"booked wrong at {threshold:.2f}: {booked_wrong}"
        )


def show_document_side(rec: dict, text: str | None) -> None:
    if text:
        st.text_area("Document text", text[:30_000], height=650)
        if len(text) > 30_000:
            st.caption(f"Showing the first 30,000 of {len(text):,} characters.")
        return
    path = source_path(rec)
    if path.suffix.lower() != ".pdf" or not path.exists():
        st.info("No text and no image for this document.")
        return
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    page = st.number_input("Page", 1, len(pdf), 1) - 1
    st.image(pdf[page].render(scale=1.6).to_pil(), caption=f"{path.name}, page {page + 1}")


def main() -> None:
    st.set_page_config(page_title="Project 1 review", layout="wide")
    st.title("Project 1: manual review")

    with st.sidebar:
        split = st.selectbox("Split", list(SPLITS))
        run_files = sorted(p.name for p in RUNS_DIR.glob("*.jsonl")) if RUNS_DIR.exists() else []
        runs = st.multiselect("Runs to compare", run_files, default=run_files)
        threshold = st.slider("Booking threshold", 0.0, 1.0, 0.95, 0.01)
        only_with_values = st.checkbox("Hide fields with no value anywhere", value=True)

    if not runs:
        st.warning(f"No prediction files found. Put JSONL files in {RUNS_DIR}.")
        return

    preds = {run: load_predictions(str(RUNS_DIR / run)) for run in runs}
    records = load_records(str(SPLITS[split]))
    docs = [r for r in records if any(r["doc_id"] in preds[run] for run in runs)]
    if not docs:
        st.warning("None of the selected runs contain documents from this split.")
        return

    reviews = load_reviews()
    summary(reviews, runs, preds, threshold)

    labels = [r["doc_id"] for r in docs]
    doc_id = st.selectbox(f"Document ({len(docs)} available)", labels)
    rec = next(r for r in docs if r["doc_id"] == doc_id)
    st.subheader(doc_id)
    st.caption(f"{rec['source_dataset']} · tags: {', '.join(rec.get('tags') or []) or 'none'}")

    text = load_text(json.dumps(rec))
    grounder = Grounder(text) if text else None
    left, right = st.columns([1, 1])
    with left:
        show_document_side(rec, text)
    with right:
        for field, spec in rec["schema"].get("properties", {}).items():
            gold = rec["gold"].get(field)
            values = [answer(preds[run].get(doc_id), field)[0] for run in runs]
            if only_with_values and gold is None and all(v is None for v in values):
                continue
            st.markdown(f"**{field}**  ·  gold: `{str(gold)[:200] if gold is not None else 'null'}`")
            for run, value in zip(runs, values):
                _, conf = answer(preds[run].get(doc_id), field)
                if value is None:
                    grounded = "no answer"
                elif grounder is None:
                    grounded = "not checkable (no text)"
                else:
                    score = grounder.score(value)
                    grounded = "not checkable" if score is None else (
                        "grounded" if score >= GROUNDED_AT else "NOT grounded")
                booked = value is not None and conf >= threshold
                cols = st.columns([3, 1, 2, 2])
                cols[0].write(f"`{run}`: {str(value)[:200] if value is not None else 'null'}")
                cols[1].write(f"conf {conf:.2f}")
                cols[2].write(f"{grounded}{' · booked' if booked else ''}")
                if value is not None:
                    current = reviews.get((run, doc_id, field), {}).get("verdict", "")
                    options = [""] + VERDICTS
                    choice = cols[3].selectbox(
                        "verdict", options, index=options.index(current),
                        key=f"verdict|{run}|{doc_id}|{field}", label_visibility="collapsed",
                        format_func=lambda v: v or "not reviewed",
                    )
                    if choice != current:
                        set_verdict(reviews, run, doc_id, field, choice)
                        st.rerun()
            st.divider()


main()

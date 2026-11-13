"""Graphviz flowchart of both pipelines for demo_app.py, optionally with live counts for one document."""
from __future__ import annotations

CONTRACT_GATES = [  # (reason substring, question, outcome when the check fails)
    ("not found in document", "value in the contract?", "REJECT"),
    ("is not a", "right clause type?", "REJECT"),
    ("clause type uncertain", "classifier sure?", "REVIEW"),
    ("probably incomplete", "answer complete?", "REVIEW"),
    ("not whole sentences", "whole sentences?", "REVIEW"),
    ("not reliable on calibration", "field trusted?", "REVIEW"),
    ("below threshold", "confidence >= 0.95?", "REVIEW"),
]
FORM_GATES = [
    ("only one reader", "both readers found it?", "REVIEW"),
    ("readers disagree", "readers agree?", "REVIEW"),
    ("not found in the text layer", "printed on the page?", "REJECT"),
    ("not next to its label", "next to its label?", "REVIEW"),
    ("not reliable on calibration", "field trusted?", "REVIEW"),
    ("below threshold", "confidence >= 0.95?", "REVIEW"),
]
OUT_COLOR = {"BOOK": "#1a7f37", "REVIEW": "#b7791f", "REJECT": "#c62828", "ABSTAIN": "#6e6e73"}
ORANGE, GREY = "#FF5F00", "#c7c7cc"


def _count(fields: dict, key: str) -> int:
    return sum(1 for f in fields.values() if key in f["reasons"][0])


def _node(name, label, on, shape="box", fill=None):
    fc = fill or ("#fff1e8" if on else "#f4f4f6")
    return (f'{name} [label="{label}", shape={shape}, style="rounded,filled", fillcolor="{fc}", '
            f'color="{ORANGE if on else GREY}", fontcolor="{"#1d1d1f" if on else "#8e8e93"}", '
            f'penwidth={2.2 if on else 1}];')


def _outcome(name, kind, n, on, live):
    c = OUT_COLOR[kind] if on else GREY
    label = f"{kind}\\n{n}" if (live and on) else kind
    return (f'{name} [label="{label}", shape=box, style="rounded,filled", fillcolor="white", color="{c}", '
            f'fontcolor="{c}", fontsize=10, penwidth=1.6];')


def flow_dot(active: str | None = None, fields: dict | None = None, usage: dict | None = None) -> str:
    """active = 'contract' | 'form' | None (show both). fields = trace fields, usage = trace usage (live counts)."""
    fields, usage = fields or {}, usage or {}
    live = bool(fields)
    on_c, on_f = active in (None, "contract"), active in (None, "form")
    n = len(fields)
    proposed = sum(1 for f in fields.values() if f["extraction"].get("value") is not None)
    abstain = sum(1 for f in fields.values() if f["decision"] == "ABSTAIN")
    book = sum(1 for f in fields.values() if f["decision"] == "BOOK")

    L = ["digraph G {", 'rankdir=LR; nodesep=0.25; ranksep=0.32; bgcolor="transparent";',
         'node [fontname="Helvetica", fontsize=11]; edge [color="#8e8e93", arrowsize=0.7];',
         _node("up", "Document\\nuploaded", True, fill="white"),
         _node("router", "Router:\\ncontract text\\nor form PDF?", True, shape="diamond", fill="white"),
         "up -> router;"]

    # ---- contract branch
    if live and active == "contract":
        c_read = f"Claude Sonnet 5.5\\nreads whole contract\\n{n} fields -> {proposed} values"
        c_split = f"split into\\n{usage.get('n_clauses', '?')} clauses"
    else:
        c_read, c_split = "Claude Sonnet 5.5\\nreads whole contract\\n(1 call)", "split into clauses\\n(offsets, headings)"
    L += ['subgraph cluster_c { label="CONTRACT  (CUAD, 41 clause fields)"; fontname="Helvetica-Bold"; '
          f'fontsize=12; color="{ORANGE if on_c else "#d1d1d6"}"; style="rounded";',
          _node("c1", c_read, on_c), _node("c2", c_split, on_c),
          _node("c3", "bge-small embeddings\\n-> clause classifier", on_c), "c1 -> c2 -> c3;"]
    prev = "c3"
    for i, (key, question, out) in enumerate(CONTRACT_GATES):
        L += [_node(f"cg{i}", question, on_c),
              _outcome(f"co{i}", out, _count(fields, key), on_c, live),
              f'{prev} -> cg{i}; cg{i} -> co{i} [label=" no", fontsize=9, color="{OUT_COLOR[out]}"];']
        prev = f"cg{i}"
    L += [_outcome("cbook", "BOOK", book, on_c, live and active == "contract"),
          _outcome("cabs", "ABSTAIN", abstain, on_c, live and active == "contract"),
          f'{prev} -> cbook [label=" yes", fontsize=9, color="#1a7f37"];',
          'c1 -> cabs [label=" null", fontsize=9]; }']

    # ---- form branch
    if live and active == "form":
        f_read = (f"pages -> images\\n{usage.get('pages', '?')} page(s), text layer: "
                  f"{'yes' if usage.get('text_layer') else 'no'}")
    else:
        f_read = "pages -> images\\n+ text layer, word positions"
    L += ['subgraph cluster_f { label="FORM / SCAN  (ExtractBench)"; fontname="Helvetica-Bold"; fontsize=12; '
          f'color="{ORANGE if on_f else "#d1d1d6"}"; style="rounded";',
          _node("f1", f_read, on_f),
          _node("fa", "Reader A\\nClaude Sonnet 5.5", on_f),
          _node("fb", "Reader B\\nGPT-5.5 (independent)", on_f, fill="#e8f0fd" if on_f else None),
          _node("fj", "compare\\nboth reads", on_f, shape="ellipse"),
          "f1 -> fa; f1 -> fb; fa -> fj; fb -> fj;",
          _outcome("fabs", "ABSTAIN", abstain, on_f, live and active == "form"),
          'fj -> fabs [label=" both null", fontsize=9];']
    prev = "fj"
    for i, (key, question, out) in enumerate(FORM_GATES):
        L += [_node(f"fg{i}", question, on_f),
              _outcome(f"fo{i}", out, _count(fields, key), on_f, live),
              f'{prev} -> fg{i}; fg{i} -> fo{i} [label=" no", fontsize=9, color="{OUT_COLOR[out]}"];']
        prev = f"fg{i}"
    L += [_outcome("fbook", "BOOK", book, on_f, live and active == "form"),
          f'{prev} -> fbook [label=" yes", fontsize=9, color="#1a7f37"]; }}']

    L += [f'router -> c1 [label=" contract", color="{ORANGE if on_c else GREY}", penwidth=2];',
          f'router -> f1 [label=" form / scan", color="{ORANGE if on_f else GREY}", penwidth=2];',
          _node("out", "output.jsonl + decision log\\nBOOK = confidence >= 0.95\\nelse: a human reviews", True,
                fill="white"),
          "cbook -> out [style=dashed]; fbook -> out [style=dashed];", "}"]
    return "\n".join(L)


# ---------------------------------------------------------------- large step-by-step storyboard (HTML)

_CSS = """<style>
.stp{font-family:inherit;margin:4px 0 10px 0}
.stp .row{display:flex;align-items:center;gap:12px;padding:8px 12px;margin:5px 0;border-radius:10px;
  border-left:4px solid #FF5F00;background:rgba(255,95,0,0.08)}
.stp .gate{border-left-color:rgba(150,150,150,0.7);background:rgba(150,150,150,0.08)}
.stp .num{flex:0 0 28px;height:28px;border-radius:50%;background:#FF5F00;color:#fff;font-weight:800;
  font-size:13px;display:flex;align-items:center;justify-content:center}
.stp .gate .num{background:rgba(150,150,150,0.85)}
.stp .txt{flex:1;min-width:0}.stp .t{font-weight:700;font-size:15px}.stp .s{font-size:12.5px;opacity:0.72}
.stp .val{font-size:17px;font-weight:800;white-space:nowrap}
.stp .pill{font-size:12px;font-weight:700;padding:2px 9px;border-radius:999px;border:1.5px solid;white-space:nowrap}
.stp .book{border-left-color:#1a7f37;background:rgba(26,127,55,0.12)}.stp .book .num{background:#1a7f37}
.stp .book .val{color:#2da44e;font-size:22px}
</style>"""

_CONTRACT_STEPS = [
    ("📄", "Contract in", "Text of the whole contract"),
    ("🤖", "Claude reads it", "1 call: a value + confidence for each of the 41 clause fields"),
    ("✂️", "Split into clauses", "Paragraphs and sentence windows, with positions"),
    ("🧠", "Clause classifier", "Local embeddings: is this clause the right type?"),
]
_FORM_STEPS = [
    ("🧾", "Form / scan in", "PDF pages rendered as images"),
    ("🅰️", "Reader A: Claude", "Reads every field from the page images"),
    ("🅱️", "Reader B: GPT-5.5", "Reads the same pages independently"),
    ("⚖️", "Compare", "Both must find the same value"),
]
_GATE_ICON = {"REJECT": "⛔", "REVIEW": "🟡"}
_GATE_HELP = {
    "value in the contract?": "Value must appear verbatim in the text",
    "right clause type?": "Classifier: is the source clause this type?",
    "classifier sure?": "Unsure clause type → a human checks",
    "answer complete?": "Other clauses of this type left out?",
    "whole sentences?": "Clause answers must be whole sentences",
    "field trusted?": "Field was error-free on calibration data",
    "confidence >= 0.95?": "The model's own confidence, checked last",
    "both readers found it?": "One reader found nothing → a human checks",
    "readers agree?": "Two model families must read the same value",
    "printed on the page?": "If the PDF has text: is the value on it?",
    "next to its label?": "Value sits next to its printed label",
}


def _row(i, icon, title, sub, cls="", value=None, pill=None):
    v = f'<div class="val">{value}</div>' if value is not None else ""
    pl = f'<div class="pill" style="color:{pill[1]};border-color:{pill[1]}">{pill[0]}</div>' if pill else ""
    return (f'<div class="row {cls}"><div class="num">{i}</div><div class="txt"><div class="t">{icon} {title}</div>'
            f'<div class="s">{sub}</div></div>{v}{pl}</div>')


def flow_cards(kind: str, fields: dict | None = None, usage: dict | None = None) -> str:
    """Compact vertical stepper for one branch ('contract' or 'form'); live numbers if fields are given."""
    fields, usage = fields or {}, usage or {}
    live = bool(fields)
    steps = _CONTRACT_STEPS if kind == "contract" else _FORM_STEPS
    gates = CONTRACT_GATES if kind == "contract" else FORM_GATES
    n_fields = len(fields)
    abstain = sum(1 for f in fields.values() if f["decision"] == "ABSTAIN")
    proposed = n_fields - abstain  # every field with a value from at least one reader goes through the checks
    vals, pills = [None] * len(steps), [None] * len(steps)
    if live and kind == "contract":
        vals = [f"{n_fields} fields", f"{proposed} values", f"{usage.get('n_clauses', '?')} clauses", None]
        pills[1] = (f"⚪ {abstain} → ABSTAIN", "#8e8e93")
    elif live:
        vals = [f"{usage.get('pages', '?')} page(s)", None, None, f"{proposed} values"]
        pills[0] = ("text layer" if usage.get("text_layer") else "scan, no text", "#8e8e93")
        pills[3] = (f"⚪ {abstain} → ABSTAIN", "#8e8e93")
    out = [_CSS, '<div class="stp">']
    i = 0
    for (ic, t, sub), v, pl in zip(steps, vals, pills):
        i += 1
        out.append(_row(i, ic, t, sub, "", v, pl))
    alive = proposed
    for key, question, res in gates:
        i += 1
        stopped = sum(1 for f in fields.values() if key in f["reasons"][0])
        color = OUT_COLOR[res]
        if live:
            alive -= stopped
            out.append(_row(i, "❓", question, _GATE_HELP.get(question, ""), "gate", f"{alive} pass",
                            (f"{_GATE_ICON[res]} {stopped} → {res}", color)))
        else:
            out.append(_row(i, "❓", question, _GATE_HELP.get(question, ""), "gate", None,
                            (f"no → {_GATE_ICON[res]} {res}", color)))
    book = sum(1 for f in fields.values() if f["decision"] == "BOOK")
    out.append(_row(i + 1, "✅", "BOOK", "Booked automatically (confidence ≥ 0.95). Everything else goes to a human.",
                    "book", book if live else None))
    out.append("</div>")
    return "".join(out)

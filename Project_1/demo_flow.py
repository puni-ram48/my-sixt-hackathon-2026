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

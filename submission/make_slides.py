"""Builds submission/slides.pdf (5 slides, 16:9) and PNG previews.  python submission/make_slides.py"""
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
ORANGE, DARK, GREY, LIGHT = "#FF5F00", "#1d1d1f", "#6e6e73", "#f4f4f6"
GREEN, AMBER, RED, BLUE = "#1a7f37", "#b7791f", "#c62828", "#1f5fbf"
plt.rcParams["font.family"] = "DejaVu Sans"


def slide(title, kicker, n):
    fig = plt.figure(figsize=(13.333, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56.25)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 54.6), 100, 1.65, color=ORANGE))
    ax.text(3, 50.6, title, fontsize=26, weight="bold", color=DARK, va="center")
    if kicker:
        ax.text(3, 46.9, kicker, fontsize=13.5, color=GREY, va="center")
    ax.text(3, 1.6, "Team Zero to SIXTy  ·  SIXT Hackathon 2026  ·  Project 1", fontsize=9, color=GREY)
    ax.text(97, 1.6, f"{n} / 5", fontsize=9, color=GREY, ha="right")
    return fig, ax


def box(ax, x, y, w, h, text, fc=LIGHT, ec="#c7c7cc", fs=10.5, color=DARK, weight="normal", ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec=ec, lw=1.2))
    tx = x + w / 2 if ha == "center" else x + 1
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=color, weight=weight, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.6))


def rules(ax, x, y, rows, w=44, rh=2.55, fs=9.6):
    """Decision-rule table: rows of (check, outcome, colour)."""
    for i, (check, outcome, c) in enumerate(rows):
        yy = y - i * rh
        ax.add_patch(plt.Rectangle((x, yy), w, rh - 0.35, color=LIGHT))
        ax.text(x + 0.8, yy + rh / 2 - 0.15, f"{i + 1}. {check}", fontsize=fs, va="center", color=DARK)
        ax.text(x + w - 0.8, yy + rh / 2 - 0.15, outcome, fontsize=fs, va="center", ha="right", color=c,
                weight="bold")


slides = []

# ------------------------------------------------------------------ 1. storyline and problem
fig, ax = slide("Teach a machine to say “I don't know”",
                "Contracts and forms are typed into SIXT's systems by hand. AI can do it in seconds — but can we trust it?",
                1)
box(ax, 3, 30.5, 29, 12.5, "1 human review\ncosts  1", fc="#eef6ee", ec=GREEN, fs=17, color=GREEN, weight="bold")
box(ax, 35, 30.5, 29, 12.5, "1 wrong value booked\ncosts  100", fc="#fdecea", ec=RED, fs=17, color=RED, weight="bold")
box(ax, 67, 30.5, 30, 12.5, "→ automation only pays off\nif booked values are right\n> 99% of the time", fc="#fff1e8",
    ec=ORANGE, fs=14, color=DARK, weight="bold")
ax.text(3, 26.3, "Where we started: trust the model's confidence", fontsize=14, weight="bold", color=DARK)
ax.text(3, 22.6, "One LLM call per document; book everything with confidence ≥ 0.9.\n"
                 "Result on 20 held-out contracts:", fontsize=12, color=DARK, va="center", linespacing=1.3)
box(ax, 3, 11.5, 44, 7.6, "52 booked · 12 wrong\nnet saving  −140%", fc="#fdecea", ec=RED, fs=16, color=RED,
    weight="bold")
ax.text(3, 7.6, "Worse than typing everything by hand:\n12 confident mistakes cost as much as 1,200 reviews.",
        fontsize=11.5, color=GREY, va="top", linespacing=1.3)
ax.text(52, 26.3, "Why this is hard (measured in our EDA)", fontsize=14, weight="bold", color=DARK)
facts = ["68% of contract fields are empty → a model that\n   always answers invents values",
         "28% of contracts are longer than the prompt limit",
         "45% of form PDFs are scans with no text to check against",
         "Keyword search finds the right clause for only ~58% of fields",
         "The model's confidence is not calibrated: right and wrong answers\n   both come back at 0.9–0.95"]
for i, f in enumerate(facts):
    ax.text(52, 23.4 - i * 3.0 - (0.9 if i > 0 else 0), "•  " + f, fontsize=11.3, color=DARK, va="top",
            linespacing=1.25)
ax.text(52, 6.0, "The real problem is not extraction.\nIt is knowing when you are right.", fontsize=13.5,
        weight="bold", color=ORANGE, va="top", linespacing=1.3)
slides.append(fig)

# ------------------------------------------------------------------ 2. contracts architecture
fig, ax = slide("Contracts: the model proposes, evidence decides",
                "Method FINAL (CUAD, 41 clause fields) · 1 LLM call per contract · declared threshold 0.95", 2)
box(ax, 3, 37, 13, 6.5, "Contract\n(text)", fc="white", fs=11, weight="bold")
box(ax, 21, 37, 22, 6.5, "Claude Sonnet 5.5\nproposes value + confidence\nfor every field", fc="#fff1e8", ec=ORANGE,
    fs=10.5)
arrow(ax, 16.3, 40.2, 20.6, 40.2)
box(ax, 21, 27.5, 22, 6.5, "Split into clauses\n(character offsets, headings)\nmedian 86 per contract", fs=10)
arrow(ax, 9.5, 36.7, 20.6, 31)
arrow(ax, 18.3, 18, 20.6, 20.5)
box(ax, 21, 18, 22, 6.5, "bge-small embeddings (local)\n→ clause classifier\nP(clause is field type)", fs=10)
arrow(ax, 32, 27.2, 32, 24.8)
box(ax, 3, 8.5, 15, 16, "Trained on\n31,969 clauses\n(160 contracts),\nlabels = CUAD's\ngold evidence\n\nheld-out\nAUC 0.88–0.91", fs=9.5,
    fc="white")
box(ax, 21, 8.5, 22, 6.5, "Per-field trust, learned on\n60 calibration contracts:\nbook only if ≤ 1% errors", fs=10,
    fc="#eef6ee", ec=GREEN)
arrow(ax, 43.3, 40.2, 50, 38.5)
arrow(ax, 43.3, 21.2, 50, 26)
arrow(ax, 43.3, 11.7, 50, 15)
ax.text(52, 43.6, "Decision rules — first match wins, nothing averaged", fontsize=12.5, weight="bold", color=DARK)
rules(ax, 52, 40.3, [("model returned null", "ABSTAIN", GREY),
                     ("value not in the contract", "REJECT", RED),
                     ("clause is another type (p < 0.02)", "REJECT", RED),
                     ("clause type uncertain (p < 0.10)", "REVIEW", AMBER),
                     ("other clauses of this type left out", "REVIEW", AMBER),
                     ("fragment, not whole sentences", "REVIEW", AMBER),
                     ("field not trusted on calibration data", "REVIEW", AMBER),
                     ("model confidence < 0.95", "REVIEW", AMBER),
                     ("every check passed", "BOOK", GREEN)])
ax.text(52, 14.3, "Confidence is the last check, never the main one.", fontsize=11.5, color=DARK, weight="bold")
ax.text(52, 11.2, "Example — Non-Disparagement: the model quoted a real sentence about a domain\n"
                  "licence; its clause scores p = 0.0005 for that type → REJECT (field truly absent).",
        fontsize=10.2, color=GREY, va="top", linespacing=1.3)
ax.text(52, 6.0, "If the classifier cannot load, every contract field goes to review (fail-safe).",
        fontsize=10.2, color=GREY)
slides.append(fig)

# ------------------------------------------------------------------ 3. forms architecture
fig, ax = slide("Forms and scans: two independent readers must agree",
                "Method forms_final (ExtractBench) · 2 vision calls per form · threshold 0.95 · no OCR stack",
                3)
box(ax, 3, 30, 13, 9, "Form PDF\n(45% are scans,\nhandwriting,\ncheckboxes)", fc="white", fs=10.5, weight="bold")
box(ax, 21, 37, 21, 6.5, "Page images\n(pypdfium2)", fs=10.5)
box(ax, 21, 25.5, 21, 6.5, "Text layer + word positions\n(pdfplumber, if any)", fs=10)
arrow(ax, 16.3, 36, 20.6, 40)
arrow(ax, 16.3, 33, 20.6, 28.8)
box(ax, 47, 39.5, 22, 6.5, "Reader A: Claude Sonnet 5.5\nvalue, confidence, page, label", fc="#fff1e8", ec=ORANGE,
    fs=10)
box(ax, 47, 31, 22, 6.5, "Reader B: GPT-5.5\n(different company, never sees A)", fc="#e8f0fd", ec=BLUE, fs=10)
arrow(ax, 42.3, 41, 46.6, 42.7)
arrow(ax, 42.3, 39.5, 46.6, 34.5)
box(ax, 47, 20, 22, 7.5, "Text-layer checks:\nvalue printed on the page?\nnext to its printed label?", fs=10)
arrow(ax, 42.3, 28.8, 46.6, 24.5)
box(ax, 47, 9.5, 22, 7.5, "Per-field trust, learned on\n40 calibration forms:\n68 of 207 field types qualify", fs=10,
    fc="#eef6ee", ec=GREEN)
for yy in (42.7, 34.2, 23.7, 13.2):
    arrow(ax, 69.3, yy, 73.5, yy)
ax.text(74.5, 44.6, "Decision rules", fontsize=12.5, weight="bold", color=DARK)
rules(ax, 74.5, 41.4, [("both readers: nothing", "ABSTAIN", GREY),
                       ("one reader only / disagree", "REVIEW", AMBER),
                       ("not printed on the page", "REJECT", RED),
                       ("not next to its label", "REVIEW", AMBER),
                       ("field not trusted", "REVIEW", AMBER),
                       ("confidence < 0.95", "REVIEW", AMBER),
                       ("all passed", "BOOK", GREEN)], w=23, fs=9.2)
ax.text(74.5, 20.5, "Why two model families?", fontsize=11.5, weight="bold", color=DARK)
ax.text(74.5, 18.6, "A scan has no text to check against.\nTwo independently trained models\n"
                    "rarely make the same mistake:\n39 of 46 disagreements were\nreal reading errors.",
        fontsize=10, color=DARK, va="top", linespacing=1.3)
ax.text(3, 21.5, "Example — scanned\nTexas well-plugging\nrecord: both read\nCounty = Archer,\nDepth = 1042'\n→ BOOK.\n\n"
                 "“TE&L Co” vs\n“T&L Co” → REVIEW.", fontsize=10, color=GREY, va="top", linespacing=1.3)
slides.append(fig)

# ------------------------------------------------------------------ 4. results
fig, ax = slide("Results: from −140% to 98 of 99 bookings right",
                "Official scorer, declared threshold 0.95 · held-out and untouched data only", 4)
methods = [("Original pipeline @0.9", -140.0), ("Agentic RAG", -119.5), ("Hybrid retrieval (C)", -70.9),
           ("BM25 retrieval (A)", -70.5), ("Embeddings (B)", -58.9), ("Original pipeline @0.95", -10.4),
           ("+ LLM verifier (H)", 1.2), ("FINAL: evidence gates", 1.3)]
bx = fig.add_axes([0.20, 0.13, 0.27, 0.62])
names = [m for m, _ in methods]
vals = [v for _, v in methods]
cols = [GREEN if v > 0 else RED for v in vals]
cols[-1] = ORANGE
bx.barh(names, vals, color=cols)
bx.axvline(0, color=DARK, lw=1)
for i, v in enumerate(vals):
    bx.text(v + (2 if v >= 0 else -2), i, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9.5)
bx.set_xlim(-200, 40)
bx.set_title("Contracts: net saving vs. all-manual (20 held-out contracts)", fontsize=11.5, loc="left")
bx.spines[["top", "right"]].set_visible(False)
bx.tick_params(labelsize=9.5)
ax.text(52, 41.5, "Untouched public test (run_hidden.sh, gold removed)", fontsize=12.5, weight="bold", color=DARK)
tbl = [("", "booked", "match gold", "net saving"), ("Contracts (24)", "11", "11", "+1.1%"),
       ("Forms (16)", "88", "87*", "−1.4%"), ("All 40 docs", "99", "98*", "−0.1%")]
for r, row in enumerate(tbl):
    for c, cell in enumerate(row):
        ax.text(52 + [0, 17, 27, 38][c], 37.8 - r * 3.1, cell, fontsize=11.5, color=DARK,
                weight="bold" if r in (0, 3) else "normal")
ax.text(52, 24.5, "* The one mismatch is a gold-label error", fontsize=12, weight="bold", color=RED)
ax.text(52, 22.4, "The form prints “3583 CASTLEHILL WAY”; both readers copied it exactly;\n"
                  "the gold says “CASTLE HILL” (checked in the text layer, the image and by hand).\n"
                  "Counted by the scorer → break-even. Corrected → 99 / 99, +5.4%.",
        fontsize=10.5, color=DARK, va="top", linespacing=1.35)
ax.text(52, 12.6, "Held-out dev data (52 docs, 2,384 fields): 92 / 92 booked values correct,\n"
                  "0 hallucinations, net saving +3.9% — the same at an error cost of 20, 100 or 500.",
        fontsize=10.5, color=DARK, va="top", linespacing=1.35)
ax.text(52, 7.4, "Forms alone (32 held-out): 81 / 81 correct, +5.2%.\n"
                 "Without the field-trust rule: 3 wrong, −1.8%.", fontsize=10, color=GREY, va="top", linespacing=1.3)
slides.append(fig)

# ------------------------------------------------------------------ 5. what we learned
fig, ax = slide("What we really learned about the problem",
                "Every point below was measured on held-out data", 5)
points = [
    ("Confidence is not evidence.", "Raising the threshold from 0.9 to 0.95 even lowered one method's precision "
                                    "(71.7% → 68.4%). Independent checks decide; confidence is the last one."),
    ("The main error was incomplete, not wrong.", "CUAD's gold lists every passage (2–5) for a clause; the model "
                                                  "quotes one. The completeness rule catches these."),
    ("More LLM passes did not help.", "Three verifier models and an agent added cost; better retrieval raised raw "
                                      "accuracy (77% → 80%) but not booking safety."),
    ("Disagreement is a strong signal.", "On forms, 39 of 46 disagreements between two model families were real "
                                         "errors. But agreement alone still let 3 wrong values through."),
    ("Learned per-field trust makes booking safe — and is now the bottleneck.",
     "Scoring the evidence sentence fixed 8–20 false “uncertain” verdicts, yet bookings stayed 11 → 11: "
     "those fields lack calibration evidence."),
    ("Gold labels have errors too.", "At a 99% target, one label error decides break-even vs. +5%."),
]
for i, (head, body) in enumerate(points):
    y = 43.2 - i * 6.15
    ax.add_patch(plt.Rectangle((3, y - 3.1), 0.7, 5.3, color=ORANGE))
    ax.text(5, y + 1.3, "\n".join(textwrap.wrap(head, 95)), fontsize=11.8, weight="bold", color=DARK, va="center")
    ax.text(5, y - 0.4, "\n".join(textwrap.wrap(body, 98)), fontsize=9.8, color=GREY, va="top", linespacing=1.25)
box(ax, 67.5, 8.5, 29.5, 34.5, "", fc="#fff1e8", ec=ORANGE)
ax.text(69, 40.3, "Next steps", fontsize=13, weight="bold", color=DARK)
nxt = ["Calibrate on all 312 / 127 training\ndocuments → more trusted fields",
       "Ask for all passages of a clause\n(the main contract error)",
       "Normalise number formats in the\nforms check (43 correct values lost)",
       "Use reader B when reader A misses\na value (229 cases)"]
for i, t in enumerate(nxt):
    ax.text(69, 37.2 - i * 5.2, "•  " + t, fontsize=10.3, color=DARK, va="top", linespacing=1.25)
ax.text(69, 15.3, "How SIXT could roll it out", fontsize=11.5, weight="bold", color=DARK)
ax.text(69, 13.4, "Shadow mode first: humans see suggestions\nand reasons; switch on booking field by field\n"
                  "once its measured error rate < 1 / error cost.", fontsize=9.8, color=DARK, va="top",
        linespacing=1.3)
ax.text(3, 4.3, "Demo: upload a document → every field's decision, reason and a citation box in the original PDF "
                "(Project_1/demo_app.py).", fontsize=10.5, color=DARK)
slides.append(fig)

with PdfPages(OUT / "slides.pdf") as pdf:
    for i, f in enumerate(slides, start=1):
        pdf.savefig(f)
        f.savefig(OUT / f"preview_{i}.png", dpi=70)
print("wrote", OUT / "slides.pdf")

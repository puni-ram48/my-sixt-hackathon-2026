# Teach a machine to say "I don't know"
**Team Zero to SIXTy · SIXT Hackathon 2026 · Project 1: Structured extraction from documents**

People type contract and form data into SIXT's systems by hand. An AI can do it in seconds, but one confidently wrong
value costs far more than a minute of human time. In the hackathon's cost model a wrong booked value costs **100** and
a human review costs **1**, so automation only pays off if booked values are right more than **99%** of the time.

**Our idea: the model proposes, evidence decides.** A value is booked automatically only when independent evidence
supports it; everything else goes to a human. The model's own confidence is the last check, never the main one.

## Results
Untouched public-test check: `run_hidden.sh` on 24 contracts and 16 forms, gold labels removed, scored with the
official `5_score.py` at our declared threshold **0.95**. This data was never used for any choice.

| | Booked automatically | Match gold | Don't match gold | Precision | Net saving vs. all-manual |
|---|---:|---:|---:|---:|---:|
| Contracts (CUAD, 24) | 11 | 11 | 0 | 100% | +1.1% |
| Forms and scans (ExtractBench, 16) | 88 | 87 | 1\* | 98.9% | −1.4% |
| **Total (40 documents, 1,845 fields)** | **99** | **98** | **1\*** | **99.0%** | **−0.1%** |
| *For comparison: original pipeline on held-out contracts, threshold 0.9* | *52* | *40* | *12* | *76.9%* | ***−140%*** |

\* **A gold-label error, not a reading error.** The form prints `3583 CASTLEHILL WAY`; both of our readers copied
exactly that, but the gold label says `CASTLE HILL`. We confirmed it in the PDF's text layer, in the rendered image
and by hand ([EDA, section 9](Project_1/EDA.md#9-a-gold-label-error-and-a-deterministic-way-to-check-gold-against-the-document)).
The scorer still counts it, and one counted error costs as much as 100 correct bookings: with it the result is
break-even (−0.1%); without it, all 99 bookings are right (+5.4%). We report the scorer's number.

Held-out dev-val (52 documents, 2,384 fields): 92 of 92 booked values correct, 0 hallucinations, net saving +3.9%.

## How it works
**Contracts → evidence-gated extraction.** One Claude Sonnet 5.5 call proposes a value for each of the 41 clause fields.
The contract is split into clauses, and a clause-type classifier (trained on 32,000 labelled clauses) scores each one.
A value is booked only if all of these hold:
- it is in the text;
- its clause is the right type;
- no other clause of that type is missing from the answer;
- it is made of whole sentences;
- its field was error-free on 60 calibration contracts;
- confidence is ≥ 0.95.

**Forms and scans → two independent readers.** Claude Sonnet 5.5 and GPT-5.5 read the page images separately, and
both must agree. On text-layer PDFs the value must also be printed on the page, next to its label. Its field must have
been error-free on 40 calibration forms.

**What did not help** (measured on the same held-out data, details in the results):
- better retrieval (BM25, embeddings, hybrid);
- LLM verifiers from three model families;
- an agentic RAG approach.

## Where to look
| Document | What it contains |
|---|---|
| [**Project_1/SUBMISSION.md**](Project_1/SUBMISSION.md) | Official submission: approach, models and why, threshold, results, how to run, next steps |
| [Project_1/results/RESULTS.md](Project_1/results/RESULTS.md) | All 12 contract methods and both form methods compared; why the winners won |
| [Project_1/EDA.md](Project_1/EDA.md) | Data analysis: what the documents look like, issues found, what the experiments answered |
| [Project_1/README.md, "Our solution"](Project_1/README.md#our-solution-evidence-gated-hybrid-extraction-hybrid) | Architecture, examples, full commands to reproduce |
| [submission/](submission/) | Slides (`slides.pdf`) |

## Run it
**No training and no manual model installation needed.** Everything the system learned is in the repo; the rest is
installed or downloaded automatically.

| Component | Where it comes from |
|---|---|
| Clause classifier + field policies (our trained parts) | committed in `Project_1/hybrid/models/`, loaded automatically |
| Python packages | `pip install -r requirements.txt` (done by `setup.sh`) |
| Embedding model `BAAI/bge-small-en-v1.5` (~130 MB) | downloaded automatically on the first run, then cached; needs internet once |
| LLMs (Claude Sonnet 5.5, GPT-5.5) | API calls to the endpoint you put in `.env` |
| Datasets | downloaded by `1_download.sh` + `2_prepare.py` (not redistributed in this repo) |

**1. Set up (once, about 10 minutes, Python 3.11+):**
```bash
cd Project_1
bash setup.sh --accept-licenses      # creates .venv, installs requirements, creates .env, downloads and prepares the data
```
Then open `Project_1/.env` and fill in `LLM_BASE_URL` and `LLM_API_KEY` (`LLM_MODEL=claude-sonnet-5.5` is pre-filled).
Never commit `.env`. On Windows without bash `.venv/bin`, `run_hidden.sh` also finds `.venv\Scripts\python.exe`.

**2. Try it on 3 documents (about 2 minutes):**
```bash
head -n 3 data/unified/test_public.jsonl > sample.jsonl
bash run_hidden.sh sample.jsonl sample_out.jsonl
python 5_score.py --gold sample.jsonl --pred sample_out.jsonl --threshold 0.95
```

**3. Run on any test file** (same JSONL format as `data/unified/*.jsonl`; source files under `data/raw/`):
```bash
bash run_hidden.sh <input.jsonl> <output.jsonl>
```
It writes two files:
- `<output>.jsonl`: one line per document, `{"doc_id", "fields": {name: {"value", "confidence"}}}`. A field counts as
  booked when its confidence is ≥ 0.95, our declared threshold.
- `<output>.trace.jsonl`: the decision and reason for every field (BOOK / REVIEW / REJECT / ABSTAIN), so any value
  can be audited.

Contracts and forms are routed automatically. Four documents run in parallel; a document takes about 10–45 s of
LLM time. If the embedding model cannot be downloaded, contract fields fail safe: they go to human review rather
than being booked with fewer checks.

Reproducing our experiments (training the classifier, calibration, all method comparisons) is optional; the commands
are in [Project_1/README.md, "Reproduce the evaluation"](Project_1/README.md#reproduce-the-evaluation).

## Repository layout
```
Project_1/
├── SUBMISSION.md, EDA.md     submission documents
├── run_hidden.sh             hidden-test entry point (routes contracts and forms)
├── hybrid/                   our system: all methods, evaluation, trained classifier and field policies
├── solution/                 the original baseline (kept for comparison)
├── results/                  method comparisons and reports
├── tests/                    offline tests (python -m unittest tests.test_hybrid tests.test_verify)
└── 1_download.sh … 5_score.py   organiser starter kit (data download, scorer)
```

Built on the [SIXT Hackathon 2026](https://github.com/sixt-hackathon-2026/sixt-hackathon-2026) starter repository.
Datasets: CUAD (CC BY 4.0) and ExtractBench (Apache-2.0); credits in [Project_1/README.md](Project_1/README.md#credits).
Data is downloaded by the scripts and not redistributed here.

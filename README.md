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
| [Project_1/README.md](Project_1/README.md) | Technical write-up: what we did step by step, both methods, decision rules, results, code map, reproduce |
| [submission/](submission/) | Slides (`slides.pdf`) |

## Setup and run (step by step)

### What you need
| Requirement | Details |
|---|---|
| Python | 3.11 or newer (we used 3.12.3) |
| Shell | `bash` (on Windows: Git Bash) and `git` |
| Internet | once, to install packages, download the data (~60 MB) and the embedding model (~130 MB) |
| **LLM endpoint + API key** | any OpenAI-compatible endpoint (`/chat/completions`) that serves the two models below; the hackathon key works |
| **Model 1: `claude-sonnet-5.5`** | contract extractor and first form reader (set as `LLM_MODEL`) |
| **Model 2: `gpt-5.5`** | second, independent form reader (`FORMS_SECOND_MODEL`, default `gpt-5.5`) |

**Nothing to train and no model to install by hand.** Our trained clause classifier and field policies are committed
in `Project_1/hybrid/models/`. The embedding model `BAAI/bge-small-en-v1.5` is downloaded automatically on the first
run, and the LLMs are called through your endpoint.

### Quick setup (about 10 minutes)
```bash
git clone https://github.com/puni-ram48/my-sixt-hackathon-2026.git
cd my-sixt-hackathon-2026/Project_1
bash setup.sh --accept-licenses     # makes .venv, installs requirements, creates .env, downloads + prepares the data
```
Then put your endpoint and key in `Project_1/.env` (see below).

### Or step by step
```bash
cd my-sixt-hackathon-2026/Project_1
python -m venv .venv
source .venv/bin/activate            # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt
bash 1_download.sh --accept-licenses # CUAD + ExtractBench into data/raw/ (licences: CC BY 4.0, Apache-2.0)
python 2_prepare.py                  # builds data/unified/dev.jsonl and test_public.jsonl
cp .env.example .env                 # then fill in the values below
```

### LLM access: `Project_1/.env`
`.env` is git-ignored; never commit it. `Project_1/.env.example` contains (comments on their own lines; do not
put comments after a value):
```bash
# Copy to .env (git-ignored, never commit it). Endpoint URL and key: from your LLM provider (hackathon: handed out on site).
LLM_BASE_URL=
LLM_API_KEY=
# Extractor for contracts and first reader for forms (what we used and validated):
LLM_MODEL=claude-sonnet-5.5
# Optional. Second, independent reader for forms (default gpt-5.5):
# FORMS_SECOND_MODEL=gpt-5.5
# Optional. Send every form field to a human instead of the forms pipeline:
# HYBRID_NONCONTRACT=review
```
| Variable | Required | Meaning |
|---|---|---|
| `LLM_BASE_URL` | yes | endpoint base URL; the code calls `<LLM_BASE_URL>/chat/completions` |
| `LLM_API_KEY` | yes | sent as `Authorization: Bearer <key>` |
| `LLM_MODEL` | yes | `claude-sonnet-5.5` (what we validated) |
| `FORMS_SECOND_MODEL` | no | default `gpt-5.5`; must be a different model from `LLM_MODEL` |
| `HYBRID_NONCONTRACT` | no | `forms_final` (default), `review`, or `baseline` (original pipeline) |

### Check the setup (no API calls)
```bash
python -m unittest tests.test_hybrid tests.test_verify     # expected: OK
```

### Try it on 3 documents (about 2 minutes, 5 API calls): 1 contract + 2 forms
```bash
{ grep '"doc_id": "cuad:' data/unified/test_public.jsonl | head -n 1
  grep '"doc_id": "extractbench:' data/unified/test_public.jsonl | head -n 2; } > sample.jsonl
bash run_hidden.sh sample.jsonl sample_out.jsonl
python 5_score.py --gold sample.jsonl --pred sample_out.jsonl --threshold 0.95
```

### Run on a test set
```bash
bash run_hidden.sh <input.jsonl> <output.jsonl>
python 5_score.py --gold <input_with_gold.jsonl> --pred <output.jsonl> --threshold 0.95   # if you have gold labels
```
- **Input:** the organisers' format, one JSON per line (`doc_id`, `pages`, `schema`, and `gold` if available), with
  source files under `Project_1/data/raw/`.
- **Output** `<output>.jsonl`: one line per document, `{"doc_id": ..., "fields": {name: {"value": ..., "confidence": ...}}}`.
  **Declared threshold: 0.95.** A field is booked when its confidence is ≥ 0.95; `null` means "sent to a human".
- **Decision log** `<output>.trace.jsonl`: for every field, the decision (BOOK / REVIEW / REJECT / ABSTAIN) and why.
- **Routing:** contracts and forms are routed automatically, and 4 documents run in parallel. Expect about 10–45 s of
  LLM time per document: 1 call per contract, 2 per form.

### Troubleshooting
| Symptom | Cause and fix |
|---|---|
| `LLM_BASE_URL, LLM_API_KEY and LLM_MODEL must be set` | fill in `Project_1/.env` |
| HTTP 429 in the log | rate limit; the client retries automatically. If it keeps failing, those fields go to review (safe) |
| `setup.sh` fails at `.venv/bin/activate` (Windows) | use the step-by-step commands with `source .venv/Scripts/activate` |
| "embeddings unavailable" / "clause classifier unavailable" warning | the embedding model could not be downloaded; contract fields then go to human review (fail-safe). Allow access to huggingface.co once |
| Form fields all `null` | `gpt-5.5` is not available on your endpoint: set `FORMS_SECOND_MODEL` to another model, or accept review |

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

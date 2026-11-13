# Teach a machine to say "I don't know"
**Team Zero to SIXTy · SIXT Hackathon 2026 · Project 1: Structured extraction from documents**

People type contract and form data into SIXT's systems by hand. An AI can do it in seconds, but one confidently wrong
value costs far more than a minute of human time. In the hackathon's cost model a wrong booked value costs **100** and
a human review costs **1**, so automation only pays off if booked values are right more than **99%** of the time.

**Our idea: the model proposes, evidence decides.** A value is booked automatically only when independent evidence
supports it; everything else goes to a human. The model's own confidence is the last check, never the main one.

## Results
Untouched public-test check: `run_hidden.sh` on 20 contracts and 12 forms, gold labels removed, scored with the
official `5_score.py` at our declared threshold **0.95**. This data was never used for any choice.

| | Booked automatically | Correct | Wrong | Net saving vs. all-manual |
|---|---:|---:|---:|---:|
| Contracts (CUAD) | 9 | 9 | 0 | +1.1% |
| Forms and scans (ExtractBench) | 73 | 73 | 0 | +11.8% |
| **Total (1,439 fields)** | **82** | **82** | **0** | **+5.7%** |
| *For comparison: original pipeline on held-out contracts, threshold 0.9* | *52* | *40* | *12* | ***−140%*** |

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
```bash
cd Project_1
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash 1_download.sh --accept-licenses && python 2_prepare.py
cp .env.example .env                                   # add LLM_BASE_URL, LLM_API_KEY, LLM_MODEL (never commit it)
bash run_hidden.sh <input.jsonl> <output.jsonl>        # the submission entry point
python 5_score.py --gold data/unified/dev.jsonl --pred <output.jsonl> --threshold 0.95
```

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

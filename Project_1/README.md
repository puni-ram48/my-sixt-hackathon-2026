# Project 1: Teach a machine to say "I don't know"
**Team Zero to SIXTy · SIXT Hackathon 2026**

This is the technical write-up of what we built. For a one-page overview and step-by-step setup, see the
[repository README](../README.md); the official submission is [SUBMISSION.md](SUBMISSION.md).

## The task (short version)
Given a document (a contract, form or scan) and a schema of fields, return a **value and a confidence** for every
field, or `null` when the field is absent or the system is not sure. A value is **booked** automatically when its
confidence is ≥ the team's declared threshold; everything else goes to a human.

| Outcome | Cost |
|---|---|
| booked and correct | saves one review (1) |
| booked but wrong, or invented for an absent field | **100** |
| sent to a human | 1 |

So automation only pays off if booked values are right more than **99%** of the time. The data comes from two public
datasets: **CUAD** (510 contracts, 41 clause fields each) and **ExtractBench** (206 real forms and scans, 2–159
fields each). The original challenge text is in the
[hackathon repository](https://github.com/sixt-hackathon-2026/sixt-hackathon-2026/blob/main/Project_1/README.md).

## What we did, step by step
1. **Baseline** (`solution/`). One LLM call per document returns value + confidence; values that cannot be found in
   the document text are nulled. On 20 held-out contracts at threshold 0.9 it booked 52 values, 12 of them wrong:
   **net saving −140%**. The model's confidence did not separate right from wrong answers.
2. **Data analysis** ([EDA.md](EDA.md)). We found that:
   - 68% of contract fields are empty, and 28% of contracts are longer than the prompt limit;
   - 45% of the form PDFs are scans with no text;
   - keyword search finds the right clause for only about half of the fields.
3. **Hybrid retrieval experiments** (configs A–D). We split contracts into clauses, retrieved candidates with BM25,
   embeddings and a supervised clause classifier, extracted from the candidates only, and tried an LLM verifier.
4. **Agentic RAG** (configs E–F). The LLM searches and reads the contract itself.
5. **Error analysis.** The main booked error was not the wrong clause but an **incomplete answer**: the gold lists
   every passage (2–5), and the model quotes one. Verifiers and agents did not fix this.
6. **Evidence gates** (config G → `FINAL`). We kept the baseline extraction and added independent checks: clause
   type, completeness, and **per-field trust learned on calibration data**. This is the first configuration that
   saves money.
7. **Forms pipeline** (`forms_final`). Two vision models from different families read each form independently,
   with text-layer checks and learned per-field trust.
8. **Validation.** We validated on held-out dev data and then ran an untouched check on public-test data through
   `run_hidden.sh`.

## Data and how we used it (no overlap between roles)
| Role | CUAD contracts | ExtractBench forms |
|---|---|---|
| train the clause classifier | dev-train 0–159 (160 contracts, 31,969 clauses) | – (nothing trained) |
| calibrate (learn which fields can be trusted) | dev-train 160–219 (60) | dev-train 0–39 (40) |
| compare methods (held-out dev-val) | first 20 of 69 (820 fields) | all 32 (1,564 fields) |
| untouched final check (test_public) | first 24 (984 fields) | first 16 (861 fields) |

`hybrid/split.py` splits the dev set 4:1 into train and val by a hash of the document id. The 10 contracts we read
by hand early on always go to train.

## Method 1: contracts → evidence-gated extraction (`FINAL`)
```
contract ─► Claude Sonnet 5.5 (original prompt, 1 call): value + confidence for 41 fields
         ─► split into clauses with character offsets (median 86 per contract)          hybrid/clauses.py
         ─► embed clauses (bge-small, local) → classifier P(clause is field type)       hybrid/embed.py, classify.py
         ─► evidence signals per value                                                   hybrid/decide.py
         ─► decision rules (first match wins, nothing averaged) ─► BOOK / REVIEW / REJECT / ABSTAIN
```
| # | Rule | Result |
|---|---|---|
| 1 | model returned null | ABSTAIN |
| 2 | value not found in the contract (grounding < 0.9) | REJECT (invented) |
| 3 | classifier: the clause holding the value is not this type (p < 0.02) | REJECT (wrong clause) |
| 4 | classifier unsure (p < 0.10) | REVIEW |
| 5 | other clauses look like this field (p ≥ 0.5) but are not in the answer | REVIEW (incomplete) |
| 6 | long answer that is not whole sentences | REVIEW (fragment) |
| 7 | field not reliable on calibration contracts (needs ≥ 8 bookings with ≤ 1% errors) | REVIEW |
| 8 | model confidence < 0.95 | REVIEW |
| 9 | everything passed | **BOOK** |

- **Clause classifier:** 41 logistic-regression heads on frozen `bge-small` embeddings. Labels come from CUAD's own
  gold evidence spans. Held-out AUC is 0.88–0.91 (Governing Law AP 0.93, Cap On Liability 0.74); it is weak on rare
  clause types with fewer than 25 examples. As a retriever it puts the gold clause in the top 8 for 89–92% of fields,
  vs 57–59% for BM25.
- **Field policy:** learned on the 60 calibration contracts. Only **Document Name (27/27)** and **Governing Law
  (21/21)** qualified; every other contract field goes to a human.
- **Fail-safe:** if the classifier cannot run (no embedding model), every contract field goes to review.

## Method 2: forms and scans → two independent readers (`forms_final`)
```
PDF ─► page images (pypdfium2) + text-layer words with positions (pdfplumber)
    ─► reader A: Claude Sonnet 5.5 (vision)     ─┐  value, confidence, page,
    ─► reader B: GPT-5.5 (vision), independent  ─┘  printed label next to the value
    ─► decision rules ─► BOOK / REVIEW / REJECT / ABSTAIN                                hybrid/forms.py
```
| # | Rule | Result |
|---|---|---|
| 1 | both readers say null | ABSTAIN |
| 2 | only one reader found a value, or the readers disagree | REVIEW |
| 3 | text-layer PDF: value not printed on the page | REJECT |
| 4 | text-layer PDF: value found, but not next to its printed label | REVIEW |
| 5 | field not reliable on calibration forms (≥ 5 bookings, ≤ 1% errors; 68 of 207 field types qualify) | REVIEW |
| 6 | min(confidence A, confidence B) < 0.95 | REVIEW |
| 7 | everything passed | **BOOK** |

For a scan there is no text to check against, so agreement between two independently trained models is the
evidence. We chose vision LLMs over an OCR stack (PaddleOCR, LayoutParser) because those are slow to install and run
on CPU and weak on historical handwriting.

## Results
### Contracts: 20 held-out contracts (dev-val, 820 fields)
| Config | Booked @0.95 | Precision | Net saving @0.95 | Net saving @0.9 | LLM calls/doc |
|---|---:|---:|---:|---:|---:|
| baseline (whole document, trust LLM confidence) | 15 | 93.3% | −10.4% | −140.0% | 1.0 |
| A: BM25 → LLM | 22 | 72.7% | −70.5% | −177.6% | 3.0 |
| B: embeddings → LLM | 17 | 70.6% | −58.9% | −214.1% | 3.0 |
| C: hybrid retrieval → LLM + clause gates | 19 | 68.4% | −70.9% | −176.5% | 3.0 |
| D: C + LLM verifier | 19 | 68.4% | −70.9% | −176.5% | 4.5 |
| E/F: agentic RAG (± verifier) | 20 | 50.0% | −119.5% | −322.3% | 3.5–4.9 |
| H: G + verifier (Claude / GPT-5.5 / Gemini 3.1 Pro) | 10 | 100% | +1.2% | −10.1% | 2.4 |
| **FINAL (= G at threshold 0.95)** | **11** | **100%** | **+1.3%** | – | **1.0** |

### Forms: 32 held-out forms (dev-val, 1,564 fields)
| Config | Booked @0.95 | Precision | Wrong booked | Net saving @0.95 | LLM calls/doc |
|---|---:|---:|---:|---:|---:|
| two readers must agree, no field policy | 372 | 98.9% | 3 | −1.8% | 2 |
| **forms_final (two readers + field policy)** | **81** | **100%** | **0** | **+5.2%** | 2 |

### Submitted system
- **Held-out dev-val** (52 documents, 2,384 fields, official `5_score.py --threshold 0.95`): precision 100%,
  0 hallucinations, coverage 7.3%, **net saving +3.9%**.
- **Untouched public-test check** (`run_hidden.sh`, gold removed, 40 documents, 1,845 fields):

| Dataset | Booked | Match gold | Precision | Net saving |
|---|---:|---:|---:|---:|
| contracts (24) | 11 | 11 | 100% | +1.1% |
| forms (16) | 88 | 87 | 98.9% | −1.4% |
| **all** | **99** | **98** | **99.0%** | **−0.1%** |

The one mismatch is a **gold-label error**. The form prints `3583 CASTLEHILL WAY`; both readers copied that exactly,
but the gold says `CASTLE HILL` (verified in the text layer, the rendered image and by hand: [EDA.md](EDA.md)
section 9, `python -m hybrid.gold_check`). The scorer counts it, so the official result is break-even; with the label
corrected, all 99 bookings are right (+5.4%).

All 12 contract configs and both forms configs, with threshold sweeps and per-rule tables:
[results/RESULTS.md](results/RESULTS.md).

## What we learned
1. **Trusting the model's confidence loses money.** Raising C's threshold from 0.9 to 0.95 even *lowered* its
   precision. Confidence works as the last check, not the main one.
2. **Incomplete answers, not wrong clauses, were the main booked error on contracts.** The completeness rule caught 20
   of them against 3 correct ones; the clause-type check rejected 31 bad values against 10 good ones.
3. **Better retrieval raised raw accuracy (79.5% vs 77.4%) but not booking safety.**
4. **LLM verifiers (three model families) and an agentic approach added cost without benefit.**
5. **Disagreement between two model families is a strong error signal on forms** (39 of 46 disagreements were real
   errors), but agreement alone is not enough.
6. **Learned per-field trust is what makes booking safe on both datasets.** The price is low coverage.
7. **Gold labels contain errors.** At a 99% target, a single label error decides whether the measured result is above
   or below break-even.

## Examples (from the decision traces)
- **Correct booking.** *Governing Law*, Hertz IP agreement: found verbatim as a whole sentence, its clause has
  p=0.97, the field is trusted (21/21), confidence 0.95 → BOOK. Matches gold.
- **Wrong-clause rejection.** *Non-Disparagement*, China Real Estate domain licence: the model proposed a real
  sentence about not tarnishing the licensed domain names, but its clause has p=0.0005 for Non-Disparagement → REJECT.
  Gold: absent.
- **Incomplete, sent to review.** *Parties*, same contract: the answer names two companies, but clauses C2, C178 and
  C1 also look like party clauses → REVIEW. The gold also lists "Licensor" and "Licensee".
- **Abstention.** *Expiration Date*, Thrivent endorsement: the best clause has p=0.02 and the model returns null →
  ABSTAIN. Gold: absent.
- **Forms.** On a scanned Texas well-plugging record both readers read County `Archer` and Total Depth `1042'` → BOOK.
  When they read `TE&L Co` vs `T&L Co` → REVIEW.

## Code map
| Path | Purpose |
|---|---|
| `run_hidden.sh` | hidden-test entry point; routes contracts to `FINAL` and forms to `forms_final`, declared threshold 0.95 |
| `hybrid/pipeline.py`, `hybrid/run.py` | configurations A–H and FINAL, routing, CLI (4 documents in parallel) |
| `hybrid/clauses.py`, `fields.py` | clause segmentation with offsets; field definitions from the dataset |
| `hybrid/bm25.py`, `embed.py`, `retrieve.py` | BM25, local embeddings, weighted RRF retrieval with explanations |
| `hybrid/classify.py`, `field_policy.py` | clause classifier (train + load); per-field trust (fit + load) |
| `hybrid/extract.py`, `verify.py`, `agent.py` | evidence extraction, LLM verifier, agentic RAG (experiments) |
| `hybrid/decide.py` | decision rules for contracts |
| `hybrid/forms.py` | forms pipeline: rendering, two readers, text-layer and label checks, decision rules |
| `hybrid/evaluate.py`, `eval_retrieval.py`, `split.py` | evaluation framework, retrieval eval, data splits |
| `hybrid/gold_check.py` | deterministic check of gold labels against the printed text |
| `hybrid/llm.py` | LLM client with on-disk cache, token and latency logging |
| `hybrid/models/` | **committed trained artefacts**: clause classifier, contract and forms field policies |
| `solution/` | original baseline, kept for comparison (`python -m solution.run`) |
| `results/` | method comparisons and reports |
| `tests/` | offline tests (`python -m unittest tests.test_hybrid tests.test_verify`) |
| `review_app.py` | optional Streamlit app for reviewing answers by hand (`pip install streamlit==1.67.0` first) |
| `1_download.sh`, `2_prepare.py`, `selection.tsv`, `3_baseline.py`, `5_score.py`, `test_score.py`, `setup.sh` | organiser starter kit (data download, no-AI reference, scorer) |

## Run
Setup, API keys and models are described step by step in the [repository README](../README.md#setup-and-run-step-by-step).
In short, from `Project_1/`:
```bash
bash setup.sh --accept-licenses                 # venv, requirements, .env, data
# fill in LLM_BASE_URL and LLM_API_KEY in .env  (LLM_MODEL=claude-sonnet-5.5; forms also use gpt-5.5)
bash run_hidden.sh <input.jsonl> <output.jsonl> # writes <output>.jsonl and <output>.trace.jsonl
python 5_score.py --gold <input_with_gold.jsonl> --pred <output.jsonl> --threshold 0.95
```
Nothing needs training: the trained classifier and field policies are in `hybrid/models/`, and the embedding model
downloads automatically on the first run.

## Reproduce the evaluation
```bash
python -m hybrid.classify                       # train the clause classifier on dev-train (CPU, ~1 h first time, then cached)
python -m hybrid.eval_retrieval --part val --limit-docs 20          # retrieval hit@k, no API
python -m hybrid.evaluate --part train --offset 160 --limit-docs 60 --configs G_baseline_gated
python -m hybrid.field_policy --run-dir data/runs/hybrid_eval/dev_train_60_from160 --config G_baseline_gated
python -m hybrid.evaluate --part val --limit-docs 20                # all contract configs -> report.md + metrics.json
python -m hybrid.evaluate --part val --limit-docs 20 --replay       # recompute metrics from the cache, no API
python -m hybrid.evaluate --part train --source extractbench --limit-docs 40 --configs forms_dual
python -m hybrid.forms fit --run-dir data/runs/hybrid_eval/dev_train_40_extractbench --config forms_dual
python -m hybrid.evaluate --part val --source extractbench --limit-docs 32 --configs forms_dual,forms_dual_policy
python -m hybrid.gold_check --survey                                # gold labels vs printed text, no API
```
LLM replies are cached in `data/cache/llm/` with their original token counts and latency, so re-runs are free.
Outputs go to `data/runs/hybrid_eval/` (git-ignored); the summaries we report are copied to `results/`.

## Limitations
- **Small evaluation sets** (20 + 32 held-out documents, 40 public-test documents). One booked error moves net saving
  by about 5–12 points.
- **Low coverage.** Only 2 of 41 contract fields and 68 of 207 form field types are trusted; everything else goes to
  a human. That is safe, but the saving is small.
- **Truncation.** The contract extractor reads the first 60,000 characters; 28% of contracts are longer.
- **New form templates** go entirely to review until they are calibrated.
- **The forms text-layer check** rejected 43 correct values on dev-val (formatting such as `$1,234` vs `1234`).
- **CPU embedding** (~55 ms per clause) is the slowest local step; it is cached per document.

## Credits
Keep these in anything you publish. The organisers converted both datasets to one format.
- Zhang, Lyjak, Stewart, Li, Suo (2026). *ExtractBench: A Benchmark for Schema-Guided Enterprise Document Extraction.* arXiv:2607.29677.
- Hendrycks, Burns, Chen, Ball (2021). *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.* NeurIPS Datasets and Benchmarks.

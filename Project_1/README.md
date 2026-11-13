# Project 1: Teach a machine to say "I don't know"
*Structured extraction from documents: turn messy PDFs and scans into JSON, and know which fields to trust.*

**Project owners:** Sruthi Radhakrishnan, Ihsan Soydemir (sruthi.radhakrishnan@sixt.com, ihsan.soydemir@sixt.com). On the day, ask on site.

## The story
Every day documents arrive: contracts, forms, scans, handwriting. Today people read them and type the values into a system. An AI can do it in seconds, but there is a catch: **one confidently wrong value that reaches the system of record costs far more than the minute a person would have spent.** So the useful system is not the one that answers most often. It is the one that **knows when it is right**: it books the fields it is sure of and hands the rest to a human.

## The task
You get a document and a schema (a list of fields, e.g. a contract clause or a form box). For every field, return a **value and a confidence**. A field that is not in the document must come back empty.

```
+---------------------------+   +---------------------------+
|          document         |   |           schema          |
|      PDF, scan, photo     |   |     the fields to fill    |
+-------------+-------------+   +-------------+-------------+
              |                               |
              +---------------+---------------+
                              |
                              v
              +-------------------------------+
              |          YOUR SYSTEM          |
              |      model, tools, checks     |
              +-------------------------------+
                              |
     a value + a confidence, for every field
                              |
                              v
              +-------------------------------+
              | confidence >= your threshold? |
              +-+---------------------------+-+
                |                           |
               yes                          no
                v                           v
    +-----------------------+   +-----------------------+
    |  booked automatically |   |    sent to a human    |
    | right: saves a review |   |     costs 1 review    |
    |    wrong: costs 100   |   |   (the safe choice)   |
    +-----------+-----------+   +-----------+-----------+
                |                           |
                +-------------+-------------+
                              |
                              v
      5_score.py: net saving vs. doing it all by hand
```

This is where models struggle: layouts vary, scans are bad, handwriting and checkboxes get misread, and models happily invent a plausible value for a blank field.

## The business case
Everyone is scored with the same fixed costs (a simplified model, not SIXT figures): a human review of one field costs **1**, a wrong value that gets booked costs **100**. So automating only pays off if you are right more than **99%** of the time.

Example with 100 fields. Doing it all by hand costs 100. Your system books 90 and sends 10 to a human (cost 10):
- all 90 booked values are right: you pay 10 and save **90%**.
- just 1 booked value is wrong: you pay 10 + 100 = 110, which is **more than by hand**.

One confident mistake can wipe out the saving. That is why knowing when you are right matters more than raw accuracy.

| Outcome | Meaning | Cost |
| --- | --- | --- |
| wrong | accepted value differs from gold | error cost |
| hallucinated | accepted value, but the field is truly absent | error cost, hurts trust most |
| missed | no value, but the field is there | one review |
| true null | no value, field truly absent | one review (no saving, no loss) |

## Quick start

<img src="../assets/time-quick-setup.svg" alt="Under 5 minutes" width="360">

**Quick setup (under 5 min):** `bash setup.sh` in this folder. It makes a virtual environment, downloads the data (add `--accept-licenses` to accept the dataset licences) and runs the baseline. Copy `.env.example` to `.env` for your LLM key. Python is our starting point; any language is fine.

Or step by step:
```bash
bash 1_download.sh        # about 60 MB, asks you to accept the dataset licences
python 2_prepare.py       # builds data/unified/dev.jsonl and test_public.jsonl
python 3_baseline.py --train data/unified/dev.jsonl --target data/unified/test_public.jsonl --out preds.jsonl
python 4_extract_example.py --data data/unified/dev.jsonl --out preds.jsonl --limit 10   # naive starter
python 5_score.py --gold data/unified/dev.jsonl --pred preds.jsonl --threshold 0.95
```
Python 3.11+, standard library only (use `python3` if `python` is not found). `pip install -r requirements.txt` is optional (`jsonschema`, `pypdf` so step 4 can read PDFs). The starter uses crude keyword rules and scores about nothing: read it before building on it.

**LLM access:** we email you an API key, endpoint URL and model list on site. Put them in `.env` as `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (any OpenAI-compatible endpoint) and the starter uses them. You can also bring your own key. Never commit secrets or `data/`, and don't run code from inside `data/raw/`.

You may start early: run steps 1 to 3 at home and look at the data. It is a one-day hackathon with a **16:45 deadline**.

## Data
**565 labelled documents (about 23,000 fields), plus a hidden test we keep.** Downloaded by you from the original hosts; this repo redistributes nothing. Licences are permissive (checked 2026-11-09, not legal advice).

| Dataset | Docs | What's in it | Licence |
| --- | --- | --- | --- |
| [CUAD](https://www.atticusprojectai.org/cuad) | 510 | Contracts, 41 clause types | CC BY 4.0 |
| [ExtractBench](https://huggingface.co/datasets/llamaindex/ExtractBench) | 206 | Real forms and scans, handwriting, checkboxes | Apache-2.0 |

Each record in `data/unified/*.jsonl` has `doc_id`, `pages` (path relative to `data/`), `schema`, `gold` (null = not in the document) and `tags`. For CUAD, `pages` is the shared `CUADv1.json` and the contract is the entry whose `title` is the `doc_id` without `cuad:`. The hidden test comes from the same datasets, so tune on dev only.

*Check before you trust the data: what does an empty checkbox mean? Does every document fit in a prompt? Where are the blanks?*

## Scoring
A field is **accepted** if its value is non-null and `confidence >= threshold` (a missing confidence counts as 1.0, so always send one). In your prediction, null means "no answer, a human reviews it". Values are compared as:

| Field type | Compared as |
| --- | --- |
| `string` (default, all ExtractBench fields) | case-insensitive, whitespace collapsed |
| `span` (CUAD) | token F1 >= 0.8; several gold answers are joined with ` \| ` |
| `json` (nested, about 2% of fields) | exact match of the whole value; null is fine |

`5_score.py` prints precision, coverage, hallucination rate and **net saving** (always at 1 and 100; the `--review-cost` and `--error-cost` flags are only for your own what-if analysis). Net saving at your declared threshold is the headline result; the sweep over thresholds is for choosing it on dev. `python test_score.py` checks the scorer. `3_baseline.py` has two no-AI references (never answer, or always the most common value); a real system must beat the second one.

## What you submit (by 16:45, pushed to your repo)
| What | Where |
| --- | --- |
| Your code, with `run_hidden.sh` calling your extractor | `Project_1/` |
| `SUBMISSION.md`: approach, results, how to run, "Models and why" table, **your threshold** ([ours](SUBMISSION.md)) | `Project_1/` |
| `EDA.md`: a short look at the data, with real numbers | `Project_1/` |
| A 1-minute video or up to 5 slides | `submission/demo.mp4` or `submission/slides.pdf` |

**Hidden test:** we run `bash run_hidden.sh <input.jsonl> <output.jsonl>` from `Project_1/` (dev format without `gold`, `data/raw/` provided) and score at exactly your threshold. List the environment variables it needs in `SUBMISSION.md`.

**Input:** one JSON per line. Dev and public test include `gold`; the hidden test has the same format without it. Shortened example:
```json
{"doc_id": "cuad:LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT", "pages": ["raw/cuad/CUADv1.json"],
 "schema": {"properties": {"Agreement Date": {"type": "string", "x-normalizer": "span"},
                           "Governing Law": {"type": "string", "x-normalizer": "span"},
                           "Non-Compete": {"type": "string", "x-normalizer": "span"}}},
 "gold": {"Agreement Date": "7th day of September, 1999.", "Governing Law": "This Agreement is to be construed according to the laws of the State of Illinois.", "Non-Compete": null}}
```
**Output (your predictions):** one JSON per line, one entry per schema field. For span fields, copy the text from the contract; null means "no answer".
```json
{"doc_id": "cuad:LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT", "fields": {
  "Agreement Date": {"value": "7th day of September, 1999.", "confidence": 0.93},
  "Governing Law": {"value": "This Agreement is to be construed according to the laws of the State of Illinois.", "confidence": 0.90},
  "Non-Compete": {"value": null}}}
```
(On disk each record is a single line; it is wrapped here to read.)

## Directions to think about
Not a checklist, just things worth asking.
- **Which model, and why?** Any model is allowed. Compare on the same dev documents, and not only on accuracy: how often does it invent a value for a blank field? Is 0.95 confidence right 95% of the time? What does a document cost in money and seconds?
- **Make the value prove itself.** Does it actually appear in the document? Can a cheap model go first and a stronger one only when it's unsure?
- **Use doubt as a signal.** Ask twice or ask two models; if they disagree, lower the confidence. Which fields would you never automate?
- **Ugly inputs.** Rotate, zoom, try another OCR, or read a page as an image. What happens when a scan is unreadable?
- **Agents and knowledge graphs** are welcome (extract, check, fix, repeat; tools such as OCR or zoom), but show before and after: did precision, coverage or savings really improve?
- **Prove the business value.** Show net saving, and how it moves if a wrong value costs 20x or 500x. Show how much runs with no human, what the model costs per document, which fields you'd keep manual, and how you'd roll it out.

## How you are judged
The general criteria are on the back of the hackathon handout. Here, "Results" means net saving, precision, coverage and hallucination rate at your threshold, on data you did not tune on.

## Credits
Keep these in anything you publish. We converted both datasets to one format.
- Zhang, Lyjak, Stewart, Li, Suo (2026). *ExtractBench: A Benchmark for Schema-Guided Enterprise Document Extraction.* arXiv:2607.29677.
- Hendrycks, Burns, Chen, Ball (2021). *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.* NeurIPS Datasets and Benchmarks.

---

## Our solution: evidence-gated hybrid extraction (`hybrid/`)

**Idea.** The baseline books a value when the LLM says it is confident. Self-reported confidence did not separate right
from wrong answers: on 20 held-out contracts it booked 12 wrong values at 0.9, a net saving of −140%. We book a value
only when *independent evidence* supports it, and say "I don't know" otherwise. The original pipeline (`solution/`) is
unchanged and still runs with `python -m solution.run`.

### Architecture (contracts, CUAD)
```
contract text ─► 1 clause units (character offsets, headings kept)          hybrid/clauses.py, fields.py
              ─► 2 candidate retrieval per field                             hybrid/retrieve.py
                    BM25 + bge-small embeddings + clause classifier, weighted RRF, every candidate explained
              ─► 3 clause classifier: P(clause is of this field type)        hybrid/classify.py
              ─► 4 extraction: value, verbatim quote, clause id, why, confidence, supported   hybrid/extract.py
              ─► 5 optional independent verifier (another model family)      hybrid/verify.py
              ─► 6 decision rules (first match wins, no averaging)           hybrid/decide.py
                    ABSTAIN  nothing supported        REJECT  not in document / wrong clause type / verifier says no
                    REVIEW   incomplete (other clauses of this type not covered), fragment, verifier ambiguous,
                             field unreliable on calibration data, confidence below threshold
                    BOOK     every check above passed
```
- **Clause classifier.** One logistic-regression head per field on frozen embeddings, trained on 31,969 clauses from
  160 dev-train contracts. Labels come from CUAD's gold evidence spans. Held-out AUC is 0.88 (Governing Law AP 0.93,
  Cap On Liability 0.74); rare fields with fewer than 25 examples are weak. It is the best retriever: the gold clause
  is in its top 8 for 89% of fields, vs 57% for BM25 (`python -m hybrid.eval_retrieval`).
- **Per-field reliability policy** (`hybrid/field_policy.py`). A field may be booked only if, on 60 *calibration*
  contracts (dev-train, not used for training), at most 1% of its booked values were wrong. That is the break-even
  point for an error cost of 100 vs a review cost of 1. The policy is learned, not hand-written; unseen fields go to review.
- **Forms and scans (ExtractBench)** (`hybrid/forms.py`). These are 1–10 page forms; 45% are scans with no text
  layer, so the clause pipeline does not apply. Pages are rendered to images and read by **two independent vision
  models from different families** (`LLM_MODEL` = Claude Sonnet 5.5, `FORMS_SECOND_MODEL` = GPT-5.5). A value can only
  be booked if both reads agree; if the PDF has a text layer, the value must also be on the page (grounding) and sit
  next to its printed label (word boxes). A per-field policy is learned on 40 dev-train forms. We chose this over an
  OCR stack (PaddleOCR / LayoutParser) because that is slow to install and run on CPU and weak on historical handwriting.

### Results on 20 held-out contracts (dev-val, 820 fields, never used to fit anything)
| config | precision@0.9 | coverage@0.9 | booked errors (wrong clause / invented / incomplete) | net saving @0.9 | net saving @0.95 | calls/doc |
|---|---:|---:|---|---:|---:|---:|
| baseline (whole document, LLM confidence) | 76.9% | 6.3% | 0 / 2 / 10 | −140.0% | −10.4% | 1.0 |
| A: BM25 → LLM | 65.9% | 5.4% | 0 / 7 / 8 | −177.6% | −70.5% | 3.0 |
| B: embeddings → LLM | 59.1% | 5.4% | 0 / 5 / 13 | −214.1% | −58.9% | 3.0 |
| C: hybrid retrieval → LLM + clause gates | 71.7% | 6.5% | 0 / 5 / 10 | −176.5% | −70.9% | 3.0 |
| D: C + verifier | 71.7% | 6.5% | 0 / 5 / 10 | −176.5% | −70.9% | 4.5 |
| E: agentic RAG (the LLM searches and reads) | 52.6% | 7.0% | 2 / 3 / 22 | −322.3% | −119.5% | 3.5 |
| **G: baseline extraction + evidence gates (submitted as FINAL, threshold 0.95)** | **95.0%** | 2.4% | **0 / 1\* / 0** | **−9.8%** | **+1.3%** | 1.0 |
| H: G + verifier (Claude / GPT-5.5 / Gemini 3.1 Pro) | 93.8–94.1% | 2.0–2.1% | 0 / 1\* / 0 | −10.1% to −10.2% | +1.2% | 2.4 |

\* The one remaining "invented" value is a label gap. The contract says it "is governed by the laws of the State of
California", but CUAD's gold label is empty. All three verifiers called it SUPPORTED. At the declared threshold 0.95,
FINAL books 11 values on these contracts, all correct.

### Results on 32 held-out forms (ExtractBench dev-val, 1,564 fields)
| config | booked | precision | booked errors | net saving @0.9 | net saving @0.95 | calls/doc |
|---|---:|---:|---:|---:|---:|---:|
| two vision readers must agree (no field policy) | 581 | 96.0% | 23 | −109.9% | −1.8% | 2 |
| **+ learned field policy (submitted as forms_final, 0.95)** | **81** | **100%** | **0** | +6.5% | **+5.2%** | 2 |

Agreement between two models from different families was not enough on its own: they still agreed on 23 wrong values,
mostly nested tables and look-alike handwriting. The field policy removes them.

### Submitted system, all held-out dev-val data (52 documents, 2,384 fields), official scorer, threshold 0.95
**precision 100% · 0 hallucinations · coverage 7.3% · net saving +3.9%**, the same at an error cost of 20, 100 or 500,
because no wrong value was booked.

### Untouched check on public-test data (never used for any choice)
`bash run_hidden.sh` on the first 20 contracts and 12 forms of `test_public.jsonl`, with gold removed, then
`5_score.py --threshold 0.95` (32 documents, 1,439 fields, 23 minutes end to end, no failures):

| dataset | fields | booked | correct | wrong | hallucinated | net saving |
|---|---:|---:|---:|---:|---:|---:|
| CUAD contracts (20) | 820 | 9 | 9 | 0 | 0 | +1.1% |
| ExtractBench forms (12) | 619 | 73 | 73 | 0 | 0 | +11.8% |
| **all** | **1,439** | **82** | **82** | **0** | **0** | **+5.7%** |

Precision 100% (scorer band 99%), coverage 10.9%. The numbers are in line with dev-val, so the method was not tuned to it.

**What we learned**
1. The main booked error was not the wrong clause but an *incomplete* answer: the gold lists every passage (2–5), and
   the model quotes one. The completeness rule and the field policy remove these.
2. Better retrieval raised raw accuracy (C: 79.5% vs 77.4%) but did not make booking safe on its own.
3. A second LLM pass as verifier, with the same model or another family, changed almost nothing and cost 2.4× the calls.
4. The agent shortened clause answers to bare values and did worst.
5. Safe booking is narrow today. Only Document Name and Governing Law qualify on calibration data, so coverage is
   small (about 1–2% of fields), but nothing booked was wrong.

### Examples (from the decision traces in `data/runs/hybrid_eval/dev_val_20/*.trace.jsonl`)
- **Correct booking.** *Governing Law*, Hertz IP agreement. The value is found verbatim as a whole sentence, the
  classifier gives its clause (C49) p=0.97, the field is bookable on calibration data (21/21) and confidence is 0.95,
  so it is BOOKED. It matches the gold.
- **Wrong-clause rejection.** *Non-Disparagement*, China Real Estate domain licence. The model proposed "Licensee shall
  not knowingly use the Licensed Domain Names in any manner that tarnishes, degrades, disparages…". The text exists,
  but the classifier gives that clause p=0.0005 for Non-Disparagement, so it is REJECTED. Gold: field absent.
- **Incomplete, sent to review.** *Parties*, same contract. The answer names two companies, but clauses C2, C178 and C1
  also look like party clauses (p 0.76 / 0.73 / 0.59), so it goes to REVIEW. The gold also lists "Licensor" and "Licensee".
- **Abstention.** *Expiration Date*, Thrivent endorsement. Retrieval's best clause has classifier p=0.02 and the model
  returns null, so the system ABSTAINS. Gold: absent.

### Run
```bash
pip install -r requirements.txt                 # adds numpy, fastembed (+onnxruntime), scikit-learn
bash run_hidden.sh in.jsonl out.jsonl           # FINAL config; also writes out.trace.jsonl with every decision
python -m hybrid.run --data data/unified/dev.jsonl --out preds.jsonl --config FINAL --trace trace.jsonl --limit 5
python 5_score.py --gold data/unified/dev.jsonl --pred preds.jsonl --threshold 0.95
```
Environment: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (the extractor and first form reader; we used
`claude-sonnet-5.5`). Optional: `FORMS_SECOND_MODEL` (second form reader, default `gpt-5.5`), `HYBRID_NONCONTRACT`
(`forms_final` default | `review` | `baseline`), `LLM_VERIFY_MODEL`, `HYBRID_EMBED_MODEL`.
The embedding model (~130 MB) downloads on first use; without `fastembed`, retrieval falls back to BM25.

### Reproduce the evaluation
```bash
python -m hybrid.classify                       # train the clause classifier on dev-train (CPU, ~1 h first time, then cached)
python -m hybrid.eval_retrieval --part val --limit-docs 20          # retrieval hit@k, no API
python -m hybrid.evaluate --part train --offset 160 --limit-docs 60 --configs G_baseline_gated
python -m hybrid.field_policy --run-dir data/runs/hybrid_eval/dev_train_60_from160 --config G_baseline_gated
python -m hybrid.evaluate --part val --limit-docs 20                # all configs -> report.md + metrics.json
python -m hybrid.evaluate --part val --limit-docs 20 --replay       # recompute metrics from the cache, no API
python -m hybrid.evaluate --part train --source extractbench --limit-docs 40 --configs forms_dual
python -m hybrid.forms fit --run-dir data/runs/hybrid_eval/dev_train_40_extractbench --config forms_dual
python -m hybrid.evaluate --part val --source extractbench --limit-docs 32 --configs forms_dual,forms_dual_policy
python -m unittest tests.test_hybrid tests.test_verify
```
LLM replies are cached in `data/cache/llm/` with their original token counts and latency, so re-runs are free.
`hybrid/split.py` splits dev documents 4:1 by hash into train and val; the 10 pilot contracts read by hand always go
to train. test_public was not used for any choice.

### Limitations
- Evaluated on 20 held-out contracts (820 fields). One booked error moves net saving by about 12 points, so the
  numbers are indicative. The threshold (0.95) is the only one that is positive on both calibration and validation data.
- Coverage is low. Most clause fields are spread over several passages, so they are left to humans.
- The classifier is weak for rare clause types; the field policy keeps those fields manual.
- Verifiers were measured but are not used: they gave no gain on this data. The team rate limit (400 requests/hour)
  also limits multi-pass designs.
- Forms: the text-layer check rejected 43 correct values on dev-val (formatting such as "$1,234" vs "1234"); fixing
  it would raise coverage. Forms need two vision calls (~45 s, ~$0.19 per document at the assumed price).
- CPU embedding is slow (~55 ms per clause). The cache makes repeat runs fast; a first run on a new contract takes seconds.

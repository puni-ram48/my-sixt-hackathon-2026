# Results: which method worked best, and why

Team **Zero to SIXTy**, Project 1. All numbers come from the reports in this folder, made by `python -m hybrid.evaluate`.
Scoring follows the project rules: a booked value is one with confidence ≥ the declared threshold **0.95**.
A wrong booked value costs 100, a review costs 1, and net saving is measured against reviewing every field by hand.

## Winners
| Dataset | Winning method | Booked on held-out data | Wrong | Net saving |
|---|---|---:|---:|---:|
| CUAD contracts | **FINAL**: baseline extraction + evidence gates | 11 / 11 correct | 0 | **+1.3%** |
| ExtractBench forms | **forms_final**: two independent vision readers + learned field policy | 81 / 81 correct | 0 | **+5.2%** |
| Both (submitted system, official scorer) | | 92 / 92 correct | 0 | **+3.9%** |
| **Untouched public test** (32 docs) | same system via `run_hidden.sh` | **82 / 82 correct** | **0** | **+5.7%** |

**Why these won:** they are the only methods that never booked a wrong value. Under a 100:1 cost, one wrong booking
cancels a hundred correct ones, so the winner is the method that *knows when it is right*, not the one that answers most.

## Data used (no overlap between roles)
| Role | CUAD | ExtractBench |
|---|---|---|
| Train the clause classifier | dev-train contracts 0–159 (31,969 clauses) | – (nothing trained) |
| Calibrate (learn which fields to trust) | dev-train contracts 160–219 (60) | dev-train forms 0–39 (40) |
| Held-out evaluation and method comparison | dev-val, first 20 contracts (820 fields) | dev-val, 32 forms (1,564 fields) |
| Untouched final check | test_public, first 20 contracts | test_public, first 12 forms |

Dev documents are split 4:1 into train/val by a hash of the document id (`hybrid/split.py`).

## All methods compared: CUAD (20 held-out contracts, 820 fields)
| Method | Raw accuracy | Booked @0.95 | Precision | Net saving @0.95 | Net saving @0.9 | LLM calls/doc | USD/doc* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline: whole document, trust LLM confidence | 77.4% | 15 | 93.3% | −10.4% | −140.0% | 1.0 | 0.070 |
| A: BM25 retrieval → LLM | 77.2% | 22 | 72.7% | −70.5% | −177.6% | 3.0 | 0.102 |
| B: embedding retrieval → LLM | 77.0% | 17 | 70.6% | −58.9% | −214.1% | 3.0 | 0.095 |
| C: hybrid retrieval (BM25 + embeddings + classifier) → LLM | **79.5%** | 19 | 68.4% | −70.9% | −176.5% | 3.0 | 0.122 |
| D: C + LLM verifier | 79.5% | 19 | 68.4% | −70.9% | −176.5% | 4.5 | 0.175 |
| E: agentic RAG (LLM searches and reads) | 77.4% | 20 | 50.0% | −119.5% | −322.3% | 3.5 | 0.189 |
| F: E + verifier | 77.4% | 20 | 50.0% | −119.5% | −322.3% | 4.9 | 0.227 |
| G: baseline + evidence gates | 77.4% | 11 | 100% | +1.3% | −9.8% | 1.0 | 0.070 |
| H: G + verifier (Claude / GPT-5.5 / Gemini 3.1 Pro) | 77.4% | 10 | 100% | +1.2% | −10.1% | 2.4 | 0.110–0.155 |
| **FINAL = G at threshold 0.95 (submitted)** | 77.4% | **11** | **100%** | **+1.3%** | – | **1.0** | **0.070** |

\* Assumed price of 3 / 15 USD per million input / output tokens; the endpoint's real price is not known to us.

## All methods compared: ExtractBench (32 held-out forms, 1,564 fields)
| Method | Raw accuracy | Booked @0.95 | Precision | Wrong booked | Net saving @0.95 | LLM calls/doc | USD/doc* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original pipeline (before this work) | not measured with an LLM; the no-AI reference had 32% precision | | | | | | |
| Two readers must agree, no field policy | 79.4% | 372 | 98.9% | 3 | −1.8% | 2.0 | 0.191 |
| **forms_final: two readers + field policy (submitted)** | 79.4% | **81** | **100%** | **0** | **+5.2%** | 2.0 | 0.191 |

## Why the losers lost
- **Trusting the LLM's confidence (baseline):** at 0.9 it booked 12 wrong values, a net saving of −140%. Confidence is
  not calibrated: raising C's threshold from 0.9 to 0.95 *lowered* its precision (71.7% → 68.4%).
- **Better retrieval (A–D):** classifier-led retrieval found the gold clause in the top 8 for 87–92% of fields, vs 57–59%
  for BM25, and C had the best raw accuracy. But better extraction does not tell you which answers
  are safe to book.
- **Verifiers (D, F, H):** three verifier models (same model, GPT-5.5, Gemini 3.1 Pro) changed almost no decision and
  cost 1.5–2.4× the calls.
- **Agentic RAG (E, F):** the agent shortened clause answers to bare values ("New York" instead of the governing-law
  sentence) and was worst.
- **Two readers without a policy (forms):** the models agreed on 3 wrong values at 0.95 (23 at 0.9). Agreement is
  strong evidence, but not enough on its own.

## Key findings (evidence from the rule tables in the reports)
1. **The main booked error on contracts is an incomplete answer, not the wrong clause.** CUAD's gold lists every
   passage (2–5) for a clause field; the model quotes one. The completeness rule caught 20 such answers, against 3
   correct ones.
2. **The clause-type classifier catches wrong-clause and invented answers.** It rejected 31 bad values against 10 good ones.
3. **Disagreement between two model families is the strongest error signal on forms:** 39 of 46 disagreements were
   real errors.
4. **Learned per-field trust is what makes booking safe on both datasets.** On contracts only Document Name (27/27) and
   Governing Law (21/21) qualified; on forms 68 of 207 field types qualified.

## Files
| File | Content |
|---|---|
| `cuad_dev_val_20.report.md` / `.metrics.json` | all 12 contract configs, threshold sweep, per-rule tables |
| `cuad_calibration_60.report.md` / `.metrics.json` | calibration run used to learn the contract field policy |
| `forms_dev_val_32.report.md` / `.metrics.json` | forms configs on held-out forms |
| `forms_calibration_40.report.md` / `.metrics.json` | calibration run used to learn the forms field policy |

Per-field decision traces are not committed, because they contain document text; regenerate them with the commands in
`../README.md`. In the forms rule tables, the "readers disagree" reasons are merged into one row so that no document
values are quoted.

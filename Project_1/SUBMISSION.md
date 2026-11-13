# Submission: Project 1, Zero to SIXTy

## Authors
**Team name:** Zero to SIXTy

**Project:** Project 1: Teach a machine to say "I don't know"

## Problem and approach
A wrong value that gets booked costs 100; a human review costs 1. A system therefore only saves money if its booked
values are right more than 99% of the time. Our starting pipeline (one LLM call, then booking anything the model was
≥ 0.9 confident about) lost money: on 20 held-out contracts it booked 12 wrong values, a net saving of −140%. The
model's self-reported confidence did not separate right from wrong.

**Main idea: the model proposes, evidence decides.** A value is booked only if independent checks support it; otherwise
it goes to a human. Self-reported confidence is the last check, never the main one. Which fields can be trusted at all
is learned from calibration data with the 1% break-even rule, not hand-picked.

- **Contracts (CUAD), method `FINAL`.** One Claude call reads the contract. The contract is split into clauses with
  character offsets, and a clause-type classifier (trained on CUAD's gold evidence) scores every clause for every
  field. A value is booked only if it is found verbatim in the text, its clause is the right type, no other clause of
  that type is left out (completeness), it is made of whole sentences, its field was error-free on 60 calibration
  contracts, and confidence is ≥ 0.95.
- **Forms and scans (ExtractBench), method `forms_final`.** Pages are rendered to images. Two models from different
  families (Claude and GPT-5.5) read them independently, and both must agree. If the PDF has a text layer, the value
  must also be printed on the page, next to its label. The field must have been error-free on 40 calibration forms,
  and confidence must be ≥ 0.95. 45% of these PDFs are scans, where agreement between two readers is the only
  available evidence.
- `run_hidden.sh` routes each record to one of the two methods by its source.

**Assumptions:** the hidden test comes from the same datasets and document templates (as the project README states);
the endpoint serves `claude-sonnet-5.5` and `gpt-5.5`. If anything is unavailable, the affected fields go to review:
the system fails safe.

## Models and why
All candidates were compared on the same held-out dev-val documents (20 contracts, 32 forms).

| Model | Open-weight or closed | Where it runs | Main metric(s) | Cost and time per item | Chosen? |
| --- | --- | --- | --- | --- | --- |
| Claude Sonnet 5.5 (`claude-sonnet-5.5`) | closed | SIXT LLM endpoint (API) | contracts: 77.4% raw field accuracy; with our gates, 100% booked precision | ~$0.07*, ~10 s per contract (1 call) | **yes**: extractor and form reader A |
| GPT-5.5 (`gpt-5.5`) | closed | SIXT LLM endpoint (API) | forms: disagreement with reader A flagged 39 of 46 reading errors | with reader A ~$0.19*, ~46 s per form (2 calls) | **yes**: independent form reader B |
| BAAI/bge-small-en-v1.5 (via fastembed) | open | local CPU | retrieval hit@8 64% (BM25 59%) | ~55 ms per clause, free | **yes**: clause embeddings |
| Logistic-regression clause classifier (our own, 41 heads) | open (ours) | local CPU | held-out AUC 0.88–0.91; gold clause in top 8 for 89–92% of fields | ms, free | **yes**: clause-type and completeness checks |
| Claude Sonnet 5.5 / GPT-5.5 / Gemini 3.1 Pro as verifier | closed | API | no gain over FINAL (+1.2% vs +1.3%) | +1.4 calls, +7 to +42 s per contract | **no**: no measurable benefit |
| Agentic RAG (Claude with search and read tools) | closed | API | 50% booked precision, −119.5% | 3.5 calls, $0.19* per contract | **no**: shortened answers, worst result |

\* Assumed price of 3 / 15 USD per million input / output tokens.

**Declared confidence threshold: 0.95** (for both datasets). It is the only threshold with a positive net saving on
both the calibration data and the held-out data. The output confidence is our system's confidence after the gates:
booked values keep the model's confidence (≥ 0.95), values sent to review are capped at 0.5, and rejected or absent
values are null.

## Results
**Held-out dev-val** (never used for training or calibration; it was used to choose between methods):

| Method | Contracts: booked / wrong / net saving | Forms: booked / wrong / net saving |
|---|---|---|
| Original pipeline, threshold 0.9 | 52 / 12 / **−140.0%** | not measured |
| Original pipeline, threshold 0.95 | 15 / 1 / −10.4% | not measured |
| Best retrieval (C, hybrid BM25 + embeddings + classifier), 0.95 | 19 / 6 / −70.9% | – |
| Two readers must agree, no field policy, 0.95 | – | 372 / 3 / −1.8% |
| **Submitted (FINAL + forms_final), 0.95** | **11 / 0 / +1.3%** | **81 / 0 / +5.2%** |

Both datasets together (52 documents, 2,384 fields), official `5_score.py --threshold 0.95`: **precision 100%,
0 hallucinations, coverage 7.3%, net saving +3.9%**, identical at an error cost of 20, 100 or 500.

**Untouched public-test check** (first 24 contracts and 16 forms of `test_public.jsonl`; `run_hidden.sh` with gold
removed; never used for any choice): **99 booked, 98 match the gold, 0 hallucinations, precision 99.0%, net saving
−0.1% (break-even)** (contracts 11/11, +1.1%; forms 87/88, −1.4%). The one mismatch is a **gold-label error**: the
form prints `3583 CASTLEHILL WAY`, both readers copied it exactly, and the gold says `CASTLE HILL`. We verified it in
the text layer, in the rendered image and by hand (`EDA.md` section 9; deterministic check: `python -m hybrid.gold_check`).
The scorer counts it, and at 100:1 one counted error offsets about 100 correct bookings. With the label corrected,
99 of 99 are right (+5.4%). We report the scorer's number.

**What did not work, honestly:**
- Better retrieval raised raw accuracy (79.5% vs 77.4%) but not booking safety.
- LLM verifiers (three model families) added cost but no gain.
- The agent was the worst method.
- Two agreeing readers still agreed on some wrong values.

**Limits:**
- **Low coverage.** Only 2 of 41 contract fields (Document Name, Governing Law) and 68 of 207 form field types are
  trusted; everything else goes to a human.
- **Small samples.** 92 of 92 held-out bookings correct gives a 95% lower bound of about 96% precision, below the
  99% target.
- **Truncation.** The contract extractor reads the first 60,000 characters; 28% of contracts are longer.
- **New templates.** Forms of a type never seen in calibration go entirely to review.
- **Label noise.** Gold labels are not always right (one confirmed case above; 0.7% of ExtractBench dev text labels
  differ from the printed text only in spacing). At a 99% precision target, rare label errors decide whether the
  measured result is above or below break-even.
- **Runtime.** Four documents run in parallel (`--doc-workers 4`, about 1.8× faster; predictions are identical to
  a sequential run). Local CPU embedding still runs one document at a time.

Details, per-rule evidence and all 12 contract configurations: `results/RESULTS.md`. Data analysis: `EDA.md`.

## How to run
Python 3.11+ (we used 3.12.3 on Windows). From `Project_1/`:
```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash 1_download.sh --accept-licenses && python 2_prepare.py   # data into data/ (not in the repo)
bash run_hidden.sh <input.jsonl> <output.jsonl>       # submission entry point; also writes <output>.trace.jsonl
python 5_score.py --gold data/unified/dev.jsonl --pred <output.jsonl> --threshold 0.95
python -m unittest tests.test_hybrid tests.test_verify # offline tests, no API
```
The trained classifier and both field policies are committed in `hybrid/models/`, so no training is needed to run.
The embedding model ([Qdrant/bge-small-en-v1.5-onnx-Q](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q), ~67 MB) downloads on first use. To reproduce the experiments, see "Reproduce the evaluation" in
`README.md`; LLM replies are cached, so repeat runs are free.

**Environment variables** (names only): `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (= `claude-sonnet-5.5`).
Optional: `FORMS_SECOND_MODEL` (default `gpt-5.5`), `HYBRID_NONCONTRACT` (`forms_final` by default; `review` sends
all form fields to a human), `LLM_VERIFY_MODEL`, `HYBRID_EMBED_MODEL`.

## Demo
[`submission/demo.mp4`](../submission/demo.mp4): demo video of the dashboard and both pipelines.

## Next steps
1. **Raise coverage without new risk.**
   - Normalise number formats before the forms text-layer check (43 correct values were rejected by it).
   - Use reader B's value when reader A misses one (A missed 229 values that B found), with B checked against the
     text layer.
   - Calibrate on all 312 / 127 training documents instead of 60 / 40, so more fields can qualify for trust.
     **This is the main bottleneck, measured.** Many correct contract values go to review as "clause type uncertain"
     because a clause unit mixes several short sub-clauses (e.g. an insurance sentence scored 0.54 on its own but 0.05
     inside a chunk with counterparts and notices). We tested scoring the evidence sentence itself (the higher of the
     two scores): it removed 8 of 78 (held-out) and 20 of 112 (public-test) "uncertain/wrong clause" stops. But
     bookings stayed at 11 and 11, because those values then stop at "field not trusted". So we kept the validated
     system; more calibration data per field is what unlocks coverage.
2. **Fix incomplete contract answers.** Ask the extractor for *all* passages of a clause field (the main error type),
   then re-learn which fields become bookable.
3. **Speed.** Embed clauses in a separate process pool. Documents already run 4 at a time; the embedding step is
   now the bottleneck.
4. **How SIXT could use it.**
   - Start in shadow mode: the system books nothing, humans see its suggestions and decision reasons, and every
     correction becomes calibration data.
   - Switch on automatic booking field by field, only when a field's measured error rate is below 1/error-cost.
   - The per-field decision log (`*.trace.jsonl`) gives auditors the reason for every booked value.

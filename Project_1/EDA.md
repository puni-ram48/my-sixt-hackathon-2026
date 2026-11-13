# EDA: Project 1, "Teach a machine to say I don't know"

Written for: the hackathon owners and jury, to show what the data looks like and what problems we found in it and in our own pipeline.

Scope: every number here was measured on the files in `data/` during this work. The hidden test is not used anywhere in this document.

## 1. What the data is

Two public datasets, converted to one format by the organisers (`data/unified/*.jsonl`, one document per line).

| Split | Documents | CUAD contracts | ExtractBench forms | Fields | Null answers |
|---|---|---|---|---|---|
| dev | 540 | 381 | 159 | 22,115 | 12,743 (58%) |
| test_public | 176 | 129 | 47 | 6,962 | 4,097 (59%) |

- **CUAD** (legal contracts, CC BY 4.0): 510 contracts in total. Each has 41 clause fields such as Governing Law, Non-Compete and Parties. A field is either a span copied from the contract or null. In dev, 68% of CUAD fields are null (10,607 of 15,621).
- **ExtractBench** (real forms and filings, Apache-2.0): 206 documents across 32 form types. Each form type has its own schema, with 2 to 159 fields per form. Forms include tax returns (Form 1040 has about 135 fields), Schedule K-1, W-2, 1099-B, a closing disclosure, a customs entry and a utility bill. In dev, 33% of ExtractBench fields are null (2,136 of 6,494).

Each record has: `doc_id`, `pages` (path to the source), `schema` (the fields to fill), `gold` (the answer key, null when absent), `tags` (difficulty labels) and, in dev only, `gold_evidence` (page and position of each answer).

## 2. What the schemas do and do not say

- **CUAD schemas give names and types only.** Each field has a name and a span marker, with no description. The definitions are in the raw file `CUADv1.json`, where every clause type has a question ending in "Details: ...". Our baseline did not read them until late in the work.
- **ExtractBench schemas give a description for every field** (6,494 of 6,494). Descriptions contain formatting rules, such as "TWO-STATE: true only if a mark is printed inside the box... Never return null."
- ExtractBench gold stores booleans as text: `'true'` and `'false'` (356 non-null booleans in dev).

## 3. What the documents look like

- **CUAD contracts** are plain text. Each contract is a single paragraph of median length 33,000 characters (maximum 338,000). Line breaks fall in the middle of sentences, and section numbers and headings are mixed into the text.
- **ExtractBench PDFs:** 159 dev documents, of which 66 have no text layer at all (scans). The other 93 have printed text, but filled-in values are often missing from it. Example: on the Schedule K-1 `07021-2016-p0014`, the printed template is extracted, but the partner's EIN `26-3839121` is not. The page has an embedded image and the PDF has no fillable form fields, so the filled values are most likely part of that image.

## 4. Gold answer quirks

- **Multi-answer values** are joined with " | ". Parties is the clearest case: the gold is a list of names and roles, such as `Distributor | Electric City Corp. | Electric City of Illinois L.L.C. | Company`, not a sentence.
- **Clause answers are full sentences**, copied word for word. Dates keep their trailing full stop.
- **Formatting differences:** gold dates and amounts can differ from the printed text in punctuation, spacing or number formatting. Our gate ignores case, punctuation and spacing, and treats `1,250` and `1250` as the same.
- **Gold is mostly right, but not always.** A sample check of gold spans against the contract text found 99.7% (4,999 of 5,014) present in the text. This doesn't show the gold is right; it only shows the answers are in the document.

## 5. Issues found

### 5.1 Data issues

1. **Scanned documents have no text.** 66 of 159 ExtractBench dev documents have no text layer, so their answers can't be checked against text.
2. **Filled-in values are not in the text layer.** Text extraction reads printed template text, but misses values that are part of an image. On ExtractBench, measured grounding rates were 67% for numbers and 53% for text, even after switching from pypdf to pdfplumber (pypdf: 59% and 51%).
3. **CUAD schemas have no definitions.** The model only sees a field name such as "Non-Compete", so the meaning has to come from the raw question text. The definitions were not used until late in the work.
4. **Long contracts.** CUAD contracts can be 338,000 characters, far larger than a prompt should carry. Long documents are truncated in the baseline, so answers beyond the cut-off cannot be found.
5. **Dataset gaps in the gold.** Parties and Document Name are often written differently from the sentence style the model produces, which makes them hard to score.
6. **Unusual formats in ExtractBench.** The schemas are large (Form 1040 has about 135 fields) and many descriptions are instructions about layout, such as coded boxes and checkboxes. These are hard to match with simple text checks.

### 5.2 Pipeline issues (what the baseline got wrong)

7. **The model's confidence is not a reliable signal.** Across 10 CUAD contracts, correct answers had confidences from 0.3 to 0.95, while the threshold is 0.95. At that threshold most correct answers are rejected.
8. **Wrong-clause answers pass the gate.** The grounding check confirms the text exists in the contract, and the sentence check confirms it is a whole sentence. Neither checks that the sentence is the right clause. Example: "Cap On Liability" returned a real liability sentence from a different clause at confidence 0.95, and it was booked.
9. **Fragments and over-long answers.** The first prompt produced fragments: "laws of the State of Illinois" instead of the full clause. A later prompt fixed that but made dates too long (whole paragraphs).
10. **Missed answers.** The later prompt returned null more often (56 missed in one run, 23 in the next), which is safe but lowers savings.
11. **Invented values.** The model gave values for fields that are absent from the contract. These cost 100 each if booked, so they matter most. They were kept below the threshold, but their confidence was still high enough to worry about.
12. **Sentence-alignment penalty hurt correct answers.** A check that required whole sentences halved the confidence of some correct answers. The cause was list markers such as "(A)" and section numbers, which the first version of the check did not recognise. This was partly fixed; four correct answers were still affected in the last check.
13. **Rule hardcoded for one field.** A rule for Parties was added in one version. It was removed because it does not generalise to other fields.
14. **Clause definitions made things worse on a small sample.** With the definitions in the prompt, one wrong answer was booked at 0.95 and net saving fell from +3.7% to -20%. On 10 contracts one answer changes precision by about 5 points, so this is not conclusive.
15. **Retrieval alone is weak.** BM25 keyword retrieval (40 CUAD contracts, 502 fields): the gold answer is inside the top 1 clause for 18.5% of fields, top 5 for 41.6%, top 10 for 55.6%. A pipeline that retrieves first cannot answer what it did not retrieve.

### 5.3 Measurement issues

16. **Small samples.** Most results come from 10 contracts (410 fields, 112 with gold answers). With 15 accepted answers, all correct, precision is only known to be at least about 80% with 95% confidence. The target is 99%.
17. **Coverage was misread at first.** The scorer's coverage is accepted answers divided by gold-present fields, not by all fields. We reconciled this: 3 accepted of 112 gold-present fields gives 2.7%, not an estimate of about 11 accepted.
18. **Scorer numbers depend on the threshold.** The "31 wrong, 3 hallucinated" figures are at threshold 0, not at the declared threshold.
19. **Risk of tuning on gold.** Several rules were first written around gold answers. They were rewritten to come from the method and the schema. The remaining gold checks are used only for scoring.
20. **Replay limits.** Saved prediction files hold answers after the gate, not the raw model replies, so gate on versus off cannot be replayed. Raw replies are needed for that.
21. **Retrieval metric.** The first retrieval evaluation used an 80% word-overlap threshold. A later version scores overlap with gold character spans. The two sets of numbers are not comparable.

### 5.4 Engineering issues

22. **Rate limits.** The endpoint returned HTTP 429 (too many requests) on about half of the first pilot's calls. Retry backoff was lengthened, and runs now take longer than the 10-minute tool limit, so they run in the background.
23. **Line endings.** Repository text files use CRLF line endings. The download script failed every size check, and bash scripts break on CRLF. The download was run from an LF copy instead; the repo itself was not changed.
24. **Windows virtual environment.** `setup.sh` activates `.venv/bin/activate`, which does not exist for a Windows Python environment. Steps were run directly with `.venv\Scripts\python.exe`.
25. **Starter grounding check.** The starter skipped the check for any value over 80 characters, which covered 77% of CUAD gold answers. That check is not used in the current pipeline.

## 6. Baseline results

| Run | Booked at 0.95 | Precision | Net saving |
|---|---|---|---|
| Starter baseline (keyword rules, no LLM) | 0 | n/a | 0% |
| Sonnet v1, old prompt (10 CUAD) | 16 | 68.8% | -118% |
| Sonnet v2, complete-sentence prompt | 3 | 100% | +0.7% |
| Sonnet v3, list-marker fix and short-value exemption | 15 | 100% (15 of 15) | +3.7% |
| Sonnet with clause definitions | 18 | 94.4% (17 of 18) | -20.0% |

All runs use the same 10 CUAD dev contracts, 410 fields and 112 gold-present fields.

## 7. Open questions

- Does retrieval plus extraction beat the baseline on the same documents, once retrieval is good enough?
- Is the confidence signal useful if it comes from an independent check rather than the model's own report?
- How often are scanned ExtractBench documents answerable at all, without OCR or a vision model?

## 8. What the experiments answered

Measured later on held-out data (CUAD dev-val, 20 contracts and 820 fields; ExtractBench dev-val, 32 forms and
1,564 fields). Full tables are in `results/RESULTS.md`.

- **Does retrieval plus extraction beat the baseline, once retrieval is good enough?** Retrieval did get good: a
  clause classifier trained on CUAD's gold evidence puts the gold clause in the top 8 for 89–92% of fields, against
  57–59% for BM25 (issue 15). Raw accuracy rose slightly (79.5% vs 77.4%), but booking did not get safer: every
  retrieval-first configuration lost money (−59% to −71% net saving at 0.95). Retrieval was not the bottleneck.
- **Is the confidence signal useful if it comes from an independent check?** Yes, and that is what worked. On contracts
  the independent checks were grounding, a clause-type classifier, a completeness check and per-field trust learned on
  calibration contracts. They booked 11 values, all correct (+1.3%), where trusting the model's own confidence booked
  12 wrong values at 0.9 (−140%). This confirms issue 7. A second LLM pass as verifier (three model families) added
  nothing.
- **Are scanned ExtractBench documents answerable without OCR?** Yes, with vision models. Two independent readers
  (Claude and GPT-5.5) read the page images; 42 of the 81 values booked on held-out forms came from scans with no text
  layer, all correct. When the readers disagreed, 39 of 46 times it was a real error. Issue 2 (filled values missing
  from the text layer) also showed up here: the text-layer check rejected 43 correct values whose printed form
  differed from the reading.
- **Gold labels can be wrong, and the scorer counts them.** The only value our system booked that did not match the
  gold, across 40 public-test documents (99 booked), is a gold-label error. See section 9.
- **New finding: the main booked error on contracts was an incomplete answer, not the wrong clause.** The gold for a
  clause field lists every passage that states it (often 2–5), and the model quoted one. This is the multi-answer
  quirk from section 4: it affects scoring of almost every clause field, not just Parties.

## 9. A gold-label error, and a deterministic way to check gold against the document

**The case.** On the public test, our system booked one value that the scorer counts as wrong:

| | |
|---|---|
| Document | `extractbench:short/ROGER-WINFIELD-SCOTT-FOUNDATION-INC-2009-580655183-p0024` (one-page 1099-type form) |
| Field | `recipient_address` |
| Gold evidence box | page 1, bbox `[0.7194, 0.1414, 0.146, 0.0385]` (x, y, width, height as page fractions) |
| **Printed in that box** (PDF text layer) | `3583 CASTLEHILL WAY` / `TUCKER, GA 30084`; the word `CASTLEHILL` sits at x 599–659, y 90 on the 794 × 624 page, inside the gold box |
| Rendered page image of the box | `3583 CASTLEHILL WAY`, no space (checked by rendering the box at 4× and by eye, and confirmed by a team member opening the PDF) |
| Our booked value | `3583 CASTLEHILL WAY, TUCKER, GA 30084` (reader A, Claude, 0.96; reader B, GPT-5.5, 0.99; both agree; found verbatim on the page) |
| Gold | `3583 CASTLE HILL WAY, TUCKER, GA 30084` |

The document prints CASTLEHILL; the gold label inserts a space. Our value is a faithful copy of the document. The
scorer compares strings after collapsing whitespace, so it cannot equate the two and counts the value as a wrong
booking (cost 100). We did not change anything to "fix" this case: that would be tuning on public-test data.

**The deterministic check** (`hybrid/gold_check.py`, no LLM and no judgement). For a document and field, it reads
the PDF's own words inside the gold evidence box and compares them with the gold value and with a prediction:
`exact` (same words in order, case and punctuation ignored), `spacing_only` (same letters and digits, different
spaces) or `different`. It can also save the box as an image (`--crop`).

```bash
python -m hybrid.gold_check --data data/unified/test_public.jsonl \
  --doc "extractbench:short/ROGER-WINFIELD-SCOTT-FOUNDATION-INC-2009-580655183-p0024" \
  --field recipient_address --pred "3583 CASTLEHILL WAY, TUCKER, GA 30084" --crop castlehill.png
# printed_in_box: "3583 CASTLEHILL WAY TUCKER, GA 30084"
# gold_vs_printed: spacing_only   pred_vs_printed: exact   pred_vs_gold: spacing_only
```

**How common is this?** `python -m hybrid.gold_check --survey` checks every non-null text label of the ExtractBench
dev set whose gold box contains printed text (83 documents, 868 labels; another 1,824 labels have no printed text in
the box, because they are scans or image-only values):

| Gold vs printed text in its own box | Labels | Share |
|---|---:|---:|
| exact | 733 | 84.4% |
| spacing only | 6 | 0.7% |
| different | 129 | 14.9% |

Spacing-only differences are rare, and they go both ways. Some are gold normalisations (tax ids printed as
`80-0294810`, stored as `800294810`); some are text-layer artefacts (`SUNTRUSTBANK` in the text layer, `SUNTRUST BANK`
in the gold). "Different" mostly means the value is not cleanly in the text layer (split words, filled values drawn
as images; see issue 2), not that the gold is wrong. So the text layer is evidence, not ground truth, and a mismatch
needs a look at the rendered image before calling the gold wrong, as we did here.

**Why it matters.** Under a 100:1 cost, one label error among 99 correct bookings turns a +5.4% saving into −0.1%.
Gold noise is a floor on measured precision that no extraction method can get below, so a system aiming at 99%
precision should expect occasional "errors" that are really label errors.

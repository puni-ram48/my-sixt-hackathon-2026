# Baseline vs hybrid on 32 contracts (dev_val_32_extractbench), threshold 0.9

| metric | forms_dual | forms_dual_policy |
|---|---:|---:|
| booking precision | 96.0% | 100.0% |
| recall | 54.2% | 9.8% |
| F1 | 0.693 | 0.179 |
| raw field accuracy | 79.4% | 79.4% |
| coverage (booked) | 37.1% | 6.5% |
| human review | 62.9% | 93.5% |
| booked wrong-clause | 8 | 0 |
| booked hallucinated | 0 | 0 |
| booked boundary err | 15 | 0 |
| wrong-clause rate (booked) | 1.38% | 0.00% |
| false acceptance | 29.9% | 0.0% |
| false rejection | 60.6% | 92.9% |
| raw wrong-clause answers | 45 | 45 |
| raw invented answers | 2 | 2 |
| net saving (1/100) | -109.9% | 6.5% |
| LLM calls / doc | 2.0 | 2.0 |
| input tokens / doc | 30,124 | 30,124 |
| output tokens / doc | 6,709 | 6,709 |
| LLM seconds / doc | 46.5 | 46.5 |
| USD / doc (assumed price) | $0.191 | $0.191 |

## Threshold sweep (replayed, no new API calls)

| config | t | booked | precision | coverage | booked wrong-clause | booked hallucinated | net saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| forms_dual | 0.8 | 658 | 95.6% | 42.1% | 9 | 0 | -143.4% |
| forms_dual | 0.85 | 632 | 95.6% | 40.4% | 9 | 0 | -138.6% |
| forms_dual | 0.9 | 581 | 96.0% | 37.1% | 8 | 0 | -109.9% |
| forms_dual | 0.95 | 372 | 98.9% | 23.8% | 3 | 0 | -1.8% |
| forms_dual_policy | 0.8 | 104 | 100.0% | 6.6% | 0 | 0 | 6.6% |
| forms_dual_policy | 0.85 | 103 | 100.0% | 6.6% | 0 | 0 | 6.6% |
| forms_dual_policy | 0.9 | 101 | 100.0% | 6.5% | 0 | 0 | 6.5% |
| forms_dual_policy | 0.95 | 81 | 100.0% | 5.2% | 0 | 0 | 5.2% |

## Decision rules fired: forms_dual

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| BOOK: all checks passed | 558 | 15 | 8 | 0 | 0 | 0 |
| ABSTAIN: both readers: not present | 0 | 0 | 0 | 0 | 16 | 490 |
| REVIEW: only one reader found a value | 1 | 0 | 0 | 2 | 229 | 43 |
| REVIEW: below threshold | 100 | 6 | 4 | 0 | 0 | 0 |
| REJECT: value not found in the text layer (groun | 43 | 0 | 3 | 0 | 0 | 0 |
| REVIEW: readers disagree | 7 | 9 | 30 | 0 | 0 | 0 |

## Decision rules fired: forms_dual_policy

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| REVIEW: not reliable on calibration | 553 | 21 | 12 | 0 | 0 | 0 |
| ABSTAIN: both readers: not present | 0 | 0 | 0 | 0 | 16 | 490 |
| REVIEW: only one reader found a value | 1 | 0 | 0 | 2 | 229 | 43 |
| BOOK: all checks passed | 101 | 0 | 0 | 0 | 0 | 0 |
| REJECT: value not found in the text layer (groun | 43 | 0 | 3 | 0 | 0 | 0 |
| REVIEW: below threshold | 4 | 0 | 0 | 0 | 0 | 0 |
| REVIEW: readers disagree | 7 | 9 | 30 | 0 | 0 | 0 |

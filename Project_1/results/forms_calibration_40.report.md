# Baseline vs hybrid on 40 contracts (dev_train_40_extractbench), threshold 0.9

| metric | forms_dual |
|---|---:|
| booking precision | 96.2% |
| recall | 73.6% |
| F1 | 0.834 |
| raw field accuracy | 94.8% |
| coverage (booked) | 52.7% |
| human review | 47.3% |
| booked wrong-clause | 10 |
| booked hallucinated | 0 |
| booked boundary err | 23 |
| wrong-clause rate (booked) | 1.14% |
| false acceptance | 40.7% |
| false rejection | 60.3% |
| raw wrong-clause answers | 31 |
| raw invented answers | 3 |
| net saving (1/100) | -145.4% |
| LLM calls / doc | 2.0 |
| input tokens / doc | 19,242 |
| output tokens / doc | 7,114 |
| LLM seconds / doc | 48.5 |
| USD / doc (assumed price) | $0.164 |

## Threshold sweep (replayed, no new API calls)

| config | t | booked | precision | coverage | booked wrong-clause | booked hallucinated | net saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| forms_dual | 0.8 | 954 | 95.5% | 57.3% | 15 | 0 | -200.8% |
| forms_dual | 0.85 | 942 | 95.5% | 56.5% | 14 | 0 | -195.6% |
| forms_dual | 0.9 | 878 | 96.2% | 52.7% | 10 | 0 | -145.4% |
| forms_dual | 0.95 | 514 | 100.0% | 30.9% | 0 | 0 | 30.9% |

## Decision rules fired: forms_dual

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| BOOK: all checks passed | 845 | 23 | 10 | 0 | 0 | 0 |
| ABSTAIN: both readers: not present | 0 | 0 | 0 | 0 | 3 | 515 |
| REJECT: value not found in the text layer (groun | 119 | 1 | 2 | 0 | 0 | 0 |
| REVIEW: below threshold | 83 | 6 | 7 | 0 | 0 | 0 |
| REVIEW: only one reader found a value | 2 | 1 | 0 | 2 | 2 | 0 |
| REVIEW: readers disagree | 15 | 16 | 12 | 1 | 0 | 0 |
| REVIEW: value found, but not next to its label ' | 1 | 0 | 0 | 0 | 0 | 0 |

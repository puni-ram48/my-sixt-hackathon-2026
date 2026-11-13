# Baseline vs hybrid on 60 contracts (dev_train_60_from160), threshold 0.9

| metric | G_baseline_gated |
|---|---:|
| booking precision | 100.0% |
| recall | 5.8% |
| F1 | 0.109 |
| raw field accuracy | 74.1% |
| coverage (booked) | 2.0% |
| human review | 98.0% |
| booked wrong-clause | 0 |
| booked hallucinated | 0 |
| booked boundary err | 0 |
| wrong-clause rate (booked) | 0.00% |
| false acceptance | 0.0% |
| false rejection | 92.3% |
| raw wrong-clause answers | 41 |
| raw invented answers | 116 |
| net saving (1/100) | 2.0% |
| LLM calls / doc | 1.0 |
| input tokens / doc | 16,209 |
| output tokens / doc | 2,104 |
| LLM seconds / doc | 11.1 |
| USD / doc (assumed price) | $0.080 |

## Threshold sweep (replayed, no new API calls)

| config | t | booked | precision | coverage | booked wrong-clause | booked hallucinated | net saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| G_baseline_gated | 0.8 | 57 | 96.5% | 2.3% | 0 | 0 | -5.8% |
| G_baseline_gated | 0.85 | 55 | 98.2% | 2.2% | 0 | 0 | -1.8% |
| G_baseline_gated | 0.9 | 48 | 100.0% | 2.0% | 0 | 0 | 2.0% |
| G_baseline_gated | 0.95 | 25 | 100.0% | 1.0% | 0 | 0 | 1.0% |

## Decision rules fired: G_baseline_gated

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| ABSTAIN: no supporting clause | 0 | 0 | 0 | 0 | 248 | 1513 |
| REVIEW: not reliable on calibration | 125 | 52 | 10 | 21 | 0 | 0 |
| REVIEW: clause type uncertain | 40 | 38 | 14 | 33 | 0 | 0 |
| REJECT: is not a | 31 | 25 | 8 | 51 | 0 | 0 |
| REVIEW: not whole sentences | 38 | 58 | 4 | 11 | 0 | 0 |
| REVIEW: probably incomplete | 18 | 54 | 5 | 0 | 0 | 0 |
| BOOK: all checks passed | 48 | 0 | 0 | 0 | 0 | 0 |
| REVIEW: below threshold | 10 | 3 | 0 | 0 | 0 | 0 |
| REJECT: not found in document | 0 | 2 | 0 | 0 | 0 | 0 |

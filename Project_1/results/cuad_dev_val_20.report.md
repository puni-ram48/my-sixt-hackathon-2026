# Baseline vs hybrid on 20 contracts (dev_val_20), threshold 0.9

| metric | baseline | A_bm25 | B_embed | C_hybrid | D_hybrid_verify | E_agent | F_agent_verify | G_baseline_gated | H_baseline_gated_verify | H_verify_gpt | H_verify_gemini | FINAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| booking precision | 76.9% | 65.9% | 59.1% | 71.7% | 71.7% | 52.6% | 52.6% | 95.0% | 94.1% | 94.1% | 93.8% | 100.0% |
| recall | 17.5% | 12.7% | 11.4% | 16.7% | 16.7% | 13.2% | 13.2% | 8.3% | 7.0% | 7.0% | 6.6% | 4.8% |
| F1 | 0.286 | 0.213 | 0.191 | 0.270 | 0.270 | 0.211 | 0.211 | 0.153 | 0.131 | 0.131 | 0.123 | 0.092 |
| raw field accuracy | 77.4% | 77.2% | 77.0% | 79.5% | 79.5% | 77.4% | 77.4% | 77.4% | 77.4% | 77.4% | 77.4% | 77.4% |
| coverage (booked) | 6.3% | 5.4% | 5.4% | 6.5% | 6.5% | 7.0% | 7.0% | 2.4% | 2.1% | 2.1% | 2.0% | 1.3% |
| human review | 93.7% | 94.6% | 94.6% | 93.5% | 93.5% | 93.0% | 93.0% | 97.6% | 97.9% | 97.9% | 98.0% | 98.7% |
| booked wrong-clause | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 0 | 0 | 0 | 0 | 0 |
| booked hallucinated | 2 | 7 | 5 | 5 | 5 | 3 | 3 | 1 | 1 | 1 | 1 | 0 |
| booked boundary err | 10 | 8 | 13 | 10 | 10 | 22 | 22 | 0 | 0 | 0 | 0 | 0 |
| wrong-clause rate (booked) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 3.51% | 3.51% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| false acceptance | 9.8% | 15.0% | 18.0% | 11.0% | 11.0% | 21.6% | 21.6% | 0.8% | 0.8% | 0.8% | 0.8% | 0.0% |
| false rejection | 76.2% | 81.4% | 83.8% | 82.1% | 82.1% | 77.9% | 77.9% | 88.7% | 90.5% | 90.5% | 91.1% | 93.5% |
| raw wrong-clause answers | 8 | 13 | 11 | 10 | 10 | 13 | 13 | 8 | 8 | 8 | 8 | 8 |
| raw invented answers | 41 | 37 | 41 | 46 | 46 | 25 | 25 | 41 | 41 | 41 | 41 | 41 |
| net saving (1/100) | -140.0% | -177.6% | -214.1% | -176.5% | -176.5% | -322.3% | -322.3% | -9.8% | -10.1% | -10.1% | -10.2% | 1.3% |
| LLM calls / doc | 1.0 | 3.0 | 3.0 | 3.0 | 4.5 | 3.5 | 4.9 | 1.0 | 2.4 | 2.4 | 2.4 | 1.0 |
| input tokens / doc | 13,672 | 19,060 | 16,490 | 21,667 | 33,830 | 42,769 | 51,116 | 13,672 | 22,343 | 18,987 | 19,173 | 13,672 |
| output tokens / doc | 1,909 | 3,014 | 3,023 | 3,770 | 4,924 | 4,041 | 4,909 | 1,909 | 2,858 | 4,701 | 6,513 | 1,909 |
| LLM seconds / doc | 10.2 | 17.2 | 17.1 | 20.6 | 28.4 | 24.1 | 32.3 | 10.2 | 17.1 | 46.0 | 52.0 | 10.2 |
| USD / doc (assumed price) | $0.070 | $0.102 | $0.095 | $0.122 | $0.175 | $0.189 | $0.227 | $0.070 | $0.110 | $0.127 | $0.155 | $0.070 |

## Threshold sweep (replayed, no new API calls)

| config | t | booked | precision | coverage | booked wrong-clause | booked hallucinated | net saving |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.8 | 79 | 69.6% | 9.6% | 2 | 7 | -283.0% |
| baseline | 0.85 | 64 | 71.9% | 7.8% | 1 | 3 | -211.7% |
| baseline | 0.9 | 52 | 76.9% | 6.3% | 0 | 2 | -140.0% |
| baseline | 0.95 | 15 | 93.3% | 1.8% | 0 | 0 | -10.4% |
| A_bm25 | 0.8 | 68 | 55.9% | 8.3% | 2 | 11 | -357.6% |
| A_bm25 | 0.85 | 56 | 62.5% | 6.8% | 1 | 8 | -249.3% |
| A_bm25 | 0.9 | 44 | 65.9% | 5.4% | 0 | 7 | -177.6% |
| A_bm25 | 0.95 | 22 | 72.7% | 2.7% | 0 | 2 | -70.5% |
| B_embed | 0.8 | 79 | 53.2% | 9.6% | 3 | 10 | -441.6% |
| B_embed | 0.85 | 64 | 56.2% | 7.8% | 2 | 7 | -333.7% |
| B_embed | 0.9 | 44 | 59.1% | 5.4% | 0 | 5 | -214.1% |
| B_embed | 0.95 | 17 | 70.6% | 2.1% | 0 | 0 | -58.9% |
| C_hybrid | 0.8 | 68 | 72.1% | 8.3% | 0 | 5 | -223.4% |
| C_hybrid | 0.85 | 65 | 70.8% | 7.9% | 0 | 5 | -223.8% |
| C_hybrid | 0.9 | 53 | 71.7% | 6.5% | 0 | 5 | -176.5% |
| C_hybrid | 0.95 | 19 | 68.4% | 2.3% | 0 | 2 | -70.9% |
| D_hybrid_verify | 0.8 | 67 | 71.6% | 8.2% | 0 | 5 | -223.5% |
| D_hybrid_verify | 0.85 | 64 | 70.3% | 7.8% | 0 | 5 | -223.9% |
| D_hybrid_verify | 0.9 | 53 | 71.7% | 6.5% | 0 | 5 | -176.5% |
| D_hybrid_verify | 0.95 | 19 | 68.4% | 2.3% | 0 | 2 | -70.9% |
| E_agent | 0.8 | 83 | 53.0% | 10.1% | 6 | 7 | -465.5% |
| E_agent | 0.85 | 71 | 54.9% | 8.7% | 3 | 6 | -381.6% |
| E_agent | 0.9 | 57 | 52.6% | 7.0% | 2 | 3 | -322.3% |
| E_agent | 0.95 | 20 | 50.0% | 2.4% | 1 | 0 | -119.5% |
| F_agent_verify | 0.8 | 81 | 54.3% | 9.9% | 5 | 7 | -441.3% |
| F_agent_verify | 0.85 | 71 | 54.9% | 8.7% | 3 | 6 | -381.6% |
| F_agent_verify | 0.9 | 57 | 52.6% | 7.0% | 2 | 3 | -322.3% |
| F_agent_verify | 0.95 | 20 | 50.0% | 2.4% | 1 | 0 | -119.5% |
| G_baseline_gated | 0.8 | 22 | 90.9% | 2.7% | 1 | 1 | -21.7% |
| G_baseline_gated | 0.85 | 21 | 90.5% | 2.6% | 1 | 1 | -21.8% |
| G_baseline_gated | 0.9 | 20 | 95.0% | 2.4% | 0 | 1 | -9.8% |
| G_baseline_gated | 0.95 | 11 | 100.0% | 1.3% | 0 | 0 | 1.3% |
| H_baseline_gated_verify | 0.8 | 17 | 94.1% | 2.1% | 0 | 1 | -10.1% |
| H_baseline_gated_verify | 0.85 | 17 | 94.1% | 2.1% | 0 | 1 | -10.1% |
| H_baseline_gated_verify | 0.9 | 17 | 94.1% | 2.1% | 0 | 1 | -10.1% |
| H_baseline_gated_verify | 0.95 | 10 | 100.0% | 1.2% | 0 | 0 | 1.2% |
| H_verify_gpt | 0.8 | 17 | 94.1% | 2.1% | 0 | 1 | -10.1% |
| H_verify_gpt | 0.85 | 17 | 94.1% | 2.1% | 0 | 1 | -10.1% |
| H_verify_gpt | 0.9 | 17 | 94.1% | 2.1% | 0 | 1 | -10.1% |
| H_verify_gpt | 0.95 | 10 | 100.0% | 1.2% | 0 | 0 | 1.2% |
| H_verify_gemini | 0.8 | 16 | 93.8% | 2.0% | 0 | 1 | -10.2% |
| H_verify_gemini | 0.85 | 16 | 93.8% | 2.0% | 0 | 1 | -10.2% |
| H_verify_gemini | 0.9 | 16 | 93.8% | 2.0% | 0 | 1 | -10.2% |
| H_verify_gemini | 0.95 | 10 | 100.0% | 1.2% | 0 | 0 | 1.2% |
| FINAL | 0.8 | 22 | 90.9% | 2.7% | 1 | 1 | -21.7% |
| FINAL | 0.85 | 21 | 90.5% | 2.6% | 1 | 1 | -21.8% |
| FINAL | 0.9 | 20 | 95.0% | 2.4% | 0 | 1 | -9.8% |
| FINAL | 0.95 | 11 | 100.0% | 1.3% | 0 | 0 | 1.3% |

## Decision rules fired: A_bm25

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| ABSTAIN: no supporting clause | 0 | 0 | 0 | 0 | 87 | 555 |
| REVIEW: below threshold | 19 | 16 | 8 | 19 | 0 | 0 |
| BOOK: all checks passed | 29 | 8 | 0 | 7 | 0 | 0 |
| REVIEW: not whole sentences | 14 | 20 | 2 | 8 | 0 | 0 |
| REVIEW: not located | 11 | 4 | 0 | 1 | 0 | 0 |
| ABSTAIN: did not mark | 5 | 2 | 3 | 2 | 0 | 0 |

## Decision rules fired: B_embed

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| ABSTAIN: no supporting clause | 0 | 0 | 0 | 0 | 89 | 551 |
| REVIEW: below threshold | 29 | 18 | 8 | 19 | 0 | 0 |
| BOOK: all checks passed | 26 | 13 | 0 | 5 | 0 | 0 |
| REVIEW: not whole sentences | 18 | 15 | 1 | 6 | 0 | 0 |
| ABSTAIN: did not mark | 1 | 1 | 2 | 9 | 0 | 0 |
| REVIEW: not located | 6 | 1 | 0 | 2 | 0 | 0 |

## Decision rules fired: C_hybrid

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| ABSTAIN: no supporting clause | 0 | 0 | 0 | 0 | 32 | 546 |
| BOOK: all checks passed | 38 | 10 | 0 | 5 | 0 | 0 |
| REVIEW: clause type uncertain | 17 | 15 | 6 | 10 | 0 | 0 |
| REJECT: is not a | 11 | 11 | 2 | 15 | 0 | 0 |
| REVIEW: not whole sentences | 12 | 23 | 0 | 3 | 0 | 0 |
| REVIEW: below threshold | 19 | 10 | 1 | 7 | 0 | 0 |
| REVIEW: probably incomplete | 6 | 10 | 0 | 0 | 0 | 0 |
| ABSTAIN: did not mark | 1 | 0 | 1 | 6 | 0 | 0 |
| REVIEW: not located | 2 | 1 | 0 | 0 | 0 | 0 |

## Decision rules fired: D_hybrid_verify

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| ABSTAIN: no supporting clause | 0 | 0 | 0 | 0 | 32 | 546 |
| BOOK: all checks passed | 38 | 10 | 0 | 5 | 0 | 0 |
| REJECT: is not a | 11 | 12 | 2 | 15 | 0 | 0 |
| REVIEW: clause type uncertain | 16 | 9 | 3 | 7 | 0 | 0 |
| REVIEW: not whole sentences | 12 | 19 | 0 | 2 | 0 | 0 |
| REVIEW: verifier: AMBIGUOUS | 6 | 10 | 4 | 6 | 0 | 0 |
| REVIEW: below threshold | 14 | 6 | 0 | 4 | 0 | 0 |
| REVIEW: probably incomplete | 4 | 9 | 0 | 0 | 0 | 0 |
| ABSTAIN: did not mark | 1 | 0 | 1 | 6 | 0 | 0 |
| REJECT: verifier: UNSUPPORTED | 2 | 3 | 0 | 1 | 0 | 0 |
| REVIEW: not located | 2 | 1 | 0 | 0 | 0 | 0 |
| REVIEW: is not a | 0 | 1 | 0 | 0 | 0 | 0 |

## Decision rules fired: E_agent

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| ABSTAIN: no supporting clause | 0 | 0 | 0 | 0 | 42 | 544 |
| REVIEW: not whole sentences | 16 | 45 | 2 | 8 | 0 | 0 |
| BOOK: all checks passed | 30 | 22 | 2 | 3 | 0 | 0 |
| REVIEW: below threshold | 20 | 16 | 8 | 13 | 0 | 0 |
| ABSTAIN: extraction failed | 0 | 0 | 0 | 0 | 18 | 23 |
| REVIEW: not located | 2 | 1 | 1 | 1 | 0 | 0 |
| REJECT: not found in document | 0 | 2 | 0 | 0 | 0 | 0 |
| ABSTAIN: did not mark | 0 | 1 | 0 | 0 | 0 | 0 |

## Decision rules fired: F_agent_verify

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| ABSTAIN: no supporting clause | 0 | 0 | 0 | 0 | 42 | 544 |
| BOOK: all checks passed | 30 | 22 | 2 | 3 | 0 | 0 |
| REVIEW: not whole sentences | 12 | 37 | 1 | 6 | 0 | 0 |
| REVIEW: below threshold | 18 | 11 | 5 | 10 | 0 | 0 |
| ABSTAIN: extraction failed | 0 | 0 | 0 | 0 | 18 | 23 |
| REVIEW: verifier: AMBIGUOUS | 5 | 13 | 3 | 3 | 0 | 0 |
| REJECT: verifier: UNSUPPORTED | 3 | 0 | 2 | 2 | 0 | 0 |
| REJECT: not found in document | 0 | 2 | 0 | 0 | 0 | 0 |
| REVIEW: not located | 0 | 1 | 0 | 1 | 0 | 0 |
| ABSTAIN: did not mark | 0 | 1 | 0 | 0 | 0 | 0 |

## Decision rules fired: G_baseline_gated

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| ABSTAIN: no supporting clause | 0 | 0 | 0 | 0 | 63 | 551 |
| REVIEW: not reliable on calibration | 25 | 14 | 2 | 6 | 0 | 0 |
| REJECT: is not a | 10 | 9 | 3 | 19 | 0 | 0 |
| REVIEW: clause type uncertain | 14 | 12 | 2 | 9 | 0 | 0 |
| REVIEW: not whole sentences | 12 | 19 | 0 | 4 | 0 | 0 |
| REVIEW: probably incomplete | 3 | 19 | 0 | 1 | 0 | 0 |
| BOOK: all checks passed | 19 | 0 | 0 | 1 | 0 | 0 |
| REVIEW: below threshold | 1 | 0 | 1 | 1 | 0 | 0 |

## Decision rules fired: H_baseline_gated_verify

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| ABSTAIN: no supporting clause | 0 | 0 | 0 | 0 | 63 | 551 |
| REJECT: is not a | 10 | 9 | 3 | 19 | 0 | 0 |
| REVIEW: not reliable on calibration | 24 | 9 | 1 | 5 | 0 | 0 |
| REVIEW: clause type uncertain | 17 | 8 | 3 | 6 | 0 | 0 |
| REVIEW: not whole sentences | 11 | 16 | 0 | 4 | 0 | 0 |
| REVIEW: verifier: AMBIGUOUS | 4 | 14 | 1 | 4 | 0 | 0 |
| BOOK: all checks passed | 16 | 0 | 0 | 1 | 0 | 0 |
| REJECT: verifier: UNSUPPORTED | 1 | 8 | 0 | 2 | 0 | 0 |
| REVIEW: probably incomplete | 1 | 9 | 0 | 0 | 0 | 0 |

## Decision rules fired: H_verify_gpt

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| ABSTAIN: no supporting clause | 0 | 0 | 0 | 0 | 63 | 551 |
| REJECT: is not a | 10 | 9 | 3 | 19 | 0 | 0 |
| REVIEW: not reliable on calibration | 23 | 10 | 0 | 5 | 0 | 0 |
| REVIEW: clause type uncertain | 15 | 10 | 2 | 5 | 0 | 0 |
| REVIEW: not whole sentences | 10 | 15 | 0 | 3 | 0 | 0 |
| REVIEW: verifier: AMBIGUOUS | 6 | 8 | 3 | 5 | 0 | 0 |
| REJECT: verifier: UNSUPPORTED | 2 | 15 | 0 | 3 | 0 | 0 |
| BOOK: all checks passed | 16 | 0 | 0 | 1 | 0 | 0 |
| REVIEW: probably incomplete | 2 | 6 | 0 | 0 | 0 | 0 |

## Decision rules fired: H_verify_gemini

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| ABSTAIN: no supporting clause | 0 | 0 | 0 | 0 | 63 | 551 |
| REJECT: is not a | 10 | 9 | 3 | 19 | 0 | 0 |
| REVIEW: not reliable on calibration | 23 | 11 | 0 | 5 | 0 | 0 |
| REVIEW: clause type uncertain | 16 | 7 | 2 | 6 | 0 | 0 |
| REVIEW: not whole sentences | 10 | 15 | 0 | 3 | 0 | 0 |
| REJECT: verifier: UNSUPPORTED | 2 | 14 | 1 | 4 | 0 | 0 |
| REVIEW: verifier: AMBIGUOUS | 6 | 9 | 2 | 3 | 0 | 0 |
| BOOK: all checks passed | 15 | 0 | 0 | 1 | 0 | 0 |
| REVIEW: probably incomplete | 2 | 8 | 0 | 0 | 0 | 0 |

## Decision rules fired: FINAL

| rule | correct | boundary | wrong_clause | hallucinated | missed | true_null |
|---|---:|---:|---:|---:|---:|---:|
| ABSTAIN: no supporting clause | 0 | 0 | 0 | 0 | 63 | 551 |
| REVIEW: not reliable on calibration | 25 | 14 | 2 | 6 | 0 | 0 |
| REJECT: is not a | 10 | 9 | 3 | 19 | 0 | 0 |
| REVIEW: clause type uncertain | 14 | 12 | 2 | 9 | 0 | 0 |
| REVIEW: not whole sentences | 12 | 19 | 0 | 4 | 0 | 0 |
| REVIEW: probably incomplete | 3 | 19 | 0 | 1 | 0 | 0 |
| REVIEW: below threshold | 9 | 0 | 1 | 2 | 0 | 0 |
| BOOK: all checks passed | 11 | 0 | 0 | 0 | 0 | 0 |

# Choice vs Ranking Comparison: pog

## Summary
| view | common_n | choice_hit_at_1 | ranking_hit_at_1 | ranking_minus_choice_hit_at_1 | ranking_hit_at_3 | ranking_mrr | top1_same_prediction_ratio | choice_valid_ratio | ranking_valid_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BIxIB | 250 | 0.3240 | 0.3800 | 0.0560 | 0.7040 | 0.5749 | 0.6280 | 1.0000 | 1.0000 |
| IUxUI | 250 | 0.2520 | 0.2960 | 0.0440 | 0.6240 | 0.5118 | 0.6560 | 1.0000 | 1.0000 |
| IBxBI | 250 | 0.2960 | 0.3240 | 0.0280 | 0.6720 | 0.5381 | 0.6640 | 1.0000 | 1.0000 |

## Outcome Counts
| view | outcome | count | ratio |
| --- | --- | --- | --- |
| BIxIB | both_fail | 141 | 0.5640 |
| BIxIB | both_hit | 67 | 0.2680 |
| BIxIB | choice_only | 14 | 0.0560 |
| BIxIB | ranking_only | 28 | 0.1120 |
| IBxBI | both_fail | 152 | 0.6080 |
| IBxBI | both_hit | 57 | 0.2280 |
| IBxBI | choice_only | 17 | 0.0680 |
| IBxBI | ranking_only | 24 | 0.0960 |
| IUxUI | both_fail | 167 | 0.6680 |
| IUxUI | both_hit | 54 | 0.2160 |
| IUxUI | choice_only | 9 | 0.0360 |
| IUxUI | ranking_only | 20 | 0.0800 |

## Interpretation
All three pairs share the same 250 sample/candidate sets, so the difference here is attributable to the choice vs ranking task/prompt rather than sample mismatch.
# CF Signal Analysis

Tie handling: `min_rank` is optimistic, `avg_rank` is neutral, `in_top_tie` accepts any candidate sharing the maximum score, and `unique_top1` only counts a single strictly highest candidate.

## CF Signal Quality

| signal | n | gt_in_top_tie_rate | gt_unique_top1_rate | gt_mean_min_rank | gt_mean_avg_rank | gt_mean_score | mean_top_score | mean_top_tie_size | all_zero_rate | all_tied_rate | positive_signal_rate | mean_top_second_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cooccurrence | 250 | 100.0% | 4.8% | 1.00 | 5.28 | 0.08 | 0.08 | 9.57 | 95.2% | 95.2% | 4.8% | 0.08 |
| user_pref | 250 | 99.6% | 2.0% | 1.00 | 5.41 | 0.35 | 0.39 | 9.78 | 97.6% | 97.6% | 2.4% | 0.39 |

## LLM Reliance On CF Score

| method | signal | n | valid_pred_n | accuracy | pred_in_top_tie_rate | pred_unique_top1_rate | pred_mean_min_rank | pred_mean_avg_rank | pred_mean_score | hit_when_pred_in_top_tie | hit_when_pred_not_in_top_tie | pred_in_top_tie_rate_when_positive | pred_unique_top1_rate_when_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | cooccurrence | 250 | 250 | 32.8% | 96.4% | 1.2% | 1.04 | 5.46 | 0.01 | 34.0% | 0.0% | 25.0% | 25.0% |
| co_occur | cooccurrence | 249 | 249 | 35.3% | 100.0% | 4.8% | 1.00 | 5.28 | 0.08 | 35.3% |  | 100.0% | 100.0% |
| user_prefer | cooccurrence | 250 | 250 | 32.4% | 96.4% | 1.2% | 1.04 | 5.46 | 0.01 | 33.6% | 0.0% | 25.0% | 25.0% |
| base | user_pref | 250 | 250 | 32.8% | 98.8% | 1.2% | 1.01 | 5.45 | 0.23 | 33.2% | 0.0% | 50.0% | 50.0% |
| co_occur | user_pref | 249 | 249 | 35.3% | 98.4% | 0.8% | 1.02 | 5.47 | 0.15 | 35.9% | 0.0% | 33.3% | 33.3% |
| user_prefer | user_pref | 250 | 250 | 32.4% | 99.6% | 2.0% | 1.00 | 5.41 | 0.35 | 32.5% | 0.0% | 83.3% | 83.3% |

## GT/Prediction CF Quadrants

| method | signal | gt_cf_group | pred_cf_group | n | accuracy | mean_gt_avg_rank | mean_pred_avg_rank | mean_top_tie_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | cooccurrence | gt_top | pred_not_top | 9 | 0.0% | 1.00 | 6.00 | 1.00 |
| base | cooccurrence | gt_top | pred_top | 241 | 34.0% | 5.44 | 5.44 | 9.89 |
| co_occur | cooccurrence | gt_top | pred_top | 249 | 35.3% | 5.28 | 5.28 | 9.57 |
| user_prefer | cooccurrence | gt_top | pred_not_top | 9 | 0.0% | 1.00 | 6.00 | 1.00 |
| user_prefer | cooccurrence | gt_top | pred_top | 241 | 33.6% | 5.44 | 5.44 | 9.89 |
| base | user_pref | gt_not_top | pred_not_top | 1 | 0.0% | 6.00 | 6.00 | 1.00 |
| base | user_pref | gt_top | pred_not_top | 2 | 0.0% | 1.00 | 6.00 | 1.00 |
| base | user_pref | gt_top | pred_top | 247 | 33.2% | 5.45 | 5.45 | 9.89 |
| co_occur | user_pref | gt_not_top | pred_not_top | 1 | 0.0% | 6.00 | 6.00 | 1.00 |
| co_occur | user_pref | gt_top | pred_not_top | 3 | 0.0% | 1.00 | 6.00 | 1.00 |
| co_occur | user_pref | gt_top | pred_top | 245 | 35.9% | 5.46 | 5.46 | 9.93 |
| user_prefer | user_pref | gt_not_top | pred_not_top | 1 | 0.0% | 6.00 | 6.00 | 1.00 |
| user_prefer | user_pref | gt_top | pred_top | 249 | 32.5% | 5.41 | 5.41 | 9.82 |

## Pairwise Method Delta

| signal | left_method | right_method | n | left_accuracy | right_accuracy | left_only | right_only | both_hit | both_fail | right_only_gt_in_top_tie_rate | right_only_gt_unique_top1_rate | right_only_pred_in_top_tie_rate | right_only_pred_unique_top1_rate | right_only_mean_gt_avg_rank | right_only_mean_pred_avg_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cooccurrence | base | co_occur | 249 | 32.9% | 35.3% | 23 | 29 | 59 | 138 | 100.0% | 31.0% | 100.0% | 31.0% | 4.10 | 4.10 |
| cooccurrence | base | user_prefer | 250 | 32.8% | 32.4% | 19 | 18 | 63 | 150 | 100.0% | 0.0% | 100.0% | 0.0% | 5.50 | 5.50 |
| cooccurrence | co_occur | user_prefer | 249 | 35.3% | 32.5% | 21 | 14 | 67 | 147 | 100.0% | 0.0% | 100.0% | 0.0% | 5.50 | 5.50 |
| user_pref | base | co_occur | 249 | 32.9% | 35.3% | 23 | 29 | 59 | 138 | 100.0% | 0.0% | 100.0% | 0.0% | 5.50 | 5.50 |
| user_pref | base | user_prefer | 250 | 32.8% | 32.4% | 19 | 18 | 63 | 150 | 100.0% | 11.1% | 100.0% | 11.1% | 5.00 | 5.00 |
| user_pref | co_occur | user_prefer | 249 | 35.3% | 32.5% | 21 | 14 | 67 | 147 | 100.0% | 21.4% | 100.0% | 21.4% | 4.54 | 4.54 |

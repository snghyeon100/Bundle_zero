# CF Signal Analysis

Tie handling: `min_rank` is optimistic, `avg_rank` is neutral, `in_top_tie` accepts any candidate sharing the maximum score, and `unique_top1` only counts a single strictly highest candidate.

## CF Signal Quality

| signal | n | gt_in_top_tie_rate | gt_unique_top1_rate | gt_mean_min_rank | gt_mean_avg_rank | gt_mean_score | mean_top_score | mean_top_tie_size | all_zero_rate | all_tied_rate | positive_signal_rate | mean_top_second_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cooccurrence | 250 | 84.0% | 44.0% | 1.25 | 3.37 | 3.94 | 4.20 | 4.24 | 34.4% | 34.4% | 65.6% | 4.00 |
| user_pref | 250 | 74.8% | 12.8% | 1.34 | 4.88 | 0.13 | 0.22 | 6.34 | 58.4% | 58.4% | 41.6% | 0.20 |

## LLM Reliance On CF Score

| method | signal | n | valid_pred_n | accuracy | pred_in_top_tie_rate | pred_unique_top1_rate | pred_mean_min_rank | pred_mean_avg_rank | pred_mean_score | hit_when_pred_in_top_tie | hit_when_pred_not_in_top_tie | pred_in_top_tie_rate_when_positive | pred_unique_top1_rate_when_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | cooccurrence | 250 | 250 | 33.6% | 58.8% | 19.6% | 1.60 | 4.61 | 1.78 | 52.4% | 6.8% | 37.2% | 29.9% |
| co_occer_cf | cooccurrence | 250 | 250 | 55.6% | 99.6% | 54.8% | 1.00 | 2.64 | 4.20 | 55.8% | 0.0% | 99.4% | 83.5% |
| user_prefer_cf | cooccurrence | 250 | 250 | 38.0% | 61.2% | 23.6% | 1.56 | 4.45 | 2.56 | 56.2% | 9.3% | 40.9% | 36.0% |
| base | user_pref | 250 | 250 | 33.6% | 70.4% | 8.4% | 1.39 | 5.09 | 0.07 | 37.5% | 24.3% | 28.8% | 20.2% |
| co_occer_cf | user_pref | 250 | 250 | 55.6% | 74.8% | 12.0% | 1.34 | 4.91 | 0.13 | 59.4% | 44.4% | 39.4% | 28.8% |
| user_prefer_cf | user_pref | 250 | 250 | 38.0% | 87.6% | 24.0% | 1.16 | 4.21 | 0.19 | 37.9% | 38.7% | 70.2% | 57.7% |

## GT/Prediction CF Quadrants

| method | signal | gt_cf_group | pred_cf_group | n | accuracy | mean_gt_avg_rank | mean_pred_avg_rank | mean_top_tie_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | cooccurrence | gt_not_top | pred_not_top | 33 | 21.2% | 5.94 | 6.21 | 1.39 |
| base | cooccurrence | gt_not_top | pred_top | 7 | 0.0% | 5.79 | 1.29 | 1.57 |
| base | cooccurrence | gt_top | pred_not_top | 70 | 0.0% | 1.04 | 5.78 | 1.07 |
| base | cooccurrence | gt_top | pred_top | 140 | 55.0% | 3.81 | 3.81 | 6.63 |
| co_occer_cf | cooccurrence | gt_not_top | pred_not_top | 1 | 0.0% | 6.00 | 6.00 | 1.00 |
| co_occer_cf | cooccurrence | gt_not_top | pred_top | 39 | 0.0% | 5.91 | 1.22 | 1.44 |
| co_occer_cf | cooccurrence | gt_top | pred_top | 210 | 66.2% | 2.89 | 2.89 | 4.78 |
| user_prefer_cf | cooccurrence | gt_not_top | pred_not_top | 33 | 27.3% | 5.92 | 5.79 | 1.36 |
| user_prefer_cf | cooccurrence | gt_not_top | pred_top | 7 | 0.0% | 5.86 | 1.36 | 1.71 |
| user_prefer_cf | cooccurrence | gt_top | pred_not_top | 64 | 0.0% | 1.11 | 5.88 | 1.22 |
| user_prefer_cf | cooccurrence | gt_top | pred_top | 146 | 58.9% | 3.67 | 3.67 | 6.34 |
| base | user_pref | gt_not_top | pred_not_top | 53 | 34.0% | 5.89 | 5.79 | 1.13 |
| base | user_pref | gt_not_top | pred_top | 10 | 0.0% | 5.85 | 1.20 | 1.40 |
| base | user_pref | gt_top | pred_not_top | 21 | 0.0% | 1.07 | 5.98 | 1.14 |
| base | user_pref | gt_top | pred_top | 166 | 39.8% | 4.98 | 4.98 | 8.96 |
| co_occer_cf | user_pref | gt_not_top | pred_not_top | 56 | 50.0% | 5.82 | 5.98 | 1.12 |
| co_occer_cf | user_pref | gt_not_top | pred_top | 7 | 0.0% | 6.36 | 1.29 | 1.57 |
| co_occer_cf | user_pref | gt_top | pred_not_top | 7 | 0.0% | 1.07 | 6.07 | 1.14 |
| co_occer_cf | user_pref | gt_top | pred_top | 180 | 61.7% | 4.68 | 4.68 | 8.36 |
| user_prefer_cf | user_pref | gt_not_top | pred_not_top | 29 | 41.4% | 5.95 | 5.48 | 1.17 |
| user_prefer_cf | user_pref | gt_not_top | pred_top | 34 | 0.0% | 5.82 | 1.09 | 1.18 |
| user_prefer_cf | user_pref | gt_top | pred_not_top | 2 | 0.0% | 1.25 | 4.25 | 1.50 |
| user_prefer_cf | user_pref | gt_top | pred_top | 185 | 44.9% | 4.58 | 4.58 | 8.16 |

## Pairwise Method Delta

| signal | left_method | right_method | n | left_accuracy | right_accuracy | left_only | right_only | both_hit | both_fail | right_only_gt_in_top_tie_rate | right_only_gt_unique_top1_rate | right_only_pred_in_top_tie_rate | right_only_pred_unique_top1_rate | right_only_mean_gt_avg_rank | right_only_mean_pred_avg_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cooccurrence | base | co_occer_cf | 250 | 33.6% | 55.6% | 19 | 74 | 65 | 92 | 100.0% | 89.2% | 100.0% | 89.2% | 1.39 | 1.39 |
| cooccurrence | base | user_prefer_cf | 250 | 33.6% | 38.0% | 23 | 34 | 61 | 132 | 88.2% | 70.6% | 88.2% | 70.6% | 2.34 | 2.34 |
| cooccurrence | co_occer_cf | user_prefer_cf | 250 | 55.6% | 38.0% | 61 | 17 | 78 | 94 | 47.1% | 0.0% | 47.1% | 0.0% | 4.88 | 4.88 |
| user_pref | base | co_occer_cf | 250 | 33.6% | 55.6% | 19 | 74 | 65 | 92 | 79.7% | 20.3% | 79.7% | 20.3% | 4.48 | 4.48 |
| user_pref | base | user_prefer_cf | 250 | 33.6% | 38.0% | 23 | 34 | 61 | 132 | 91.2% | 50.0% | 91.2% | 50.0% | 2.82 | 2.82 |
| user_pref | co_occer_cf | user_prefer_cf | 250 | 55.6% | 38.0% | 61 | 17 | 78 | 94 | 82.4% | 35.3% | 82.4% | 35.3% | 3.85 | 3.85 |

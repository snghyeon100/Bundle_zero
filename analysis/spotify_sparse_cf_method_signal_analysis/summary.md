# CF Signal Analysis

Tie handling: `min_rank` is optimistic, `avg_rank` is neutral, `in_top_tie` accepts any candidate sharing the maximum score, and `unique_top1` only counts a single strictly highest candidate.

## CF Signal Quality

| signal | n | gt_in_top_tie_rate | gt_unique_top1_rate | gt_mean_min_rank | gt_mean_avg_rank | gt_mean_score | mean_top_score | mean_top_tie_size | all_zero_rate | all_tied_rate | positive_signal_rate | mean_top_second_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cooccurrence | 250 | 74.4% | 19.6% | 1.50 | 4.29 | 2.75 | 3.21 | 5.33 | 44.4% | 44.4% | 55.6% | 2.13 |
| user_pref | 250 | 38.0% | 16.8% | 3.00 | 5.06 | 4.33 | 10.40 | 2.91 | 20.8% | 20.8% | 79.2% | 4.65 |

## LLM Reliance On CF Score

| method | signal | n | valid_pred_n | accuracy | pred_in_top_tie_rate | pred_unique_top1_rate | pred_mean_min_rank | pred_mean_avg_rank | pred_mean_score | hit_when_pred_in_top_tie | hit_when_pred_not_in_top_tie | pred_in_top_tie_rate_when_positive | pred_unique_top1_rate_when_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HN_base | cooccurrence | 250 | 250 | 46.0% | 70.0% | 17.2% | 1.59 | 4.58 | 2.59 | 50.9% | 34.7% | 46.0% | 30.9% |
| HN_co_occur | cooccurrence | 250 | 250 | 46.4% | 99.6% | 38.4% | 1.00 | 3.17 | 3.20 | 46.6% | 0.0% | 99.3% | 69.1% |
| HN_user_prefer | cooccurrence | 250 | 250 | 32.8% | 62.8% | 11.2% | 1.82 | 4.95 | 2.08 | 43.9% | 14.0% | 33.1% | 20.1% |
| HN_base | user_pref | 250 | 250 | 46.0% | 33.6% | 12.4% | 3.26 | 5.37 | 3.44 | 53.6% | 42.2% | 16.2% | 15.7% |
| HN_co_occur | user_pref | 250 | 250 | 46.4% | 39.2% | 17.6% | 2.98 | 4.96 | 4.29 | 48.0% | 45.4% | 23.2% | 22.2% |
| HN_user_prefer | user_pref | 250 | 250 | 32.8% | 92.8% | 68.8% | 1.13 | 2.26 | 10.22 | 29.3% | 77.8% | 90.9% | 86.9% |

## GT/Prediction CF Quadrants

| method | signal | gt_cf_group | pred_cf_group | n | accuracy | mean_gt_avg_rank | mean_pred_avg_rank | mean_top_tie_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HN_base | cooccurrence | gt_not_top | pred_not_top | 49 | 53.1% | 5.80 | 6.01 | 1.37 |
| HN_base | cooccurrence | gt_not_top | pred_top | 15 | 0.0% | 5.00 | 1.40 | 1.80 |
| HN_base | cooccurrence | gt_top | pred_not_top | 26 | 0.0% | 1.19 | 5.69 | 1.38 |
| HN_base | cooccurrence | gt_top | pred_top | 160 | 55.6% | 4.26 | 4.26 | 7.52 |
| HN_co_occur | cooccurrence | gt_not_top | pred_top | 64 | 0.0% | 5.61 | 1.23 | 1.47 |
| HN_co_occur | cooccurrence | gt_top | pred_not_top | 1 | 0.0% | 1.00 | 2.00 | 1.00 |
| HN_co_occur | cooccurrence | gt_top | pred_top | 185 | 62.7% | 3.85 | 3.85 | 6.69 |
| HN_user_prefer | cooccurrence | gt_not_top | pred_not_top | 48 | 27.1% | 5.62 | 6.30 | 1.40 |
| HN_user_prefer | cooccurrence | gt_not_top | pred_top | 16 | 0.0% | 5.56 | 1.34 | 1.69 |
| HN_user_prefer | cooccurrence | gt_top | pred_not_top | 45 | 0.0% | 1.24 | 5.73 | 1.49 |
| HN_user_prefer | cooccurrence | gt_top | pred_top | 141 | 48.9% | 4.66 | 4.66 | 8.31 |
| HN_base | user_pref | gt_not_top | pred_not_top | 142 | 49.3% | 6.14 | 6.19 | 1.05 |
| HN_base | user_pref | gt_not_top | pred_top | 13 | 0.0% | 4.88 | 1.04 | 1.08 |
| HN_base | user_pref | gt_top | pred_not_top | 24 | 0.0% | 1.04 | 6.06 | 1.08 |
| HN_base | user_pref | gt_top | pred_top | 71 | 63.4% | 4.30 | 4.30 | 7.59 |
| HN_co_occur | user_pref | gt_not_top | pred_not_top | 129 | 53.5% | 6.16 | 5.94 | 1.05 |
| HN_co_occur | user_pref | gt_not_top | pred_top | 26 | 0.0% | 5.46 | 1.04 | 1.08 |
| HN_co_occur | user_pref | gt_top | pred_not_top | 23 | 0.0% | 1.04 | 6.11 | 1.09 |
| HN_co_occur | user_pref | gt_top | pred_top | 72 | 65.3% | 4.25 | 4.25 | 7.50 |
| HN_user_prefer | user_pref | gt_not_top | pred_not_top | 18 | 77.8% | 5.67 | 5.22 | 1.06 |
| HN_user_prefer | user_pref | gt_not_top | pred_top | 137 | 0.0% | 6.09 | 1.03 | 1.05 |
| HN_user_prefer | user_pref | gt_top | pred_top | 95 | 71.6% | 3.47 | 3.47 | 5.95 |

## Pairwise Method Delta

| signal | left_method | right_method | n | left_accuracy | right_accuracy | left_only | right_only | both_hit | both_fail | right_only_gt_in_top_tie_rate | right_only_gt_unique_top1_rate | right_only_pred_in_top_tie_rate | right_only_pred_unique_top1_rate | right_only_mean_gt_avg_rank | right_only_mean_pred_avg_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cooccurrence | HN_base | HN_co_occur | 250 | 46.0% | 46.4% | 35 | 36 | 80 | 99 | 100.0% | 47.2% | 100.0% | 47.2% | 2.61 | 2.61 |
| cooccurrence | HN_base | HN_user_prefer | 250 | 46.0% | 32.8% | 61 | 28 | 54 | 107 | 85.7% | 7.1% | 85.7% | 7.1% | 4.46 | 4.46 |
| cooccurrence | HN_co_occur | HN_user_prefer | 250 | 46.4% | 32.8% | 61 | 27 | 55 | 107 | 51.9% | 0.0% | 51.9% | 0.0% | 5.31 | 5.31 |
| user_pref | HN_base | HN_co_occur | 250 | 46.0% | 46.4% | 35 | 36 | 80 | 99 | 38.9% | 19.4% | 38.9% | 19.4% | 5.04 | 5.04 |
| user_pref | HN_base | HN_user_prefer | 250 | 46.0% | 32.8% | 61 | 28 | 54 | 107 | 92.9% | 82.1% | 92.9% | 82.1% | 1.91 | 1.91 |
| user_pref | HN_co_occur | HN_user_prefer | 250 | 46.4% | 32.8% | 61 | 27 | 55 | 107 | 96.3% | 81.5% | 96.3% | 81.5% | 1.74 | 1.74 |

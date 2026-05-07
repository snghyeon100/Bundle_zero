# CF Signal Analysis

Tie handling: `min_rank` is optimistic, `avg_rank` is neutral, `in_top_tie` accepts any candidate sharing the maximum score, and `unique_top1` only counts a single strictly highest candidate.

## CF Signal Quality

| signal | n | gt_in_top_tie_rate | gt_unique_top1_rate | gt_mean_min_rank | gt_mean_avg_rank | gt_mean_score | mean_top_score | mean_top_tie_size | all_zero_rate | all_tied_rate | positive_signal_rate | mean_top_second_margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cooccurrence | 250 | 66.4% | 46.0% | 1.73 | 3.02 | 35.70 | 39.30 | 2.45 | 14.4% | 14.4% | 85.6% | 26.85 |
| user_pref | 250 | 36.8% | 22.0% | 3.10 | 4.97 | 4.91 | 8.99 | 2.26 | 13.6% | 13.6% | 86.4% | 4.82 |

## LLM Reliance On CF Score

| method | signal | n | valid_pred_n | accuracy | pred_in_top_tie_rate | pred_unique_top1_rate | pred_mean_min_rank | pred_mean_avg_rank | pred_mean_score | hit_when_pred_in_top_tie | hit_when_pred_not_in_top_tie | pred_in_top_tie_rate_when_positive | pred_unique_top1_rate_when_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HN_base | cooccurrence | 250 | 250 | 52.0% | 64.8% | 44.4% | 1.79 | 2.98 | 33.51 | 68.5% | 21.6% | 58.9% | 51.9% |
| HN_co_occur | cooccurrence | 250 | 250 | 56.4% | 99.2% | 74.4% | 1.01 | 1.73 | 39.28 | 56.5% | 50.0% | 99.1% | 86.9% |
| HN_user_prefer | cooccurrence | 250 | 250 | 30.8% | 46.0% | 26.8% | 2.43 | 4.00 | 18.61 | 57.4% | 8.1% | 36.9% | 31.3% |
| HN_base | user_pref | 250 | 250 | 52.0% | 32.8% | 18.4% | 3.14 | 5.07 | 4.70 | 63.4% | 46.4% | 22.2% | 21.3% |
| HN_co_occur | user_pref | 250 | 250 | 56.4% | 39.6% | 24.8% | 3.01 | 4.81 | 5.15 | 58.6% | 55.0% | 30.1% | 28.7% |
| HN_user_prefer | user_pref | 250 | 250 | 30.8% | 97.2% | 80.4% | 1.05 | 1.72 | 8.92 | 30.0% | 57.1% | 96.8% | 93.1% |

## GT/Prediction CF Quadrants

| method | signal | gt_cf_group | pred_cf_group | n | accuracy | mean_gt_avg_rank | mean_pred_avg_rank | mean_top_tie_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HN_base | cooccurrence | gt_not_top | pred_not_top | 58 | 32.8% | 4.88 | 4.54 | 1.14 |
| HN_base | cooccurrence | gt_not_top | pred_top | 26 | 0.0% | 5.06 | 1.13 | 1.27 |
| HN_base | cooccurrence | gt_top | pred_not_top | 30 | 0.0% | 1.08 | 4.80 | 1.17 |
| HN_base | cooccurrence | gt_top | pred_top | 136 | 81.6% | 2.26 | 2.26 | 3.51 |
| HN_co_occur | cooccurrence | gt_not_top | pred_not_top | 2 | 50.0% | 2.50 | 2.00 | 1.00 |
| HN_co_occur | cooccurrence | gt_not_top | pred_top | 82 | 0.0% | 4.99 | 1.09 | 1.18 |
| HN_co_occur | cooccurrence | gt_top | pred_top | 166 | 84.3% | 2.05 | 2.05 | 3.09 |
| HN_user_prefer | cooccurrence | gt_not_top | pred_not_top | 60 | 18.3% | 5.00 | 5.24 | 1.08 |
| HN_user_prefer | cooccurrence | gt_not_top | pred_top | 24 | 0.0% | 4.77 | 1.21 | 1.42 |
| HN_user_prefer | cooccurrence | gt_top | pred_not_top | 75 | 0.0% | 1.09 | 5.33 | 1.19 |
| HN_user_prefer | cooccurrence | gt_top | pred_top | 91 | 72.5% | 2.83 | 2.83 | 4.66 |
| HN_base | user_pref | gt_not_top | pred_not_top | 145 | 53.8% | 6.27 | 6.38 | 1.05 |
| HN_base | user_pref | gt_not_top | pred_top | 13 | 0.0% | 6.69 | 1.00 | 1.00 |
| HN_base | user_pref | gt_top | pred_not_top | 23 | 0.0% | 1.02 | 4.61 | 1.04 |
| HN_base | user_pref | gt_top | pred_top | 69 | 75.4% | 3.23 | 3.23 | 5.46 |
| HN_co_occur | user_pref | gt_not_top | pred_not_top | 135 | 61.5% | 6.41 | 6.43 | 1.02 |
| HN_co_occur | user_pref | gt_not_top | pred_top | 23 | 0.0% | 5.67 | 1.09 | 1.17 |
| HN_co_occur | user_pref | gt_top | pred_not_top | 16 | 0.0% | 1.06 | 5.09 | 1.12 |
| HN_co_occur | user_pref | gt_top | pred_top | 76 | 76.3% | 3.02 | 3.02 | 5.04 |
| HN_user_prefer | user_pref | gt_not_top | pred_not_top | 6 | 66.7% | 6.17 | 4.25 | 1.00 |
| HN_user_prefer | user_pref | gt_not_top | pred_top | 152 | 0.0% | 6.31 | 1.02 | 1.05 |
| HN_user_prefer | user_pref | gt_top | pred_not_top | 1 | 0.0% | 1.00 | 2.50 | 1.00 |
| HN_user_prefer | user_pref | gt_top | pred_top | 91 | 80.2% | 2.70 | 2.70 | 4.40 |

## Pairwise Method Delta

| signal | left_method | right_method | n | left_accuracy | right_accuracy | left_only | right_only | both_hit | both_fail | right_only_gt_in_top_tie_rate | right_only_gt_unique_top1_rate | right_only_pred_in_top_tie_rate | right_only_pred_unique_top1_rate | right_only_mean_gt_avg_rank | right_only_mean_pred_avg_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cooccurrence | HN_base | HN_co_occur | 250 | 52.0% | 56.4% | 23 | 34 | 107 | 86 | 97.1% | 73.5% | 97.1% | 73.5% | 1.63 | 1.63 |
| cooccurrence | HN_base | HN_user_prefer | 250 | 52.0% | 30.8% | 77 | 24 | 53 | 96 | 66.7% | 37.5% | 66.7% | 37.5% | 2.92 | 2.92 |
| cooccurrence | HN_co_occur | HN_user_prefer | 250 | 56.4% | 30.8% | 82 | 18 | 59 | 91 | 38.9% | 0.0% | 38.9% | 0.0% | 4.19 | 4.19 |
| user_pref | HN_base | HN_co_occur | 250 | 52.0% | 56.4% | 23 | 34 | 107 | 86 | 38.2% | 32.4% | 38.2% | 32.4% | 4.54 | 4.54 |
| user_pref | HN_base | HN_user_prefer | 250 | 52.0% | 30.8% | 77 | 24 | 53 | 96 | 100.0% | 87.5% | 100.0% | 87.5% | 1.23 | 1.23 |
| user_pref | HN_co_occur | HN_user_prefer | 250 | 56.4% | 30.8% | 82 | 18 | 59 | 91 | 100.0% | 72.2% | 100.0% | 72.2% | 1.81 | 1.81 |

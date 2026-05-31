# Ranking View Complementarity Analysis: pog_dense

## Input files
- IBxBI: `C:\Users\wotjs\Desktop\bundle\LLM-ZeroShot\results\pog_dense\results_pog_dense_RANK_ITEMAFF_HN_C10_T5_20260528_035357.csv`
- IUxUI: `C:\Users\wotjs\Desktop\bundle\LLM-ZeroShot\results\pog_dense\results_pog_dense_RANK_USERPUR_HN_C10_T5_20260528_031400.csv`
- BIxIB: `C:\Users\wotjs\Desktop\bundle\LLM-ZeroShot\results\pog_dense\results_pog_dense_RANK_BGRAPH_HN_C10_T5_20260527_132341.csv`

## Core claim
The oracle per-sample selector is an upper bound for a future view-selection agent. If it outperforms the best fixed view, then the three graph-derived views contain complementary evidence.

- Best fixed view by MRR: `BIxIB`
- Oracle Hit@1 gain over best fixed view: 0.2160
- Oracle MRR gain over best fixed view: 0.1613

## Summary
| method | type | n | hit_at_1 | hit_at_3 | hit_at_5 | mrr | ndcg_at_3 | ndcg_at_5 | ndcg_at_10 | mean_rank | median_rank | valid_rank_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IBxBI | single_view | 250 | 0.4280 | 0.7480 | 0.8600 | 0.6125 | 0.6131 | 0.6600 | 0.7059 | 2.7280 | 2.0000 | 1.0000 |
| IUxUI | single_view | 250 | 0.2920 | 0.6040 | 0.7960 | 0.5006 | 0.4737 | 0.5532 | 0.6195 | 3.4240 | 3.0000 | 1.0000 |
| BIxIB | single_view | 250 | 0.4360 | 0.7400 | 0.9000 | 0.6226 | 0.6147 | 0.6812 | 0.7142 | 2.5680 | 2.0000 | 1.0000 |
| OracleSelector | oracle | 250 | 0.6520 | 0.8960 | 0.9720 | 0.7839 | 0.7965 | 0.8280 | 0.8376 | 1.7320 | 1.0000 | 1.0000 |

## Unique oracle winner distribution
| winner | count | ratio |
| --- | --- | --- |
| Tie | 114 | 0.4560 |
| BIxIB | 56 | 0.2240 |
| IBxBI | 51 | 0.2040 |
| IUxUI | 29 | 0.1160 |

## Best-view tie cardinality
| tie_type | num_best_views | count | ratio |
| --- | --- | --- | --- |
| unique_best | 1 | 136 | 0.5440 |
| two_views_tied_best | 2 | 69 | 0.2760 |
| all_three_tied_best | 3 | 45 | 0.1800 |

## Exact best-view sets
| best_view_set | num_best_views | count | ratio |
| --- | --- | --- | --- |
| BIxIB | 1 | 56 | 0.2240 |
| IBxBI | 1 | 51 | 0.2040 |
| IBxBI|IUxUI|BIxIB | 3 | 45 | 0.1800 |
| IBxBI|BIxIB | 2 | 40 | 0.1600 |
| IUxUI | 1 | 29 | 0.1160 |
| IUxUI|BIxIB | 2 | 20 | 0.0800 |
| IBxBI|IUxUI | 2 | 9 | 0.0360 |

## Success overlap
| k | success_pattern | count | ratio |
| --- | --- | --- | --- |
| 1 | BIxIB | 27 | 0.1080 |
| 1 | IBxBI | 33 | 0.1320 |
| 1 | IBxBI|BIxIB | 30 | 0.1200 |
| 1 | IBxBI|IUxUI | 8 | 0.0320 |
| 1 | IBxBI|IUxUI|BIxIB | 36 | 0.1440 |
| 1 | IUxUI | 13 | 0.0520 |
| 1 | IUxUI|BIxIB | 16 | 0.0640 |
| 1 | None | 87 | 0.3480 |
| 3 | BIxIB | 15 | 0.0600 |
| 3 | IBxBI | 18 | 0.0720 |
| 3 | IBxBI|BIxIB | 40 | 0.1600 |
| 3 | IBxBI|IUxUI | 12 | 0.0480 |
| 3 | IBxBI|IUxUI|BIxIB | 117 | 0.4680 |
| 3 | IUxUI | 9 | 0.0360 |
| 3 | IUxUI|BIxIB | 13 | 0.0520 |
| 3 | None | 26 | 0.1040 |
| 5 | BIxIB | 13 | 0.0520 |
| 5 | IBxBI | 5 | 0.0200 |
| 5 | IBxBI|BIxIB | 26 | 0.1040 |
| 5 | IBxBI|IUxUI | 9 | 0.0360 |
| 5 | IBxBI|IUxUI|BIxIB | 175 | 0.7000 |
| 5 | IUxUI | 4 | 0.0160 |
| 5 | IUxUI|BIxIB | 11 | 0.0440 |
| 5 | None | 7 | 0.0280 |

## Success cardinality
| k | success_type | num_successful_views | count | ratio |
| --- | --- | --- | --- | --- |
| 1 | all_three_failed | 0 | 87 | 0.3480 |
| 1 | one_view_success | 1 | 73 | 0.2920 |
| 1 | two_views_success | 2 | 54 | 0.2160 |
| 1 | all_three_success | 3 | 36 | 0.1440 |
| 3 | all_three_failed | 0 | 26 | 0.1040 |
| 3 | one_view_success | 1 | 42 | 0.1680 |
| 3 | two_views_success | 2 | 65 | 0.2600 |
| 3 | all_three_success | 3 | 117 | 0.4680 |
| 5 | all_three_failed | 0 | 7 | 0.0280 |
| 5 | one_view_success | 1 | 22 | 0.0880 |
| 5 | two_views_success | 2 | 46 | 0.1840 |
| 5 | all_three_success | 3 | 175 | 0.7000 |

## Hit@1 margin by focal view
| view | hit_at_1_count | focal_only_hit_at_1_count | focal_only_hit_at_1_ratio | shared_with_one_other_count | shared_with_two_others_count | best_other_rank_mean | best_other_rank_median | best_other_margin_mean | best_other_margin_median | best_other_rank_ge_3_count | best_other_rank_ge_3_ratio | best_other_rank_ge_5_count | best_other_rank_ge_5_ratio | worst_other_rank_mean | worst_other_rank_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IBxBI | 107 | 33 | 0.3084 | 38 | 36 | 1.4860 | 1.0000 | 0.4860 | 0.0000 | 11 | 0.1028 | 1 | 0.0093 | 2.8224 | 2.0000 |
| IUxUI | 73 | 13 | 0.1781 | 24 | 36 | 1.3699 | 1.0000 | 0.3699 | 0.0000 | 7 | 0.0959 | 1 | 0.0137 | 2.4521 | 2.0000 |
| BIxIB | 109 | 27 | 0.2477 | 46 | 36 | 1.4220 | 1.0000 | 0.4220 | 0.0000 | 9 | 0.0826 | 3 | 0.0275 | 2.9908 | 2.0000 |

## Unique Hit@1 Other-View Ranks
| unique_hit_view | count | avg_rank_IUxUI | median_rank_IUxUI | avg_rank_BIxIB | median_rank_BIxIB | avg_rank_IBxBI | median_rank_IBxBI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IBxBI | 33 | 4.0000 | 4.0000 | 2.8788 | 2.0000 | nan | nan |
| IUxUI | 13 | nan | nan | 4.7692 | 4.0000 | 3.4615 | 3.0000 |
| BIxIB | 27 | 4.4074 | 4.0000 | nan | nan | 3.2593 | 2.0000 |

## Two-View Hit@1 Remaining-View Ranks
| hit_view_pair | remaining_view | count | avg_rank_remaining_view | median_rank_remaining_view |
| --- | --- | --- | --- | --- |
| IBxBI|IUxUI | BIxIB | 8 | 2.7500 | 2.5000 |
| IBxBI|BIxIB | IUxUI | 30 | 3.4000 | 3.0000 |
| IUxUI|BIxIB | IBxBI | 16 | 3.3750 | 3.0000 |

## Pairwise rank comparison
| view_a | view_b | a_better_count | b_better_count | tie_count | mean_rank_delta_a_minus_b | mean_abs_rank_delta |
| --- | --- | --- | --- | --- | --- | --- |
| IBxBI | IUxUI | 116 | 64 | 70 | -0.6960 | 1.8480 |
| IBxBI | BIxIB | 72 | 83 | 95 | 0.1600 | 1.4080 |
| IUxUI | BIxIB | 48 | 121 | 81 | 0.8560 | 1.8240 |

## Best-fixed regret
| best_fixed_view | mean_regret | median_regret | regret_gt_0_count | regret_gt_0_ratio | regret_ge_3_count | regret_ge_3_ratio | max_regret |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BIxIB | 0.8360 | 0.0000 | 89 | 0.3560 | 33 | 0.1320 | 8.0000 |

## Suggested paper/report wording
A fixed relational view is not uniformly optimal across test instances. The oracle selector, which chooses the best view per instance using the ground-truth rank, substantially improves over the best single view. This gap motivates a final sample-aware agent that decides which relational evidence view to expose to the LLM before producing the final ranking.
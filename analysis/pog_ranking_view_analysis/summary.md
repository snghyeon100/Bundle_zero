# Ranking View Complementarity Analysis: pog

## Input files
- IBxBI: `C:\Users\wotjs\Desktop\bundle\LLM-ZeroShot\results\pog\results_pog_RANK_ITEMAFF_HN_C10_T5_20260527_132007.csv`
- IUxUI: `C:\Users\wotjs\Desktop\bundle\LLM-ZeroShot\results\pog\results_pog_RANK_USERPUR_HN_C10_T5_20260527_132119.csv`
- BIxIB: `C:\Users\wotjs\Desktop\bundle\LLM-ZeroShot\results\pog\results_pog_RANK_BGRAPH_HN_C10_T5_20260527_132228.csv`

## Core claim
The oracle per-sample selector is an upper bound for a future view-selection agent. If it outperforms the best fixed view, then the three graph-derived views contain complementary evidence.

- Best fixed view by MRR: `BIxIB`
- Oracle Hit@1 gain over best fixed view: 0.1440
- Oracle MRR gain over best fixed view: 0.1194

## Summary
| method | type | n | hit_at_1 | hit_at_3 | hit_at_5 | mrr | ndcg_at_3 | ndcg_at_5 | ndcg_at_10 | mean_rank | median_rank | valid_rank_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IBxBI | single_view | 250 | 0.3240 | 0.6720 | 0.8520 | 0.5381 | 0.5258 | 0.6006 | 0.6496 | 3.0160 | 2.0000 | 1.0000 |
| IUxUI | single_view | 250 | 0.2960 | 0.6240 | 0.8480 | 0.5118 | 0.4862 | 0.5786 | 0.6292 | 3.1520 | 3.0000 | 1.0000 |
| BIxIB | single_view | 250 | 0.3800 | 0.7040 | 0.8680 | 0.5749 | 0.5656 | 0.6329 | 0.6777 | 2.8080 | 2.0000 | 1.0000 |
| OracleSelector | oracle | 250 | 0.5240 | 0.8240 | 0.9560 | 0.6944 | 0.6997 | 0.7549 | 0.7698 | 2.0920 | 1.0000 | 1.0000 |

## Unique oracle winner distribution
| winner | count | ratio |
| --- | --- | --- |
| Tie | 145 | 0.5800 |
| IBxBI | 41 | 0.1640 |
| BIxIB | 38 | 0.1520 |
| IUxUI | 26 | 0.1040 |

## Best-view tie cardinality
| tie_type | num_best_views | count | ratio |
| --- | --- | --- | --- |
| unique_best | 1 | 105 | 0.4200 |
| two_views_tied_best | 2 | 81 | 0.3240 |
| all_three_tied_best | 3 | 64 | 0.2560 |

## Exact best-view sets
| best_view_set | num_best_views | count | ratio |
| --- | --- | --- | --- |
| IBxBI|IUxUI|BIxIB | 3 | 64 | 0.2560 |
| IBxBI | 1 | 41 | 0.1640 |
| IUxUI|BIxIB | 2 | 39 | 0.1560 |
| BIxIB | 1 | 38 | 0.1520 |
| IBxBI|BIxIB | 2 | 29 | 0.1160 |
| IUxUI | 1 | 26 | 0.1040 |
| IBxBI|IUxUI | 2 | 13 | 0.0520 |

## Success overlap
| k | success_pattern | count | ratio |
| --- | --- | --- | --- |
| 1 | BIxIB | 21 | 0.0840 |
| 1 | IBxBI | 17 | 0.0680 |
| 1 | IBxBI|BIxIB | 19 | 0.0760 |
| 1 | IBxBI|IUxUI | 5 | 0.0200 |
| 1 | IBxBI|IUxUI|BIxIB | 40 | 0.1600 |
| 1 | IUxUI | 14 | 0.0560 |
| 1 | IUxUI|BIxIB | 15 | 0.0600 |
| 1 | None | 119 | 0.4760 |
| 3 | BIxIB | 10 | 0.0400 |
| 3 | IBxBI | 13 | 0.0520 |
| 3 | IBxBI|BIxIB | 27 | 0.1080 |
| 3 | IBxBI|IUxUI | 11 | 0.0440 |
| 3 | IBxBI|IUxUI|BIxIB | 117 | 0.4680 |
| 3 | IUxUI | 6 | 0.0240 |
| 3 | IUxUI|BIxIB | 22 | 0.0880 |
| 3 | None | 44 | 0.1760 |
| 5 | BIxIB | 5 | 0.0200 |
| 5 | IBxBI | 8 | 0.0320 |
| 5 | IBxBI|BIxIB | 14 | 0.0560 |
| 5 | IBxBI|IUxUI | 11 | 0.0440 |
| 5 | IBxBI|IUxUI|BIxIB | 180 | 0.7200 |
| 5 | IUxUI | 3 | 0.0120 |
| 5 | IUxUI|BIxIB | 18 | 0.0720 |
| 5 | None | 11 | 0.0440 |

## Success cardinality
| k | success_type | num_successful_views | count | ratio |
| --- | --- | --- | --- | --- |
| 1 | all_three_failed | 0 | 119 | 0.4760 |
| 1 | one_view_success | 1 | 52 | 0.2080 |
| 1 | two_views_success | 2 | 39 | 0.1560 |
| 1 | all_three_success | 3 | 40 | 0.1600 |
| 3 | all_three_failed | 0 | 44 | 0.1760 |
| 3 | one_view_success | 1 | 29 | 0.1160 |
| 3 | two_views_success | 2 | 60 | 0.2400 |
| 3 | all_three_success | 3 | 117 | 0.4680 |
| 5 | all_three_failed | 0 | 11 | 0.0440 |
| 5 | one_view_success | 1 | 16 | 0.0640 |
| 5 | two_views_success | 2 | 43 | 0.1720 |
| 5 | all_three_success | 3 | 180 | 0.7200 |

## Hit@1 margin by focal view
| view | hit_at_1_count | focal_only_hit_at_1_count | focal_only_hit_at_1_ratio | shared_with_one_other_count | shared_with_two_others_count | best_other_rank_mean | best_other_rank_median | best_other_margin_mean | best_other_margin_median | best_other_rank_ge_3_count | best_other_rank_ge_3_ratio | best_other_rank_ge_5_count | best_other_rank_ge_5_ratio | worst_other_rank_mean | worst_other_rank_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IBxBI | 81 | 17 | 0.2099 | 24 | 40 | 1.5432 | 1.0000 | 0.5432 | 0.0000 | 12 | 0.1481 | 4 | 0.0494 | 2.5062 | 2.0000 |
| IUxUI | 74 | 14 | 0.1892 | 20 | 40 | 1.2838 | 1.0000 | 0.2838 | 0.0000 | 5 | 0.0676 | 0 | 0.0000 | 1.9730 | 1.0000 |
| BIxIB | 95 | 21 | 0.2211 | 34 | 40 | 1.4000 | 1.0000 | 0.4000 | 0.0000 | 11 | 0.1158 | 1 | 0.0105 | 2.4316 | 2.0000 |

## Unique Hit@1 Other-View Ranks
| unique_hit_view | count | avg_rank_IUxUI | median_rank_IUxUI | avg_rank_BIxIB | median_rank_BIxIB | avg_rank_IBxBI | median_rank_IBxBI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IBxBI | 17 | 4.3529 | 4.0000 | 4.0000 | 3.0000 | nan | nan |
| IUxUI | 14 | nan | nan | 2.9286 | 2.0000 | 3.2857 | 3.0000 |
| BIxIB | 21 | 3.5714 | 3.0000 | nan | nan | 3.3810 | 3.0000 |

## Two-View Hit@1 Remaining-View Ranks
| hit_view_pair | remaining_view | count | avg_rank_remaining_view | median_rank_remaining_view |
| --- | --- | --- | --- | --- |
| IBxBI|IUxUI | BIxIB | 5 | 3.2000 | 2.0000 |
| IBxBI|BIxIB | IUxUI | 19 | 3.4737 | 3.0000 |
| IUxUI|BIxIB | IBxBI | 15 | 2.5333 | 2.0000 |

## Pairwise rank comparison
| view_a | view_b | a_better_count | b_better_count | tie_count | mean_rank_delta_a_minus_b | mean_abs_rank_delta |
| --- | --- | --- | --- | --- | --- | --- |
| IBxBI | IUxUI | 80 | 78 | 92 | -0.1360 | 1.4480 |
| IBxBI | BIxIB | 62 | 85 | 103 | 0.2080 | 1.3040 |
| IUxUI | BIxIB | 49 | 81 | 120 | 0.3440 | 1.1440 |

## Best-fixed regret
| best_fixed_view | mean_regret | median_regret | regret_gt_0_count | regret_gt_0_ratio | regret_ge_3_count | regret_ge_3_ratio | max_regret |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BIxIB | 0.7160 | 0.0000 | 80 | 0.3200 | 26 | 0.1040 | 7.0000 |

## Suggested paper/report wording
A fixed relational view is not uniformly optimal across test instances. The oracle selector, which chooses the best view per instance using the ground-truth rank, substantially improves over the best single view. This gap motivates a final sample-aware agent that decides which relational evidence view to expose to the LLM before producing the final ranking.
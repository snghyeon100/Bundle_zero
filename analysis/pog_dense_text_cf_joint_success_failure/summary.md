# Joint Success/Failure Analysis

Methods: base, co_occur, user_prefer

## Joint Outcome Counts

| joint_outcome | n | rate | base_acc | co_occur_acc | user_prefer_acc |
| --- | --- | --- | --- | --- | --- |
| mixed | 114 | 45.6% | 28.9% | 77.2% | 38.6% |
| all_fail | 85 | 34.0% | 0.0% | 0.0% | 0.0% |
| all_hit | 51 | 20.4% | 100.0% | 100.0% | 100.0% |

## Pairwise Outcomes

| left_method | right_method | n | left_acc | right_acc | both_hit | both_fail | left_only | right_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | co_occur | 250 | 33.6% | 55.6% | 65 | 92 | 19 | 74 |
| base | user_prefer | 250 | 33.6% | 38.0% | 61 | 132 | 23 | 34 |
| co_occur | user_prefer | 250 | 55.6% | 38.0% | 78 | 94 | 61 | 17 |

## By Primary Semantic Tag

| primary_tag | n | all_hit | all_hit_rate | all_fail | all_fail_rate | mixed | mixed_rate | mean_hit_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| style_theme_match | 79 | 9 | 11.4% | 33 | 41.8% | 37 | 46.8% | 1.0 |
| category_completion | 93 | 18 | 19.4% | 33 | 35.5% | 42 | 45.2% | 1.2258064516129032 |
| brand_or_collection_match | 7 | 5 | 71.4% | 2 | 28.6% | 0 | 0.0% | 2.142857142857143 |
| season_match | 63 | 15 | 23.8% | 17 | 27.0% | 31 | 49.2% | 1.4444444444444444 |
| gender_or_age_filtering | 6 | 4 | 66.7% | 0 | 0.0% | 2 | 33.3% | 2.6666666666666665 |
| ambiguous_or_counterintuitive_gt | 1 | 0 | 0.0% | 0 | 0.0% | 1 | 100.0% | 1.0 |
| color_material_pattern_match | 1 | 0 | 0.0% | 0 | 0.0% | 1 | 100.0% | 2.0 |

## By Semantic Tag In Primary Or Secondary

| tag_primary_or_secondary | n | all_hit | all_hit_rate | all_fail | all_fail_rate | mixed | mixed_rate | mean_hit_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| style_theme_match | 161 | 24 | 14.9% | 63 | 39.1% | 74 | 46.0% | 1.093167701863354 |
| category_completion | 237 | 49 | 20.7% | 81 | 34.2% | 107 | 45.1% | 1.2742616033755274 |
| season_match | 85 | 20 | 23.5% | 24 | 28.2% | 41 | 48.2% | 1.423529411764706 |
| brand_or_collection_match | 8 | 5 | 62.5% | 2 | 25.0% | 1 | 12.5% | 2.125 |
| gender_or_age_filtering | 6 | 4 | 66.7% | 0 | 0.0% | 2 | 33.3% | 2.6666666666666665 |
| color_material_pattern_match | 1 | 0 | 0.0% | 0 | 0.0% | 1 | 100.0% | 2.0 |
| ambiguous_or_counterintuitive_gt | 1 | 0 | 0.0% | 0 | 0.0% | 1 | 100.0% | 1.0 |

## By Distractor Hardness

| distractor_hardness | n | all_hit | all_hit_rate | all_fail | all_fail_rate | mixed | mixed_rate | mean_hit_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.0 | 28 | 3 | 10.7% | 14 | 50.0% | 11 | 39.3% | 0.8214285714285714 |
| 2.0 | 221 | 47 | 21.3% | 71 | 32.1% | 103 | 46.6% | 1.3212669683257918 |
| 1.0 | 1 | 1 | 100.0% | 0 | 0.0% | 0 | 0.0% | 3.0 |

## By GT Plausibility

| gt_plausibility | n | all_hit | all_hit_rate | all_fail | all_fail_rate | mixed | mixed_rate | mean_hit_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3.0 | 25 | 2 | 8.0% | 16 | 64.0% | 7 | 28.0% | 0.56 |
| 4.0 | 175 | 26 | 14.9% | 65 | 37.1% | 84 | 48.0% | 1.1142857142857143 |
| 5.0 | 49 | 23 | 46.9% | 4 | 8.2% | 22 | 44.9% | 2.204081632653061 |
| 1.0 | 1 | 0 | 0.0% | 0 | 0.0% | 1 | 100.0% | 1.0 |

## By Primary Rule Tag

| primary_rule_tag | n | all_hit | all_hit_rate | all_fail | all_fail_rate | mixed | mixed_rate | mean_hit_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_negative_like | 11 | 0 | 0.0% | 10 | 90.9% | 1 | 9.1% | 0.09090909090909091 |
| weak_or_no_rule_signal | 88 | 10 | 11.4% | 56 | 63.6% | 22 | 25.0% | 0.7045454545454546 |
| popularity_shortcut | 37 | 9 | 24.3% | 17 | 45.9% | 11 | 29.7% | 1.1081081081081081 |
| cooccurrence_shortcut | 114 | 32 | 28.1% | 2 | 1.8% | 80 | 70.2% | 1.8771929824561404 |

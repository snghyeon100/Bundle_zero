# Tagged Result Analysis

Result columns: base

## Accuracy By Primary Semantic Tag

| primary_tag | n | hit_base_n | hit_base_acc |
| --- | --- | --- | --- |
| category_completion | 93 | 93 | 26.9% |
| style_theme_match | 79 | 79 | 25.3% |
| season_match | 63 | 63 | 31.7% |
| brand_or_collection_match | 7 | 7 | 71.4% |
| gender_or_age_filtering | 6 | 6 | 83.3% |
| ambiguous_or_counterintuitive_gt | 1 | 1 | 0.0% |
| color_material_pattern_match | 1 | 1 | 0.0% |

## Accuracy By Semantic Tag In Primary Or Secondary

| tag_in_primary_or_secondary | n | hit_base_n | hit_base_acc |
| --- | --- | --- | --- |
| category_completion | 237 | 237 | 28.7% |
| style_theme_match | 161 | 161 | 26.1% |
| season_match | 85 | 85 | 34.1% |
| brand_or_collection_match | 8 | 8 | 75.0% |
| gender_or_age_filtering | 6 | 6 | 83.3% |
| color_material_pattern_match | 1 | 1 | 0.0% |
| ambiguous_or_counterintuitive_gt | 1 | 1 | 0.0% |

## Accuracy By Distractor Hardness

| distractor_hardness | n | hit_base_n | hit_base_acc |
| --- | --- | --- | --- |
| 2.0 | 221 | 221 | 30.8% |
| 3.0 | 28 | 28 | 21.4% |
| 1.0 | 1 | 1 | 100.0% |

## Accuracy By GT Plausibility

| gt_plausibility | n | hit_base_n | hit_base_acc |
| --- | --- | --- | --- |
| 4.0 | 175 | 175 | 25.1% |
| 5.0 | 49 | 49 | 59.2% |
| 3.0 | 25 | 25 | 8.0% |
| 1.0 | 1 | 1 | 0.0% |

## Accuracy By Primary Rule Tag

| primary_rule_tag | n | hit_base_n | hit_base_acc |
| --- | --- | --- | --- |
| cooccurrence_shortcut | 114 | 114 | 34.2% |
| weak_or_no_rule_signal | 88 | 88 | 30.7% |
| popularity_shortcut | 37 | 37 | 24.3% |
| hard_negative_like | 11 | 11 | 0.0% |

## Semantic x Rule Cross-Tab

| semantic_rule | n | hit_base_n | hit_base_acc |
| --- | --- | --- | --- |
| category_completion / cooccurrence_shortcut | 36 | 36 | 27.8% |
| category_completion / weak_or_no_rule_signal | 35 | 35 | 28.6% |
| season_match / cooccurrence_shortcut | 35 | 35 | 31.4% |
| style_theme_match / cooccurrence_shortcut | 33 | 33 | 33.3% |
| style_theme_match / weak_or_no_rule_signal | 27 | 27 | 25.9% |
| season_match / weak_or_no_rule_signal | 22 | 22 | 31.8% |
| category_completion / popularity_shortcut | 17 | 17 | 29.4% |
| style_theme_match / popularity_shortcut | 15 | 15 | 13.3% |
| gender_or_age_filtering / cooccurrence_shortcut | 6 | 6 | 83.3% |
| season_match / popularity_shortcut | 5 | 5 | 40.0% |
| category_completion / hard_negative_like | 5 | 5 | 0.0% |
| brand_or_collection_match / weak_or_no_rule_signal | 4 | 4 | 75.0% |
| style_theme_match / hard_negative_like | 4 | 4 | 0.0% |
| brand_or_collection_match / cooccurrence_shortcut | 2 | 2 | 100.0% |
| ambiguous_or_counterintuitive_gt / cooccurrence_shortcut | 1 | 1 | 0.0% |
| brand_or_collection_match / hard_negative_like | 1 | 1 | 0.0% |
| color_material_pattern_match / cooccurrence_shortcut | 1 | 1 | 0.0% |
| season_match / hard_negative_like | 1 | 1 | 0.0% |

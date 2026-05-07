# Tagged Result Analysis

Result columns: baseline, intent_prompt, delete_prompt, image_text, delete_image_text, intent_image_text, co_occer_cf, user_prefer_cf

## Accuracy By Primary Semantic Tag

| primary_tag | n | hit_baseline_n | hit_baseline_acc | hit_intent_prompt_n | hit_intent_prompt_acc | hit_delete_prompt_n | hit_delete_prompt_acc | hit_image_text_n | hit_image_text_acc | hit_delete_image_text_n | hit_delete_image_text_acc | hit_intent_image_text_n | hit_intent_image_text_acc | hit_co_occer_cf_n | hit_co_occer_cf_acc | hit_user_prefer_cf_n | hit_user_prefer_cf_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| category_completion | 93 | 93 | 35.5% | 93 | 35.5% | 93 | 38.7% | 93 | 32.3% | 93 | 37.6% | 93 | 26.9% | 93 | 52.7% | 93 | 34.4% |
| style_theme_match | 79 | 79 | 21.5% | 79 | 24.1% | 79 | 32.9% | 79 | 27.8% | 79 | 22.8% | 79 | 25.3% | 79 | 46.8% | 79 | 31.6% |
| season_match | 63 | 63 | 39.7% | 63 | 36.5% | 63 | 36.5% | 63 | 31.7% | 63 | 31.7% | 63 | 31.7% | 63 | 63.5% | 63 | 41.3% |
| brand_or_collection_match | 7 | 7 | 71.4% | 7 | 71.4% | 7 | 71.4% | 7 | 71.4% | 7 | 71.4% | 7 | 71.4% | 7 | 71.4% | 7 | 71.4% |
| gender_or_age_filtering | 6 | 6 | 66.7% | 6 | 83.3% | 6 | 66.7% | 6 | 66.7% | 6 | 66.7% | 6 | 83.3% | 6 | 100.0% | 6 | 100.0% |
| ambiguous_or_counterintuitive_gt | 1 | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 100.0% | 1 | 0.0% |
| color_material_pattern_match | 1 | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 100.0% | 1 | 100.0% |

## Accuracy By Semantic Tag In Primary Or Secondary

| tag_in_primary_or_secondary | n | hit_baseline_n | hit_baseline_acc | hit_intent_prompt_n | hit_intent_prompt_acc | hit_delete_prompt_n | hit_delete_prompt_acc | hit_image_text_n | hit_image_text_acc | hit_delete_image_text_n | hit_delete_image_text_acc | hit_intent_image_text_n | hit_intent_image_text_acc | hit_co_occer_cf_n | hit_co_occer_cf_acc | hit_user_prefer_cf_n | hit_user_prefer_cf_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| category_completion | 237 | 237 | 33.8% | 237 | 33.8% | 237 | 37.1% | 237 | 31.2% | 237 | 32.1% | 237 | 28.7% | 237 | 55.3% | 237 | 38.4% |
| style_theme_match | 161 | 161 | 28.6% | 161 | 29.2% | 161 | 36.0% | 161 | 30.4% | 161 | 30.4% | 161 | 26.1% | 161 | 49.1% | 161 | 31.7% |
| season_match | 85 | 85 | 37.6% | 85 | 37.6% | 85 | 37.6% | 85 | 34.1% | 85 | 34.1% | 85 | 34.1% | 85 | 63.5% | 85 | 41.2% |
| brand_or_collection_match | 8 | 8 | 75.0% | 8 | 75.0% | 8 | 75.0% | 8 | 75.0% | 8 | 75.0% | 8 | 75.0% | 8 | 62.5% | 8 | 75.0% |
| gender_or_age_filtering | 6 | 6 | 66.7% | 6 | 83.3% | 6 | 66.7% | 6 | 66.7% | 6 | 66.7% | 6 | 83.3% | 6 | 100.0% | 6 | 100.0% |
| color_material_pattern_match | 1 | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 100.0% | 1 | 100.0% |
| ambiguous_or_counterintuitive_gt | 1 | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 100.0% | 1 | 0.0% |

## Accuracy By Distractor Hardness

| distractor_hardness | n | hit_baseline_n | hit_baseline_acc | hit_intent_prompt_n | hit_intent_prompt_acc | hit_delete_prompt_n | hit_delete_prompt_acc | hit_image_text_n | hit_image_text_acc | hit_delete_image_text_n | hit_delete_image_text_acc | hit_intent_image_text_n | hit_intent_image_text_acc | hit_co_occer_cf_n | hit_co_occer_cf_acc | hit_user_prefer_cf_n | hit_user_prefer_cf_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2.0 | 221 | 221 | 35.7% | 221 | 35.7% | 221 | 38.5% | 221 | 33.9% | 221 | 33.9% | 221 | 30.8% | 221 | 57.0% | 221 | 39.4% |
| 3.0 | 28 | 28 | 14.3% | 28 | 17.9% | 28 | 28.6% | 28 | 17.9% | 28 | 21.4% | 28 | 21.4% | 28 | 42.9% | 28 | 25.0% |
| 1.0 | 1 | 1 | 100.0% | 1 | 100.0% | 1 | 100.0% | 1 | 100.0% | 1 | 100.0% | 1 | 100.0% | 1 | 100.0% | 1 | 100.0% |

## Accuracy By GT Plausibility

| gt_plausibility | n | hit_baseline_n | hit_baseline_acc | hit_intent_prompt_n | hit_intent_prompt_acc | hit_delete_prompt_n | hit_delete_prompt_acc | hit_image_text_n | hit_image_text_acc | hit_delete_image_text_n | hit_delete_image_text_acc | hit_intent_image_text_n | hit_intent_image_text_acc | hit_co_occer_cf_n | hit_co_occer_cf_acc | hit_user_prefer_cf_n | hit_user_prefer_cf_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.0 | 175 | 175 | 27.4% | 175 | 28.6% | 175 | 31.4% | 175 | 26.9% | 175 | 29.1% | 175 | 25.1% | 175 | 52.6% | 175 | 31.4% |
| 5.0 | 49 | 49 | 65.3% | 49 | 65.3% | 49 | 67.3% | 49 | 57.1% | 49 | 59.2% | 49 | 59.2% | 49 | 79.6% | 49 | 75.5% |
| 3.0 | 25 | 25 | 16.0% | 25 | 12.0% | 25 | 24.0% | 25 | 24.0% | 25 | 8.0% | 25 | 8.0% | 25 | 28.0% | 25 | 12.0% |
| 1.0 | 1 | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 100.0% | 1 | 0.0% |

## Accuracy By Primary Rule Tag

| primary_rule_tag | n | hit_baseline_n | hit_baseline_acc | hit_intent_prompt_n | hit_intent_prompt_acc | hit_delete_prompt_n | hit_delete_prompt_acc | hit_image_text_n | hit_image_text_acc | hit_delete_image_text_n | hit_delete_image_text_acc | hit_intent_image_text_n | hit_intent_image_text_acc | hit_co_occer_cf_n | hit_co_occer_cf_acc | hit_user_prefer_cf_n | hit_user_prefer_cf_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cooccurrence_shortcut | 114 | 114 | 40.4% | 114 | 40.4% | 114 | 42.1% | 114 | 37.7% | 114 | 38.6% | 114 | 34.2% | 114 | 98.2% | 114 | 49.1% |
| weak_or_no_rule_signal | 88 | 88 | 28.4% | 88 | 29.5% | 88 | 31.8% | 88 | 29.5% | 88 | 31.8% | 88 | 30.7% | 88 | 14.8% | 88 | 27.3% |
| popularity_shortcut | 37 | 37 | 35.1% | 37 | 32.4% | 37 | 40.5% | 37 | 32.4% | 37 | 27.0% | 37 | 24.3% | 37 | 35.1% | 37 | 40.5% |
| hard_negative_like | 11 | 11 | 0.0% | 11 | 9.1% | 11 | 27.3% | 11 | 0.0% | 11 | 0.0% | 11 | 0.0% | 11 | 9.1% | 11 | 0.0% |

## Semantic x Rule Cross-Tab

| semantic_rule | n | hit_baseline_n | hit_baseline_acc | hit_intent_prompt_n | hit_intent_prompt_acc | hit_delete_prompt_n | hit_delete_prompt_acc | hit_image_text_n | hit_image_text_acc | hit_delete_image_text_n | hit_delete_image_text_acc | hit_intent_image_text_n | hit_intent_image_text_acc | hit_co_occer_cf_n | hit_co_occer_cf_acc | hit_user_prefer_cf_n | hit_user_prefer_cf_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| category_completion / cooccurrence_shortcut | 36 | 36 | 47.2% | 36 | 47.2% | 36 | 47.2% | 36 | 41.7% | 36 | 47.2% | 36 | 27.8% | 36 | 100.0% | 36 | 47.2% |
| category_completion / weak_or_no_rule_signal | 35 | 35 | 25.7% | 35 | 28.6% | 35 | 31.4% | 35 | 25.7% | 35 | 34.3% | 35 | 28.6% | 35 | 17.1% | 35 | 25.7% |
| season_match / cooccurrence_shortcut | 35 | 35 | 42.9% | 35 | 40.0% | 35 | 37.1% | 35 | 34.3% | 35 | 34.3% | 35 | 31.4% | 35 | 97.1% | 35 | 45.7% |
| style_theme_match / cooccurrence_shortcut | 33 | 33 | 24.2% | 33 | 24.2% | 33 | 36.4% | 33 | 30.3% | 33 | 27.3% | 33 | 33.3% | 33 | 97.0% | 33 | 42.4% |
| style_theme_match / weak_or_no_rule_signal | 27 | 27 | 18.5% | 27 | 18.5% | 27 | 22.2% | 27 | 25.9% | 27 | 22.2% | 27 | 25.9% | 27 | 3.7% | 27 | 22.2% |
| season_match / weak_or_no_rule_signal | 22 | 22 | 36.4% | 22 | 36.4% | 22 | 36.4% | 22 | 31.8% | 22 | 31.8% | 22 | 31.8% | 22 | 13.6% | 22 | 27.3% |
| category_completion / popularity_shortcut | 17 | 17 | 41.2% | 17 | 35.3% | 17 | 41.2% | 17 | 35.3% | 17 | 35.3% | 17 | 29.4% | 17 | 35.3% | 17 | 35.3% |
| style_theme_match / popularity_shortcut | 15 | 15 | 26.7% | 15 | 33.3% | 15 | 40.0% | 15 | 33.3% | 15 | 20.0% | 15 | 13.3% | 15 | 26.7% | 15 | 33.3% |
| gender_or_age_filtering / cooccurrence_shortcut | 6 | 6 | 66.7% | 6 | 83.3% | 6 | 66.7% | 6 | 66.7% | 6 | 66.7% | 6 | 83.3% | 6 | 100.0% | 6 | 100.0% |
| season_match / popularity_shortcut | 5 | 5 | 40.0% | 5 | 20.0% | 5 | 40.0% | 5 | 20.0% | 5 | 20.0% | 5 | 40.0% | 5 | 60.0% | 5 | 80.0% |
| category_completion / hard_negative_like | 5 | 5 | 0.0% | 5 | 0.0% | 5 | 20.0% | 5 | 0.0% | 5 | 0.0% | 5 | 0.0% | 5 | 20.0% | 5 | 0.0% |
| brand_or_collection_match / weak_or_no_rule_signal | 4 | 4 | 75.0% | 4 | 75.0% | 4 | 75.0% | 4 | 75.0% | 4 | 75.0% | 4 | 75.0% | 4 | 75.0% | 4 | 75.0% |
| style_theme_match / hard_negative_like | 4 | 4 | 0.0% | 4 | 25.0% | 4 | 50.0% | 4 | 0.0% | 4 | 0.0% | 4 | 0.0% | 4 | 0.0% | 4 | 0.0% |
| brand_or_collection_match / cooccurrence_shortcut | 2 | 2 | 100.0% | 2 | 100.0% | 2 | 100.0% | 2 | 100.0% | 2 | 100.0% | 2 | 100.0% | 2 | 100.0% | 2 | 100.0% |
| ambiguous_or_counterintuitive_gt / cooccurrence_shortcut | 1 | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 100.0% | 1 | 0.0% |
| brand_or_collection_match / hard_negative_like | 1 | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% |
| color_material_pattern_match / cooccurrence_shortcut | 1 | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 100.0% | 1 | 100.0% |
| season_match / hard_negative_like | 1 | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% | 1 | 0.0% |

## Pairwise Result Comparison

| comparison | n | baseline_only | intent_prompt_only | both_hit | both_fail | baseline_acc | intent_prompt_acc | delete_prompt_only | delete_prompt_acc | image_text_only | image_text_acc | delete_image_text_only | delete_image_text_acc | intent_image_text_only | intent_image_text_acc | co_occer_cf_only | co_occer_cf_acc | user_prefer_cf_only | user_prefer_cf_acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_vs_intent_prompt | 250 | 12.0 | 13.0 | 72 | 153 | 33.6% | 34.0% |  |  |  |  |  |  |  |  |  |  |  |  |
| baseline_vs_delete_prompt | 250 | 12.0 |  | 72 | 144 | 33.6% |  | 22.0 | 37.6% |  |  |  |  |  |  |  |  |  |  |
| baseline_vs_image_text | 250 | 21.0 |  | 63 | 148 | 33.6% |  |  |  | 18.0 | 32.4% |  |  |  |  |  |  |  |  |
| baseline_vs_delete_image_text | 250 | 21.0 |  | 63 | 147 | 33.6% |  |  |  |  |  | 19.0 | 32.8% |  |  |  |  |  |  |
| baseline_vs_intent_image_text | 250 | 25.0 |  | 59 | 150 | 33.6% |  |  |  |  |  |  |  | 16.0 | 30.0% |  |  |  |  |
| baseline_vs_co_occer_cf | 250 | 19.0 |  | 65 | 92 | 33.6% |  |  |  |  |  |  |  |  |  | 74.0 | 55.6% |  |  |
| baseline_vs_user_prefer_cf | 250 | 23.0 |  | 61 | 132 | 33.6% |  |  |  |  |  |  |  |  |  |  |  | 34.0 | 38.0% |
| intent_prompt_vs_delete_prompt | 250 |  | 11.0 | 74 | 145 |  | 34.0% | 20.0 | 37.6% |  |  |  |  |  |  |  |  |  |  |
| intent_prompt_vs_image_text | 250 |  | 23.0 | 62 | 146 |  | 34.0% |  |  | 19.0 | 32.4% |  |  |  |  |  |  |  |  |
| intent_prompt_vs_delete_image_text | 250 |  | 22.0 | 63 | 146 |  | 34.0% |  |  |  |  | 19.0 | 32.8% |  |  |  |  |  |  |
| intent_prompt_vs_intent_image_text | 250 |  | 24.0 | 61 | 151 |  | 34.0% |  |  |  |  |  |  | 14.0 | 30.0% |  |  |  |  |
| intent_prompt_vs_co_occer_cf | 250 |  | 19.0 | 66 | 92 |  | 34.0% |  |  |  |  |  |  |  |  | 73.0 | 55.6% |  |  |
| intent_prompt_vs_user_prefer_cf | 250 |  | 19.0 | 66 | 136 |  | 34.0% |  |  |  |  |  |  |  |  |  |  | 29.0 | 38.0% |
| delete_prompt_vs_image_text | 250 |  |  | 63 | 138 |  |  | 31.0 | 37.6% | 18.0 | 32.4% |  |  |  |  |  |  |  |  |
| delete_prompt_vs_delete_image_text | 250 |  |  | 65 | 139 |  |  | 29.0 | 37.6% |  |  | 17.0 | 32.8% |  |  |  |  |  |  |
| delete_prompt_vs_intent_image_text | 250 |  |  | 62 | 143 |  |  | 32.0 | 37.6% |  |  |  |  | 13.0 | 30.0% |  |  |  |  |
| delete_prompt_vs_co_occer_cf | 250 |  |  | 69 | 86 |  |  | 25.0 | 37.6% |  |  |  |  |  |  | 70.0 | 55.6% |  |  |
| delete_prompt_vs_user_prefer_cf | 250 |  |  | 67 | 128 |  |  | 27.0 | 37.6% |  |  |  |  |  |  |  |  | 28.0 | 38.0% |
| image_text_vs_delete_image_text | 250 |  |  | 66 | 153 |  |  |  |  | 15.0 | 32.4% | 16.0 | 32.8% |  |  |  |  |  |  |
| image_text_vs_intent_image_text | 250 |  |  | 63 | 157 |  |  |  |  | 18.0 | 32.4% |  |  | 12.0 | 30.0% |  |  |  |  |
| image_text_vs_co_occer_cf | 250 |  |  | 60 | 90 |  |  |  |  | 21.0 | 32.4% |  |  |  |  | 79.0 | 55.6% |  |  |
| image_text_vs_user_prefer_cf | 250 |  |  | 60 | 134 |  |  |  |  | 21.0 | 32.4% |  |  |  |  |  |  | 35.0 | 38.0% |
| delete_image_text_vs_intent_image_text | 250 |  |  | 64 | 157 |  |  |  |  |  |  | 18.0 | 32.8% | 11.0 | 30.0% |  |  |  |  |
| delete_image_text_vs_co_occer_cf | 250 |  |  | 61 | 90 |  |  |  |  |  |  | 21.0 | 32.8% |  |  | 78.0 | 55.6% |  |  |
| delete_image_text_vs_user_prefer_cf | 250 |  |  | 62 | 135 |  |  |  |  |  |  | 20.0 | 32.8% |  |  |  |  | 33.0 | 38.0% |
| intent_image_text_vs_co_occer_cf | 250 |  |  | 56 | 92 |  |  |  |  |  |  |  |  | 19.0 | 30.0% | 83.0 | 55.6% |  |  |
| intent_image_text_vs_user_prefer_cf | 250 |  |  | 60 | 140 |  |  |  |  |  |  |  |  | 15.0 | 30.0% |  |  | 35.0 | 38.0% |
| co_occer_cf_vs_user_prefer_cf | 250 |  |  | 78 | 94 |  |  |  |  |  |  |  |  |  |  | 61.0 | 55.6% | 17.0 | 38.0% |

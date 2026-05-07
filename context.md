# LLM-ZeroShot Context

## 1. Project Purpose

This repository evaluates bundle construction and playlist continuation as zero-shot multiple-choice tasks for LLMs.

Task format:

- Given partial bundle or playlist input items.
- Choose the ground-truth item from candidates A-J.
- `pog` and `pog_dense`: fashion outfit bundle completion.
- `spotify` and `spotify_sparse`: playlist continuation.

Current research goal:

- Understand what LLMs do well and poorly in zero-shot bundle construction.
- Check whether high Spotify accuracy is caused by shortcut signals from random negatives.
- Compare model success/failure rates by hard negatives, rule-based shortcuts, and problem-level semantic tags.
- Build a research story such as: LLMs are strong at broad semantic filtering but weak at fine-grained compatibility and collaborative preference modeling.

## 2. Main Folder Structure

- `src/`
  - `main.py`: Runs Gemini zero-shot evaluation.
  - `dataset.py`: Loads datasets and creates evaluation samples.
  - `generate_hard_negatives.py`: Generates Spotify hard negative samples.
  - `analyze_artist_bias.py`: Analyzes Spotify artist-overlap bias.
  - `analyze_rule_based_baselines.py`: Computes rule-based baseline accuracies for result CSVs.
  - `create_tag_meta.py`: Creates reusable problem meta and rule-based tag meta from result CSVs.
  - `tag_fashion_problem_meta.py`: Uses an LLM to tag POG/POG-dense fashion problems.

- `datasets/`
  - `pog/`, `pog_dense/`, `spotify/`, `spotify_sparse/`
  - Each dataset contains files such as `item_info.json`, `bi_train.txt`, `bi_test_input.txt`, and `bi_test_gt.txt`.

- `results/`
  - Dataset-specific result CSVs.
  - `problem_meta_clean.csv`: Reusable problem definition metadata.
  - `problem_tag_meta.csv`: Problem metadata plus deterministic rule-based tags/features.
  - `problem_fashion_semantic_tags*.csv`: LLM semantic tags for POG-style fashion problems.

- `analysis/`
  - Comparison-ready CSVs and LLM-generated meta-analysis markdown reports.

## 3. Files Modified Or Created So Far

### Main scripts

- `src/analyze_rule_based_baselines.py`
  - Added to compute popularity, artist/album overlap, and co-occurrence baselines.
  - Used to show that Spotify accuracy can be strongly affected by candidate-set shortcuts.

- `src/create_tag_meta.py`
  - Added/updated to extract reusable problem-level metadata from any result CSV.
  - Adds `true_item_str` by using `true_option_char` to extract the matching option text from `target_str`.
  - With `--no_rule`, saves clean problem metadata only.
  - Without `--no_rule`, also adds rule-based tags/features.

- `src/tag_fashion_problem_meta.py`
  - Added for POG/POG-dense fashion semantic tagging.
  - Reads `problem_meta_clean.csv`.
  - Sends exactly one problem per API call.
  - Saves after every tagged problem, so it can resume after interruption.
  - Does not use `prediction`, `hit`, `raw_response`, `difficulty`, or `reason`.
  - Current taxonomy:
    - `category_completion`
    - `season_match`
    - `brand_or_collection_match`
    - `style_theme_match`
    - `color_material_pattern_match`
    - `gender_or_age_filtering`
    - `fine_grained_hard_choice`
    - `ambiguous_or_counterintuitive_gt`

### Generated or updated metadata CSVs

- `results/pog/problem_meta_clean.csv`
- `results/pog/problem_tag_meta.csv`
- `results/pog/problem_fashion_semantic_tags.csv`
- `results/pog_dense/problem_meta_clean.csv`
- `results/pog_dense/problem_tag_meta.csv`
- `results/pog_dense/problem_fashion_semantic_tags.csv`
- `results/pog_dense/problem_fashion_semantic_tags_v2.csv`
- `results/spotify/problem_meta_clean.csv`
- `results/spotify/problem_tag_meta.csv`
- `results/spotify_sparse/problem_meta_clean.csv`
- `results/spotify_sparse/problem_tag_meta.csv`

Clean problem metadata columns:

```text
index,bundle_id,true_indice,true_option_idx,true_option_char,input_indices,candidate_indices,input_str,target_str,true_item_str
```

## 4. Current Issues Or Notes

- Early `tag_fashion_problem_meta.py` prompt overused `category_completion`.
  - Reason: almost every fashion bundle can be described as completing a category.
  - Fixed by tightening the prompt:
    - Use `category_completion` as primary only when the functional category gap is the strongest signal.
    - Prefer season, brand, gender, color/material/pattern, or style when those are more decisive.
  - Test output with the updated prompt: `results/pog_dense/problem_fashion_semantic_tags_v2.csv`.

- `true_item_str` was missing from the original clean metadata.
  - Fixed in `create_tag_meta.py`.
  - `tag_fashion_problem_meta.py` now uses `true_item_str` first, and falls back to parsing `target_str` if needed.

- POG text is not actually corrupted in the source data.
  - `item_info.json` contains valid Chinese product titles.
  - Some console output can look broken because of Windows/PowerShell encoding.

- Git working tree is dirty.
  - There are user/generated result files and new scripts mixed together.
  - Do not revert unrelated changes.

## 5. Next Steps

1. Inspect the tag distribution of `results/pog_dense/problem_fashion_semantic_tags_v2.csv`.
   - Confirm that `category_completion` is no longer over-dominant.

2. If the v2 prompt distribution looks good, rerun POG semantic tagging with the same prompt.
   - Suggested output: `results/pog/problem_fashion_semantic_tags_v2.csv`.

3. Merge semantic tags with result CSVs by `index`.
   - Compute hit rate by `primary_tag`.
   - Compute hit rate by `distractor_hardness`.
   - Compare text-only vs multimodal if both results are available.

4. Merge rule-based tags with result CSVs by `index`.
   - Use `problem_tag_meta.csv`.
   - Check whether shortcut-heavy cases have higher accuracy.
   - Check whether `weak_or_no_rule_signal` and `fine_grained_hard_choice` have lower accuracy.

5. Build the research interpretation:
   - LLMs are strong at explicit semantic filtering: season, gender, brand, style, easy outlier removal.
   - LLMs are weaker at fine-grained hard choices, ambiguous ground truth cases, and cases requiring collaborative preference signals.

## 6. Important Commands

### Create clean problem metadata from the selected 250-row result CSVs

```powershell
python src/create_tag_meta.py --csv results/pog/results_pog_HN_C10_T5_20260503_002458.csv --output results/pog/problem_meta_clean.csv --no_rule
python src/create_tag_meta.py --csv results/pog_dense/results_pog_dense_C10_T5_20260505_112706.csv --output results/pog_dense/problem_meta_clean.csv --no_rule
python src/create_tag_meta.py --csv results/spotify/results_spotify_C10_T5_20260504_133400.csv --output results/spotify/problem_meta_clean.csv --no_rule
python src/create_tag_meta.py --csv results/spotify_sparse/results_spotify_sparse_20260417_210146.csv --output results/spotify_sparse/problem_meta_clean.csv --no_rule
```

### Create rule-based tag metadata

```powershell
python src/create_tag_meta.py --csv results/pog/results_pog_HN_C10_T5_20260503_002458.csv --output results/pog/problem_tag_meta.csv
python src/create_tag_meta.py --csv results/pog_dense/results_pog_dense_C10_T5_20260505_112706.csv --output results/pog_dense/problem_tag_meta.csv
python src/create_tag_meta.py --csv results/spotify/results_spotify_C10_T5_20260504_133400.csv --output results/spotify/problem_tag_meta.csv
python src/create_tag_meta.py --csv results/spotify_sparse/results_spotify_sparse_20260417_210146.csv --output results/spotify_sparse/problem_tag_meta.csv
```

### Run fashion semantic tagging for POG/POG-dense

Small test:

```powershell
python src/tag_fashion_problem_meta.py --input results/pog_dense/problem_meta_clean.csv --dataset pog_dense --limit 5
```

Full run:

```powershell
python src/tag_fashion_problem_meta.py --input results/pog_dense/problem_meta_clean.csv --dataset pog_dense
python src/tag_fashion_problem_meta.py --input results/pog/problem_meta_clean.csv --dataset pog
```

Run updated prompt into a separate output file:

```powershell
python src/tag_fashion_problem_meta.py --input results/pog_dense/problem_meta_clean.csv --dataset pog_dense --limit 20 --output results/pog_dense/problem_fashion_semantic_tags_v2.csv
```

Resume after interruption:

```powershell
python src/tag_fashion_problem_meta.py --input results/pog_dense/problem_meta_clean.csv --dataset pog_dense --output results/pog_dense/problem_fashion_semantic_tags_v2.csv
```

### Check rule-based baseline accuracy

```powershell
python src/analyze_rule_based_baselines.py --csv results/spotify/results_spotify_20260411_191411.csv
python src/analyze_rule_based_baselines.py --csv results/spotify_sparse/results_spotify_sparse_20260411_195706.csv
python src/analyze_rule_based_baselines.py --csv results/spotify/results_spotify_20260416_172608.csv
```

### Selected latest 250-row result CSVs

```text
pog:            results/pog/results_pog_HN_C10_T5_20260503_002458.csv
pog_dense:      results/pog_dense/results_pog_dense_C10_T5_20260505_112706.csv
spotify:        results/spotify/results_spotify_C10_T5_20260504_133400.csv
spotify_sparse: results/spotify_sparse/results_spotify_sparse_20260417_210146.csv
```

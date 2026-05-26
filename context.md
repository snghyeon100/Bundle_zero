# LLM-ZeroShot Context

Last updated: 2026-05-18

## 1. Project Purpose

This repository studies zero-shot LLM performance on bundle construction and playlist continuation.

Task format:

- Given a partial bundle or playlist.
- Choose the ground-truth item from candidates A-J.
- `pog` and `pog_dense`: fashion outfit bundle completion.
- `spotify` and `spotify_sparse`: playlist continuation.

Current research question:

- What does an LLM solve well in zero-shot bundle construction?
- What does it consistently fail on?
- Are high scores caused by dataset shortcuts such as popularity, co-occurrence, artist/album overlap, or candidate construction?
- Does adding collaborative filtering signals help reasoning, or does the LLM simply follow numeric shortcuts?

Current narrative:

- Spotify's earlier high accuracy was likely inflated by candidate shortcuts.
- Hard negatives reduce obvious shortcuts and lower Spotify accuracy.
- Co-occurrence can help when it aligns with the ground truth, but the LLM tends to over-follow the score.
- User preference often hurts because the signal is not reliably aligned with bundle/playlist completion.
- For POG/POG-dense, semantic tags plus empirical difficulty are the most useful way to explain what the LLM can and cannot do.
- For POG/POG-dense, item category structure is highly informative, but not as embedding similarity. The stronger signal is train-derived category co-occurrence: observed bundle categories predict likely missing/complementary categories.

### Why extend `I-B'` with category structure

The motivation for extending the original item-bundle interaction space is sparsity in exact item-level collaborative evidence.

- In sparse POG-style settings, many test input items and candidate items have little or no direct co-occurrence history in `bi_train.txt`.
- When this happens, raw `BI x IB` item-item evidence often collapses into all-zero or tie-heavy scores, so candidate items are hard to distinguish.
- `UI/IU` user-preference evidence can also be weak or misaligned with bundle completion: it may describe broad user taste, but not necessarily which missing item completes the current outfit.
- POG-dense behaves differently: item-level co-occurrence is much more predictive, which is why `use_cooccurrence` works strongly there. This suggests the issue is not that collaborative structure is useless, but that exact item-level structure is too sparse in POG.

`I-B'` is therefore intended to enrich the interaction space with category-aware bundle structure. Instead of relying only on exact item co-occurrence, the model can use repeated category-composition patterns. For example, if input item categories are `A, B, C`, train bundles may show that category `D` frequently completes this category set, even when the exact candidate item has never co-occurred with the exact input items.

This connects directly to the category-completion findings:

- Full bundles behave like category slot templates: repeated-category rates are almost zero, while category set patterns repeat.
- Category-only train-to-test completion works surprisingly well without an LLM.
- Text/category embedding similarity is weak as a direct scorer, implying the key signal is not semantic closeness but complementary category completion.

Methodologically, this supports category-aware extensions in two forms:

- Use category completion priors as LLM-facing context, verbalized through representative items rather than opaque category IDs or raw numeric scores.
- Use category structure to expand item-bundle evidence, so sparse exact `BI` signals can be backed by category-level bundle patterns.

## 2. Main Folder Structure

- `src/`
  - `main.py`: Runs Gemini evaluation. Supports multimodal images, ICL retrieval, and user context retrieval.
  - `dataset.py`: Loads datasets and builds eval samples.
  - `generate_hard_negatives.py`: Generates hard negative samples.
  - `analyze_rule_based_baselines.py`: Rule-based baselines for shortcut analysis.
  - `create_tag_meta.py`: Builds reusable problem metadata and optional rule tags.
  - `tag_fashion_problem_meta.py`: LLM semantic tagging for fashion problems.
  - `analyze_tagged_results.py`: Merges semantic/rule tags with result CSVs.
  - `analyze_cf_signal.py`: CF signal quality and LLM reliance analysis.
  - `make_cf_overleaf_tables.py`: Builds Overleaf tables for CF analysis across datasets.
  - `analyze_joint_success_failure.py`: Analyzes all-hit/all-fail/mixed patterns across methods.
  - `deduplicate_dataset.py`: Builds deduplicated dataset copy, especially for POG user-item data.

- `datasets/`
  - `pog/`, `pog_dense/`, `pog_dedup/`, `spotify/`, `spotify_sparse/`
  - Important files include `item_info.json`, `bi_train.txt`, `bi_test_input.txt`, `bi_test_gt.txt`, `ui_full.txt`, and `images/`.

- `results/`
  - Dataset-specific result CSVs.
  - `problem_meta_clean.csv`: Reusable problem definition metadata.
  - `problem_tag_meta.csv`: Rule-based problem metadata.
  - `problem_fashion_semantic_tags*.csv`: LLM semantic tag outputs.

- `analysis/`
  - Generated analysis outputs and Overleaf table files.

- `analyzer_utility/`
  - Analysis and experiment helper scripts used after the core evaluation runs.
  - Recent category-related scripts live here, including category embedding, full-bundle category co-occurrence, and train-to-test category completion evaluation.

## 3. Files Modified Or Created

### Core evaluation and image handling

- `src/main.py`
  - Multimodal image lookup was fixed:
    - Previously only searched `<item_id>.jpg`.
    - Now searches `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif`, and falls back to `<item_id>.*`.
  - Debug output now prints:
    - image count
    - image directory
    - requested item ids
    - missing ids
    - failed image opens
    - found image filenames

- `download_images.py`
  - Rewritten to support result-CSV-based downloading:
    - `--csv <result.csv>` reads `input_indices` and `candidate_indices` directly from the result file.
    - This avoids config/seed/HN mismatch.
  - Still supports config-based downloading when `--csv` is omitted.
  - Skips existing `<item_id>.*` regardless of extension.
  - Uses `https:` for protocol-relative image URLs.
  - Checks `Content-Type` contains `image`.
  - Saves reports:
    - `results/<dataset>/image_download_failed.csv`
    - `results/<dataset>/image_download_no_url.csv`
    - `results/<dataset>/image_required_missing_after_download.csv`

### Rule and semantic metadata

- `src/analyze_rule_based_baselines.py`
  - Added to compute popularity, artist/album overlap, category/album/artist overlap, and co-occurrence baselines.

- `src/create_tag_meta.py`
  - Creates `problem_meta_clean.csv`.
  - Adds `true_item_str` by reading `true_option_char` from `target_str`.
  - With `--no_rule`, outputs pure problem metadata.
  - Without `--no_rule`, also creates deterministic rule tags/features.

- `src/tag_fashion_problem_meta.py`
  - One API call per problem.
  - Saves after every row and resumes by skipping already-tagged rows.
  - Does not use result prediction/hit/raw_response.
  - Uses `true_item_str` if available.
  - Current taxonomy:
    - `category_completion`
    - `season_match`
    - `brand_or_collection_match`
    - `style_theme_match`
    - `color_material_pattern_match`
    - `gender_or_age_filtering`
    - `fine_grained_hard_choice`
    - `ambiguous_or_counterintuitive_gt`
  - Prompt was tightened because early runs overused `category_completion`.

- `src/analyze_tagged_results.py`
  - Merges multiple result CSVs with semantic and rule tags.
  - Outputs accuracy by:
    - primary semantic tag
    - primary or secondary semantic tag
    - distractor hardness
    - GT plausibility
    - rule tag
    - semantic x rule cross-tab
    - pairwise method comparison

### CF and joint success/failure analysis

- `src/analyze_cf_signal.py`
  - Reads result CSVs and `cf_scores_<dataset>.json`.
  - Computes per-sample CF rank statistics for GT and LLM prediction:
    - `min_rank`
    - `avg_rank`
    - `in_top_tie`
    - `unique_top1`
    - `all_zero`
    - `all_tied`
    - `top_tie_size`
  - Outputs:
    - `cf_detailed.csv`
    - `cf_signal_quality.csv`
    - `cf_method_reliance.csv`
    - `cf_quadrants.csv`
    - `cf_pairwise.csv`
    - `summary.md`

- `src/make_cf_overleaf_tables.py`
  - Creates one Overleaf file for CF statistics across four datasets:
    - `analysis/cf_signal_all_datasets_overleaf.tex`

- `src/analyze_joint_success_failure.py`
  - Compares multiple methods problem-by-problem.
  - Labels each problem as:
    - `all_hit`
    - `all_fail`
    - `mixed`
  - Optional merge with semantic tags, rule tags, and CF details.
  - Outputs examples and summary tables.

- `src/deduplicate_dataset.py`
  - Created `datasets/pog_dedup`.
  - Deduplicates repeated user-item entries in `ui_full.txt`.
  - Copies/dedupes dataset files and links/copies images.

### Category signal analysis

- `analyzer_utility/build_category_embedding_cache.py`
  - Builds category prototype embeddings from existing all-item OpenAI text embedding caches.
  - For each category, averages all item text embeddings assigned to that category.
  - Automatically resolves category fields:
    - `pog`: `cate`
    - `pog_dense`: `cate_id`
  - Outputs category embedding `.npz`, category counts, and metadata under `analysis/category_embedding_cache/`.

- `analyzer_utility/analyze_category_embedding_signal.py`
  - Evaluates whether category prototype embeddings provide a useful problem-level signal.
  - Compares item text embedding signal and category embedding signal on POG/POG-dense result rows.
  - Outputs category quality, per-problem detail, summary, and text/category hit-miss quadrants.

- `analyzer_utility/analyze_bundle_category_cooccurrence.py`
  - Analyzes full original bundles from `bi_full.txt` without using GT/candidate result rows.
  - Converts each bundle to a category set/multiset.
  - Outputs category frequency, pair co-occurrence, full category set patterns, itemset patterns, and association rules.

- `analyzer_utility/evaluate_category_completion_selfsup.py`
  - Learns category co-occurrence rules from `bi_train.txt`.
  - Evaluates on `bi_test_input.txt` as observed categories and `bi_test_gt.txt` as held-out categories.
  - Uses no LLM; this is a category-only self-supervised completion prior.

## 4. Generated Metadata And Analysis Outputs

### Problem metadata

Clean problem metadata columns:

```text
index,bundle_id,true_indice,true_option_idx,true_option_char,input_indices,candidate_indices,input_str,target_str,true_item_str
```

Generated files include:

- `results/pog/problem_meta_clean.csv`
- `results/pog/problem_tag_meta.csv`
- `results/pog_dense/problem_meta_clean.csv`
- `results/pog_dense/problem_tag_meta.csv`
- `results/spotify/problem_meta_clean.csv`
- `results/spotify/problem_tag_meta.csv`
- `results/spotify_sparse/problem_meta_clean.csv`
- `results/spotify_sparse/problem_tag_meta.csv`

Fashion semantic tag files:

- `results/pog/problem_fashion_semantic_tags.csv`
- `results/pog_dense/problem_fashion_semantic_tags.csv`
- `results/pog_dense/problem_fashion_semantic_tags_v2.csv`

### Main analysis folders

- `analysis/pog_dense_all_methods_tag_analysis/`
  - Main POG-dense semantic/rule analysis across all methods.
  - Important files:
    - `summary.md`
    - `overleaf_tables.tex`
    - `overleaf_tables_safe.tex`
    - `tagged_results_merged.csv`
    - `primary_semantic.csv`
    - `multi_semantic.csv`
    - `hardness.csv`
    - `plausibility.csv`
    - `rule.csv`
    - `semantic_rule_cross.csv`
    - `pairwise.csv`

- `analysis/pog_dense_cf_signal_analysis/`
  - CF signal analysis for POG-dense.

- `analysis/pog_cf_method_signal_analysis/`
  - CF method signal analysis for POG.

- `analysis/spotify_cf_method_signal_analysis/`
  - CF method signal analysis for Spotify.

- `analysis/spotify_sparse_cf_method_signal_analysis/`
  - CF method signal analysis for Spotify Sparse.

- `analysis/pog_dense_text_cf_joint_success_failure/`
  - Joint success/failure analysis for POG-dense using `base`, `co_occur`, and `user_prefer`.

- `analysis/cf_signal_all_datasets_overleaf.tex`
  - Overleaf tables for CF signal statistics across POG, POG-dense, Spotify, and Spotify-Sparse.

### Category analysis folders

- `analysis/category_embedding_cache/text-embedding-3-large/all_items/`
  - Category prototype embeddings built by averaging all item text embeddings within each item category.
  - Current outputs:
    - `pog/category_embeddings_text-embedding-3-large_float32.npz`
    - `pog/category_summary.csv`
    - `pog_dense/category_embeddings_text-embedding-3-large_float32.npz`
    - `pog_dense/category_summary.csv`

- `analysis/category_embedding_signal/text-embedding-3-large/`
  - Tests category prototype embeddings as a direct semantic signal.
  - Important files:
    - `category_embedding_quality.csv`
    - `category_signal_detail.csv`
    - `category_signal_summary.csv`
    - `category_text_joint_quadrants.csv`
    - `summary.md`

- `analysis/bundle_category_cooccurrence/bi_full/`
  - Full-bundle category grammar analysis from `bi_full.txt`.
  - Important files:
    - `bundle_category_detail.csv`
    - `category_frequency.csv`
    - `category_pair_cooccur.csv`
    - `category_set_patterns.csv`
    - `category_itemset_patterns.csv`
    - `category_association_rules.csv`
    - `bundle_category_summary.csv`
    - `summary.md`

- `analysis/category_completion_selfsup/train_to_test/`
  - Category-only train-to-test completion evaluation.
  - Rules are learned from `bi_train.txt`.
  - Evaluation uses `bi_test_input.txt` as observed categories and `bi_test_gt.txt` as held-out categories.
  - Important files:
    - `category_completion_selfsup_detail.csv`
    - `category_completion_selfsup_summary.csv`
    - `summary.md`

- `analysis/category_graph_space/`
  - Non-LLM category graph completion analysis.
  - Builds bundle-category and category-category graphs:
    - `BI`: bundle-item matrix.
    - `IC`: item-category matrix.
    - `BC = BI @ IC`: bundle-category matrix.
    - `CC = BC^T @ BC`: category-category co-occurrence graph.
  - Important files:
    - `category_graph_completion_summary_all.csv`
    - `category_graph_stats_all.csv`
    - `category_method_quadrants_all.csv`
    - `summary.md`

- `datasets/pog/category_graph/` and `datasets/pog_dense/category_graph/`
  - Stored category graph matrices generated from train/full splits.
  - Includes count and binary variants of `BI`, `IC`, `BC`, and `CC`.

## 5. Important Result CSV Mapping

### POG-dense method mapping

- `Base`: `results/pog_dense/results_pog_dense_HN_C10_T5_20260430_172343.csv`
- `Intent`: `results/pog_dense/results_pog_dense_20260420_203840.csv`
- `Del`: `results/pog_dense/results_pog_dense_HN_C10_T5_20260426_194737.csv`
- `Img`: `results/pog_dense/results_pog_dense_HN_C10_T5_20260502_154223.csv`
- `Del+Img`: `results/pog_dense/results_pog_dense_C10_T5_20260504_151523.csv`
- `Int+Img`: `results/pog_dense/results_pog_dense_C10_T5_20260505_112706.csv`
- `CoCF`: `results/pog_dense/results_pog_dense_HN_C10_T5_20260506_133202.csv`
- `UserCF`: `results/pog_dense/results_pog_dense_HN_C10_T5_20260506_163507.csv`

Current overall POG-dense accuracies:

- Base: 33.6%
- Intent: 34.0%
- Del: 37.6%
- Img: 32.4%
- Del+Img: 32.8%
- Int+Img: 30.0%
- CoCF: 55.6%
- UserCF: 38.0%

Note: image-related result CSVs were produced before the image-loading bug was fixed, so they should be treated cautiously until rerun.

### Other dataset method mapping

POG:

- Base: `results/pog/results_pog_20260416_142034.csv`
- CoCF: `results/pog/results_pog_HN_C10_T5_20260506_172746.csv`
- UserCF: `results/pog_dedup/results_pog_dedup_C10_T5_20260507_194302.csv`

Spotify:

- HN Base: `results/spotify/results_spotify_20260416_172608.csv`
- HN CoCF: `results/spotify/results_spotify_HN_C10_T5_20260506_195749.csv`
- HN UserCF: `results/spotify/results_spotify_HN_C10_T5_20260507_181903.csv`

Spotify Sparse:

- HN Base: `results/spotify_sparse/results_spotify_sparse_HN_C10_T5_20260506_115545.csv`
- HN CoCF: `results/spotify_sparse/results_spotify_sparse_HN_C10_T5_20260507_153017.csv`
- HN UserCF: `results/spotify_sparse/results_spotify_sparse_HN_C10_T5_20260507_164806.csv`

## 6. Key Findings So Far

### Spotify shortcut analysis

Early Spotify/Sparse result files showed high accuracy around 82%. Rule-based baselines suggested this was partly due to candidate shortcuts.

Examples from rule baseline analysis:

- Spotify random negatives:
  - popularity baseline around 55.2%
  - co-occurrence fallback around 66.4%
- Spotify Sparse random negatives:
  - popularity baseline around 43.6%
  - co-occurrence fallback around 52.8%
- Spotify hard negative version dropped to around 52%.

Interpretation:

- The high random-negative Spotify score likely reflected dataset/candidate shortcuts rather than pure LLM playlist reasoning.
- Hard negatives reduce these shortcuts.

### CF signal analysis

Main conclusion:

- The LLM can use collaborative signals, but tends to over-follow them.
- Good signals help; bad signals hurt.
- The LLM does not reliably judge signal trustworthiness.

Important CF findings:

- POG:
  - CF signals are mostly all-zero/all-tied.
  - Co-occurrence all-zero: 95.2%.
  - User preference all-zero: 97.6%.
  - CF is not very discriminative.

- POG-dense:
  - CoCF strongly improves Base: 33.6% to 55.6%.
  - CoCF prediction follows co-occurrence top-tie around 99.6%.
  - Base vs CoCF right-only cases often have GT as co-occurrence unique top-1.

- Spotify:
  - CoCF improves HN Base: 52.0% to 56.4%.
  - UserCF hurts badly: 52.0% to 30.8%.
  - UserCF prediction follows user-preference top-tie around 97.2%, showing over-reliance on a bad/misaligned signal.

- Spotify Sparse:
  - CoCF barely improves: 46.0% to 46.4%.
  - Co-occurrence exists but is sparse/tie-heavy.
  - UserCF hurts: 46.0% to 32.8%.

### Semantic/rule tag findings for POG-dense

From `analysis/pog_dense_all_methods_tag_analysis/overleaf_tables_safe.tex`:

LLM tends to do well when the answer is explicit or textually obvious:

- `brand_or_collection_match`: around 71.4% for most methods.
- `gender_or_age_filtering`: Base 66.7%, CoCF/UserCF 100%.
- `gt_plausibility=5`: Base 65.3%, CoCF 79.6%, UserCF 75.5%.

LLM struggles when the problem requires subtle compatibility:

- `style_theme_match`: Base 21.5%.
- `distractor_hardness=3`: Base 14.3%.
- `gt_plausibility=3`: Base 16.0%.
- `hard_negative_like`: Base 0.0%.
- `weak_or_no_rule_signal`: Base 28.4%, CoCF 14.8%.

Rule analysis:

- `cooccurrence_shortcut`: CoCF 98.2%.
- `weak_or_no_rule_signal`: CoCF 14.8%.
- `hard_negative_like`: CoCF 9.1%.

Interpretation:

- CoCF's gain is largely from strong co-occurrence shortcuts, not necessarily deeper reasoning.
- Without rule/CF shortcut signals, LLM performance remains weak.
- Delete/elimination prompting gives small but meaningful gains, especially in hard negative-like cases.

### Joint success/failure findings

For POG-dense text/CF methods (`base`, `co_occur`, `user_prefer`):

- `all_hit`: 51/250 (20.4%)
- `all_fail`: 85/250 (34.0%)
- `mixed`: 114/250 (45.6%)

High all-hit / stable success:

- `brand_or_collection_match`
- `gender_or_age_filtering`
- high GT plausibility

High all-fail / stable failure:

- `style_theme_match`
- `category_completion` with hard distractors
- low GT plausibility
- `weak_or_no_rule_signal`
- `hard_negative_like`

Interpretation:

- Stable successes are often explicit lexical or strong-cue problems.
- Stable failures are fine-grained compatibility or weak-signal problems.
- Mixed cases are intervention-sensitive and are good targets for prompt/CF/image analysis.

### POG deduplication

POG `ui_full.txt` had massive duplicate user-item entries:

- raw pairs: 237,519
- unique pairs: 61,987
- removed duplicates: 175,532
- rows with duplicates: 14,744

Created `datasets/pog_dedup`.

### Category embedding analysis

Category prototype embeddings were built from existing all-item OpenAI text embedding caches:

- `pog`: 72 categories, category field `cate`
- `pog_dense`: 67 categories, category field `cate_id`

Problem-level category embedding signal was weak when used as semantic similarity:

- `pog`:
  - category top-1: 5.2%
  - category top-3: 26.4%
  - text-miss/category-hit: 2.4%
- `pog_dense`:
  - category top-1: 2.0%
  - category top-3: 10.8%
  - text-miss/category-hit: 1.2%

Interpretation:

- Category embeddings are useful for category quality/prototype inspection, but not strong as a direct similarity scorer from input categories to candidate categories.
- Fashion bundle completion is not mainly "find a semantically similar category"; it is closer to "infer the complementary category slot missing from the current bundle."

### Full-bundle category co-occurrence analysis

Using `bi_full.txt`, each original bundle was converted into a category set/multiset without using GT/candidate rows.

Key statistics:

- `pog`:
  - bundles: 20,000
  - categories: 72
  - average item count: 3.61
  - average unique category count: 3.61
  - repeated-category bundle rate: 0.003
  - unique category set ratio: 0.251

- `pog_dense`:
  - bundles: 29,686
  - categories: 67
  - average item count: 3.56
  - average unique category count: 3.56
  - repeated-category bundle rate: 0.001
  - unique category set ratio: 0.113

Interpretation:

- Bundle categories almost never repeat inside the same bundle.
- The original bundles look like category slot templates rather than arbitrary item collections.
- `pog_dense` has stronger repeated category templates than `pog`, because it has more bundles but fewer unique category sets.
- Raw co-occurrence counts are dominated by high-frequency categories, so confidence/lift/PMI should be interpreted separately.

### Category completion self-supervised evaluation

Rules were trained only on `bi_train.txt` and evaluated on test split categories:

- observed: categories from `bi_test_input.txt`
- held-out target: categories from `bi_test_gt.txt`
- no LLM and no result predictions were used

`direct_confidence` scoring:

```text
score(c | observed S) = count_train(S union {c}) / count_train(S)
```

Main results:

- `pog direct_confidence`:
  - coverage: 0.996
  - hit@1: 0.455
  - hit@3: 0.765
  - hit@5: 0.879
  - MRR: 0.634

- `pog_dense direct_confidence`:
  - coverage: 0.995
  - hit@1: 0.465
  - hit@3: 0.787
  - hit@5: 0.875
  - MRR: 0.643

Observed-size breakdown for confidence:

- `pog`:
  - observed size 1: direct hit@3 0.698
  - observed size 2: direct hit@3 0.832
- `pog_dense`:
  - most test inputs have observed size 2
  - observed size 2: direct hit@3 0.787

Interpretation:

- Train-derived category co-occurrence alone recovers held-out test categories surprisingly well.
- Confidence works much better than lift/PMI, meaning the frequent category templates are not just noise; they are useful priors in this dataset.
- This supports a method direction based on category completion priors rather than category embedding similarity.

### Non-LLM C-C category graph completion

In addition to direct category-set confidence, we built an explicit category graph from bundle-item and item-category matrices:

```text
BI: bundle-item matrix
IC: item-category matrix
BC = BI @ IC
CC = BC^T @ BC
```

Interpretation:

- `BC` represents which categories appear in each bundle.
- `CC` represents how often two categories co-occur in the same train bundle.
- Count and binary variants were both built.
  - Count keeps repeated category counts inside a bundle.
  - Binary clips bundle-category membership to 0/1.
- Since repeated-category bundle rate is almost zero, count and binary results are nearly identical.

For test completion, given observed input category set `S`, each candidate category `c` is scored by averaging pairwise conditional category co-occurrence:

```text
score(c | S) = average_{s in S} P(c | s)
P(c | s) = CC[s, c] / sum_{c'} CC[s, c']
```

This is different from `set_direct_confidence`.

- `CC graph` methods use pairwise `s`-`c` co-occurrence and average over `s in S`.
- `set_direct_confidence` checks whether the whole observed category set `S` appears with `c`:

```text
score(c | S) = count_train(S union {c}) / count_train(S)
```

Full-test split results, no LLM:

- `pog`:
  - `graph_conditional_count`: hit@1 0.461, hit@3 0.760, hit@5 0.868, MRR 0.635
  - `graph_conditional_binary`: hit@1 0.461, hit@3 0.760, hit@5 0.869, MRR 0.634
  - `text_centroid_similarity`: hit@1 0.008, hit@3 0.029

- `pog_dense`:
  - `graph_conditional_count`: hit@1 0.450, hit@3 0.767, hit@5 0.851, MRR 0.627
  - `graph_conditional_binary`: hit@1 0.450, hit@3 0.767, hit@5 0.851, MRR 0.627
  - `text_centroid_similarity`: hit@1 0.001, hit@3 0.008

Implication:

- Category completion is strong even without an LLM.
- The useful signal is structural category complementarity, not semantic category similarity.
- This supports category-aware prompt methods, although current representative-item verbalizations did not transfer the signal strongly to LLM accuracy.

Methodology implication:

- Prompting the LLM with raw numeric counts/probabilities may look ad-hoc.
- A more natural paper framing is:
  - learn a category-level co-occurrence prior from train bundles,
  - convert it into LLM-readable structured hints such as top-k likely complementary categories, verbal strength buckets, or candidate-level prior annotations.
- Candidate alternatives:
  - use category prior as prompt hint,
  - use it for candidate annotation,
  - use it as external reranking,
  - use category pattern similarity for few-shot example selection.

### Implemented category prior prompt method

Implemented the first category-aware baseline extension as `use_category_completion_prior_desc`.

Design:

- For each test sample, use the full input category set `S`.
- Learn category completion confidence from `bi_train.txt` only:

```text
score(c | S) = count_train(S union {c}) / count_train(S)
```

- Select top-k complementary categories by this score.
- Default:
  - `category_prior_top_k: 3`
  - `category_prior_rep_items_per_category: 3`
  - `category_prior_min_support: 3`
- Do not expose raw scores or category hash IDs to the LLM.
- For each selected category, show representative item titles instead.
- Representative items are selected from train bundle items in that category.
- Ranking criterion for representatives:
  - use all-items category centroid from `analysis/category_embedding_cache/...`,
  - compute cosine similarity between train item text embeddings and the category centroid,
  - take centroid-nearest train items.
- Input items and candidate items are excluded from representative examples for the current sample.

Prompt block wording:

```text
Additional outfit context:
The current outfit is commonly completed by item categories similar to the examples below.
1. Similar category examples: ...
2. Similar category examples: ...
3. Similar category examples: ...
Use these examples only as a soft hint about plausible missing item types.
```

Prompt placement:

- ICL example, if any, remains before the target question.
- Current-sample context blocks now appear after the input question and before candidate options:
  - `user_context_block`
  - `bundle_graph_context_block`
  - `category_prior_context_block`
- This means `use_bundle_graph_context`, `use_user_context`, and `use_category_completion_prior_desc` are affected by the prompt placement change.
- Inline item/candidate augmentations are not affected:
  - `use_item_bundle_affiliation_desc`
  - `use_item_user_copurchase_desc`
  - `use_cooccurrence`
  - `use_soft_cooccurrence`
  - `use_category_item_text_aug`
  - `input_category_co_occur`

Result file naming:

- Runs with category prior enabled include `CATPRIOR_` in the output filename.

Implementation files:

- `src/dataset.py`
  - Builds category completion prior from `bi_train.txt`.
  - Loads category and item embedding caches.
  - Selects centroid-nearest representative train items.
  - Formats the category prior context block.
- `src/main.py`
  - Adds `category_prior_context_block` to `generate_prompt`.
  - Places current-sample context after the input question and before options.
  - Saves category-prior config fields and metadata in result CSVs.
- `config.yaml`
  - Added category prior options.

### Additional category prompt methods

After the first category prior prompt result was weak, two lighter category verbalization methods were added to test whether category information should be attached directly to item text rather than as a separate global context block.

#### Same-category item text augmentation

Implemented as `use_category_item_text_aug`.

Design:

- For each item, detect its category from `item_info.json`.
- Select representative train items from the same category.
- Representative item ranking uses the already-built category centroid:
  - load all-item text embeddings,
  - load all-items category centroid embeddings,
  - rank train items in the category by cosine similarity to the category centroid.
- Exclude the current sample's input and candidate items from representative examples.
- This is intended to verbalize what the current item category looks like, without showing opaque category IDs.

Config:

```yaml
use_category_item_text_aug: false
category_item_aug_apply_to: both  # inputs | candidates | both
category_item_aug_rep_items_per_category: 2
```

Prompt form:

```text
{item title} [Additional context: Item Category examples: {rep item 1}; {rep item 2}.]
```

Notes:

- This method is semantic/category-grounding oriented.
- It does not directly encode category co-occurrence.
- Runs with this option enabled include `CATITEMAUG_` in the output filename.

#### Category-name item text augmentation

Implemented as `use_category_name_aug`.

Design:

- Use the generated human-readable category names from `analysis/category_names/gemini/<dataset>/category_names.json`.
- For each input/candidate item, detect its category from `item_info.json`.
- Append the selected category-name field as natural text.
- This gives the LLM an explicit semantic label without exposing numeric category IDs.

Config:

```yaml
use_category_name_aug: false
category_name_aug_apply_to: both  # inputs | candidates | both
category_name_field: category_name_en  # category_name_en | category_name_ko | short_description_en
category_name_root: ./analysis/category_names/gemini
```

Prompt form:

```text
{item title} [Additional context: Item category: {generated category name}.]
```

Notes:

- Runs with this option enabled include `CATNAMEAUG_` in the output filename.
- Result CSVs save `cfg_use_category_name_aug`, `cfg_category_name_aug_apply_to`, `cfg_category_name_field`, and `cfg_category_name_root`.

#### Pairwise category co-occurrence item augmentation

Implemented as `input_category_co_occur`.

Design:

- Learn category-category pair co-occurrence from `bi_train.txt`.
- For each train bundle, convert items to a unique category set.
- For every category pair in the set, increment a symmetric pair count.
- For an item category `c`, retrieve the top-k categories that most often co-occur with `c`.
- For each co-occurring category, show one representative item title.
- Do not expose raw counts or category IDs.
- Current wording explicitly says that each example item represents one co-occurring category, so the LLM does not read the examples as three items from one category.

Config:

```yaml
input_category_co_occur: true
input_category_co_occur_apply_to: inputs  # inputs | candidates | both
input_category_co_occur_verbalization: representative_items  # representative_items | category_names
input_category_co_occur_top_k: 3
input_category_co_occur_rep_items_per_category: 1
```

Prompt form with `representative_items`:

```text
{item title} [Additional context: Category context: this item's category often appears with the following other categories, each represented by one example item: {rep item 1}; {rep item 2}; {rep item 3}.]
```

Prompt form with `category_names`:

```text
{item title} [Additional context: Category context: this item's category often appears with these other categories: {category name 1}; {category name 2}; {category name 3}.]
```

Apply-to variants:

- `inputs`: query-side augmentation only. This is the cleanest first variant because it enriches the partial outfit without directly annotating options.
- `candidates`: option-side augmentation only. This tests whether candidate category-neighborhood descriptions help option comparison.
- `both`: query and option augmentation. This is the strongest verbalized category graph variant but may add more prompt noise.

Notes:

- This method is closer to the strong category co-occurrence signal observed in self-supervised analysis than same-category examples.
- It uses pairwise category co-occurrence, not the full input-set completion score `score(c | S)`.
- Runs with this option enabled include `INPCATCOOC_` in the output filename for representative-item verbalization and `INPCATNAMECOOC_` for category-name verbalization.
- Result CSVs save:
  - `cfg_input_category_co_occur`
  - `cfg_input_category_co_occur_apply_to`
  - `cfg_input_category_co_occur_verbalization`
  - `cfg_input_category_co_occur_top_k`
  - `cfg_input_category_co_occur_rep_items_per_category`

Implementation files:

- `src/dataset.py`
  - Builds category pair co-occurrence counts from `bi_train.txt`.
  - Reuses category representative item ranking.
  - Loads generated category names when `use_category_name_aug` is enabled or when co-occurrence verbalization is `category_names`.
  - Adds item-level category co-occurrence context according to `input_category_co_occur_apply_to`.
- `src/main.py`
  - Saves the new config fields.
  - Adds `CATNAMEAUG_`, `INPCATCOOC_`, or `INPCATNAMECOOC_` to result filenames.
- `config.yaml`
  - Added same-category, generated category-name, and pairwise category co-occurrence augmentation options.

## 7. Current Issues Or Cautions

- Existing image/multimodal result CSVs should be treated cautiously.
  - Earlier `main.py` only loaded `.jpg`, while most downloaded images were `.png`.
  - This was fixed, but old image-result files should be rerun if used in the paper/PPT.

- `download_images.py` should be run with `--csv` when matching images to an existing result file.
  - This avoids mismatch from changed `config.yaml`, seed, HN, or dataset.

- POG/POG-dense HN naming can be confusing.
  - HN matters mainly for Spotify experiments.
  - POG/POG-dense result names may contain `HN`, but if no hard-negative JSON exists, `main.py` falls back to `dataset.get_eval_samples()`.

- Running multiple experiments simultaneously is OK if:
  - output/partial files do not collide,
  - already-running processes are restarted after code changes.

- Windows console encoding can make some Korean comments or Chinese text look broken.
  - Source item text itself is generally fine.

- Git worktree is dirty with user/generated files. Do not revert unrelated changes.

- POG and POG-dense category IDs are opaque hashes.
  - They are useful for deterministic analysis.
  - They are not directly meaningful to an LLM.
  - Any prompt-facing category signal needs category verbalization, such as representative item titles or generated category descriptions.

- Category co-occurrence confidence is a strong prior, but it is still a dataset prior.
  - It should be evaluated separately from LLM reasoning.
  - If used in prompts, compare raw category labels, top-k complementary hints, verbalized confidence buckets, raw numeric confidence, and external reranking.

- Category prior prompt uses all-items category centroids for representative selection, but representatives themselves are selected only from train bundle items and exclude the current sample's input/candidate items.
  - This was chosen for analysis-oriented experimentation.
  - A stricter later variant could rebuild train-only category centroids.

- Because prompt block placement changed, rerun methods that use separate context blocks before using old results:
  - `use_category_completion_prior_desc`
  - `use_bundle_graph_context`
  - `use_user_context`
  - Methods with inline item/candidate augmentation do not need rerun solely due to this prompt placement change.

- Category item augmentation methods are inline item-text changes.
  - `use_category_item_text_aug` appends same-category representative examples.
  - `input_category_co_occur` appends co-occurring-category representative examples.
  - Their behavior depends on `*_apply_to` options, so compare `inputs`, `candidates`, and `both` as separate ablations.

## 8. Important Commands

### Run main evaluation

```powershell
python src\main.py
```

Resume:

```powershell
python src\main.py --resume results\<dataset>\<partial_file>.csv
```

### Download images

Best for existing result CSV:

```powershell
python download_images.py --csv results\pog_dense\results_pog_dense_HN_C10_T5_20260430_172343.csv
```

Config-based:

```powershell
python download_images.py
```

### Analyze CF signal

POG-dense:

```powershell
python src\analyze_cf_signal.py --results results\pog_dense\results_pog_dense_HN_C10_T5_20260430_172343.csv results\pog_dense\results_pog_dense_HN_C10_T5_20260506_133202.csv results\pog_dense\results_pog_dense_HN_C10_T5_20260506_163507.csv --names base co_occur user_prefer --dataset pog_dense --output_dir analysis\pog_dense_cf_signal_analysis
```

All-dataset Overleaf CF tables:

```powershell
python src\make_cf_overleaf_tables.py --output analysis\cf_signal_all_datasets_overleaf.tex
```

### Analyze POG-dense tags across all methods

The output already exists at `analysis/pog_dense_all_methods_tag_analysis/`.

It was produced with `src/analyze_tagged_results.py` using:

- all POG-dense method CSVs
- `results/pog_dense/problem_fashion_semantic_tags_v2.csv`
- `results/pog_dense/problem_tag_meta.csv`

Overleaf file to use:

```text
analysis/pog_dense_all_methods_tag_analysis/overleaf_tables_safe.tex
```

### Joint success/failure analysis

```powershell
python src\analyze_joint_success_failure.py --results results\pog_dense\results_pog_dense_HN_C10_T5_20260430_172343.csv results\pog_dense\results_pog_dense_HN_C10_T5_20260506_133202.csv results\pog_dense\results_pog_dense_HN_C10_T5_20260506_163507.csv --names base co_occur user_prefer --semantic results\pog_dense\problem_fashion_semantic_tags_v2.csv --rule results\pog_dense\problem_tag_meta.csv --output_dir analysis\pog_dense_text_cf_joint_success_failure
```

### Category embedding and co-occurrence analysis

Build category embedding cache:

```powershell
python analyzer_utility\build_category_embedding_cache.py --datasets pog pog_dense
```

Analyze category embedding signal:

```powershell
python analyzer_utility\analyze_category_embedding_signal.py
```

Analyze full-bundle category co-occurrence from `bi_full.txt`:

```powershell
python analyzer_utility\analyze_bundle_category_cooccurrence.py
```

Evaluate train-to-test category completion prior:

```powershell
python analyzer_utility\evaluate_category_completion_selfsup.py
```

Run category prior prompt experiment:

```yaml
use_category_completion_prior_desc: true
category_prior_top_k: 3
category_prior_rep_items_per_category: 3
category_prior_min_support: 3
```

Then run:

```powershell
python src\main.py
```

Resume works as usual:

```powershell
python src\main.py --resume results\<dataset>\results_<dataset>_CATPRIOR_..._partial.csv
```

Run same-category item text augmentation:

```yaml
use_category_item_text_aug: true
category_item_aug_apply_to: both
category_item_aug_rep_items_per_category: 2
input_category_co_occur: false
```

Then run:

```powershell
python src\main.py
```

Run pairwise category co-occurrence item augmentation:

```yaml
use_category_item_text_aug: false
input_category_co_occur: true
input_category_co_occur_apply_to: inputs  # inputs | candidates | both
input_category_co_occur_verbalization: representative_items  # representative_items | category_names
input_category_co_occur_top_k: 3
input_category_co_occur_rep_items_per_category: 1
```

Run generated category-name augmentation:

```yaml
use_category_name_aug: true
category_name_aug_apply_to: both  # inputs | candidates | both
category_name_field: category_name_en
category_name_root: ./analysis/category_names/gemini
```

Then run:

```powershell
python src\main.py
```

### Create clean problem metadata

```powershell
python src\create_tag_meta.py --csv <result_csv> --output results\<dataset>\problem_meta_clean.csv --no_rule
```

### Create rule metadata

```powershell
python src\create_tag_meta.py --csv <result_csv> --output results\<dataset>\problem_tag_meta.csv
```

### Fashion semantic tagging

```powershell
python src\tag_fashion_problem_meta.py --input results\pog_dense\problem_meta_clean.csv --dataset pog_dense --output results\pog_dense\problem_fashion_semantic_tags_v2.csv
```

Resume uses the same command; completed rows are skipped.

### Rule-based baselines

```powershell
python src\analyze_rule_based_baselines.py --csv results\spotify\results_spotify_20260411_191411.csv
python src\analyze_rule_based_baselines.py --csv results\spotify_sparse\results_spotify_sparse_20260411_195706.csv
python src\analyze_rule_based_baselines.py --csv results\spotify\results_spotify_20260416_172608.csv
```

## 9. Suggested Next Work

1. Rerun multimodal POG/POG-dense after image loading fix.
   - First run `download_images.py --csv <target result csv>` if using existing sample ids.
   - Confirm first debug prints `[Image Count]` equal to input items plus candidates.

2. Decide whether PPT main body separates:
   - method analysis: Base, Image, CoCF, UserCF
   - prompt engineering: Intent, Delete, Intent+Image, Delete+Image

3. Use semantic tags + difficulty for the main "what LLM does well/poorly" story.
   - Show accuracy by semantic tag.
   - Show accuracy by hardness and GT plausibility.
   - Show rule shortcut table to explain CoCF.
   - Use joint all-hit/all-fail/mixed to identify stable strengths and failures.

4. Keep CF signal analysis as a separate section:
   - It explains whether external collaborative numbers help.
   - Main conclusion: LLM over-follows CF scores and needs reliability-aware fusion.

5. Add case studies:
   - all-hit examples: clear brand/gender/plausibility.
   - all-fail examples: style/theme, low plausibility, weak rule signal.
   - CoCF-only examples: GT is co-occurrence unique top-1.

6. Turn category co-occurrence prior into LLM-facing signals.
   - First build a readable category map from hashed category IDs to representative item titles/descriptions.
   - Compare:
     - raw item category labels,
     - top-k complementary category hints,
     - verbal strength buckets,
     - raw confidence numbers,
     - external co-occurrence reranking.
   - Use the self-supervised result as justification that the prior is strong before exposing it to the LLM.

7. Category method ablations to run next.
   - `input_category_co_occur_apply_to: inputs`
   - `input_category_co_occur_apply_to: candidates`
   - `input_category_co_occur_apply_to: both`
   - `use_category_item_text_aug: true` with `category_item_aug_apply_to: both`
   - Compare these against `use_category_completion_prior_desc` and base.

## 10. 2026-05-25 Updates: Category/Neighborhood Summary Methods

### POG input count / latest result sanity

- Latest POG result checked: `results/pog/results_pog_CATSUM_HN_C10_T5_20260525_151003.csv`.
- Rows: 250.
- Input item count distribution:
  - 1 input item: 115 samples.
  - 2 input items: 135 samples.
  - 3+ input items: 0 samples.
- Latest CATSUM hit by input count:
  - 1 input item: 33/115 = 28.70%.
  - 2 input items: 34/135 = 25.19%.
  - overall: 67/250 = 26.80%.
- Unique input items in that latest POG result: 375.

### `cc_retrieval_context_k`

- Added support for retrieving multiple C-C completion train outfits.
- Config:

```yaml
use_cc_retrieval_context: false
cc_retrieval_context_k: 3
cc_retrieval_context_seed: 45
cc_retrieval_overlap_weight: 1.0
cc_retrieval_extra_weight: 1.0
```

- `src/dataset.py` now returns plural metadata such as:
  - `cc_retrieval_context_bundle_ids`
  - `cc_retrieval_context_scores`
  - `cc_retrieval_context_overlap_counts`
  - `cc_retrieval_context_extra_priors`
  - `cc_retrieval_context_jaccards`
  - `cc_retrieval_context_item_texts_by_bundle`
  - `cc_retrieval_context_selected_count`
- Existing single first-result metadata is preserved for backward compatibility.

### Category names as category prior verbalization

- Category names were generated under:

```text
analysis/category_names/gemini/{pog,pog_dense}/category_names.json
```

- Added config:

```yaml
category_prior_verbalization: category_names  # representative_items | category_names
category_name_root: ./analysis/category_names/gemini
category_name_field: category_name_en
```

- When `use_category_completion_prior_desc: true` and `category_prior_verbalization: category_names`, the prior now inserts human-readable category names instead of representative item titles.
- Filename prefix becomes `CATPRIORNAME_` for category-name prior runs.

### CATSUM: category evidence summary

- Added candidate-agnostic category evidence summary method.
- Config:

```yaml
use_category_evidence_summary: false
category_evidence_summary_k: 5
category_evidence_summary_include_evidence: false
category_evidence_summary_model: ""  # empty = use main model
category_evidence_summary_api_key_env: ""  # empty = use main client
category_evidence_summary_max_output_tokens: 180
```

- Retrieval:
  - Uses train outfit category sets.
  - Prioritizes full input category set evidence, then pairwise, then single-category fallback.
  - Evidence block contains category names, not item titles.
- Summary agent prompt asks for 2-4 sentences about likely missing roles/categories and duplicate roles to avoid.
- Final selector receives:

```text
Historical category summary:
...
Use this as a soft historical hint, while still choosing the candidate that best completes the given items.
```

- Separate summary API key support is available via `category_evidence_summary_api_key_env`.
- Filename prefix: `CATSUM_`.
- Observed POG run was weak: overall hit rate around 0.2680, likely because the summary is candidate-agnostic and too generic.

### Important method decision

- Candidate-agnostic category summary is probably not enough.
- Better multi-agent directions discussed:
  - Candidate-aware category evidence summary.
  - Candidate-aware co-occurrence summary.
  - Input-item neighborhood summary.
- Current implemented next method is input-item neighborhood summary:

```text
input item text
+ exact IB x BI co-affiliated items
+ title text embedding I-I'-B soft co-affiliated items
-> summary agent
-> item-level neighborhood summary
-> append to input item text in final selector prompt
```

### Title-only description cache: generated but not the intended method

- A title-only description generator was first created:

```text
analyzer_utility/generate_input_item_descriptions_gemini.py
```

- It generated descriptions from item title only:

```text
item title -> Gemini -> concise product description
```

- Completed cache:

```text
analysis/input_item_descriptions/gemini/pog/input_item_descriptions.json
```

- Count: 375 cached input items.
- This cache is not the intended neighborhood-summary method. It should be treated as a possible ablation only.

### Correct method: input item neighborhood summary

- Added script:

```text
analyzer_utility/generate_input_item_neighborhood_summaries_gemini.py
```

- For each unique input item, the script builds:
  - Input item title.
  - Exact co-affiliated items from `IB x BI`.
  - Soft co-affiliated items from title text embedding based `I-I'-B`.
- Existing title-text I-I'-B mapping is used:

```text
datasets/pog/item_smoothing_i2bprime_text_top1.json
```

- No new OpenAI embedding cache is needed for this method.
- Default evidence size:

```text
exact IB x BI top-k: 5
soft I-I'-B top-k: 5
```

- Summary prompt constraints:
  - Do not choose an answer.
  - Do not mention candidates.
  - Do not invent unsupported facts.
  - Treat soft evidence as approximate.
  - Return one concise English sentence under 35 words.

- Output cache path:

```text
analysis/input_item_neighborhood_summaries/gemini/pog/input_item_descriptions.json
```

- Output field:

```text
summary
```

- This correct neighborhood summary cache has not been generated yet at the time of this note.

### API key behavior for neighborhood summary generation

- Default single-key option:

```text
--api-key-env GEMINI_API_KEY_2
```

- Multi-key fallback option:

```text
--api-key-envs GEMINI_API_KEY,GEMINI_API_KEY_2,GEMINI_API_KEY_3
```

- The first key in `--api-key-envs` is used first.
- Error handling:
  - `high demand`, `overloaded`, `503`, `unavailable`: retry the same key.
  - `quota`, `429`, `resource_exhausted`, `rate limit`: switch to the next key.

### Command to generate the correct neighborhood summaries

Run from repo root:

```powershell
python analyzer_utility\generate_input_item_neighborhood_summaries_gemini.py `
  --dataset pog `
  --result-csv results\pog\results_pog_CATSUM_HN_C10_T5_20260525_151003.csv `
  --soft-source item_smoothing_text `
  --api-key-envs GEMINI_API_KEY,GEMINI_API_KEY_2,GEMINI_API_KEY_3 `
  --model gemini-3.1-flash-lite-preview `
  --sleep 1.0
```

Equivalent one-line command:

```powershell
python analyzer_utility\generate_input_item_neighborhood_summaries_gemini.py --dataset pog --result-csv results\pog\results_pog_CATSUM_HN_C10_T5_20260525_151003.csv --soft-source item_smoothing_text --api-key-envs GEMINI_API_KEY,GEMINI_API_KEY_2,GEMINI_API_KEY_3 --model gemini-3.1-flash-lite-preview --sleep 1.0
```

### Config to append generated neighborhood summaries to input item text

After the correct cache is generated, use:

```yaml
use_input_item_description_aug: true
input_item_description_cache_root: ./analysis/input_item_neighborhood_summaries/gemini
input_item_description_field: summary
```

- The final prompt input item becomes:

```text
{original input item title} [Additional context: Generated item summary: {summary}.]
```

- For the clean first experiment, keep other contexts off:

```yaml
use_category_evidence_summary: false
use_cc_retrieval_context: false
use_cooccurrence: false
use_soft_cooccurrence: false
use_item_bundle_affiliation_desc: false
use_bundle_graph_context: false
use_category_completion_prior_desc: false
use_category_item_text_aug: false
use_category_name_aug: false
input_category_co_occur: false
```

### Additional implementation notes

- `config.yaml` now defaults the input description root to the intended neighborhood-summary cache:

```yaml
use_input_item_description_aug: false
input_item_description_cache_root: ./analysis/input_item_neighborhood_summaries/gemini
input_item_description_field: summary
```

- `src/dataset.py` loads `input_item_descriptions.json` from that root and appends the configured field only for `role == "input"`.
- `src/main.py` records these fields in result CSV:
  - `cfg_use_input_item_description_aug`
  - `cfg_input_item_description_cache_root`
  - `cfg_input_item_description_field`
- Result filename gets `INPDESC_` when this augmentation is enabled.

### Verification already run

```powershell
venv\Scripts\python.exe -m py_compile src\dataset.py src\main.py analyzer_utility\generate_input_item_descriptions_gemini.py analyzer_utility\generate_input_item_neighborhood_summaries_gemini.py
```

- `git diff --check` was also run for the touched files; only CRLF warnings appeared.

import argparse
import ast
import json
import os
import re
import time
from collections import Counter

import pandas as pd


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

BASE_COLUMNS = [
    "bundle_id",
    "true_indice",
    "true_option_idx",
    "true_option_char",
    "input_indices",
    "candidate_indices",
    "input_str",
    "target_str",
]

SEMANTIC_TAG_DEFINITIONS = {
    "easy_outlier_elimination": "Most wrong candidates are obviously incompatible with the input bundle.",
    "explicit_anchor_matching": "The ground truth is supported by explicit text anchors such as brand, artist, album, category, season, gender, color, or style keywords.",
    "category_completion": "The problem is mainly about identifying the missing functional category, such as top/bottom/shoes/accessory or the next song role.",
    "broad_style_or_genre_match": "The problem can be solved by broad style, aesthetic, genre, era, mood, or occasion matching.",
    "fine_grained_similarity_confusion": "Multiple candidates share the right broad category/style, so subtle distinctions are needed.",
    "same_category_hard_negative": "Several candidates are from the same category/type as the ground truth.",
    "counterintuitive_ground_truth": "The ground truth looks less intuitive than at least one distractor.",
    "multiple_plausible_candidates": "Several candidates could reasonably fit the input bundle.",
    "collaborative_preference_needed": "Text/common-sense alone seems insufficient; historical co-occurrence or bundle preference would help.",
    "visual_signal_needed": "Visual details such as color, pattern, silhouette, texture, or material are needed.",
    "metadata_shortcut": "The problem contains shortcut metadata such as exact keyword, brand, artist, album, category, or popularity-like cues.",
    "ambiguous_or_unknown": "The cause is unclear from the provided text.",
}


def normalize(value):
    return str(value or "").strip().lower()


def parse_list(value):
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []


def short_text(value, limit=1600):
    value = re.sub(r"\s+", " ", str(value)).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def load_config(config_path):
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_dataset_from_csv(csv_path):
    parts = os.path.normpath(csv_path).split(os.sep)
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    base = os.path.basename(csv_path).lower()
    for dataset in ("spotify_sparse", "spotify", "pog_dense", "pog"):
        if dataset in base:
            return dataset
    return None


def find_latest_result_csv(output_dir, dataset):
    dataset_dir = os.path.join(output_dir, dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset output directory not found: {dataset_dir}")

    candidates = []
    for name in os.listdir(dataset_dir):
        lowered = name.lower()
        if not lowered.endswith(".csv"):
            continue
        if "meta" in lowered or "tag" in lowered or "semantic" in lowered or "difficulty" in lowered:
            continue
        candidates.append(os.path.join(dataset_dir, name))

    if not candidates:
        raise FileNotFoundError(f"No result CSV found in {dataset_dir}")
    return max(candidates, key=os.path.getctime)


def get_true_position(row):
    if "true_option_idx" in row and pd.notna(row["true_option_idx"]):
        return int(row["true_option_idx"])

    true_char = str(row.get("true_option_char", "")).strip().upper()
    if true_char in LETTERS:
        return LETTERS.index(true_char)
    return None


def load_item_info(data_path, dataset):
    path = os.path.join(data_path, dataset, "item_info.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_item_frequency(data_path, dataset, split):
    path = os.path.join(data_path, dataset, split)
    counts = Counter()
    if not os.path.exists(path):
        return counts

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = [v for v in line.strip().split(", ") if v]
            for item_id in values[1:]:
                counts[int(item_id)] += 1
    return counts


def load_relevant_bundle_sets(data_path, dataset, split, relevant_items):
    path = os.path.join(data_path, dataset, split)
    bundle_sets = {int(item_id): set() for item_id in relevant_items}
    if not os.path.exists(path):
        return bundle_sets

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values = [int(v) for v in line.strip().split(", ") if v]
            if not values:
                continue
            bundle_id = values[0]
            for item_id in values[1:]:
                if item_id in bundle_sets:
                    bundle_sets[item_id].add(bundle_id)
    return bundle_sets


def item_meta(item_info, item_id):
    info = item_info.get(str(int(item_id)), {})
    return {
        "artist": normalize(info.get("artist_name")),
        "album": normalize(info.get("album_name")),
        "category": normalize(info.get("cate_id") or info.get("cate")),
        "title": normalize(info.get("title") or info.get("track_name")),
    }


def top_index(scores, fallback_scores=None):
    if not scores:
        return None
    if fallback_scores is None:
        fallback_scores = [0] * len(scores)
    max_score = max(scores)
    top = [idx for idx, score in enumerate(scores) if score == max_score]
    return max(top, key=lambda idx: fallback_scores[idx])


def rank_of_true(scores, true_pos):
    if true_pos is None or true_pos >= len(scores):
        return None
    sorted_scores = sorted(scores, reverse=True)
    return 1 + sorted_scores.index(scores[true_pos])


def primary_rule_tag(labels):
    priority = [
        "exact_album_anchor",
        "exact_artist_anchor",
        "exact_category_anchor",
        "cooccurrence_shortcut",
        "popularity_shortcut",
        "hard_negative_like",
        "weak_or_no_rule_signal",
    ]
    for tag in priority:
        if tag in labels:
            return tag
    return "weak_or_no_rule_signal"


def build_base_meta(df):
    missing = [col for col in BASE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in result CSV: {missing}")

    meta = df[BASE_COLUMNS].copy()
    meta.insert(0, "index", df.index)
    return meta


def add_rule_tags(meta_df, dataset, data_path, split):
    item_info = load_item_info(data_path, dataset)
    frequency = load_item_frequency(data_path, dataset, split)

    relevant_items = set()
    parsed = []
    for _, row in meta_df.iterrows():
        input_ids = [int(v) for v in parse_list(row["input_indices"])]
        candidate_ids = [int(v) for v in parse_list(row["candidate_indices"])]
        parsed.append((row, input_ids, candidate_ids))
        relevant_items.update(input_ids)
        relevant_items.update(candidate_ids)

    bundle_sets = load_relevant_bundle_sets(data_path, dataset, split, relevant_items)
    rule_rows = []

    for row, input_ids, candidate_ids in parsed:
        true_pos = get_true_position(row)
        input_metas = [item_meta(item_info, item_id) for item_id in input_ids]
        candidate_metas = [item_meta(item_info, item_id) for item_id in candidate_ids]
        true_meta = candidate_metas[true_pos] if true_pos is not None and true_pos < len(candidate_metas) else {}

        input_artists = {meta["artist"] for meta in input_metas if meta["artist"]}
        input_albums = {meta["album"] for meta in input_metas if meta["album"]}
        input_categories = {meta["category"] for meta in input_metas if meta["category"]}

        true_artist_overlap = bool(true_meta.get("artist") and true_meta["artist"] in input_artists)
        true_album_overlap = bool(true_meta.get("album") and true_meta["album"] in input_albums)
        true_category_overlap = bool(true_meta.get("category") and true_meta["category"] in input_categories)

        same_true_artist_candidates = sum(
            bool(true_meta.get("artist") and meta["artist"] == true_meta["artist"])
            for meta in candidate_metas
        )
        same_true_album_candidates = sum(
            bool(true_meta.get("album") and meta["album"] == true_meta["album"])
            for meta in candidate_metas
        )
        same_true_category_candidates = sum(
            bool(true_meta.get("category") and meta["category"] == true_meta["category"])
            for meta in candidate_metas
        )

        popularity_scores = [frequency[int(item_id)] for item_id in candidate_ids]
        popularity_top = top_index(popularity_scores)
        popularity_rank = rank_of_true(popularity_scores, true_pos)

        cooccurrence_scores = []
        for candidate_id in candidate_ids:
            candidate_bundles = bundle_sets.get(int(candidate_id), set())
            score = 0
            for input_id in input_ids:
                score += len(candidate_bundles & bundle_sets.get(int(input_id), set()))
            cooccurrence_scores.append(score)

        cooccurrence_top = top_index(cooccurrence_scores, popularity_scores)
        cooccurrence_rank = rank_of_true(cooccurrence_scores, true_pos)
        cooccurrence_available = any(score > 0 for score in cooccurrence_scores)

        labels = []
        if true_album_overlap:
            labels.append("exact_album_anchor")
        if true_artist_overlap:
            labels.append("exact_artist_anchor")
        if true_category_overlap:
            labels.append("exact_category_anchor")
        if popularity_top == true_pos:
            labels.append("popularity_shortcut")
        if cooccurrence_top == true_pos and cooccurrence_available:
            labels.append("cooccurrence_shortcut")
        if (
            same_true_artist_candidates > 1
            or same_true_album_candidates > 1
            or same_true_category_candidates > 2
        ):
            labels.append("hard_negative_like")
        if not labels:
            labels.append("weak_or_no_rule_signal")

        rule_rows.append(
            {
                "index": int(row["index"]),
                "primary_rule_tag": primary_rule_tag(labels),
                "rule_tags": json.dumps(labels, ensure_ascii=False),
                "true_artist_overlap": true_artist_overlap,
                "true_album_overlap": true_album_overlap,
                "true_category_overlap": true_category_overlap,
                "same_true_artist_candidates": same_true_artist_candidates,
                "same_true_album_candidates": same_true_album_candidates,
                "same_true_category_candidates": same_true_category_candidates,
                "true_popularity_rank": popularity_rank,
                "true_popularity_score": popularity_scores[true_pos] if true_pos is not None else None,
                "popularity_top_is_gt": popularity_top == true_pos,
                "true_cooccurrence_rank": cooccurrence_rank,
                "true_cooccurrence_score": cooccurrence_scores[true_pos] if true_pos is not None else None,
                "cooccurrence_available": cooccurrence_available,
                "cooccurrence_top_is_gt": cooccurrence_top == true_pos and cooccurrence_available,
            }
        )

    rule_df = pd.DataFrame(rule_rows)
    drop_cols = [col for col in rule_df.columns if col != "index" and col in meta_df.columns]
    if drop_cols:
        meta_df = meta_df.drop(columns=drop_cols)
    return meta_df.merge(rule_df, on="index", how="left")


def clean_json_text(text):
    text = (text or "").strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and start < end:
        text = text[start : end + 1]
    return text


def build_semantic_prompt(rows, dataset):
    if "spotify" in dataset:
        task_name = "playlist continuation"
        bundle_name = "playlist"
        item_name = "song"
    else:
        task_name = "fashion bundle construction"
        bundle_name = "outfit bundle"
        item_name = "fashion item"

    tag_lines = "\n".join([f"- {tag}: {desc}" for tag, desc in SEMANTIC_TAG_DEFINITIONS.items()])
    case_blocks = []

    for _, row in rows:
        block = (
            f"Case Index: {row['index']}\n"
            f"Bundle ID: {row['bundle_id']}\n"
            f"Input {bundle_name}: {short_text(row['input_str'])}\n"
            f"Candidate {item_name}s: {short_text(row['target_str'])}\n"
            f"Ground Truth Option: {row['true_option_char']}\n"
        )
        case_blocks.append(block)

    return f"""
You are an expert researcher creating reusable metadata for zero-shot LLM {task_name} problems.
Assign semantic problem-structure tags to each case. Do not use any model prediction, hit/miss, or result-specific information.

Important:
- Tag the problem itself, not a particular model's behavior.
- Use only the input bundle, candidates, and ground-truth option.
- primary_semantic_tag must be exactly one tag from the taxonomy.
- secondary_semantic_tags may contain 0 to 3 tags from the taxonomy.
- evidence must be short and grounded in the provided input/candidates/ground truth.

Taxonomy:
{tag_lines}

Return ONLY a JSON list with exactly {len(rows)} objects:
[
  {{
    "index": 0,
    "primary_semantic_tag": "fine_grained_similarity_confusion",
    "secondary_semantic_tags": ["multiple_plausible_candidates"],
    "requires_collaborative_signal": true,
    "ground_truth_plausibility": 3,
    "distractor_hardness": 4,
    "confidence": 0.75,
    "evidence": "Several candidates share the same broad category/style, so the ground truth requires subtle compatibility judgment."
  }}
]

Field rules:
- ground_truth_plausibility: integer 1-5, where 1 means counterintuitive and 5 means very obvious.
- distractor_hardness: integer 1-5, where 1 means easy distractors and 5 means very hard distractors.
- confidence: number from 0.0 to 1.0.
- requires_collaborative_signal: true when text/common-sense alone seems insufficient and historical co-occurrence/preference would help.

Cases:
{chr(10).join(case_blocks)}
""".strip()


def validate_semantic_result(result):
    primary = result.get("primary_semantic_tag", "ambiguous_or_unknown")
    if primary not in SEMANTIC_TAG_DEFINITIONS:
        result["primary_semantic_tag"] = "ambiguous_or_unknown"

    secondary = result.get("secondary_semantic_tags", [])
    if not isinstance(secondary, list):
        secondary = []
    result["secondary_semantic_tags"] = [
        tag for tag in secondary if tag in SEMANTIC_TAG_DEFINITIONS
    ][:3]

    for key in ("ground_truth_plausibility", "distractor_hardness"):
        try:
            result[key] = max(1, min(5, int(result.get(key, 3))))
        except (TypeError, ValueError):
            result[key] = 3

    try:
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    except (TypeError, ValueError):
        result["confidence"] = 0.5

    result["requires_collaborative_signal"] = bool(result.get("requires_collaborative_signal", False))
    result["evidence"] = str(result.get("evidence", "")).strip()
    return result


def tag_semantic_batch(client, model, rows, dataset, max_retries):
    prompt = build_semantic_prompt(rows, dataset)
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            )
            parsed = json.loads(clean_json_text(response.text))
            if not isinstance(parsed, list):
                raise ValueError("Model response is not a JSON list.")
            return parsed
        except Exception as exc:
            last_error = exc
            wait = 10 * (attempt + 1)
            print(f">>> Semantic batch failed: {exc} | retrying in {wait}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)

    raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")


def ensure_semantic_columns(df):
    defaults = {
        "primary_semantic_tag": pd.NA,
        "secondary_semantic_tags": pd.NA,
        "requires_collaborative_signal": pd.NA,
        "ground_truth_plausibility": pd.NA,
        "distractor_hardness": pd.NA,
        "confidence": pd.NA,
        "evidence": pd.NA,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def add_semantic_tags(meta_df, output_path, dataset, model, batch_size, limit, max_retries):
    from dotenv import load_dotenv
    from google import genai

    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path, encoding="utf-8-sig")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY or GOOGLE_API_KEY not found in .env.")

    client = genai.Client(api_key=api_key)
    meta_df = ensure_semantic_columns(meta_df)

    tagged = set(
        meta_df[meta_df["primary_semantic_tag"].notna()]["index"].astype(int).tolist()
    )
    unprocessed = [int(idx) for idx in meta_df["index"].tolist() if int(idx) not in tagged]
    if limit > 0:
        unprocessed = unprocessed[:limit]

    if not unprocessed:
        print(">>> All rows already have semantic tags.")
        return meta_df

    print(f">>> Semantic tagging rows: {len(unprocessed)}")
    for start in range(0, len(unprocessed), batch_size):
        batch_indices = unprocessed[start : start + batch_size]
        rows = [
            (idx, meta_df.loc[meta_df["index"].astype(int) == idx].iloc[0])
            for idx in batch_indices
        ]
        print(f">>> Semantic batch {start // batch_size + 1} ({len(rows)} cases)")

        results = tag_semantic_batch(client, model, rows, dataset, max_retries)
        by_index = {}
        for result in results:
            if "index" not in result:
                continue
            result = validate_semantic_result(result)
            by_index[int(result["index"])] = result

        for idx in batch_indices:
            result = by_index.get(int(idx))
            if not result:
                print(f">>> Warning: missing semantic result for index {idx}")
                continue

            mask = meta_df["index"].astype(int) == idx
            meta_df.loc[mask, "primary_semantic_tag"] = result["primary_semantic_tag"]
            meta_df.loc[mask, "secondary_semantic_tags"] = json.dumps(
                result["secondary_semantic_tags"], ensure_ascii=False
            )
            meta_df.loc[mask, "requires_collaborative_signal"] = result[
                "requires_collaborative_signal"
            ]
            meta_df.loc[mask, "ground_truth_plausibility"] = result[
                "ground_truth_plausibility"
            ]
            meta_df.loc[mask, "distractor_hardness"] = result["distractor_hardness"]
            meta_df.loc[mask, "confidence"] = result["confidence"]
            meta_df.loc[mask, "evidence"] = result["evidence"]

        meta_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"    Saved progress: {output_path}")

        if start + batch_size < len(unprocessed):
            time.sleep(5)

    return meta_df


def summarize(meta_df):
    print("\n" + "=" * 64)
    print("Tag Meta Summary")
    print("=" * 64)
    if "primary_rule_tag" in meta_df.columns:
        print("Rule tags")
        for tag, count in meta_df["primary_rule_tag"].value_counts().items():
            print(f"- {tag}: {count} ({count / len(meta_df) * 100:.1f}%)")
    if "primary_semantic_tag" in meta_df.columns:
        completed = meta_df[meta_df["primary_semantic_tag"].notna()]
        if not completed.empty:
            print("\nSemantic tags")
            for tag, count in completed["primary_semantic_tag"].value_counts().items():
                print(f"- {tag}: {count} ({count / len(completed) * 100:.1f}%)")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="Create reusable problem-level tag metadata from any result CSV."
    )
    parser.add_argument("--config", default="config.yaml", help="Config YAML path.")
    parser.add_argument("--csv", default=None, help="Result CSV path.")
    parser.add_argument("--dataset", default=None, help="Dataset name. If omitted, inferred from CSV or config.")
    parser.add_argument("--data-path", default="./datasets", help="Dataset root directory.")
    parser.add_argument("--split", default="bi_train.txt", help="Interaction split for rule tags.")
    parser.add_argument("--output", default=None, help="Output tag meta CSV path.")
    parser.add_argument("--semantic", action="store_true", help="Call Gemini to add semantic problem tags.")
    parser.add_argument("--no_rule", action="store_true", help="Do not add deterministic rule tags.")
    parser.add_argument("--batch_size", type=int, default=8, help="Cases per semantic API call.")
    parser.add_argument("--limit", type=int, default=-1, help="Only semantic-tag first N untagged rows.")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model for semantic tags.")
    parser.add_argument("--max_retries", type=int, default=5, help="Max retries for semantic API calls.")
    args = parser.parse_args()

    conf = load_config(args.config)
    output_dir = conf.get("output_dir", "./results")
    dataset = args.dataset or (infer_dataset_from_csv(args.csv) if args.csv else None) or conf.get("dataset")
    if not dataset:
        raise ValueError("Dataset not provided and could not be inferred.")

    csv_path = args.csv or find_latest_result_csv(output_dir, dataset)
    dataset = args.dataset or infer_dataset_from_csv(csv_path) or dataset
    actual_output_dir = os.path.join(output_dir, dataset)
    os.makedirs(actual_output_dir, exist_ok=True)

    output_path = args.output or os.path.join(actual_output_dir, "problem_tag_meta.csv")

    df = pd.read_csv(csv_path)
    current_meta = build_base_meta(df)

    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        keep_cols = [col for col in existing.columns if col not in current_meta.columns or col == "index"]
        current_meta = current_meta.merge(existing[keep_cols], on="index", how="left")
        print(f">>> Resuming existing tag meta: {output_path}")

    if not args.no_rule:
        current_meta = add_rule_tags(current_meta, dataset, args.data_path, args.split)

    current_meta.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f">>> Saved tag meta base: {output_path}")

    if args.semantic:
        current_meta = ensure_semantic_columns(current_meta)
        current_meta = add_semantic_tags(
            current_meta,
            output_path,
            dataset,
            args.model,
            args.batch_size,
            args.limit,
            args.max_retries,
        )
        current_meta.to_csv(output_path, index=False, encoding="utf-8-sig")

    summarize(current_meta)


if __name__ == "__main__":
    main()

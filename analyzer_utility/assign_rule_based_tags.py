import argparse
import ast
import json
import os
from collections import Counter

import pandas as pd


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def normalize(value):
    return str(value or "").strip().lower()


def parse_list(value):
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []


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


def load_config(config_path):
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
    except ImportError:
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_latest_result_csv(output_dir, dataset):
    dataset_dir = os.path.join(output_dir, dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset output directory not found: {dataset_dir}")

    candidates = []
    for name in os.listdir(dataset_dir):
        lowered = name.lower()
        if not lowered.endswith(".csv"):
            continue
        if "meta" in lowered or "semantic" in lowered or "rule" in lowered or "difficulty" in lowered:
            continue
        candidates.append(os.path.join(dataset_dir, name))

    if not candidates:
        raise FileNotFoundError(f"No result CSV found in {dataset_dir}")
    return max(candidates, key=os.path.getctime)


def load_item_info(data_path, dataset):
    path = os.path.join(data_path, dataset, "item_info.json")
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


def get_true_position(row):
    if "true_option_idx" in row and pd.notna(row["true_option_idx"]):
        return int(row["true_option_idx"])

    true_char = str(row.get("true_option_char", "")).strip().upper()
    if true_char in LETTERS:
        return LETTERS.index(true_char)
    return None


def get_prediction_position(row):
    pred = str(row.get("prediction", "")).strip().upper()
    if len(pred) == 1 and pred in LETTERS:
        return LETTERS.index(pred)
    return None


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


def assign_primary_rule_tag(labels):
    priority = [
        "prompt_or_response_error",
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


def analyze(csv_path, dataset, data_path, split, output_path, merge_output):
    df = pd.read_csv(csv_path)
    df["index"] = df.index

    item_info = load_item_info(data_path, dataset)
    frequency = load_item_frequency(data_path, dataset, split)

    parsed_rows = []
    relevant_items = set()
    for _, row in df.iterrows():
        input_ids = [int(v) for v in parse_list(row["input_indices"])]
        candidate_ids = [int(v) for v in parse_list(row["candidate_indices"])]
        parsed_rows.append((row, input_ids, candidate_ids))
        relevant_items.update(input_ids)
        relevant_items.update(candidate_ids)

    bundle_sets = load_relevant_bundle_sets(data_path, dataset, split, relevant_items)

    out_rows = []
    for row, input_ids, candidate_ids in parsed_rows:
        true_pos = get_true_position(row)
        pred_pos = get_prediction_position(row)

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
        if pred_pos is None or pred_pos >= len(candidate_ids):
            labels.append("prompt_or_response_error")
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
        if pred_pos == popularity_top:
            labels.append("llm_followed_popularity_top")
        if pred_pos == cooccurrence_top and cooccurrence_available:
            labels.append("llm_followed_cooccurrence_top")
        if not labels:
            labels.append("weak_or_no_rule_signal")

        out_rows.append(
            {
                "index": int(row["index"]),
                "hit": int(row.get("hit", 0)),
                "prediction": row.get("prediction", ""),
                "primary_rule_tag": assign_primary_rule_tag(labels),
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
                "llm_matches_popularity_top": pred_pos == popularity_top,
                "true_cooccurrence_rank": cooccurrence_rank,
                "true_cooccurrence_score": cooccurrence_scores[true_pos] if true_pos is not None else None,
                "cooccurrence_available": cooccurrence_available,
                "cooccurrence_top_is_gt": cooccurrence_top == true_pos and cooccurrence_available,
                "llm_matches_cooccurrence_top": pred_pos == cooccurrence_top and cooccurrence_available,
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f">>> Saved rule-based tags: {output_path}")

    print("\n" + "=" * 64)
    print("Rule-Based Tag Summary")
    print("=" * 64)
    for tag, count in out_df["primary_rule_tag"].value_counts().items():
        hit_rate = out_df.loc[out_df["primary_rule_tag"] == tag, "hit"].mean()
        print(f"- {tag}: {count} ({count / len(out_df) * 100:.1f}%), hit={hit_rate * 100:.1f}%")
    print("=" * 64)

    if merge_output:
        merged_path = output_path.replace(".csv", "_merged.csv")
        merged = df.merge(out_df, on=["index", "hit", "prediction"], how="left")
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        print(f">>> Saved merged file: {merged_path}")


def main():
    parser = argparse.ArgumentParser(description="Assign deterministic rule-based tags to result CSV cases.")
    parser.add_argument("--config", default="config.yaml", help="Config YAML path.")
    parser.add_argument("--csv", default=None, help="Result CSV path. If omitted, latest result CSV for config dataset is used.")
    parser.add_argument("--dataset", default=None, help="Dataset name. If omitted, inferred from CSV or config.")
    parser.add_argument("--data-path", default="./datasets", help="Dataset root directory.")
    parser.add_argument("--split", default="bi_train.txt", help="Interaction split for popularity/co-occurrence.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    parser.add_argument("--merge_output", action="store_true", help="Also save result+rule merged CSV.")
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

    output_path = args.output or os.path.join(actual_output_dir, "problem_rule_based_tags.csv")
    analyze(csv_path, dataset, args.data_path, args.split, output_path, args.merge_output)


if __name__ == "__main__":
    main()

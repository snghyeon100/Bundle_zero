import argparse
import ast
import json
import os
from collections import Counter, defaultdict

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
    for name in ("spotify_sparse", "spotify", "pog_dense", "pog"):
        if name in base:
            return name
    return None


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


def candidate_meta(item_info, item_id):
    info = item_info.get(str(int(item_id)), {})
    return {
        "artist": normalize(info.get("artist_name")),
        "album": normalize(info.get("album_name")),
        "track": normalize(info.get("track_name")),
    }


def choose_by_scores(scores):
    if not scores:
        return None
    return max(range(len(scores)), key=lambda idx: scores[idx])


def evaluate_choice(rows, name, choices):
    total = 0
    hits = 0
    ties = 0
    for row, choice_info in zip(rows, choices):
        true_pos = row["true_pos"]
        if choice_info["choice"] is None or true_pos is None:
            continue
        total += 1
        hits += int(choice_info["choice"] == true_pos)
        ties += int(choice_info.get("top_tie_count", 1) > 1)

    rate = hits / total if total else 0.0
    tie_rate = ties / total if total else 0.0
    return {
        "baseline": name,
        "evaluated": total,
        "hit": hits,
        "accuracy": rate,
        "top_tie_count": ties,
        "top_tie_rate": tie_rate,
    }


def make_score_choice(scores, fallback_scores=None):
    if fallback_scores is None:
        fallback_scores = [0] * len(scores)

    max_score = max(scores) if scores else 0
    top = [idx for idx, score in enumerate(scores) if score == max_score]
    if len(top) == 1:
        return {"choice": top[0], "top_tie_count": 1}

    choice = max(top, key=lambda idx: fallback_scores[idx])
    return {"choice": choice, "top_tie_count": len(top)}


def analyze(csv_path, dataset, data_path, split):
    df = pd.read_csv(csv_path)
    item_info = load_item_info(data_path, dataset)
    frequency = load_item_frequency(data_path, dataset, split)

    rows = []
    relevant_items = set()
    for _, row in df.iterrows():
        input_ids = [int(v) for v in parse_list(row["input_indices"])]
        candidate_ids = [int(v) for v in parse_list(row["candidate_indices"])]
        true_pos = get_true_position(row)
        pred_pos = get_prediction_position(row)
        rows.append(
            {
                "input_ids": input_ids,
                "candidate_ids": candidate_ids,
                "true_pos": true_pos,
                "pred_pos": pred_pos,
                "hit": int(row.get("hit", 0)),
            }
        )
        relevant_items.update(input_ids)
        relevant_items.update(candidate_ids)

    bundle_sets = load_relevant_bundle_sets(data_path, dataset, split, relevant_items)

    choices = defaultdict(list)
    diagnostics = Counter()

    for row in rows:
        input_meta = [candidate_meta(item_info, item_id) for item_id in row["input_ids"]]
        input_artists = {meta["artist"] for meta in input_meta if meta["artist"]}
        input_albums = {meta["album"] for meta in input_meta if meta["album"]}

        candidate_ids = row["candidate_ids"]
        candidate_metas = [candidate_meta(item_info, item_id) for item_id in candidate_ids]
        pop_scores = [frequency[int(item_id)] for item_id in candidate_ids]

        artist_scores = [
            int(meta["artist"] in input_artists) if meta["artist"] else 0
            for meta in candidate_metas
        ]
        album_scores = [
            int(meta["album"] in input_albums) if meta["album"] else 0
            for meta in candidate_metas
        ]
        artist_album_scores = [
            int(artist_scores[idx] > 0 or album_scores[idx] > 0)
            for idx in range(len(candidate_ids))
        ]

        cooc_scores = []
        for candidate_id in candidate_ids:
            candidate_bundles = bundle_sets.get(int(candidate_id), set())
            score = 0
            for input_id in row["input_ids"]:
                score += len(candidate_bundles & bundle_sets.get(int(input_id), set()))
            cooc_scores.append(score)

        choices["popularity"].append(make_score_choice(pop_scores))
        choices["artist_overlap_pop_fallback"].append(make_score_choice(artist_scores, pop_scores))
        choices["album_overlap_pop_fallback"].append(make_score_choice(album_scores, pop_scores))
        choices["artist_or_album_pop_fallback"].append(make_score_choice(artist_album_scores, pop_scores))
        choices["cooccurrence_pop_fallback"].append(make_score_choice(cooc_scores, pop_scores))

        true_pos = row["true_pos"]
        if true_pos is not None and true_pos < len(candidate_ids):
            diagnostics["gt_artist_overlap"] += artist_scores[true_pos] > 0
            diagnostics["gt_album_overlap"] += album_scores[true_pos] > 0
            diagnostics["gt_artist_or_album_overlap"] += artist_album_scores[true_pos] > 0
            diagnostics["gt_popularity_top"] += make_score_choice(pop_scores)["choice"] == true_pos
            diagnostics["gt_cooccurrence_top"] += make_score_choice(cooc_scores, pop_scores)["choice"] == true_pos

        diagnostics["artist_overlap_available"] += any(artist_scores)
        diagnostics["album_overlap_available"] += any(album_scores)
        diagnostics["artist_or_album_available"] += any(artist_album_scores)
        diagnostics["cooccurrence_available"] += any(cooc_scores)

        if row["pred_pos"] is not None and row["pred_pos"] < len(candidate_ids):
            diagnostics["llm_pred_popularity_top"] += choices["popularity"][-1]["choice"] == row["pred_pos"]
            diagnostics["llm_pred_cooccurrence_top"] += choices["cooccurrence_pop_fallback"][-1]["choice"] == row["pred_pos"]
            diagnostics["valid_predictions"] += 1

    results = [evaluate_choice(rows, name, choice) for name, choice in choices.items()]
    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)

    total = len(rows)
    llm_hits = sum(row["hit"] for row in rows)

    print("=" * 72)
    print(f"CSV      : {csv_path}")
    print(f"Dataset  : {dataset}")
    print(f"Item info: {os.path.join(data_path, dataset, 'item_info.json')}")
    print(f"Split    : {split}")
    print(f"Samples  : {total}")
    print(f"LLM hit  : {llm_hits}/{total} ({llm_hits / total * 100:.1f}%)")
    print("=" * 72)
    print(results_df.to_string(index=False, formatters={
        "accuracy": lambda value: f"{value * 100:.1f}%",
        "top_tie_rate": lambda value: f"{value * 100:.1f}%",
    }))

    print("-" * 72)
    print("Diagnostics")
    for key in [
        "gt_artist_overlap",
        "gt_album_overlap",
        "gt_artist_or_album_overlap",
        "artist_overlap_available",
        "album_overlap_available",
        "artist_or_album_available",
        "gt_popularity_top",
        "gt_cooccurrence_top",
        "cooccurrence_available",
        "llm_pred_popularity_top",
        "llm_pred_cooccurrence_top",
    ]:
        denom = diagnostics["valid_predictions"] if key.startswith("llm_pred") else total
        value = diagnostics[key]
        print(f"{key:30s}: {value}/{denom} ({value / denom * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate simple rule-based baselines for bundle completion result CSVs."
    )
    parser.add_argument("--csv", required=True, help="Path to a result CSV.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name. If omitted, inferred from results/<dataset>/... or the CSV filename.",
    )
    parser.add_argument("--data-path", default="./datasets", help="Dataset root directory.")
    parser.add_argument(
        "--split",
        default="bi_train.txt",
        help="Interaction split used for popularity/co-occurrence baselines.",
    )
    args = parser.parse_args()

    dataset = args.dataset or infer_dataset_from_csv(args.csv)
    if not dataset:
        raise ValueError("Could not infer dataset. Pass --dataset explicitly.")

    analyze(args.csv, dataset, args.data_path, args.split)


if __name__ == "__main__":
    main()

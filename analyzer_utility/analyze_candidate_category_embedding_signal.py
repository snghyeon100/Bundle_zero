import argparse
import json
import os

import numpy as np
import pandas as pd


def load_count(dataset_dir):
    with open(os.path.join(dataset_dir, "count.json"), "r", encoding="utf-8") as f:
        stat = json.load(f)
    return int(stat["#B"]), int(stat["#I"])


def detect_category_field(item_info):
    fields = ["cate_id", "cate", "category"]
    counts = {field: 0 for field in fields}
    for item in item_info.values():
        for field in fields:
            value = item.get(field)
            if value is not None and str(value).strip():
                counts[field] += 1
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise ValueError("No category field found")
    return best


def load_item_info(dataset_dir):
    with open(os.path.join(dataset_dir, "item_info.json"), "r", encoding="utf-8") as f:
        item_info = json.load(f)
    category_field = detect_category_field(item_info)
    item_to_category = {}
    for item_id_str, item in item_info.items():
        value = item.get(category_field)
        if value is None or not str(value).strip():
            continue
        item_to_category[int(item_id_str)] = str(value).strip()
    return item_info, category_field, item_to_category


def item_text(dataset, item_info, item_id):
    item = item_info.get(str(int(item_id)), {})
    if "pog" in dataset:
        return " ".join(str(item.get("title", f"Item {item_id}")).split())
    return f"Item {item_id}"


def load_bundle_file(path):
    bundles = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = [int(v) for v in line.strip().split(", ") if v]
            if len(vals) < 2:
                continue
            bundles[int(vals[0])] = vals[1:]
    return bundles


def load_gt_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = [int(v) for v in line.strip().split(", ") if v]
            if len(vals) < 2:
                continue
            bundle_id = int(vals[0])
            for item_id in vals[1:]:
                pairs.append((bundle_id, int(item_id)))
    return pairs


def unique_categories(item_ids, item_to_category):
    out = []
    seen = set()
    for item_id in item_ids:
        category = item_to_category.get(int(item_id))
        if category and category not in seen:
            out.append(category)
            seen.add(category)
    return sorted(out)


def load_category_embeddings(repo_root, dataset, embedding_name, dtype):
    path = os.path.join(
        repo_root,
        "analysis",
        "category_embedding_cache",
        embedding_name,
        "all_items",
        dataset,
        f"category_embeddings_{embedding_name}_{dtype}.npz",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        category_ids = [str(c) for c in data["category_ids"].astype(str).tolist()]
        embeddings = data["embeddings_normed"].astype(np.float64)
    return path, category_ids, embeddings, {category: idx for idx, category in enumerate(category_ids)}


def build_candidates(bundle_id, true_item_id, input_items, gt_items, num_items, num_cans, seed):
    rng = np.random.default_rng(int(bundle_id) + int(seed))
    blocked = set(int(i) for i in input_items) | set(int(i) for i in gt_items)
    false_pool = np.asarray([i for i in range(num_items) if i not in blocked], dtype=np.int64)
    false_indices = rng.choice(false_pool, size=int(num_cans) - 1, replace=False)
    candidates = np.concatenate([[int(true_item_id)], false_indices.astype(np.int64)])
    rng.shuffle(candidates)
    true_option_idx = int(np.flatnonzero(candidates == int(true_item_id))[0])
    return [int(i) for i in candidates.tolist()], true_option_idx


def rank_candidates(candidate_scores):
    order = sorted(
        range(len(candidate_scores)),
        key=lambda idx: (-float(candidate_scores[idx]), idx),
    )
    return {idx: rank + 1 for rank, idx in enumerate(order)}, order


def rank_unique_categories(candidate_categories, candidate_scores):
    best_by_category = {}
    for idx, (category, score) in enumerate(zip(candidate_categories, candidate_scores)):
        if category is None:
            continue
        if category not in best_by_category or score > best_by_category[category]["score"]:
            best_by_category[category] = {"score": float(score), "first_idx": idx}
    ordered = sorted(
        best_by_category.items(),
        key=lambda x: (-x[1]["score"], x[1]["first_idx"], x[0]),
    )
    return {category: rank + 1 for rank, (category, _) in enumerate(ordered)}, ordered


def analyze_dataset(repo_root, dataset, embedding_name, dtype, num_cans, seed, output_root):
    dataset_dir = os.path.join(repo_root, "datasets", dataset)
    _, num_items = load_count(dataset_dir)
    item_info, category_field, item_to_category = load_item_info(dataset_dir)
    emb_path, category_ids, embeddings, category_to_idx = load_category_embeddings(
        repo_root,
        dataset,
        embedding_name,
        dtype,
    )
    input_bundles = load_bundle_file(os.path.join(dataset_dir, "bi_test_input.txt"))
    gt_bundles = load_bundle_file(os.path.join(dataset_dir, "bi_test_gt.txt"))
    gt_pairs = load_gt_pairs(os.path.join(dataset_dir, "bi_test_gt.txt"))

    rows = []
    for sample_idx, (bundle_id, true_item_id) in enumerate(gt_pairs):
        input_items = input_bundles.get(bundle_id, [])
        gt_items = gt_bundles.get(bundle_id, [])
        input_categories = unique_categories(input_items, item_to_category)
        input_category_indices = [category_to_idx[c] for c in input_categories if c in category_to_idx]
        gt_category = item_to_category.get(int(true_item_id))
        if not input_category_indices or gt_category not in category_to_idx:
            continue

        query = embeddings[input_category_indices].mean(axis=0)
        norm = np.linalg.norm(query)
        if norm <= 0:
            continue
        query = query / norm

        candidates, true_option_idx = build_candidates(
            bundle_id,
            true_item_id,
            input_items,
            gt_items,
            num_items,
            num_cans,
            seed,
        )
        candidate_categories = [item_to_category.get(int(item_id)) for item_id in candidates]
        candidate_category_indices = [
            category_to_idx.get(category) if category is not None else None
            for category in candidate_categories
        ]
        candidate_scores = [
            float(embeddings[idx] @ query) if idx is not None else float("-inf")
            for idx in candidate_category_indices
        ]
        gt_score = candidate_scores[true_option_idx]
        distractor_scores = [
            score
            for idx, score in enumerate(candidate_scores)
            if idx != true_option_idx and np.isfinite(score)
        ]
        max_distractor_score = max(distractor_scores) if distractor_scores else float("nan")
        mean_distractor_score = float(np.mean(distractor_scores)) if distractor_scores else float("nan")
        rank_by_item, item_order = rank_candidates(candidate_scores)
        item_rank = rank_by_item[true_option_idx]
        unique_rank_by_category, unique_order = rank_unique_categories(candidate_categories, candidate_scores)
        category_rank = unique_rank_by_category.get(gt_category, "")
        same_gt_category_distractors = sum(
            1
            for idx, category in enumerate(candidate_categories)
            if idx != true_option_idx and category == gt_category
        )
        pairwise_win_rate = float(np.mean([gt_score > score for score in distractor_scores])) if distractor_scores else float("nan")
        pairwise_tie_rate = float(np.mean([gt_score == score for score in distractor_scores])) if distractor_scores else float("nan")

        top_items = [
            {
                "rank": rank + 1,
                "option": chr(ord("A") + idx),
                "item_id": candidates[idx],
                "category": candidate_categories[idx],
                "score": candidate_scores[idx],
                "is_gt": int(idx == true_option_idx),
                "text": item_text(dataset, item_info, candidates[idx]),
            }
            for rank, idx in enumerate(item_order)
        ]
        top_unique_categories = [
            {
                "rank": rank + 1,
                "category": category,
                "score": payload["score"],
                "is_gt_category": int(category == gt_category),
            }
            for rank, (category, payload) in enumerate(unique_order)
        ]

        rows.append({
            "dataset": dataset,
            "sample_idx": sample_idx,
            "bundle_id": bundle_id,
            "true_item_id": true_item_id,
            "true_option_idx": true_option_idx,
            "true_option_char": chr(ord("A") + true_option_idx),
            "input_items": json.dumps(input_items, ensure_ascii=False),
            "input_categories": json.dumps(input_categories, ensure_ascii=False),
            "gt_category": gt_category,
            "candidate_items": json.dumps(candidates, ensure_ascii=False),
            "candidate_categories": json.dumps(candidate_categories, ensure_ascii=False),
            "candidate_scores": json.dumps(candidate_scores, ensure_ascii=False),
            "gt_score": gt_score,
            "max_distractor_score": max_distractor_score,
            "mean_distractor_score": mean_distractor_score,
            "margin_vs_max_distractor": gt_score - max_distractor_score,
            "margin_vs_mean_distractor": gt_score - mean_distractor_score,
            "pairwise_win_rate": pairwise_win_rate,
            "pairwise_tie_rate": pairwise_tie_rate,
            "gt_item_rank_among_candidates": item_rank,
            "gt_category_rank_among_unique_candidate_categories": category_rank,
            "hit_at_1_item": int(item_rank == 1),
            "hit_at_3_item": int(item_rank <= 3),
            "hit_at_5_item": int(item_rank <= 5),
            "hit_at_1_category": int(category_rank == 1),
            "hit_at_3_category": int(category_rank != "" and category_rank <= 3),
            "hit_at_5_category": int(category_rank != "" and category_rank <= 5),
            "same_gt_category_distractor_count": same_gt_category_distractors,
            "unique_candidate_category_count": len(set(c for c in candidate_categories if c is not None)),
            "input_category_count": len(input_categories),
            "top_items_by_score": json.dumps(top_items, ensure_ascii=False),
            "top_unique_categories_by_score": json.dumps(top_unique_categories, ensure_ascii=False),
        })

    detail = pd.DataFrame(rows)
    output_dir = os.path.join(output_root, dataset)
    os.makedirs(output_dir, exist_ok=True)
    detail_path = os.path.join(output_dir, "candidate_category_embedding_signal_detail.csv")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    summary_rows = []
    for group_name, group in [("overall", detail)]:
        summary_rows.append({
            "dataset": dataset,
            "group": group_name,
            "n": int(len(group)),
            "mean_gt_score": float(group["gt_score"].mean()),
            "mean_max_distractor_score": float(group["max_distractor_score"].mean()),
            "mean_margin_vs_max_distractor": float(group["margin_vs_max_distractor"].mean()),
            "median_margin_vs_max_distractor": float(group["margin_vs_max_distractor"].median()),
            "mean_pairwise_win_rate": float(group["pairwise_win_rate"].mean()),
            "hit_at_1_item": float(group["hit_at_1_item"].mean()),
            "hit_at_3_item": float(group["hit_at_3_item"].mean()),
            "hit_at_5_item": float(group["hit_at_5_item"].mean()),
            "hit_at_1_category": float(group["hit_at_1_category"].mean()),
            "hit_at_3_category": float(group["hit_at_3_category"].mean()),
            "hit_at_5_category": float(group["hit_at_5_category"].mean()),
            "mean_item_rank": float(group["gt_item_rank_among_candidates"].mean()),
            "mean_category_rank": float(pd.to_numeric(group["gt_category_rank_among_unique_candidate_categories"]).mean()),
            "same_gt_category_distractor_rate": float((group["same_gt_category_distractor_count"] > 0).mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "candidate_category_embedding_signal_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    metadata = {
        "dataset": dataset,
        "embedding_name": embedding_name,
        "embedding_path": emb_path,
        "category_field": category_field,
        "num_cans": num_cans,
        "seed": seed,
        "definition": "q=normalized mean of input category embeddings; candidate score=cos(q,candidate category embedding)",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return detail, summary


def write_summary_md(path, summary):
    lines = [
        "# Candidate Category Embedding Signal",
        "",
        "Scores are cosine similarities between the input-category mean embedding and each candidate item's category embedding.",
        "",
        "| dataset | n | mean GT score | mean max distractor | mean margin | item hit@1 | item hit@3 | category hit@1 | category hit@3 | mean item rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['n'])} | {row['mean_gt_score']:.3f} | "
            f"{row['mean_max_distractor_score']:.3f} | {row['mean_margin_vs_max_distractor']:.3f} | "
            f"{row['hit_at_1_item']:.3f} | {row['hit_at_3_item']:.3f} | "
            f"{row['hit_at_1_category']:.3f} | {row['hit_at_3_category']:.3f} | "
            f"{row['mean_item_rank']:.2f} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze candidate-level category embedding cosine signal.")
    parser.add_argument("--datasets", nargs="+", default=["pog", "pog_dense"])
    parser.add_argument("--embedding_name", default="LightGCN_bi")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--num_cans", type=int, default=10)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--output_dir", default=os.path.join("analysis", "candidate_category_embedding_signal"))
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_root = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_root, exist_ok=True)
    details = []
    summaries = []
    for dataset in args.datasets:
        detail, summary = analyze_dataset(
            repo_root,
            dataset,
            args.embedding_name,
            args.dtype,
            args.num_cans,
            args.seed,
            output_root,
        )
        details.append(detail)
        summaries.append(summary)
    detail_all = pd.concat(details, ignore_index=True)
    summary_all = pd.concat(summaries, ignore_index=True)
    detail_all.to_csv(os.path.join(output_root, "candidate_category_embedding_signal_detail_all.csv"), index=False, encoding="utf-8-sig")
    summary_all.to_csv(os.path.join(output_root, "candidate_category_embedding_signal_summary_all.csv"), index=False, encoding="utf-8-sig")
    write_summary_md(os.path.join(output_root, "summary.md"), summary_all)
    print(f"[Done] Wrote candidate category embedding signal analysis to {output_root}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd


def load_item_info(dataset_dir):
    with open(os.path.join(dataset_dir, "item_info.json"), "r", encoding="utf-8") as f:
        item_info = json.load(f)
    field_counts = {field: 0 for field in ["cate_id", "cate", "category"]}
    for item in item_info.values():
        for field in field_counts:
            value = item.get(field)
            if value is not None and str(value).strip():
                field_counts[field] += 1
    category_field = max(field_counts, key=field_counts.get)
    if field_counts[category_field] == 0:
        raise ValueError(f"No category field found in {dataset_dir}/item_info.json")

    item_to_category = {}
    category_to_items = {}
    for item_id_str, item in item_info.items():
        value = item.get(category_field)
        if value is None or not str(value).strip():
            continue
        category = str(value).strip()
        item_id = int(item_id_str)
        item_to_category[item_id] = category
        category_to_items.setdefault(category, []).append(item_id)
    for category in category_to_items:
        category_to_items[category].sort()
    return item_info, category_field, item_to_category, category_to_items


def item_text(dataset, item_info, item_id):
    info = item_info.get(str(int(item_id)), {})
    if "pog" in dataset:
        return " ".join(str(info.get("title", f"Item {item_id}")).split())
    return f"Item {item_id}"


def representative_items(dataset, item_info, category_to_items, category, k=3):
    return [
        item_text(dataset, item_info, item_id)
        for item_id in category_to_items.get(category, [])[:k]
    ]


def load_bundle_file(path):
    bundles = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = [int(v) for v in line.strip().split(", ") if v]
            if len(vals) < 2:
                continue
            bundle_id = vals[0]
            seen = set()
            items = []
            for item_id in vals[1:]:
                if item_id not in seen:
                    items.append(item_id)
                    seen.add(item_id)
            bundles[bundle_id] = items
    return bundles


def items_to_categories(item_ids, item_to_category):
    out = []
    seen = set()
    missing = []
    for item_id in item_ids:
        category = item_to_category.get(int(item_id))
        if category is None:
            missing.append(int(item_id))
            continue
        if category not in seen:
            out.append(category)
            seen.add(category)
    return sorted(out), missing


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
        counts = data["counts"].astype(np.int64) if "counts" in data else np.zeros(len(category_ids), dtype=np.int64)
    return path, category_ids, embeddings, counts


def rank_categories(scores, observed_indices):
    observed = set(observed_indices)
    ranked = [
        (idx, float(score))
        for idx, score in enumerate(scores)
        if idx not in observed
    ]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked


def summarize(detail):
    rows = []
    groupings = [
        ["dataset"],
        ["dataset", "observed_category_count"],
        ["dataset", "heldout_category_count"],
    ]
    for grouping in groupings:
        for keys, sub in detail.groupby(grouping):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(grouping, keys))
            row["grouping"] = "+".join(grouping)
            row["n"] = int(len(sub))
            row["coverage"] = float(sub["covered"].mean()) if len(sub) else 0.0
            covered = sub[sub["covered"] == 1]
            row["mean_gt_best_cosine"] = float(covered["gt_best_cosine"].mean()) if len(covered) else 0.0
            row["median_gt_best_cosine"] = float(covered["gt_best_cosine"].median()) if len(covered) else 0.0
            row["mean_gt_mean_cosine"] = float(covered["gt_mean_cosine"].mean()) if len(covered) else 0.0
            row["mean_gt_best_rank"] = float(covered["gt_best_rank"].mean()) if len(covered) else 0.0
            row["hit_at_1"] = float(covered["hit_at_1"].mean()) if len(covered) else 0.0
            row["hit_at_3"] = float(covered["hit_at_3"].mean()) if len(covered) else 0.0
            row["hit_at_5"] = float(covered["hit_at_5"].mean()) if len(covered) else 0.0
            row["mrr"] = float(covered["reciprocal_rank"].mean()) if len(covered) else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_dataset(repo_root, dataset, embedding_name, dtype, output_root):
    dataset_dir = os.path.join(repo_root, "datasets", dataset)
    item_info, category_field, item_to_category, category_to_items = load_item_info(dataset_dir)
    emb_path, category_ids, embeddings, counts = load_category_embeddings(repo_root, dataset, embedding_name, dtype)
    category_to_idx = {category: idx for idx, category in enumerate(category_ids)}

    input_bundles = load_bundle_file(os.path.join(dataset_dir, "bi_test_input.txt"))
    gt_bundles = load_bundle_file(os.path.join(dataset_dir, "bi_test_gt.txt"))
    common_bundle_ids = sorted(set(input_bundles) & set(gt_bundles))

    detail_rows = []
    per_gt_rows = []
    for sample_idx, bundle_id in enumerate(common_bundle_ids):
        input_categories, input_missing = items_to_categories(input_bundles[bundle_id], item_to_category)
        heldout_categories, heldout_missing = items_to_categories(gt_bundles[bundle_id], item_to_category)
        observed_indices = [category_to_idx[c] for c in input_categories if c in category_to_idx]
        heldout_indices = [category_to_idx[c] for c in heldout_categories if c in category_to_idx]
        if not observed_indices or not heldout_indices:
            detail_rows.append({
                "dataset": dataset,
                "bundle_id": bundle_id,
                "sample_idx": sample_idx,
                "covered": 0,
                "input_categories": json.dumps(input_categories, ensure_ascii=False),
                "heldout_categories": json.dumps(heldout_categories, ensure_ascii=False),
                "observed_category_count": len(observed_indices),
                "heldout_category_count": len(heldout_indices),
                "input_missing_item_count": len(input_missing),
                "heldout_missing_item_count": len(heldout_missing),
            })
            continue

        query = embeddings[observed_indices].mean(axis=0)
        query_norm = np.linalg.norm(query)
        if query_norm <= 0:
            covered = 0
            scores = np.zeros(len(category_ids), dtype=np.float64)
        else:
            covered = 1
            query = query / query_norm
            scores = embeddings @ query

        ranked = rank_categories(scores, observed_indices)
        rank_by_idx = {idx: rank + 1 for rank, (idx, _) in enumerate(ranked)}
        heldout_ranks = [rank_by_idx[idx] for idx in heldout_indices if idx in rank_by_idx]
        heldout_scores = [float(scores[idx]) for idx in heldout_indices]
        best_rank = min(heldout_ranks) if heldout_ranks else np.nan
        best_score = max(heldout_scores) if heldout_scores else np.nan
        mean_score = float(np.mean(heldout_scores)) if heldout_scores else np.nan
        top_categories = [
            {
                "rank": rank + 1,
                "category_id": category_ids[idx],
                "score": float(score),
                "representative_items": representative_items(dataset, item_info, category_to_items, category_ids[idx], k=1),
            }
            for rank, (idx, score) in enumerate(ranked[:10])
        ]

        for gt_idx in heldout_indices:
            per_gt_rows.append({
                "dataset": dataset,
                "bundle_id": bundle_id,
                "sample_idx": sample_idx,
                "gt_category": category_ids[gt_idx],
                "gt_cosine": float(scores[gt_idx]),
                "gt_rank": int(rank_by_idx[gt_idx]) if gt_idx in rank_by_idx else "",
                "gt_representative_items": json.dumps(
                    representative_items(dataset, item_info, category_to_items, category_ids[gt_idx], k=3),
                    ensure_ascii=False,
                ),
            })

        detail_rows.append({
            "dataset": dataset,
            "bundle_id": bundle_id,
            "sample_idx": sample_idx,
            "covered": covered,
            "input_categories": json.dumps(input_categories, ensure_ascii=False),
            "heldout_categories": json.dumps(heldout_categories, ensure_ascii=False),
            "input_representative_items": json.dumps(
                {
                    category: representative_items(dataset, item_info, category_to_items, category, k=1)
                    for category in input_categories
                },
                ensure_ascii=False,
            ),
            "heldout_representative_items": json.dumps(
                {
                    category: representative_items(dataset, item_info, category_to_items, category, k=1)
                    for category in heldout_categories
                },
                ensure_ascii=False,
            ),
            "observed_category_count": len(observed_indices),
            "heldout_category_count": len(heldout_indices),
            "gt_best_cosine": best_score,
            "gt_mean_cosine": mean_score,
            "gt_best_rank": best_rank,
            "hit_at_1": int(best_rank == 1),
            "hit_at_3": int(best_rank <= 3) if not np.isnan(best_rank) else 0,
            "hit_at_5": int(best_rank <= 5) if not np.isnan(best_rank) else 0,
            "reciprocal_rank": float(1.0 / best_rank) if best_rank and not np.isnan(best_rank) else 0.0,
            "top_categories_by_cosine": json.dumps(top_categories, ensure_ascii=False),
            "input_missing_item_count": len(input_missing),
            "heldout_missing_item_count": len(heldout_missing),
        })

    detail = pd.DataFrame(detail_rows)
    per_gt = pd.DataFrame(per_gt_rows)
    summary = summarize(detail)

    output_dir = os.path.join(output_root, dataset)
    os.makedirs(output_dir, exist_ok=True)
    detail.to_csv(os.path.join(output_dir, "lightgcn_category_input_gt_cosine_detail.csv"), index=False, encoding="utf-8-sig")
    per_gt.to_csv(os.path.join(output_dir, "lightgcn_category_input_gt_cosine_per_gt.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(os.path.join(output_dir, "lightgcn_category_input_gt_cosine_summary.csv"), index=False, encoding="utf-8-sig")
    metadata = {
        "dataset": dataset,
        "embedding_name": embedding_name,
        "embedding_path": emb_path,
        "category_field": category_field,
        "num_categories": len(category_ids),
        "embedding_dim": int(embeddings.shape[1]),
        "test_bundles": len(common_bundle_ids),
        "definition": "cosine(mean(input category embeddings), heldout GT category embedding)",
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return detail, per_gt, summary


def write_summary_md(path, summary_all):
    lines = [
        "# LightGCN Category Input-GT Cosine Signal",
        "",
        "This evaluates cosine similarity between the mean BI-LightGCN category embedding of input categories and held-out GT category embeddings.",
        "",
        "| dataset | grouping | n | coverage | mean cosine | median cosine | hit@1 | hit@3 | hit@5 | MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overall = summary_all[summary_all["grouping"] == "dataset"].copy()
    for _, row in overall.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['grouping']} | {int(row['n'])} | "
            f"{row['coverage']:.3f} | {row['mean_gt_best_cosine']:.3f} | "
            f"{row['median_gt_best_cosine']:.3f} | {row['hit_at_1']:.3f} | "
            f"{row['hit_at_3']:.3f} | {row['hit_at_5']:.3f} | {row['mrr']:.3f} |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze input mean vs GT category cosine using BI-LightGCN category embeddings.")
    parser.add_argument("--datasets", nargs="+", default=["pog", "pog_dense"])
    parser.add_argument("--embedding_name", default="LightGCN_bi")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--output_dir", default=os.path.join("analysis", "lightgcn_category_embedding_signal"))
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_root = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_root, exist_ok=True)
    details = []
    per_gts = []
    summaries = []
    for dataset in args.datasets:
        detail, per_gt, summary = analyze_dataset(repo_root, dataset, args.embedding_name, args.dtype, output_root)
        details.append(detail)
        per_gts.append(per_gt)
        summaries.append(summary)

    detail_all = pd.concat(details, ignore_index=True)
    per_gt_all = pd.concat(per_gts, ignore_index=True)
    summary_all = pd.concat(summaries, ignore_index=True)
    detail_all.to_csv(os.path.join(output_root, "lightgcn_category_input_gt_cosine_detail_all.csv"), index=False, encoding="utf-8-sig")
    per_gt_all.to_csv(os.path.join(output_root, "lightgcn_category_input_gt_cosine_per_gt_all.csv"), index=False, encoding="utf-8-sig")
    summary_all.to_csv(os.path.join(output_root, "lightgcn_category_input_gt_cosine_summary_all.csv"), index=False, encoding="utf-8-sig")
    write_summary_md(os.path.join(output_root, "summary.md"), summary_all)
    print(f"[Done] Wrote LightGCN category cosine analysis to {output_root}")


if __name__ == "__main__":
    main()

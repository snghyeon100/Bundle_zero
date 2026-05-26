import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import BundleZeroShotDataset, set_seed  # noqa: E402


def mean(values):
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def jaccard(a, b):
    a = set(a)
    b = set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_category_names(dataset_name):
    path = REPO_ROOT / "analysis" / "category_names" / "gemini" / dataset_name / "category_names.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for row in data.get("categories", []):
        category_id = str(row.get("category_id", "")).strip()
        name = str(row.get("category_name_en", "")).strip()
        if category_id and name:
            out[category_id] = name
    return out


def load_category_embeddings(dataset_name):
    path = (
        REPO_ROOT
        / "analysis"
        / "category_embedding_cache"
        / "LightGCN_bi"
        / "all_items"
        / dataset_name
        / "category_embeddings_LightGCN_bi_float32.npz"
    )
    with np.load(path, allow_pickle=True) as data:
        category_ids = [str(x) for x in data["category_ids"].tolist()]
        embeddings = data["embeddings_normed"].astype(np.float32)
    return path, category_ids, embeddings


def category_role_top_categories(dataset, category_to_row, embeddings, input_categories, top_k):
    rows = [category_to_row[c] for c in input_categories if c in category_to_row]
    if not rows:
        return [], []
    query = embeddings[np.asarray(rows, dtype=np.int64)].mean(axis=0)
    norm = float(np.linalg.norm(query))
    if norm <= 0:
        return [], []
    query = query / norm
    scores = embeddings @ query
    input_set = set(input_categories)
    ranked = [
        (category, float(scores[row]))
        for category, row in category_to_row.items()
        if category not in input_set
    ]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    top = ranked[:top_k]
    return [c for c, _ in top], [s for _, s in top]


def category_prior_top_categories(dataset, input_categories, top_k):
    scores, observed_support = dataset._category_prior_scores(input_categories)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [c for c, _ in ranked], [float(s) for _, s in ranked], int(observed_support)


def candidate_items_supported_by_categories(dataset, candidate_items, categories):
    category_set = set(categories)
    return [
        int(item_id)
        for item_id in candidate_items
        if dataset._get_item_category(int(item_id)) in category_set
    ]


def build_train_outfit_category_embeddings(dataset, category_to_row, embeddings):
    bundle_ids = []
    bundle_vectors = []
    bundle_categories = {}
    for bundle_id, categories in sorted(dataset.cc_retrieval_train_categories.items()):
        valid = [c for c in sorted(categories) if c in category_to_row]
        if not valid:
            continue
        rows = np.asarray([category_to_row[c] for c in valid], dtype=np.int64)
        vec = embeddings[rows].mean(axis=0)
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            continue
        bundle_ids.append(int(bundle_id))
        bundle_vectors.append((vec / norm).astype(np.float32))
        bundle_categories[int(bundle_id)] = valid
    return np.asarray(bundle_ids, dtype=np.int64), np.vstack(bundle_vectors), bundle_categories


def graph_role_outfit_top_bundles(
    dataset,
    category_to_row,
    embeddings,
    train_bundle_ids,
    train_bundle_vectors,
    input_categories,
    top_k,
):
    rows = [category_to_row[c] for c in input_categories if c in category_to_row]
    if not rows:
        return [], []
    query = embeddings[np.asarray(rows, dtype=np.int64)].mean(axis=0)
    norm = float(np.linalg.norm(query))
    if norm <= 0:
        return [], []
    query = query / norm
    scores = train_bundle_vectors @ query
    order = np.lexsort((train_bundle_ids, -scores))
    top_indices = order[:top_k]
    return [int(train_bundle_ids[i]) for i in top_indices], [float(scores[i]) for i in top_indices]


def cc_retrieval_top_bundles(dataset, sample):
    result = dataset.retrieve_cc_retrieval_context(sample)
    if not result:
        return [], []
    meta = result.get("metadata", {})
    bundle_ids = json.loads(meta.get("cc_retrieval_context_bundle_ids", "[]"))
    scores = json.loads(meta.get("cc_retrieval_context_scores", "[]"))
    return [int(x) for x in bundle_ids], [float(x) for x in scores]


def names_for(categories, category_names):
    return [category_names.get(str(c), str(c)) for c in categories]


def analyze_dataset(args, dataset_name):
    conf = {
        "dataset": dataset_name,
        "data_path": str(REPO_ROOT / "datasets"),
        "num_cans": args.num_cans,
        "toy_eval": args.toy_eval,
        "num_token": args.num_token,
        "seed": args.seed,
        "shuffle_seed": args.shuffle_seed,
        "use_hard_negative": False,
        "use_category_completion_prior_desc": True,
        "category_prior_top_k": args.category_top_k,
        "category_prior_verbalization": "category_names",
        "category_prior_min_support": args.category_prior_min_support,
        "category_prior_max_itemset_size": args.category_prior_max_itemset_size,
        "use_cc_retrieval_context": True,
        "cc_retrieval_context_k": args.outfit_top_k,
        "cc_retrieval_context_seed": args.seed,
        "cc_retrieval_overlap_weight": args.cc_overlap_weight,
        "cc_retrieval_extra_weight": args.cc_extra_weight,
        "use_category_name_aug": False,
        "use_ui_category_purchase_prior": False,
    }

    set_seed(args.seed)
    dataset = BundleZeroShotDataset(conf)
    samples = dataset.get_eval_samples()
    emb_path, category_ids, category_embeddings = load_category_embeddings(dataset_name)
    category_to_row = {category: idx for idx, category in enumerate(category_ids)}
    category_names = load_category_names(dataset_name)
    train_bundle_ids, train_bundle_vectors, train_bundle_categories = build_train_outfit_category_embeddings(
        dataset,
        category_to_row,
        category_embeddings,
    )

    rows = []
    for sample_idx, sample in enumerate(samples):
        input_items = [int(x) for x in sample.get("input_indices", [])]
        candidate_items = [int(x) for x in sample.get("candidate_indices", [])]
        input_categories = dataset._items_to_unique_categories(input_items)

        role_categories, role_scores = category_role_top_categories(
            dataset,
            category_to_row,
            category_embeddings,
            input_categories,
            args.category_top_k,
        )
        prior_categories, prior_scores, prior_support = category_prior_top_categories(
            dataset,
            input_categories,
            args.category_top_k,
        )
        role_supported_items = candidate_items_supported_by_categories(dataset, candidate_items, role_categories)
        prior_supported_items = candidate_items_supported_by_categories(dataset, candidate_items, prior_categories)

        rag_bundles, rag_scores = graph_role_outfit_top_bundles(
            dataset,
            category_to_row,
            category_embeddings,
            train_bundle_ids,
            train_bundle_vectors,
            input_categories,
            args.outfit_top_k,
        )
        cc_bundles, cc_scores = cc_retrieval_top_bundles(dataset, sample)
        rag_category_union = set()
        for bundle_id in rag_bundles:
            rag_category_union.update(train_bundle_categories.get(int(bundle_id), []))
        cc_category_union = set()
        for bundle_id in cc_bundles:
            cc_category_union.update(dataset.cc_retrieval_train_categories.get(int(bundle_id), set()))

        rows.append({
            "dataset": dataset_name,
            "sample_idx": sample_idx,
            "bundle_id": int(sample.get("bundle_id")),
            "input_count": len(input_items),
            "input_items": json.dumps(input_items),
            "input_categories": json.dumps(input_categories, ensure_ascii=False),
            "input_category_names": json.dumps(names_for(input_categories, category_names), ensure_ascii=False),
            "role_top_categories": json.dumps(role_categories, ensure_ascii=False),
            "role_top_category_names": json.dumps(names_for(role_categories, category_names), ensure_ascii=False),
            "role_top_scores": json.dumps(role_scores),
            "prior_top_categories": json.dumps(prior_categories, ensure_ascii=False),
            "prior_top_category_names": json.dumps(names_for(prior_categories, category_names), ensure_ascii=False),
            "prior_top_scores": json.dumps(prior_scores),
            "prior_observed_support": prior_support,
            "category_overlap_count": len(set(role_categories) & set(prior_categories)),
            "category_jaccard": jaccard(role_categories, prior_categories),
            "role_supported_candidate_items": json.dumps(role_supported_items),
            "prior_supported_candidate_items": json.dumps(prior_supported_items),
            "candidate_item_overlap_count": len(set(role_supported_items) & set(prior_supported_items)),
            "candidate_item_jaccard": jaccard(role_supported_items, prior_supported_items),
            "role_candidate_supported_count": len(role_supported_items),
            "prior_candidate_supported_count": len(prior_supported_items),
            "rag_bundle_ids": json.dumps(rag_bundles),
            "rag_bundle_scores": json.dumps(rag_scores),
            "cc_bundle_ids": json.dumps(cc_bundles),
            "cc_bundle_scores": json.dumps(cc_scores),
            "bundle_overlap_count": len(set(rag_bundles) & set(cc_bundles)),
            "bundle_jaccard": jaccard(rag_bundles, cc_bundles),
            "same_top1_bundle": int(bool(rag_bundles and cc_bundles and rag_bundles[0] == cc_bundles[0])),
            "retrieved_category_union_jaccard": jaccard(rag_category_union, cc_category_union),
        })

    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"{dataset_name}_bi_category_rag_vs_existing_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped["all"].append(row)
        grouped[f"input_{row['input_count']}"].append(row)

    summary_rows = []
    for group, group_rows in grouped.items():
        summary_rows.append({
            "dataset": dataset_name,
            "group": group,
            "n": len(group_rows),
            "category_overlap_mean": mean(row["category_overlap_count"] for row in group_rows),
            "category_jaccard_mean": mean(row["category_jaccard"] for row in group_rows),
            "category_no_overlap_rate": mean(row["category_overlap_count"] == 0 for row in group_rows),
            "candidate_item_overlap_mean": mean(row["candidate_item_overlap_count"] for row in group_rows),
            "candidate_item_jaccard_mean": mean(row["candidate_item_jaccard"] for row in group_rows),
            "role_supported_candidate_count_mean": mean(row["role_candidate_supported_count"] for row in group_rows),
            "prior_supported_candidate_count_mean": mean(row["prior_candidate_supported_count"] for row in group_rows),
            "bundle_overlap_mean": mean(row["bundle_overlap_count"] for row in group_rows),
            "bundle_jaccard_mean": mean(row["bundle_jaccard"] for row in group_rows),
            "bundle_no_overlap_rate": mean(row["bundle_overlap_count"] == 0 for row in group_rows),
            "same_top1_bundle_rate": mean(row["same_top1_bundle"] for row in group_rows),
            "retrieved_category_union_jaccard_mean": mean(row["retrieved_category_union_jaccard"] for row in group_rows),
        })

    return {
        "dataset": dataset_name,
        "detail_path": str(detail_path),
        "summary_rows": summary_rows,
        "embedding_path": str(emb_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="pog,pog_dense")
    parser.add_argument("--toy-eval", type=int, default=250)
    parser.add_argument("--num-cans", type=int, default=10)
    parser.add_argument("--num-token", type=int, default=5)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--shuffle-seed", type=int, default=41)
    parser.add_argument("--category-top-k", type=int, default=3)
    parser.add_argument("--outfit-top-k", type=int, default=3)
    parser.add_argument("--category-prior-min-support", type=int, default=3)
    parser.add_argument("--category-prior-max-itemset-size", type=int, default=6)
    parser.add_argument("--cc-overlap-weight", type=float, default=1.0)
    parser.add_argument("--cc-extra-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", default=os.path.join("analysis", "bi_category_rag_vs_existing"))
    args = parser.parse_args()

    all_summary_rows = []
    outputs = []
    for dataset_name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        result = analyze_dataset(args, dataset_name)
        all_summary_rows.extend(result["summary_rows"])
        outputs.append(result)

    out_dir = REPO_ROOT / args.output_dir
    summary_path = out_dir / "bi_category_rag_vs_existing_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_summary_rows)

    print(f"Saved summary: {summary_path}")
    for result in outputs:
        print(f"Saved detail ({result['dataset']}): {result['detail_path']}")
    print("")
    for row in all_summary_rows:
        if row["group"] != "all":
            continue
        print(
            f"{row['dataset']}: "
            f"cat_jaccard={row['category_jaccard_mean']:.3f}, "
            f"candidate_item_jaccard={row['candidate_item_jaccard_mean']:.3f}, "
            f"bundle_jaccard={row['bundle_jaccard_mean']:.3f}, "
            f"same_top1_bundle={row['same_top1_bundle_rate']:.3f}, "
            f"bundle_no_overlap={row['bundle_no_overlap_rate']:.3f}"
        )


if __name__ == "__main__":
    main()

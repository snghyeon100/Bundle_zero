import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASETS = ["pog", "pog_dense"]
DEFAULT_OUTPUT_ROOT = r"analysis\category_completion_selfsup"
CATEGORY_FIELD_CANDIDATES = ["cate_id", "cate", "category"]


def detect_category_field(item_info):
    counts = {field: 0 for field in CATEGORY_FIELD_CANDIDATES}
    for item in item_info.values():
        for field in CATEGORY_FIELD_CANDIDATES:
            value = item.get(field)
            if value is not None and str(value).strip():
                counts[field] += 1
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise ValueError("No category-like field found in item_info.json")
    return best


def load_item_categories(dataset_dir, category_field):
    path = dataset_dir / "item_info.json"
    with open(path, encoding="utf-8") as f:
        item_info = json.load(f)
    resolved = detect_category_field(item_info) if category_field == "auto" else category_field
    item_to_category = {}
    for item_id, item in item_info.items():
        value = item.get(resolved)
        if value is not None and str(value).strip():
            item_to_category[int(item_id)] = str(value).strip()
    return item_to_category, resolved, path


def load_bundle_file(path):
    bundles = {}
    with open(path, encoding="utf-8") as f:
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
    categories = []
    missing = []
    for item_id in item_ids:
        category = item_to_category.get(item_id)
        if category is None:
            missing.append(item_id)
        elif category not in categories:
            categories.append(category)
    return sorted(categories), missing


def canonical(categories):
    return "|".join(sorted(categories))


def nonempty_subsets(categories, max_size=None):
    max_size = len(categories) if max_size is None else min(max_size, len(categories))
    for size in range(max_size, 0, -1):
        for combo in combinations(sorted(categories), size):
            yield combo


def build_train_counts(train_bundles, item_to_category, max_itemset_size):
    itemset_counts = defaultdict(Counter)
    category_counts = Counter()
    usable_bundles = 0
    missing_occurrences = 0

    for item_ids in train_bundles.values():
        categories, missing = items_to_categories(item_ids, item_to_category)
        missing_occurrences += len(missing)
        if not categories:
            continue
        usable_bundles += 1
        for category in categories:
            category_counts[category] += 1
        for size in range(1, min(max_itemset_size, len(categories)) + 1):
            for combo in combinations(categories, size):
                itemset_counts[size][canonical(combo)] += 1

    return {
        "itemset_counts": itemset_counts,
        "category_counts": category_counts,
        "num_train_bundles": usable_bundles,
        "missing_occurrences": missing_occurrences,
        "categories": sorted(category_counts),
    }


def direct_scores(observed, counts, rule_type, min_support):
    observed_key = canonical(observed)
    observed_size = len(observed)
    observed_count = counts["itemset_counts"][observed_size].get(observed_key, 0)
    if observed_count < min_support:
        return {}, {"coverage": 0, "used_level": 0, "used_support": observed_count}

    scores = {}
    for category in counts["categories"]:
        if category in observed:
            continue
        joint = canonical(list(observed) + [category])
        joint_count = counts["itemset_counts"][observed_size + 1].get(joint, 0)
        confidence = joint_count / observed_count
        score = transform_score(confidence, category, counts, rule_type)
        scores[category] = score
    return scores, {"coverage": 1, "used_level": observed_size, "used_support": observed_count}


def backoff_scores(observed, counts, rule_type, min_support, max_backoff_size):
    observed = sorted(observed)
    accum = defaultdict(float)
    weights = defaultdict(float)
    used_supports = []
    used_level = 0

    for subset in nonempty_subsets(observed, max_backoff_size):
        subset = list(subset)
        subset_size = len(subset)
        subset_key = canonical(subset)
        subset_count = counts["itemset_counts"][subset_size].get(subset_key, 0)
        if subset_count < min_support:
            continue
        used_level = max(used_level, subset_size)
        used_supports.append(subset_count)
        subset_weight = float(subset_size * subset_count)
        for category in counts["categories"]:
            if category in observed:
                continue
            joint = canonical(subset + [category])
            joint_count = counts["itemset_counts"][subset_size + 1].get(joint, 0)
            confidence = joint_count / subset_count
            score = transform_score(confidence, category, counts, rule_type)
            accum[category] += subset_weight * score
            weights[category] += subset_weight

    if not weights:
        return {}, {"coverage": 0, "used_level": 0, "used_support": 0}

    scores = {category: accum[category] / weights[category] for category in weights}
    return scores, {
        "coverage": 1,
        "used_level": used_level,
        "used_support": int(max(used_supports) if used_supports else 0),
    }


def transform_score(confidence, category, counts, rule_type):
    if rule_type == "confidence":
        return confidence
    p_category = counts["category_counts"][category] / counts["num_train_bundles"]
    if p_category <= 0:
        return 0.0
    lift = confidence / p_category
    if rule_type == "lift":
        return lift
    if rule_type == "pmi":
        return math.log2(lift) if lift > 0 else -100.0
    raise ValueError(f"Unknown rule_type={rule_type}")


def rank_predictions(scores):
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def evaluate_prediction(ranked, heldout, k_values):
    heldout = set(heldout)
    ranks = [idx + 1 for idx, (category, _) in enumerate(ranked) if category in heldout]
    best_rank = min(ranks) if ranks else np.nan
    out = {
        "heldout_best_rank": best_rank,
        "mrr": 1.0 / best_rank if ranks else 0.0,
    }
    for k in k_values:
        top_k = {category for category, _ in ranked[:k]}
        hits = len(heldout & top_k)
        out[f"hit_at_{k}"] = int(hits > 0)
        out[f"recall_at_{k}"] = hits / len(heldout) if heldout else np.nan
    return out


def evaluate_dataset(repo_root, dataset, args):
    dataset_dir = repo_root / args.data_path / dataset
    item_to_category, category_field, item_info_path = load_item_categories(dataset_dir, args.category_field)
    train_bundles = load_bundle_file(dataset_dir / args.train_file)
    test_input_bundles = load_bundle_file(dataset_dir / args.test_input_file)
    test_gt_bundles = load_bundle_file(dataset_dir / args.test_gt_file)

    counts = build_train_counts(
        train_bundles=train_bundles,
        item_to_category=item_to_category,
        max_itemset_size=args.max_itemset_size,
    )
    rows = []
    missing_test_bundles = sorted(set(test_input_bundles) ^ set(test_gt_bundles))
    common_bundle_ids = sorted(set(test_input_bundles) & set(test_gt_bundles))

    base_rule_types = args.rule_types
    methods = []
    for rule_type in base_rule_types:
        methods.append((f"direct_{rule_type}", rule_type, "direct"))
        methods.append((f"backoff_{rule_type}", rule_type, "backoff"))

    for bundle_id in common_bundle_ids:
        observed, observed_missing = items_to_categories(test_input_bundles[bundle_id], item_to_category)
        heldout, heldout_missing = items_to_categories(test_gt_bundles[bundle_id], item_to_category)
        if not observed or not heldout:
            continue

        for method_name, rule_type, mode in methods:
            if mode == "direct":
                if len(observed) + 1 > args.max_itemset_size:
                    scores, coverage = {}, {"coverage": 0, "used_level": len(observed), "used_support": 0}
                else:
                    scores, coverage = direct_scores(observed, counts, rule_type, args.min_support)
            else:
                scores, coverage = backoff_scores(
                    observed,
                    counts,
                    rule_type,
                    args.min_support,
                    max_backoff_size=min(args.max_backoff_size, args.max_itemset_size - 1),
                )

            ranked = rank_predictions(scores)
            metrics = evaluate_prediction(ranked, heldout, args.k_values)
            top_n = ranked[: args.store_top_n]
            rows.append(
                {
                    "dataset": dataset,
                    "bundle_id": bundle_id,
                    "method": method_name,
                    "rule_type": rule_type,
                    "scoring_mode": mode,
                    "observed_size": len(observed),
                    "heldout_size": len(heldout),
                    "observed_categories": json.dumps(observed, ensure_ascii=False),
                    "heldout_categories": json.dumps(heldout, ensure_ascii=False),
                    "coverage": coverage["coverage"],
                    "used_level": coverage["used_level"],
                    "used_support": coverage["used_support"],
                    "num_predictions": len(ranked),
                    "top_predictions": json.dumps([category for category, _ in top_n], ensure_ascii=False),
                    "top_scores": json.dumps([float(score) for _, score in top_n], ensure_ascii=False),
                    "observed_missing_item_count": len(observed_missing),
                    "heldout_missing_item_count": len(heldout_missing),
                    **metrics,
                }
            )

    metadata = {
        "dataset": dataset,
        "item_info_path": str(item_info_path),
        "category_field": category_field,
        "train_file": str(dataset_dir / args.train_file),
        "test_input_file": str(dataset_dir / args.test_input_file),
        "test_gt_file": str(dataset_dir / args.test_gt_file),
        "num_train_bundles": len(train_bundles),
        "num_usable_train_bundles": counts["num_train_bundles"],
        "num_train_categories": len(counts["categories"]),
        "train_missing_category_occurrences": counts["missing_occurrences"],
        "num_test_input_bundles": len(test_input_bundles),
        "num_test_gt_bundles": len(test_gt_bundles),
        "num_common_test_bundles": len(common_bundle_ids),
        "missing_test_bundle_count": len(missing_test_bundles),
        "min_support": args.min_support,
        "max_itemset_size": args.max_itemset_size,
        "max_backoff_size": args.max_backoff_size,
    }
    return pd.DataFrame(rows), metadata


def summarize(detail):
    rows = []
    groupings = [
        ["dataset", "method"],
        ["dataset", "method", "observed_size"],
        ["dataset", "method", "observed_size", "heldout_size"],
    ]
    for grouping in groupings:
        for keys, sub in detail.groupby(grouping):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(grouping, keys))
            row["grouping"] = "+".join(grouping)
            row["n"] = int(len(sub))
            row["coverage"] = float(sub["coverage"].mean())
            covered = sub[sub["coverage"] == 1]
            eval_sub = covered if not covered.empty else sub
            row["n_covered"] = int(len(covered))
            row["mrr"] = float(eval_sub["mrr"].mean())
            row["heldout_best_rank_mean"] = float(eval_sub["heldout_best_rank"].dropna().mean())
            for col in sorted([c for c in detail.columns if c.startswith("hit_at_")], key=lambda x: int(x.split("_")[-1])):
                row[col] = float(eval_sub[col].mean())
            for col in sorted([c for c in detail.columns if c.startswith("recall_at_")], key=lambda x: int(x.split("_")[-1])):
                row[col] = float(eval_sub[col].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(path, summary):
    overall = summary[summary["grouping"] == "dataset+method"].copy()
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Category Completion Self-supervised Evaluation\n\n")
        f.write("Rules are learned from `bi_train.txt`; evaluation uses `bi_test_input.txt` as observed categories and `bi_test_gt.txt` as held-out categories.\n\n")
        f.write("| dataset | method | n | coverage | hit@1 | hit@3 | hit@5 | recall@3 | MRR |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in overall.iterrows():
            f.write(
                f"| {row['dataset']} | {row['method']} | {int(row['n'])} | {row['coverage']:.3f} | "
                f"{row.get('hit_at_1', np.nan):.3f} | {row.get('hit_at_3', np.nan):.3f} | "
                f"{row.get('hit_at_5', np.nan):.3f} | {row.get('recall_at_3', np.nan):.3f} | "
                f"{row['mrr']:.3f} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Evaluate category-only completion rules learned from train bundles.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--data-path", default="datasets")
    parser.add_argument("--category-field", default="auto")
    parser.add_argument("--train-file", default="bi_train.txt")
    parser.add_argument("--test-input-file", default="bi_test_input.txt")
    parser.add_argument("--test-gt-file", default="bi_test_gt.txt")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rule-types", nargs="+", default=["confidence", "lift", "pmi"], choices=["confidence", "lift", "pmi"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--max-itemset-size", type=int, default=4)
    parser.add_argument("--max-backoff-size", type=int, default=3)
    parser.add_argument("--store-top-n", type=int, default=10)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    details = []
    metadata = []
    for dataset in args.datasets:
        detail, meta = evaluate_dataset(repo_root, dataset, args)
        details.append(detail)
        metadata.append(meta)
        print(f"\n=== {dataset} ===")
        print(f"category field     : {meta['category_field']}")
        print(f"train bundles      : {meta['num_train_bundles']}")
        print(f"test bundles       : {meta['num_common_test_bundles']}")
        print(f"train categories   : {meta['num_train_categories']}")

    detail_all = pd.concat(details, ignore_index=True)
    summary = summarize(detail_all)

    out_dir = repo_root / args.output_root / "train_to_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "category_completion_selfsup_detail.csv"
    summary_path = out_dir / "category_completion_selfsup_summary.csv"
    sanity_path = out_dir / "sanity.json"
    md_path = out_dir / "summary.md"

    detail_all.to_csv(detail_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    write_markdown(md_path, summary)

    print("\n=== outputs ===")
    print(f"detail  : {detail_path}")
    print(f"summary : {summary_path}")
    print(f"sanity  : {sanity_path}")
    print(f"markdown: {md_path}")
    print("\n=== overall ===")
    print(summary[summary["grouping"] == "dataset+method"].to_string(index=False))


if __name__ == "__main__":
    main()

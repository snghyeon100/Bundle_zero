import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd


DEFAULT_DATASETS = ["pog", "pog_dense"]
DEFAULT_OUTPUT_ROOT = r"analysis\bundle_category_cooccurrence"
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
    item_info_path = dataset_dir / "item_info.json"
    if not item_info_path.exists():
        raise FileNotFoundError(item_info_path)
    with open(item_info_path, encoding="utf-8") as f:
        item_info = json.load(f)

    resolved_field = detect_category_field(item_info) if category_field == "auto" else category_field
    item_to_category = {}
    missing_category = 0
    for item_id, item in item_info.items():
        value = item.get(resolved_field)
        if value is None or not str(value).strip():
            missing_category += 1
            continue
        item_to_category[int(item_id)] = str(value).strip()
    return item_to_category, resolved_field, item_info_path, missing_category


def load_bundles(dataset_dir, bundle_file):
    path = dataset_dir / bundle_file
    if not path.exists():
        raise FileNotFoundError(path)
    bundles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [int(v) for v in line.split(", ") if v]
            if len(vals) < 2:
                continue
            bundle_id = vals[0]
            seen = set()
            items = []
            for item_id in vals[1:]:
                if item_id not in seen:
                    items.append(item_id)
                    seen.add(item_id)
            if items:
                bundles.append((bundle_id, items))
    return bundles, path


def canonical_set(values):
    return "|".join(sorted(values))


def entropy_from_counts(counts):
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def make_bundle_detail(dataset, bundles, item_to_category):
    rows = []
    missing_items = Counter()
    for bundle_id, item_ids in bundles:
        categories = []
        for item_id in item_ids:
            category = item_to_category.get(item_id)
            if category is None:
                missing_items[item_id] += 1
                continue
            categories.append(category)

        multiset = Counter(categories)
        unique_categories = sorted(multiset)
        rows.append(
            {
                "dataset": dataset,
                "bundle_id": bundle_id,
                "item_count": len(item_ids),
                "categorized_item_count": len(categories),
                "category_count": len(categories),
                "unique_category_count": len(unique_categories),
                "category_set": canonical_set(unique_categories),
                "category_multiset": json.dumps(dict(sorted(multiset.items())), ensure_ascii=False),
                "has_repeated_category": int(any(count > 1 for count in multiset.values())),
                "max_same_category_count": max(multiset.values(), default=0),
                "missing_category_item_count": len(item_ids) - len(categories),
            }
        )
    return pd.DataFrame(rows), missing_items


def make_category_frequency(detail):
    rows = []
    for dataset, ds in detail.groupby("dataset"):
        num_bundles = len(ds)
        bundle_count = Counter()
        item_count = Counter()
        for _, row in ds.iterrows():
            multiset = json.loads(row["category_multiset"])
            for category, count in multiset.items():
                item_count[category] += int(count)
                bundle_count[category] += 1
        total_items = sum(item_count.values())
        for category in sorted(bundle_count):
            rows.append(
                {
                    "dataset": dataset,
                    "category": category,
                    "bundle_count": bundle_count[category],
                    "item_count": item_count[category],
                    "bundle_ratio": bundle_count[category] / num_bundles,
                    "item_ratio": item_count[category] / total_items if total_items else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "bundle_count"], ascending=[True, False])


def make_pair_cooccurrence(detail, category_frequency):
    rows = []
    freq_by_dataset = {
        dataset: group.set_index("category")["bundle_count"].to_dict()
        for dataset, group in category_frequency.groupby("dataset")
    }
    for dataset, ds in detail.groupby("dataset"):
        num_bundles = len(ds)
        pair_counts = Counter()
        for category_set in ds["category_set"]:
            categories = category_set.split("|") if category_set else []
            for a, b in combinations(categories, 2):
                pair_counts[(a, b)] += 1

        bundle_counts = freq_by_dataset[dataset]
        for (a, b), count in pair_counts.items():
            count_a = bundle_counts[a]
            count_b = bundle_counts[b]
            p_a = count_a / num_bundles
            p_b = count_b / num_bundles
            p_ab = count / num_bundles
            lift = p_ab / (p_a * p_b) if p_a and p_b else 0.0
            pmi = math.log2(lift) if lift > 0 else float("-inf")
            rows.append(
                {
                    "dataset": dataset,
                    "category_a": a,
                    "category_b": b,
                    "bundle_count": count,
                    "support": p_ab,
                    "prob_b_given_a": count / count_a,
                    "prob_a_given_b": count / count_b,
                    "lift": lift,
                    "pmi": pmi,
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "bundle_count"], ascending=[True, False])


def make_set_patterns(detail):
    rows = []
    for dataset, ds in detail.groupby("dataset"):
        num_bundles = len(ds)
        counts = Counter(ds["category_set"])
        for category_set, count in counts.items():
            set_size = len(category_set.split("|")) if category_set else 0
            rows.append(
                {
                    "dataset": dataset,
                    "category_set": category_set,
                    "set_size": set_size,
                    "support": count,
                    "support_ratio": count / num_bundles,
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "support"], ascending=[True, False])


def make_itemset_patterns(detail, sizes):
    rows = []
    for dataset, ds in detail.groupby("dataset"):
        num_bundles = len(ds)
        for size in sizes:
            counts = Counter()
            for category_set in ds["category_set"]:
                categories = category_set.split("|") if category_set else []
                if len(categories) < size:
                    continue
                for combo in combinations(categories, size):
                    counts[canonical_set(combo)] += 1
            for itemset, count in counts.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "itemset_size": size,
                        "category_itemset": itemset,
                        "support": count,
                        "support_ratio": count / num_bundles,
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset", "itemset_size", "support"], ascending=[True, True, False])


def make_association_rules(detail, max_antecedent_size, min_support_count):
    rows = []
    for dataset, ds in detail.groupby("dataset"):
        num_bundles = len(ds)
        itemset_counts = defaultdict(Counter)
        bundle_sets = []
        for category_set in ds["category_set"]:
            categories = category_set.split("|") if category_set else []
            bundle_sets.append(categories)
            for size in range(1, max_antecedent_size + 2):
                if len(categories) < size:
                    continue
                for combo in combinations(categories, size):
                    itemset_counts[size][canonical_set(combo)] += 1

        for antecedent_size in range(1, max_antecedent_size + 1):
            for antecedent, antecedent_count in itemset_counts[antecedent_size].items():
                antecedent_categories = antecedent.split("|") if antecedent else []
                consequent_counts = Counter()
                for categories in bundle_sets:
                    category_set = set(categories)
                    if not set(antecedent_categories).issubset(category_set):
                        continue
                    for consequent in category_set - set(antecedent_categories):
                        consequent_counts[consequent] += 1

                for consequent, joint_count in consequent_counts.items():
                    if joint_count < min_support_count:
                        continue
                    consequent_count = itemset_counts[1][consequent]
                    support = joint_count / num_bundles
                    confidence = joint_count / antecedent_count
                    p_consequent = consequent_count / num_bundles
                    lift = confidence / p_consequent if p_consequent else 0.0
                    pmi = math.log2(lift) if lift > 0 else float("-inf")
                    rows.append(
                        {
                            "dataset": dataset,
                            "antecedent_size": antecedent_size,
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "joint_count": joint_count,
                            "antecedent_count": antecedent_count,
                            "consequent_count": consequent_count,
                            "support": support,
                            "confidence": confidence,
                            "lift": lift,
                            "pmi": pmi,
                        }
                    )
    if not rows:
        return pd.DataFrame(
            columns=[
                "dataset",
                "antecedent_size",
                "antecedent",
                "consequent",
                "joint_count",
                "antecedent_count",
                "consequent_count",
                "support",
                "confidence",
                "lift",
                "pmi",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["dataset", "antecedent_size", "joint_count", "confidence"],
        ascending=[True, True, False, False],
    )


def make_summary(detail, category_frequency, pair_cooccur, set_patterns):
    rows = []
    for dataset, ds in detail.groupby("dataset"):
        num_bundles = len(ds)
        set_counts = Counter(ds["category_set"])
        item_count_entropy = entropy_from_counts(ds["item_count"].value_counts().tolist())
        category_set_entropy = entropy_from_counts(set_counts.values())
        top_pair = ""
        top_pair_count = 0
        pair_ds = pair_cooccur[pair_cooccur["dataset"] == dataset]
        if not pair_ds.empty:
            top = pair_ds.iloc[0]
            top_pair = f"{top['category_a']}|{top['category_b']}"
            top_pair_count = int(top["bundle_count"])
        top_set = ""
        top_set_count = 0
        set_ds = set_patterns[set_patterns["dataset"] == dataset]
        if not set_ds.empty:
            top = set_ds.iloc[0]
            top_set = top["category_set"]
            top_set_count = int(top["support"])
        rows.append(
            {
                "dataset": dataset,
                "num_bundles": num_bundles,
                "num_categories": int((category_frequency["dataset"] == dataset).sum()),
                "avg_item_count": float(ds["item_count"].mean()),
                "avg_unique_category_count": float(ds["unique_category_count"].mean()),
                "median_unique_category_count": float(ds["unique_category_count"].median()),
                "repeated_category_bundle_rate": float(ds["has_repeated_category"].mean()),
                "unique_category_set_count": len(set_counts),
                "unique_category_set_ratio": len(set_counts) / num_bundles,
                "category_set_entropy": category_set_entropy,
                "item_count_entropy": item_count_entropy,
                "top_pair": top_pair,
                "top_pair_count": top_pair_count,
                "top_category_set": top_set,
                "top_category_set_count": top_set_count,
            }
        )
    return pd.DataFrame(rows)


def write_markdown(path, summary, category_frequency, pair_cooccur, set_patterns):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Bundle Category Co-occurrence Summary\n\n")
        f.write("| dataset | bundles | categories | avg items | avg unique cats | repeated-cat rate | unique set ratio |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for _, row in summary.iterrows():
            f.write(
                f"| {row['dataset']} | {int(row['num_bundles'])} | {int(row['num_categories'])} | "
                f"{row['avg_item_count']:.2f} | {row['avg_unique_category_count']:.2f} | "
                f"{row['repeated_category_bundle_rate']:.3f} | {row['unique_category_set_ratio']:.3f} |\n"
            )

        for dataset in summary["dataset"]:
            f.write(f"\n## {dataset}\n\n")
            f.write("Top categories by bundle count:\n\n")
            f.write("| category | bundle count | bundle ratio | item count |\n")
            f.write("|---|---:|---:|---:|\n")
            top_cats = category_frequency[category_frequency["dataset"] == dataset].head(10)
            for _, row in top_cats.iterrows():
                f.write(
                    f"| {row['category']} | {int(row['bundle_count'])} | "
                    f"{row['bundle_ratio']:.3f} | {int(row['item_count'])} |\n"
                )

            f.write("\nTop category pairs by bundle count:\n\n")
            f.write("| category A | category B | count | P(B|A) | lift | PMI |\n")
            f.write("|---|---|---:|---:|---:|---:|\n")
            top_pairs = pair_cooccur[pair_cooccur["dataset"] == dataset].head(10)
            for _, row in top_pairs.iterrows():
                f.write(
                    f"| {row['category_a']} | {row['category_b']} | {int(row['bundle_count'])} | "
                    f"{row['prob_b_given_a']:.3f} | {row['lift']:.3f} | {row['pmi']:.3f} |\n"
                )

            f.write("\nTop full category set patterns:\n\n")
            f.write("| category set | support | ratio | size |\n")
            f.write("|---|---:|---:|---:|\n")
            top_sets = set_patterns[set_patterns["dataset"] == dataset].head(10)
            for _, row in top_sets.iterrows():
                f.write(
                    f"| {row['category_set']} | {int(row['support'])} | "
                    f"{row['support_ratio']:.3f} | {int(row['set_size'])} |\n"
                )


def run_dataset(repo_root, dataset, args):
    dataset_dir = repo_root / args.data_path / dataset
    item_to_category, category_field, item_info_path, missing_category_items = load_item_categories(
        dataset_dir,
        args.category_field,
    )
    bundles, bundle_path = load_bundles(dataset_dir, args.bundle_file)
    detail, missing_items = make_bundle_detail(dataset, bundles, item_to_category)
    metadata = {
        "dataset": dataset,
        "bundle_file": str(bundle_path),
        "item_info_path": str(item_info_path),
        "category_field": category_field,
        "num_item_info_items": len(item_to_category) + missing_category_items,
        "num_categorized_items": len(item_to_category),
        "missing_category_items_in_item_info": missing_category_items,
        "num_bundles": len(bundles),
        "missing_item_ids_in_bundles": len(missing_items),
        "missing_item_occurrences_in_bundles": int(sum(missing_items.values())),
    }
    return detail, metadata


def main():
    parser = argparse.ArgumentParser(description="Analyze full bundle category co-occurrence without GT/candidate structure.")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--data-path", default="datasets")
    parser.add_argument("--bundle-file", default="bi_full.txt")
    parser.add_argument("--category-field", default="auto")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-antecedent-size", type=int, default=2)
    parser.add_argument("--min-rule-support", type=int, default=5)
    parser.add_argument("--itemset-sizes", nargs="+", type=int, default=[2, 3])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    details = []
    metadata = []
    for dataset in args.datasets:
        detail, meta = run_dataset(repo_root, dataset, args)
        details.append(detail)
        metadata.append(meta)
        print(f"\n=== {dataset} ===")
        print(f"bundle file    : {args.bundle_file}")
        print(f"category field : {meta['category_field']}")
        print(f"bundles        : {meta['num_bundles']}")
        print(f"missing items  : {meta['missing_item_occurrences_in_bundles']}")

    detail_all = pd.concat(details, ignore_index=True)
    category_frequency = make_category_frequency(detail_all)
    pair_cooccur = make_pair_cooccurrence(detail_all, category_frequency)
    set_patterns = make_set_patterns(detail_all)
    itemset_patterns = make_itemset_patterns(detail_all, args.itemset_sizes)
    rules = make_association_rules(detail_all, args.max_antecedent_size, args.min_rule_support)
    summary = make_summary(detail_all, category_frequency, pair_cooccur, set_patterns)

    out_dir = repo_root / args.output_root / Path(args.bundle_file).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "bundle_category_detail.csv"
    frequency_path = out_dir / "category_frequency.csv"
    pair_path = out_dir / "category_pair_cooccur.csv"
    set_path = out_dir / "category_set_patterns.csv"
    itemset_path = out_dir / "category_itemset_patterns.csv"
    rules_path = out_dir / "category_association_rules.csv"
    summary_path = out_dir / "bundle_category_summary.csv"
    sanity_path = out_dir / "sanity.json"
    md_path = out_dir / "summary.md"

    detail_all.to_csv(detail_path, index=False, quoting=csv.QUOTE_MINIMAL)
    category_frequency.to_csv(frequency_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pair_cooccur.to_csv(pair_path, index=False, quoting=csv.QUOTE_MINIMAL)
    set_patterns.to_csv(set_path, index=False, quoting=csv.QUOTE_MINIMAL)
    itemset_patterns.to_csv(itemset_path, index=False, quoting=csv.QUOTE_MINIMAL)
    rules.to_csv(rules_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    write_markdown(md_path, summary, category_frequency, pair_cooccur, set_patterns)

    print("\n=== outputs ===")
    print(f"detail    : {detail_path}")
    print(f"frequency : {frequency_path}")
    print(f"pairs     : {pair_path}")
    print(f"sets      : {set_path}")
    print(f"itemsets  : {itemset_path}")
    print(f"rules     : {rules_path}")
    print(f"summary   : {summary_path}")
    print(f"markdown  : {md_path}")
    print("\n=== summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

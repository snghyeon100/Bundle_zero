import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd


VIEWS = ("IBxBI", "IUxUI", "BIxIB")


def read_csv(path):
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return []


def item_category(item_info, item_id):
    info = item_info.get(str(int(item_id)), {})
    return info.get("cate") or info.get("cate_id") or info.get("category") or ""


def load_category_names(repo_root, dataset):
    path = repo_root / "analysis" / "category_names" / "gemini" / dataset / "category_names.csv"
    if not path.exists():
        return {}
    df = read_csv(path)
    return dict(zip(df["category_id"].astype(str), df["category_name_en"].astype(str)))


def category_label(category_id, category_names):
    if not category_id:
        return "Unknown"
    return category_names.get(str(category_id), str(category_id))


def top_items(counter, top_k=5):
    return "; ".join(f"{name} ({count})" for name, count in counter.most_common(top_k))


def analyze_dataset(repo_root, dataset):
    per_sample_path = repo_root / "analysis" / f"{dataset}_ranking_view_analysis" / "per_sample_oracle.csv"
    item_info_path = repo_root / "datasets" / dataset / "item_info.json"
    if not per_sample_path.exists():
        raise FileNotFoundError(per_sample_path)
    if not item_info_path.exists():
        raise FileNotFoundError(item_info_path)

    per = read_csv(per_sample_path)
    item_info = json.loads(item_info_path.read_text(encoding="utf-8"))
    category_names = load_category_names(repo_root, dataset)

    rows = []
    for _, row in per.iterrows():
        input_indices = parse_list(row.get("input_indices_IBxBI", row.get("input_indices", "[]")))
        candidate_indices = parse_list(row.get("candidate_indices", "[]"))
        true_item = int(row["true_indice"])
        true_category = item_category(item_info, true_item)
        true_category_name = category_label(true_category, category_names)
        input_categories = [item_category(item_info, item_id) for item_id in input_indices]
        input_category_names = [category_label(cat, category_names) for cat in input_categories]
        candidate_categories = [item_category(item_info, item_id) for item_id in candidate_indices]
        same_gt_category_candidates = sum(1 for cat in candidate_categories if cat == true_category)

        for view in VIEWS:
            if int(row.get(f"{view}_hit_at_1_calc", 0)) != 1:
                continue
            if int(row.get("num_views_hit_at_1", 0)) != 1:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "unique_hit_view": view,
                    "bundle_id": int(row["bundle_id"]),
                    "true_indice": true_item,
                    "true_category": true_category,
                    "true_category_name": true_category_name,
                    "input_indices": input_indices,
                    "input_category_names": "; ".join(input_category_names),
                    "candidate_indices": candidate_indices,
                    "same_gt_category_candidates": int(same_gt_category_candidates),
                    "num_input_items": int(len(input_indices)),
                    "num_distinct_input_categories": int(len(set(input_categories))),
                    "rank_IBxBI": int(row["true_rank_IBxBI"]),
                    "rank_IUxUI": int(row["true_rank_IUxUI"]),
                    "rank_BIxIB": int(row["true_rank_BIxIB"]),
                    "input_text": row.get(f"input_str_{view}", ""),
                    "target_text": row.get(f"target_str_{view}", ""),
                }
            )
    detail = pd.DataFrame(rows)

    summary_rows = []
    gt_category_rows = []
    input_gt_rows = []
    if not detail.empty:
        for view, group in detail.groupby("unique_hit_view"):
            gt_counter = Counter(group["true_category_name"])
            input_counter = Counter()
            pair_counter = Counter()
            for _, case in group.iterrows():
                input_names = [name for name in str(case["input_category_names"]).split("; ") if name]
                input_counter.update(input_names)
                for input_name in input_names:
                    pair_counter[(input_name, case["true_category_name"])] += 1
            summary_rows.append(
                {
                    "dataset": dataset,
                    "unique_hit_view": view,
                    "count": int(len(group)),
                    "avg_same_gt_category_candidates": float(group["same_gt_category_candidates"].mean()),
                    "avg_num_input_items": float(group["num_input_items"].mean()),
                    "avg_distinct_input_categories": float(group["num_distinct_input_categories"].mean()),
                    "top_gt_categories": top_items(gt_counter),
                    "top_input_categories": top_items(input_counter),
                }
            )
            for name, count in gt_counter.most_common(20):
                gt_category_rows.append(
                    {
                        "dataset": dataset,
                        "unique_hit_view": view,
                        "true_category_name": name,
                        "count": int(count),
                        "ratio_within_view": float(count / len(group)),
                    }
                )
            for (input_name, gt_name), count in pair_counter.most_common(30):
                input_gt_rows.append(
                    {
                        "dataset": dataset,
                        "unique_hit_view": view,
                        "input_category_name": input_name,
                        "true_category_name": gt_name,
                        "count": int(count),
                    }
                )

    return detail, pd.DataFrame(summary_rows), pd.DataFrame(gt_category_rows), pd.DataFrame(input_gt_rows)


def markdown_table(df):
    if df.empty:
        return "(empty)"
    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            values.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize unique-view Hit@1 case studies by category.")
    parser.add_argument("--datasets", nargs="+", default=["pog", "pog_dense"])
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "analysis" / "unique_view_case_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    details = []
    summaries = []
    gt_categories = []
    input_gt_pairs = []
    for dataset in args.datasets:
        detail, summary, gt_category, input_gt = analyze_dataset(repo_root, dataset)
        details.append(detail)
        summaries.append(summary)
        gt_categories.append(gt_category)
        input_gt_pairs.append(input_gt)

    detail_df = pd.concat(details, ignore_index=True)
    summary_df = pd.concat(summaries, ignore_index=True)
    gt_category_df = pd.concat(gt_categories, ignore_index=True)
    input_gt_pair_df = pd.concat(input_gt_pairs, ignore_index=True)

    detail_df.to_csv(out_dir / "unique_hit_case_detail.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "unique_hit_case_summary.csv", index=False, encoding="utf-8-sig")
    gt_category_df.to_csv(out_dir / "unique_hit_gt_categories.csv", index=False, encoding="utf-8-sig")
    input_gt_pair_df.to_csv(out_dir / "unique_hit_input_gt_pairs.csv", index=False, encoding="utf-8-sig")

    report = [
        "# Unique-View Hit@1 Case Study",
        "",
        "## Summary by Dataset and View",
        markdown_table(summary_df),
        "",
        "## Top GT Categories",
        markdown_table(gt_category_df.groupby(["dataset", "unique_hit_view"]).head(5).reset_index(drop=True)),
        "",
        "## Top Input-to-GT Category Pairs",
        markdown_table(input_gt_pair_df.groupby(["dataset", "unique_hit_view"]).head(5).reset_index(drop=True)),
    ]
    (out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Saved unique-view case study to: {out_dir}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

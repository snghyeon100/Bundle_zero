import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analyze_relation_feature_signal import analyze_dataset


DEFAULT_EXPERIMENTS = {
    "pog": {
        "base": r"results\pog\results_pog_20260416_142034.csv",
        "IBxBI_item_desc": r"results\pog\results_pog_USER_ITEMAFF_HN_C10_T5_20260512_135419.csv",
        "IUxUI_item_desc": r"results\pog\results_pog_USERPUR_HN_C10_T5_20260512_141640.csv",
        "BIxIB_bundle_context": r"results\pog\results_pog_BGRAPH_HN_C10_T5_20260512_164752.csv",
        "co_occur": r"results\pog\results_pog_COOC_HN_C10_T5_20260512_170847.csv",
    },
    "pog_dense": {
        "base": r"results\pog_dense\results_pog_dense_HN_C10_T5_20260430_172343.csv",
        "IBxBI_item_desc": r"results\pog_dense\results_pog_dense_USER_ITEMAFF_HN_C10_T5_20260512_135517.csv",
        "IUxUI_item_desc": r"results\pog_dense\results_pog_dense_USERPUR_HN_C10_T5_20260512_141345.csv",
        "BIxIB_bundle_context": r"results\pog_dense\results_pog_dense_BGRAPH_HN_C10_T5_20260512_164908.csv",
        "co_occur": r"results\pog_dense\results_pog_dense_COOC_HN_C10_T5_20260512_170906.csv",
    },
}


def read_graph_stats(repo_root, data_path, dataset):
    dataset_dir = repo_root / data_path / dataset
    with open(dataset_dir / "count.json", encoding="utf-8") as f:
        count = json.load(f)

    train_bundle_sizes = []
    item_freq = {}
    with open(dataset_dir / "bi_train.txt", encoding="utf-8") as f:
        for line in f:
            vals = [int(v) for v in line.strip().split(", ") if v]
            if len(vals) < 2:
                continue
            items = list(dict.fromkeys(vals[1:]))
            train_bundle_sizes.append(len(items))
            for item_id in items:
                item_freq[item_id] = item_freq.get(item_id, 0) + 1

    freq_values = pd.Series(list(item_freq.values()), dtype="float64")
    size_values = pd.Series(train_bundle_sizes, dtype="float64")
    return {
        "num_items": int(count["#I"]),
        "num_bundles": int(count["#B"]),
        "num_train_bundles": int(len(train_bundle_sizes)),
        "train_bundle_size_mean": float(size_values.mean()),
        "train_bundle_size_median": float(size_values.median()),
        "train_bundle_size_max": int(size_values.max()),
        "train_active_items": int(len(item_freq)),
        "train_item_coverage": float(len(item_freq) / count["#I"]),
        "train_item_freq_mean": float(freq_values.mean()),
        "train_item_freq_median": float(freq_values.median()),
        "train_item_freq_max": int(freq_values.max()),
    }


def summarize_cooccurrence_column(result_csv):
    df = pd.read_csv(result_csv)
    if "cooccurrence_stats" not in df.columns:
        return {}

    shared_values = []
    denom_values = []
    true_shared = []
    true_denom = []
    shared_top1 = []
    ratio_top1 = []
    candidate_positive_rates = []
    true_positive = []
    for _, row in df.iterrows():
        try:
            stats = json.loads(str(row["cooccurrence_stats"]).replace("'", '"'))
        except json.JSONDecodeError:
            try:
                import ast

                stats = ast.literal_eval(str(row["cooccurrence_stats"]))
            except (ValueError, SyntaxError):
                continue
        if not isinstance(stats, list):
            continue
        option_idx = int(row["true_option_idx"]) if "true_option_idx" in row and pd.notna(row["true_option_idx"]) else None
        row_shared = []
        row_denom = []
        for stat in stats:
            shared = float(stat.get("shared_train_bundles", 0))
            denom = float(stat.get("candidate_train_bundles", 0))
            shared_values.append(shared)
            denom_values.append(denom)
            row_shared.append(shared)
            row_denom.append(denom)
        if option_idx is not None and 0 <= option_idx < len(stats):
            true_stat = stats[option_idx]
            true_shared.append(float(true_stat.get("shared_train_bundles", 0)))
            true_denom.append(float(true_stat.get("candidate_train_bundles", 0)))
            shared_arr = np.asarray(row_shared, dtype=np.float64)
            denom_arr = np.asarray(row_denom, dtype=np.float64)
            ratio_arr = np.divide(shared_arr, denom_arr, out=np.zeros_like(shared_arr), where=denom_arr > 0)
            shared_top1.append(float(int(shared_arr.argmax()) == option_idx))
            ratio_top1.append(float(int(ratio_arr.argmax()) == option_idx))
            candidate_positive_rates.append(float((shared_arr > 0).mean()))
            true_positive.append(float(row_shared[option_idx] > 0))

    if not shared_values:
        return {}
    shared = pd.Series(shared_values, dtype="float64")
    denom = pd.Series(denom_values, dtype="float64")
    true_shared_s = pd.Series(true_shared, dtype="float64")
    true_denom_s = pd.Series(true_denom, dtype="float64")
    return {
        "cooc_all_shared_mean": float(shared.mean()),
        "cooc_all_denom_mean": float(denom.mean()),
        "cooc_all_zero_denom_rate": float((denom == 0).mean()),
        "cooc_true_shared_mean": float(true_shared_s.mean()),
        "cooc_true_denom_mean": float(true_denom_s.mean()),
        "cooc_true_zero_denom_rate": float((true_denom_s == 0).mean()),
        "cooc_shared_top1_rate": float(pd.Series(shared_top1, dtype="float64").mean()),
        "cooc_ratio_top1_rate": float(pd.Series(ratio_top1, dtype="float64").mean()),
        "cooc_candidate_positive_rate": float(pd.Series(candidate_positive_rates, dtype="float64").mean()),
        "cooc_true_positive_rate": float(pd.Series(true_positive, dtype="float64").mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Run BI-LightGCN item embedding analysis for POG experiment CSVs.")
    parser.add_argument("--datasets", nargs="+", default=["pog", "pog_dense"], choices=["pog", "pog_dense"])
    parser.add_argument("--feature", default="bi_lgcn", choices=["bi_lgcn"])
    parser.add_argument("--data-path", default="datasets")
    parser.add_argument("--output-root", default=r"analysis\bi_lgcn_experiment_signal")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    analysis_args = SimpleNamespace(data_path=args.data_path, output_root=args.output_root)

    rows = []
    for dataset in args.datasets:
        graph_stats = read_graph_stats(repo_root, args.data_path, dataset)
        for method, rel_path in DEFAULT_EXPERIMENTS[dataset].items():
            result_csv = repo_root / rel_path
            if not result_csv.exists():
                raise FileNotFoundError(result_csv)

            result = analyze_dataset(repo_root, dataset, result_csv, args.feature, analysis_args)
            summary = result["summary"]
            all_row = summary[summary["split"] == "all"].iloc[0].to_dict()
            all_row.update({
                "dataset": dataset,
                "method": method,
                "result_csv": str(result_csv),
                "detail_path": str(result["detail_path"]),
                "summary_path": str(result["summary_path"]),
            })
            all_row.update(graph_stats)
            all_row.update(summarize_cooccurrence_column(result_csv))
            rows.append(all_row)
            print(f"{dataset} / {method}: hit={all_row['llm_hit_rate']:.3f}, "
                  f"GT top1={all_row['input_gt_top1_rate']:.3f}, "
                  f"margin={all_row['input_gt_margin_vs_best_neg_mean']:.3f}")

    combined = pd.DataFrame(rows)
    front_cols = ["dataset", "method", "n", "llm_hit_rate", "input_gt_top1_rate", "input_gt_top3_rate",
                  "input_gt_mrr", "input_gt_rank_mean", "input_gt_margin_vs_best_neg_mean",
                  "pred_is_input_semantic_top1_rate", "pred_input_rank_mean",
                  "pred_to_gt_sim_mean", "pred_to_other_candidate_sim_max_mean",
                  "pred_margin_vs_best_other_candidate_mean"]
    ordered_cols = [c for c in front_cols if c in combined.columns] + [c for c in combined.columns if c not in front_cols]

    out_dir = repo_root / args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "pog_bi_lgcn_experiment_summary.csv"
    combined[ordered_cols].to_csv(combined_path, index=False, quoting=csv.QUOTE_MINIMAL)

    md_path = out_dir / "summary.md"
    all_rows = combined[ordered_cols].copy()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# POG BI-LightGCN Experiment Signal\n\n")
        f.write("| dataset | method | hit | GT top1 | GT top3 | MRR | rank mean | margin vs best neg | pred=input top1 |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in all_rows.iterrows():
            f.write(
                f"| {row['dataset']} | {row['method']} | {row['llm_hit_rate']:.3f} | "
                f"{row['input_gt_top1_rate']:.3f} | {row['input_gt_top3_rate']:.3f} | "
                f"{row['input_gt_mrr']:.3f} | {row['input_gt_rank_mean']:.2f} | "
                f"{row['input_gt_margin_vs_best_neg_mean']:.3f} | "
                f"{row['pred_is_input_semantic_top1_rate']:.3f} |\n"
            )
    print(f"\ncombined: {combined_path}")
    print(f"summary : {md_path}")


if __name__ == "__main__":
    main()

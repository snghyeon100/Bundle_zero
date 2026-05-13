import argparse
import ast
import json
import os
from itertools import combinations

import pandas as pd


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def infer_dataset_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]

    base = os.path.basename(path).lower()
    for dataset in ("spotify_sparse", "spotify", "pog_dense", "pog_dedup", "pog"):
        if dataset in base:
            return dataset
    return None


def parse_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def parse_bool_hit(value):
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().lower()
    if text in ("1", "1.0", "true", "t", "yes"):
        return 1
    if text in ("0", "0.0", "false", "f", "no"):
        return 0
    try:
        return int(float(text))
    except ValueError:
        return pd.NA


def option_to_index(value):
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if len(text) == 1 and text in LETTERS:
        return LETTERS.index(text)
    try:
        idx = int(float(text))
        return idx if idx >= 0 else None
    except ValueError:
        return None


def get_true_index(row, candidate_ids):
    if "true_option_idx" in row and pd.notna(row["true_option_idx"]):
        return int(row["true_option_idx"])

    idx = option_to_index(row.get("true_option_char", ""))
    if idx is not None:
        return idx

    true_id = row.get("true_indice")
    if pd.notna(true_id):
        try:
            return candidate_ids.index(int(true_id))
        except ValueError:
            return None
    return None


def get_prediction_index(row):
    idx = option_to_index(row.get("prediction", ""))
    if idx is not None:
        return idx

    raw = str(row.get("raw_response", "")).strip().upper()
    for char in raw:
        if char in LETTERS:
            return LETTERS.index(char)
    return None


def margin_top_to_second(scores):
    unique_scores = sorted(set(scores), reverse=True)
    if len(unique_scores) <= 1:
        return 0
    return unique_scores[0] - unique_scores[1]


def rank_stats(scores, choice_idx):
    if not scores:
        return {}

    numeric_scores = [float(score) for score in scores]
    max_score = max(numeric_scores)
    min_score = min(numeric_scores)
    top_indices = [idx for idx, score in enumerate(numeric_scores) if score == max_score]
    all_tied = len(set(numeric_scores)) == 1
    all_zero = all(score == 0 for score in numeric_scores)

    base = {
        "valid_choice": 0,
        "score": pd.NA,
        "min_rank": pd.NA,
        "max_rank": pd.NA,
        "avg_rank": pd.NA,
        "in_top_tie": pd.NA,
        "unique_top1": pd.NA,
        "gap_to_top": pd.NA,
        "top_score": max_score,
        "second_margin": margin_top_to_second(numeric_scores),
        "top_tie_size": len(top_indices),
        "all_tied": int(all_tied),
        "all_zero": int(all_zero),
        "positive_signal": int(max_score > 0),
    }

    if choice_idx is None or choice_idx < 0 or choice_idx >= len(numeric_scores):
        return base

    score = numeric_scores[choice_idx]
    greater = sum(other > score for other in numeric_scores)
    equal = sum(other == score for other in numeric_scores)
    min_rank = greater + 1
    max_rank = greater + equal
    avg_rank = (min_rank + max_rank) / 2

    base.update(
        {
            "valid_choice": 1,
            "score": score,
            "min_rank": min_rank,
            "max_rank": max_rank,
            "avg_rank": avg_rank,
            "in_top_tie": int(choice_idx in top_indices),
            "unique_top1": int(len(top_indices) == 1 and choice_idx == top_indices[0]),
            "gap_to_top": max_score - score,
        }
    )
    return base


def prefixed(prefix, stats):
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def load_cf_cache(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cache_key(row):
    if pd.isna(row.get("bundle_id")) or pd.isna(row.get("true_indice")):
        return None
    return f"{int(row['bundle_id'])}_{int(row['true_indice'])}"


def build_detailed_rows(result_paths, names, cf_cache, signals):
    rows = []
    for result_path, method in zip(result_paths, names):
        df = pd.read_csv(result_path)
        for row_idx, row in df.iterrows():
            candidate_ids = [int(v) for v in parse_list(row.get("candidate_indices", []))]
            true_idx = get_true_index(row, candidate_ids)
            pred_idx = get_prediction_index(row)
            key = cache_key(row)
            if key is None:
                continue
            cache_entry = cf_cache.get(key, {})

            for signal in signals:
                scores = cache_entry.get(signal)
                if scores is None:
                    continue
                if len(scores) != len(candidate_ids):
                    score_len_ok = 0
                else:
                    score_len_ok = 1

                gt_stats = rank_stats(scores, true_idx)
                pred_stats = rank_stats(scores, pred_idx)
                record = {
                    "method": method,
                    "signal": signal,
                    "source_csv": result_path,
                    "index": row_idx,
                    "bundle_id": row.get("bundle_id"),
                    "true_indice": row.get("true_indice"),
                    "true_option_idx": true_idx,
                    "true_option_char": row.get("true_option_char"),
                    "prediction": row.get("prediction"),
                    "prediction_idx": pred_idx,
                    "hit": parse_bool_hit(row.get("hit")),
                    "score_len_ok": score_len_ok,
                    "candidate_count": len(candidate_ids),
                    "scores": json.dumps(scores, ensure_ascii=False),
                    "candidate_indices": json.dumps(candidate_ids, ensure_ascii=False),
                }
                record.update(prefixed("gt", gt_stats))
                record.update(prefixed("pred", pred_stats))
                rows.append(record)

    return pd.DataFrame(rows)


def mean(series):
    valid = pd.to_numeric(series, errors="coerce").dropna()
    return valid.mean() if len(valid) else pd.NA


def summarize_signal_quality(detail):
    rows = []
    base = detail.drop_duplicates(["signal", "bundle_id", "true_indice"])
    for signal, group in base.groupby("signal"):
        rows.append(
            {
                "signal": signal,
                "n": len(group),
                "gt_in_top_tie_rate": mean(group["gt_in_top_tie"]),
                "gt_unique_top1_rate": mean(group["gt_unique_top1"]),
                "gt_mean_min_rank": mean(group["gt_min_rank"]),
                "gt_mean_avg_rank": mean(group["gt_avg_rank"]),
                "gt_mean_score": mean(group["gt_score"]),
                "mean_top_score": mean(group["gt_top_score"]),
                "mean_top_tie_size": mean(group["gt_top_tie_size"]),
                "all_zero_rate": mean(group["gt_all_zero"]),
                "all_tied_rate": mean(group["gt_all_tied"]),
                "positive_signal_rate": mean(group["gt_positive_signal"]),
                "mean_top_second_margin": mean(group["gt_second_margin"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_method_reliance(detail):
    rows = []
    for (method, signal), group in detail.groupby(["method", "signal"]):
        valid_pred = group[group["pred_valid_choice"] == 1]
        positive = group[group["gt_positive_signal"] == 1]
        pred_top = valid_pred[valid_pred["pred_in_top_tie"] == 1]
        pred_not_top = valid_pred[valid_pred["pred_in_top_tie"] == 0]
        rows.append(
            {
                "method": method,
                "signal": signal,
                "n": len(group),
                "valid_pred_n": len(valid_pred),
                "accuracy": mean(group["hit"]),
                "pred_in_top_tie_rate": mean(valid_pred["pred_in_top_tie"]),
                "pred_unique_top1_rate": mean(valid_pred["pred_unique_top1"]),
                "pred_mean_min_rank": mean(valid_pred["pred_min_rank"]),
                "pred_mean_avg_rank": mean(valid_pred["pred_avg_rank"]),
                "pred_mean_score": mean(valid_pred["pred_score"]),
                "hit_when_pred_in_top_tie": mean(pred_top["hit"]),
                "hit_when_pred_not_in_top_tie": mean(pred_not_top["hit"]),
                "pred_in_top_tie_rate_when_positive": mean(positive["pred_in_top_tie"]),
                "pred_unique_top1_rate_when_positive": mean(positive["pred_unique_top1"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["signal", "method"])


def summarize_quadrants(detail):
    rows = []
    valid = detail[detail["pred_valid_choice"] == 1].copy()
    valid["gt_top_group"] = valid["gt_in_top_tie"].map({1: "gt_top", 0: "gt_not_top"})
    valid["pred_top_group"] = valid["pred_in_top_tie"].map({1: "pred_top", 0: "pred_not_top"})

    for (method, signal, gt_group, pred_group), group in valid.groupby(
        ["method", "signal", "gt_top_group", "pred_top_group"]
    ):
        rows.append(
            {
                "method": method,
                "signal": signal,
                "gt_cf_group": gt_group,
                "pred_cf_group": pred_group,
                "n": len(group),
                "accuracy": mean(group["hit"]),
                "mean_gt_avg_rank": mean(group["gt_avg_rank"]),
                "mean_pred_avg_rank": mean(group["pred_avg_rank"]),
                "mean_top_tie_size": mean(group["gt_top_tie_size"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["signal", "method", "gt_cf_group", "pred_cf_group"])


def summarize_pairwise(detail, names):
    rows = []
    key_cols = ["signal", "bundle_id", "true_indice"]
    compact = detail[
        key_cols
        + [
            "method",
            "hit",
            "gt_in_top_tie",
            "gt_unique_top1",
            "gt_avg_rank",
            "pred_in_top_tie",
            "pred_unique_top1",
            "pred_avg_rank",
        ]
    ].copy()

    for signal in sorted(compact["signal"].dropna().unique()):
        signal_df = compact[compact["signal"] == signal]
        for left, right in combinations(names, 2):
            a = signal_df[signal_df["method"] == left].set_index(["bundle_id", "true_indice"])
            b = signal_df[signal_df["method"] == right].set_index(["bundle_id", "true_indice"])
            joined = a.join(b, how="inner", lsuffix="_left", rsuffix="_right")
            if joined.empty:
                continue

            left_hit = joined["hit_left"].astype(float)
            right_hit = joined["hit_right"].astype(float)
            right_only = joined[(left_hit == 0) & (right_hit == 1)]
            left_only = joined[(left_hit == 1) & (right_hit == 0)]
            both_hit = joined[(left_hit == 1) & (right_hit == 1)]
            both_fail = joined[(left_hit == 0) & (right_hit == 0)]

            rows.append(
                {
                    "signal": signal,
                    "left_method": left,
                    "right_method": right,
                    "n": len(joined),
                    "left_accuracy": mean(joined["hit_left"]),
                    "right_accuracy": mean(joined["hit_right"]),
                    "left_only": len(left_only),
                    "right_only": len(right_only),
                    "both_hit": len(both_hit),
                    "both_fail": len(both_fail),
                    "right_only_gt_in_top_tie_rate": mean(right_only["gt_in_top_tie_right"]),
                    "right_only_gt_unique_top1_rate": mean(right_only["gt_unique_top1_right"]),
                    "right_only_pred_in_top_tie_rate": mean(right_only["pred_in_top_tie_right"]),
                    "right_only_pred_unique_top1_rate": mean(right_only["pred_unique_top1_right"]),
                    "right_only_mean_gt_avg_rank": mean(right_only["gt_avg_rank_right"]),
                    "right_only_mean_pred_avg_rank": mean(right_only["pred_avg_rank_right"]),
                }
            )
    return pd.DataFrame(rows)


def format_rate(value):
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}%"


def format_num(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def df_to_markdown(df, max_rows=30):
    if df is None or df.empty:
        return "_No data._\n"
    shown = df.head(max_rows).copy()
    for col in shown.columns:
        if "_rate" in col or col.startswith("hit_when") or col.endswith("_accuracy") or col == "accuracy":
            shown[col] = shown[col].map(format_rate)
        elif "mean" in col or col.endswith("_rank"):
            shown[col] = shown[col].map(format_num)

    headers = [str(col) for col in shown.columns]
    rows = []
    for _, row in shown.iterrows():
        rows.append([str(row[col]) if not pd.isna(row[col]) else "" for col in shown.columns])

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in rows:
        escaped = [value.replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    return "\n".join(lines) + "\n"


def write_summary(path, tables):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# CF Signal Analysis\n\n")
        f.write("Tie handling: `min_rank` is optimistic, `avg_rank` is neutral, ")
        f.write("`in_top_tie` accepts any candidate sharing the maximum score, ")
        f.write("and `unique_top1` only counts a single strictly highest candidate.\n\n")

        f.write("## CF Signal Quality\n\n")
        f.write(df_to_markdown(tables["signal_quality"]))
        f.write("\n## LLM Reliance On CF Score\n\n")
        f.write(df_to_markdown(tables["method_reliance"]))
        f.write("\n## GT/Prediction CF Quadrants\n\n")
        f.write(df_to_markdown(tables["quadrants"]))
        f.write("\n## Pairwise Method Delta\n\n")
        f.write(df_to_markdown(tables["pairwise"]))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze whether LLM predictions follow co-occurrence/user-preference CF scores."
    )
    parser.add_argument("--results", nargs="+", required=True, help="One or more result CSV files.")
    parser.add_argument("--names", nargs="+", required=True, help="Method names matching --results.")
    parser.add_argument("--dataset", default=None, help="Dataset name. Inferred from result path if omitted.")
    parser.add_argument("--data_path", default="./datasets", help="Dataset root directory.")
    parser.add_argument("--cf_cache", default=None, help="Optional path to cf_scores_<dataset>.json.")
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["cooccurrence", "user_pref"],
        help="CF signals to analyze.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for output CSV/MD files. Defaults to analysis/<dataset>_cf_signal_analysis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.results) != len(args.names):
        raise ValueError("--results and --names must have the same length.")

    dataset = args.dataset or infer_dataset_from_path(args.results[0])
    if not dataset:
        raise ValueError("Could not infer dataset. Pass --dataset explicitly.")

    cf_cache_path = args.cf_cache or os.path.join(
        args.data_path, dataset, f"cf_scores_{dataset}.json"
    )
    if not os.path.exists(cf_cache_path):
        raise FileNotFoundError(
            f"CF cache not found: {cf_cache_path}."
        )

    output_dir = args.output_dir or os.path.join("analysis", f"{dataset}_cf_signal_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Dataset : {dataset}")
    print(f"CF cache: {cf_cache_path}")
    print(f"Output  : {output_dir}")

    cf_cache = load_cf_cache(cf_cache_path)
    detail = build_detailed_rows(args.results, args.names, cf_cache, args.signals)
    if detail.empty:
        raise ValueError("No analyzable rows were produced. Check --signals and CF cache contents.")

    tables = {
        "signal_quality": summarize_signal_quality(detail),
        "method_reliance": summarize_method_reliance(detail),
        "quadrants": summarize_quadrants(detail),
        "pairwise": summarize_pairwise(detail, args.names),
    }

    detail_path = os.path.join(output_dir, "cf_detailed.csv")
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")

    for name, table in tables.items():
        table.to_csv(os.path.join(output_dir, f"cf_{name}.csv"), index=False, encoding="utf-8-sig")

    summary_path = os.path.join(output_dir, "summary.md")
    write_summary(summary_path, tables)

    print(f"Saved detail : {detail_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

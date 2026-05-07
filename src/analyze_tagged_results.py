import argparse
import ast
import json
import os
from collections import defaultdict

import pandas as pd


def infer_dataset_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    base = os.path.basename(path).lower()
    for dataset in ("spotify_sparse", "spotify", "pog_dense", "pog"):
        if dataset in base:
            return dataset
    return None


def safe_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        return ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(str(value))
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []


def load_result(path, name):
    df = pd.read_csv(path)
    df["index"] = df.index
    keep = [
        "index",
        "bundle_id",
        "true_indice",
        "true_option_char",
        "input_str",
        "target_str",
        "prediction",
        "raw_response",
        "hit",
    ]
    keep = [col for col in keep if col in df.columns]
    df = df[keep].copy()
    df = df.rename(
        columns={
            "prediction": f"prediction_{name}",
            "raw_response": f"raw_response_{name}",
            "hit": f"hit_{name}",
        }
    )
    return df


def merge_inputs(result_paths, names, semantic_path, rule_path):
    merged = None
    for path, name in zip(result_paths, names):
        result_df = load_result(path, name)
        if merged is None:
            merged = result_df
        else:
            overlap = [col for col in result_df.columns if col in merged.columns and col != "index"]
            result_df = result_df.drop(columns=overlap)
            merged = merged.merge(result_df, on="index", how="outer")

    semantic_df = pd.read_csv(semantic_path)
    semantic_keep = [
        "index",
        "primary_tag",
        "secondary_tags",
        "gt_plausibility",
        "distractor_hardness",
        "confidence",
        "evidence",
    ]
    semantic_keep = [col for col in semantic_keep if col in semantic_df.columns]
    merged = merged.merge(semantic_df[semantic_keep], on="index", how="left")

    if rule_path and os.path.exists(rule_path):
        rule_df = pd.read_csv(rule_path)
        rule_keep = [
            "index",
            "primary_rule_tag",
            "rule_tags",
            "true_popularity_rank",
            "popularity_top_is_gt",
            "true_cooccurrence_rank",
            "cooccurrence_top_is_gt",
            "true_category_overlap",
        ]
        rule_keep = [col for col in rule_keep if col in rule_df.columns]
        merged = merged.merge(rule_df[rule_keep], on="index", how="left")

    return merged


def summarize_by_column(df, group_col, hit_cols):
    rows = []
    grouped = df[df[group_col].notna()].groupby(group_col)
    for group_value, group in grouped:
        row = {
            group_col: group_value,
            "n": len(group),
        }
        for hit_col in hit_cols:
            valid = group[group[hit_col].notna()]
            row[f"{hit_col}_n"] = len(valid)
            row[f"{hit_col}_acc"] = valid[hit_col].astype(float).mean() if len(valid) else pd.NA
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def summarize_multilabel(df, tag_col, secondary_col, hit_cols):
    tag_to_indices = defaultdict(set)

    for idx, row in df.iterrows():
        primary = row.get(tag_col)
        if pd.notna(primary):
            tag_to_indices[str(primary)].add(idx)

        for tag in safe_list(row.get(secondary_col)):
            tag_to_indices[str(tag)].add(idx)

    rows = []
    for tag, indices in tag_to_indices.items():
        subset = df.loc[sorted(indices)]
        row = {"tag_in_primary_or_secondary": tag, "n": len(subset)}
        for hit_col in hit_cols:
            valid = subset[subset[hit_col].notna()]
            row[f"{hit_col}_n"] = len(valid)
            row[f"{hit_col}_acc"] = valid[hit_col].astype(float).mean() if len(valid) else pd.NA
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n", ascending=False)


def make_pairwise_comparison(df, names):
    if len(names) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]
            hit_a = f"hit_{a}"
            hit_b = f"hit_{b}"
            if hit_a not in df.columns or hit_b not in df.columns:
                continue
            valid = df[df[hit_a].notna() & df[hit_b].notna()].copy()
            if valid.empty:
                continue
            rows.append(
                {
                    "comparison": f"{a}_vs_{b}",
                    "n": len(valid),
                    f"{a}_only": int(((valid[hit_a] == 1) & (valid[hit_b] == 0)).sum()),
                    f"{b}_only": int(((valid[hit_a] == 0) & (valid[hit_b] == 1)).sum()),
                    "both_hit": int(((valid[hit_a] == 1) & (valid[hit_b] == 1)).sum()),
                    "both_fail": int(((valid[hit_a] == 0) & (valid[hit_b] == 0)).sum()),
                    f"{a}_acc": valid[hit_a].astype(float).mean(),
                    f"{b}_acc": valid[hit_b].astype(float).mean(),
                }
            )
    return pd.DataFrame(rows)


def pick_examples(df, hit_cols, max_examples):
    rows = []
    for tag, group in df[df["primary_tag"].notna()].groupby("primary_tag"):
        for hit_col in hit_cols:
            if hit_col not in group.columns:
                continue
            fail = group[group[hit_col] == 0].sort_values(
                by=["distractor_hardness", "gt_plausibility"],
                ascending=[False, True],
                na_position="last",
            )
            success = group[group[hit_col] == 1].sort_values(
                by=["confidence", "gt_plausibility"],
                ascending=[False, False],
                na_position="last",
            )
            for label, subset in [("fail", fail), ("success", success)]:
                for _, row in subset.head(max_examples).iterrows():
                    rows.append(
                        {
                            "primary_tag": tag,
                            "result": hit_col.replace("hit_", ""),
                            "case_type": label,
                            "index": row.get("index"),
                            "bundle_id": row.get("bundle_id"),
                            "true_option_char": row.get("true_option_char"),
                            "prediction": row.get(f"prediction_{hit_col.replace('hit_', '')}", ""),
                            "gt_plausibility": row.get("gt_plausibility"),
                            "distractor_hardness": row.get("distractor_hardness"),
                            "evidence": row.get("evidence"),
                            "input_str": row.get("input_str"),
                            "target_str": row.get("target_str"),
                        }
                    )
    return pd.DataFrame(rows)


def write_markdown_report(out_path, result_names, tables):
    def df_to_md(df, max_rows=30):
        if df is None or df.empty:
            return "_No data._\n"
        shown = df.head(max_rows).copy()
        for col in shown.columns:
            if col.endswith("_acc"):
                shown[col] = shown[col].map(lambda x: "" if pd.isna(x) else f"{x * 100:.1f}%")
        shown = shown.fillna("")
        headers = [str(col) for col in shown.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in shown.iterrows():
            values = [str(row[col]).replace("\n", " ") for col in shown.columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines) + "\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Tagged Result Analysis\n\n")
        f.write(f"Result columns: {', '.join(result_names)}\n\n")
        f.write("## Accuracy By Primary Semantic Tag\n\n")
        f.write(df_to_md(tables["primary_semantic"]))
        f.write("\n## Accuracy By Semantic Tag In Primary Or Secondary\n\n")
        f.write(df_to_md(tables["multi_semantic"]))
        f.write("\n## Accuracy By Distractor Hardness\n\n")
        f.write(df_to_md(tables["hardness"]))
        f.write("\n## Accuracy By GT Plausibility\n\n")
        f.write(df_to_md(tables["plausibility"]))
        if "rule" in tables:
            f.write("\n## Accuracy By Primary Rule Tag\n\n")
            f.write(df_to_md(tables["rule"]))
        if "semantic_rule_cross" in tables:
            f.write("\n## Semantic x Rule Cross-Tab\n\n")
            f.write(df_to_md(tables["semantic_rule_cross"], max_rows=50))
        if "pairwise" in tables:
            f.write("\n## Pairwise Result Comparison\n\n")
            f.write(df_to_md(tables["pairwise"]))


def main():
    parser = argparse.ArgumentParser(description="Analyze result CSVs by reusable problem semantic/rule tags.")
    parser.add_argument("--results", nargs="+", required=True, help="One or more result CSV paths.")
    parser.add_argument("--names", nargs="+", default=None, help="Names for result CSVs. Defaults to result1/result2...")
    parser.add_argument("--semantic", required=True, help="problem_fashion_semantic_tags*.csv path.")
    parser.add_argument("--rule", default=None, help="Optional problem_tag_meta.csv path.")
    parser.add_argument("--output_dir", default=None, help="Output analysis directory.")
    parser.add_argument("--example_count", type=int, default=2, help="Examples per tag/result success/fail.")
    args = parser.parse_args()

    if args.names and len(args.names) != len(args.results):
        raise ValueError("--names must have the same length as --results.")

    names = args.names or [f"result{i + 1}" for i in range(len(args.results))]
    dataset = infer_dataset_from_path(args.semantic) or infer_dataset_from_path(args.results[0]) or "dataset"
    output_dir = args.output_dir or os.path.join("analysis", f"{dataset}_tagged_result_analysis")
    os.makedirs(output_dir, exist_ok=True)

    merged = merge_inputs(args.results, names, args.semantic, args.rule)
    hit_cols = [f"hit_{name}" for name in names if f"hit_{name}" in merged.columns]

    tables = {}
    tables["primary_semantic"] = summarize_by_column(merged, "primary_tag", hit_cols)
    tables["multi_semantic"] = summarize_multilabel(merged, "primary_tag", "secondary_tags", hit_cols)
    tables["hardness"] = summarize_by_column(merged, "distractor_hardness", hit_cols)
    tables["plausibility"] = summarize_by_column(merged, "gt_plausibility", hit_cols)

    if "primary_rule_tag" in merged.columns:
        tables["rule"] = summarize_by_column(merged, "primary_rule_tag", hit_cols)
        cross = summarize_by_column(
            merged.assign(
                semantic_rule=merged["primary_tag"].astype(str) + " / " + merged["primary_rule_tag"].astype(str)
            ),
            "semantic_rule",
            hit_cols,
        )
        tables["semantic_rule_cross"] = cross

    pairwise = make_pairwise_comparison(merged, names)
    if not pairwise.empty:
        tables["pairwise"] = pairwise

    examples = pick_examples(merged, hit_cols, args.example_count)

    merged.to_csv(os.path.join(output_dir, "tagged_results_merged.csv"), index=False, encoding="utf-8-sig")
    for name, table in tables.items():
        table.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False, encoding="utf-8-sig")
    examples.to_csv(os.path.join(output_dir, "examples_by_tag.csv"), index=False, encoding="utf-8-sig")
    write_markdown_report(os.path.join(output_dir, "summary.md"), names, tables)

    print(f">>> Saved analysis to: {output_dir}")
    print("\nAccuracy by primary semantic tag:")
    display = tables["primary_semantic"].copy()
    for col in display.columns:
        if col.endswith("_acc"):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x * 100:.1f}%")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()

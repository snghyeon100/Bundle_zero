import argparse
import ast
import json
import os
from itertools import combinations

import pandas as pd


def safe_list(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(str(value))
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []


def parse_hit(value):
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


def load_result(path, name):
    df = pd.read_csv(path)
    df["index"] = df.index
    keep = [
        "index",
        "bundle_id",
        "true_indice",
        "true_option_idx",
        "true_option_char",
        "input_indices",
        "candidate_indices",
        "input_str",
        "target_str",
        "prediction",
        "raw_response",
        "hit",
    ]
    keep = [col for col in keep if col in df.columns]
    df = df[keep].copy()
    df[f"hit_{name}"] = df["hit"].map(parse_hit)
    rename = {
        "prediction": f"prediction_{name}",
        "raw_response": f"raw_response_{name}",
        "hit": f"hit_raw_{name}",
    }
    df = df.rename(columns=rename)
    return df


def merge_results(paths, names):
    merged = None
    for path, name in zip(paths, names):
        part = load_result(path, name)
        if merged is None:
            merged = part
            continue
        overlap = [col for col in part.columns if col in merged.columns and col != "index"]
        part = part.drop(columns=overlap)
        merged = merged.merge(part, on="index", how="outer")
    return merged


def add_joint_outcome(df, names):
    hit_cols = [f"hit_{name}" for name in names]
    valid = df[hit_cols].notna().all(axis=1)
    hit_sum = df[hit_cols].fillna(0).astype(int).sum(axis=1)

    df = df.copy()
    df["valid_all_methods"] = valid.astype(int)
    df["hit_count"] = hit_sum
    df["method_count"] = len(names)
    df["joint_outcome"] = "invalid"
    df.loc[valid & (hit_sum == len(names)), "joint_outcome"] = "all_hit"
    df.loc[valid & (hit_sum == 0), "joint_outcome"] = "all_fail"
    df.loc[valid & (hit_sum > 0) & (hit_sum < len(names)), "joint_outcome"] = "mixed"
    return df


def add_optional_meta(df, semantic_path=None, rule_path=None, cf_detail_path=None):
    merged = df
    if semantic_path and os.path.exists(semantic_path):
        semantic = pd.read_csv(semantic_path)
        keep = [
            "index",
            "primary_tag",
            "secondary_tags",
            "gt_plausibility",
            "distractor_hardness",
            "confidence",
            "evidence",
        ]
        keep = [col for col in keep if col in semantic.columns]
        merged = merged.merge(semantic[keep], on="index", how="left")

    if rule_path and os.path.exists(rule_path):
        rule = pd.read_csv(rule_path)
        keep = [
            "index",
            "primary_rule_tag",
            "rule_tags",
            "true_popularity_rank",
            "popularity_top_is_gt",
            "true_cooccurrence_rank",
            "cooccurrence_top_is_gt",
            "true_category_overlap",
        ]
        keep = [col for col in keep if col in rule.columns]
        merged = merged.merge(rule[keep], on="index", how="left")

    if cf_detail_path and os.path.exists(cf_detail_path):
        cf = pd.read_csv(cf_detail_path)
        keep = [
            "index",
            "signal",
            "gt_in_top_tie",
            "gt_unique_top1",
            "gt_avg_rank",
            "gt_top_tie_size",
            "gt_all_zero",
            "pred_in_top_tie",
            "pred_unique_top1",
            "pred_avg_rank",
        ]
        keep = [col for col in keep if col in cf.columns]
        cf = cf[keep].drop_duplicates(["index", "signal"])
        cf_wide = cf.pivot(index="index", columns="signal")
        cf_wide.columns = [f"{signal}_{col}" for col, signal in cf_wide.columns]
        cf_wide = cf_wide.reset_index()
        merged = merged.merge(cf_wide, on="index", how="left")

    return merged


def summarize_outcomes(df, names):
    rows = []
    valid = df[df["valid_all_methods"] == 1]
    for outcome, group in valid.groupby("joint_outcome"):
        row = {"joint_outcome": outcome, "n": len(group), "rate": len(group) / len(valid) if len(valid) else pd.NA}
        for name in names:
            row[f"{name}_acc"] = group[f"hit_{name}"].astype(float).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def summarize_pairwise(df, names):
    rows = []
    for left, right in combinations(names, 2):
        left_col = f"hit_{left}"
        right_col = f"hit_{right}"
        valid = df[df[left_col].notna() & df[right_col].notna()].copy()
        if valid.empty:
            continue
        rows.append(
            {
                "left_method": left,
                "right_method": right,
                "n": len(valid),
                "left_acc": valid[left_col].astype(float).mean(),
                "right_acc": valid[right_col].astype(float).mean(),
                "both_hit": int(((valid[left_col] == 1) & (valid[right_col] == 1)).sum()),
                "both_fail": int(((valid[left_col] == 0) & (valid[right_col] == 0)).sum()),
                "left_only": int(((valid[left_col] == 1) & (valid[right_col] == 0)).sum()),
                "right_only": int(((valid[left_col] == 0) & (valid[right_col] == 1)).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_group_column(df, group_col):
    if group_col not in df.columns:
        return pd.DataFrame()
    valid = df[(df["valid_all_methods"] == 1) & df[group_col].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    rows = []
    for value, group in valid.groupby(group_col):
        row = {
            group_col: value,
            "n": len(group),
            "all_hit": int((group["joint_outcome"] == "all_hit").sum()),
            "all_hit_rate": (group["joint_outcome"] == "all_hit").mean(),
            "all_fail": int((group["joint_outcome"] == "all_fail").sum()),
            "all_fail_rate": (group["joint_outcome"] == "all_fail").mean(),
            "mixed": int((group["joint_outcome"] == "mixed").sum()),
            "mixed_rate": (group["joint_outcome"] == "mixed").mean(),
            "mean_hit_count": group["hit_count"].mean(),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["all_fail_rate", "n"], ascending=[False, False])


def summarize_multilabel_tags(df):
    if "primary_tag" not in df.columns and "secondary_tags" not in df.columns:
        return pd.DataFrame()

    tag_to_indices = {}
    valid = df[df["valid_all_methods"] == 1]
    for idx, row in valid.iterrows():
        tags = set()
        if pd.notna(row.get("primary_tag")):
            tags.add(str(row.get("primary_tag")))
        for tag in safe_list(row.get("secondary_tags")):
            tags.add(str(tag))
        for tag in tags:
            tag_to_indices.setdefault(tag, set()).add(idx)

    rows = []
    for tag, indices in tag_to_indices.items():
        group = df.loc[sorted(indices)]
        rows.append(
            {
                "tag_primary_or_secondary": tag,
                "n": len(group),
                "all_hit": int((group["joint_outcome"] == "all_hit").sum()),
                "all_hit_rate": (group["joint_outcome"] == "all_hit").mean(),
                "all_fail": int((group["joint_outcome"] == "all_fail").sum()),
                "all_fail_rate": (group["joint_outcome"] == "all_fail").mean(),
                "mixed": int((group["joint_outcome"] == "mixed").sum()),
                "mixed_rate": (group["joint_outcome"] == "mixed").mean(),
                "mean_hit_count": group["hit_count"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["all_fail_rate", "n"], ascending=[False, False])


def pick_examples(df, names, max_per_outcome=12):
    rows = []
    valid = df[df["valid_all_methods"] == 1].copy()
    sort_cols = [col for col in ["distractor_hardness", "gt_plausibility", "confidence"] if col in valid.columns]
    ascending = [False, True, False][: len(sort_cols)]

    for outcome in ["all_hit", "all_fail", "mixed"]:
        subset = valid[valid["joint_outcome"] == outcome].copy()
        if sort_cols:
            subset = subset.sort_values(sort_cols, ascending=ascending, na_position="last")
        for _, row in subset.head(max_per_outcome).iterrows():
            record = {
                "joint_outcome": outcome,
                "index": row.get("index"),
                "bundle_id": row.get("bundle_id"),
                "true_option_char": row.get("true_option_char"),
                "hit_count": row.get("hit_count"),
                "input_str": row.get("input_str"),
                "target_str": row.get("target_str"),
            }
            for col in [
                "primary_tag",
                "secondary_tags",
                "gt_plausibility",
                "distractor_hardness",
                "evidence",
                "primary_rule_tag",
            ]:
                if col in row:
                    record[col] = row.get(col)
            for name in names:
                record[f"prediction_{name}"] = row.get(f"prediction_{name}")
                record[f"hit_{name}"] = row.get(f"hit_{name}")
            rows.append(record)
    return pd.DataFrame(rows)


def format_pct(value):
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.1f}%"


def table_to_md(df, max_rows=40):
    if df is None or df.empty:
        return "_No data._\n"
    shown = df.head(max_rows).copy()
    for col in shown.columns:
        if col.endswith("_rate") or col.endswith("_acc") or col in ("rate", "left_acc", "right_acc"):
            shown[col] = shown[col].map(format_pct)
    shown = shown.fillna("")
    headers = [str(col) for col in shown.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in shown.iterrows():
        values = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in shown.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_summary(path, names, tables):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Joint Success/Failure Analysis\n\n")
        f.write(f"Methods: {', '.join(names)}\n\n")
        f.write("## Joint Outcome Counts\n\n")
        f.write(table_to_md(tables["joint_outcomes"]))
        f.write("\n## Pairwise Outcomes\n\n")
        f.write(table_to_md(tables["pairwise"]))
        for key, title in [
            ("primary_tag", "By Primary Semantic Tag"),
            ("semantic_multilabel", "By Semantic Tag In Primary Or Secondary"),
            ("distractor_hardness", "By Distractor Hardness"),
            ("gt_plausibility", "By GT Plausibility"),
            ("primary_rule_tag", "By Primary Rule Tag"),
        ]:
            if key in tables and not tables[key].empty:
                f.write(f"\n## {title}\n\n")
                f.write(table_to_md(tables[key]))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze all-hit/all-fail patterns across result CSVs.")
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--semantic", default="", help="Optional semantic tag CSV.")
    parser.add_argument("--rule", default="", help="Optional rule tag CSV.")
    parser.add_argument("--cf_detail", default="", help="Optional cf_detailed.csv.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_examples", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.results) != len(args.names):
        raise ValueError("--results and --names must have the same length.")

    os.makedirs(args.output_dir, exist_ok=True)
    merged = merge_results(args.results, args.names)
    merged = add_joint_outcome(merged, args.names)
    merged = add_optional_meta(
        merged,
        semantic_path=args.semantic or None,
        rule_path=args.rule or None,
        cf_detail_path=args.cf_detail or None,
    )

    tables = {
        "joint_outcomes": summarize_outcomes(merged, args.names),
        "pairwise": summarize_pairwise(merged, args.names),
        "primary_tag": summarize_group_column(merged, "primary_tag"),
        "semantic_multilabel": summarize_multilabel_tags(merged),
        "distractor_hardness": summarize_group_column(merged, "distractor_hardness"),
        "gt_plausibility": summarize_group_column(merged, "gt_plausibility"),
        "primary_rule_tag": summarize_group_column(merged, "primary_rule_tag"),
    }
    examples = pick_examples(merged, args.names, max_per_outcome=args.max_examples)

    merged.to_csv(os.path.join(args.output_dir, "joint_results_merged.csv"), index=False, encoding="utf-8-sig")
    examples.to_csv(os.path.join(args.output_dir, "joint_examples.csv"), index=False, encoding="utf-8-sig")
    for name, table in tables.items():
        table.to_csv(os.path.join(args.output_dir, f"{name}.csv"), index=False, encoding="utf-8-sig")
    write_summary(os.path.join(args.output_dir, "summary.md"), args.names, tables)

    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

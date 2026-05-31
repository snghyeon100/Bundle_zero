import argparse
import itertools
import math
import re
from pathlib import Path

import pandas as pd


VIEW_PATTERNS = [
    ("IBxBI", re.compile(r"ITEMAFF", re.IGNORECASE)),
    ("IUxUI", re.compile(r"USERPUR", re.IGNORECASE)),
    ("BIxIB", re.compile(r"BGRAPH", re.IGNORECASE)),
]


def infer_view_name(path):
    name = Path(path).name
    for view, pattern in VIEW_PATTERNS:
        if pattern.search(name):
            return view
    stem = Path(path).stem
    return re.sub(r"^results_[^_]+_RANK_", "", stem)


def read_csv_fallback(path):
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def as_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def ndcg_from_rank(rank, k):
    if pd.isna(rank) or rank > k:
        return 0.0
    return 1.0 / math.log2(float(rank) + 1.0)


def metrics_from_rank(rank_series, num_candidates):
    rank = as_numeric(rank_series)
    return {
        "n": int(len(rank)),
        "hit_at_1": float((rank <= 1).mean()),
        "hit_at_3": float((rank <= 3).mean()),
        "hit_at_5": float((rank <= 5).mean()),
        "mrr": float((1.0 / rank).fillna(0.0).mean()),
        "ndcg_at_3": float(rank.apply(lambda x: ndcg_from_rank(x, 3)).mean()),
        "ndcg_at_5": float(rank.apply(lambda x: ndcg_from_rank(x, 5)).mean()),
        "ndcg_at_10": float(rank.apply(lambda x: ndcg_from_rank(x, min(10, num_candidates))).mean()),
        "mean_rank": float(rank.mean()),
        "median_rank": float(rank.median()),
        "valid_rank_ratio": float(rank.notna().mean()),
    }


def find_latest_rank_files(results_dir, dataset):
    results_dir = Path(results_dir)
    files = sorted(results_dir.glob(f"results_{dataset}_RANK_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    selected = {}
    for path in files:
        view = infer_view_name(path)
        if view in {"IBxBI", "IUxUI", "BIxIB"} and view not in selected:
            selected[view] = path
    missing = [view for view in ("IBxBI", "IUxUI", "BIxIB") if view not in selected]
    if missing:
        raise FileNotFoundError(f"Missing RANK result files for views: {missing} in {results_dir}")
    return [selected["IBxBI"], selected["IUxUI"], selected["BIxIB"]]


def load_view_result(path, view_name):
    df = read_csv_fallback(path)
    required = {"bundle_id", "true_indice", "true_option_char", "candidate_indices", "true_rank"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    keep = [
        "bundle_id",
        "true_indice",
        "true_option_char",
        "candidate_indices",
        "input_indices",
        "input_str",
        "target_str",
        "prediction",
        "ranking",
        "ranking_valid",
        "true_rank",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "mrr",
        "ndcg_at_10",
        "raw_response",
    ]
    keep = [col for col in keep if col in df.columns]
    out = df[keep].copy()
    out["sample_key"] = (
        out["bundle_id"].astype(str)
        + "::"
        + out["true_indice"].astype(str)
        + "::"
        + out["true_option_char"].astype(str)
        + "::"
        + out["candidate_indices"].astype(str)
    )
    key_cols = {"sample_key", "bundle_id", "true_indice", "true_option_char", "candidate_indices"}
    rename = {col: f"{col}_{view_name}" for col in out.columns if col not in key_cols}
    out = out.rename(columns=rename)
    out[f"true_rank_{view_name}"] = as_numeric(out[f"true_rank_{view_name}"])
    return out


def merge_view_results(paths):
    frames = []
    view_paths = {}
    for path in paths:
        view = infer_view_name(path)
        if view in view_paths:
            raise ValueError(f"Duplicate view name {view}: {view_paths[view]} and {path}")
        view_paths[view] = Path(path)
        frames.append(load_view_result(path, view))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame,
            on=["sample_key", "bundle_id", "true_indice", "true_option_char", "candidate_indices"],
            how="inner",
            validate="one_to_one",
        )
    return merged, view_paths


def build_per_sample(merged, views, num_candidates, best_fixed_view):
    per = merged.copy()
    rank_cols = [f"true_rank_{view}" for view in views]
    per["oracle_true_rank"] = per[rank_cols].min(axis=1)
    per["worst_true_rank"] = per[rank_cols].max(axis=1)
    per["rank_gap"] = per["worst_true_rank"] - per["oracle_true_rank"]

    def winners(row):
        best = row["oracle_true_rank"]
        return [view for view in views if row[f"true_rank_{view}"] == best]

    winner_lists = per.apply(winners, axis=1)
    per["oracle_winners"] = winner_lists.apply(lambda xs: "|".join(xs))
    per["num_oracle_winners"] = winner_lists.apply(len)
    per["unique_oracle_winner"] = winner_lists.apply(lambda xs: xs[0] if len(xs) == 1 else "Tie")

    per["best_fixed_true_rank"] = per[f"true_rank_{best_fixed_view}"]
    per["best_fixed_regret"] = per["best_fixed_true_rank"] - per["oracle_true_rank"]
    per["oracle_hit_at_1"] = (per["oracle_true_rank"] <= 1).astype(int)
    per["oracle_hit_at_3"] = (per["oracle_true_rank"] <= 3).astype(int)
    per["oracle_hit_at_5"] = (per["oracle_true_rank"] <= 5).astype(int)
    per["oracle_mrr"] = (1.0 / per["oracle_true_rank"]).fillna(0.0)
    per["oracle_ndcg_at_10"] = per["oracle_true_rank"].apply(lambda x: ndcg_from_rank(x, min(10, num_candidates)))

    for k in (1, 3, 5):
        hit_cols = []
        for view in views:
            col = f"{view}_hit_at_{k}_calc"
            per[col] = (per[f"true_rank_{view}"] <= k).astype(int)
            hit_cols.append(col)
        per[f"num_views_hit_at_{k}"] = per[hit_cols].sum(axis=1)
        per[f"unique_success_at_{k}"] = per[f"num_views_hit_at_{k}"].eq(1).astype(int)

    return per


def build_view_summary(merged, views, num_candidates):
    rows = []
    for view in views:
        row = {"method": view, "type": "single_view"}
        row.update(metrics_from_rank(merged[f"true_rank_{view}"], num_candidates))
        rows.append(row)
    return pd.DataFrame(rows)


def build_oracle_summary(per_sample, num_candidates):
    row = {"method": "OracleSelector", "type": "oracle"}
    row.update(metrics_from_rank(per_sample["oracle_true_rank"], num_candidates))
    return pd.DataFrame([row])


def build_winner_distribution(per_sample):
    total = len(per_sample)
    unique = (
        per_sample["unique_oracle_winner"]
        .value_counts(dropna=False)
        .rename_axis("winner")
        .reset_index(name="count")
    )
    unique["ratio"] = unique["count"] / total

    all_winners = (
        per_sample["oracle_winners"]
        .value_counts(dropna=False)
        .rename_axis("winner_set")
        .reset_index(name="count")
    )
    all_winners["ratio"] = all_winners["count"] / total
    return unique, all_winners


def build_best_tie_distribution(per_sample):
    total = len(per_sample)

    cardinality = (
        per_sample["num_oracle_winners"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("num_best_views")
        .reset_index(name="count")
    )
    labels = {
        1: "unique_best",
        2: "two_views_tied_best",
        3: "all_three_tied_best",
    }
    cardinality["tie_type"] = cardinality["num_best_views"].map(labels).fillna("other")
    cardinality["ratio"] = cardinality["count"] / total
    cardinality = cardinality[["tie_type", "num_best_views", "count", "ratio"]]

    exact_sets = (
        per_sample["oracle_winners"]
        .value_counts(dropna=False)
        .rename_axis("best_view_set")
        .reset_index(name="count")
    )
    exact_sets["num_best_views"] = exact_sets["best_view_set"].astype(str).str.count(r"\|") + 1
    exact_sets["ratio"] = exact_sets["count"] / total
    exact_sets = exact_sets[["best_view_set", "num_best_views", "count", "ratio"]]

    return cardinality, exact_sets


def build_success_overlap(per_sample, views):
    rows = []
    for k in (1, 3, 5):
        patterns = []
        for _, row in per_sample.iterrows():
            hits = [view for view in views if row[f"{view}_hit_at_{k}_calc"] == 1]
            patterns.append("|".join(hits) if hits else "None")
        counts = pd.Series(patterns).value_counts().sort_index()
        for pattern, count in counts.items():
            rows.append({"k": k, "success_pattern": pattern, "count": int(count), "ratio": float(count / len(per_sample))})
    return pd.DataFrame(rows)


def build_success_cardinality(per_sample, views):
    rows = []
    label_by_count = {
        0: "all_three_failed",
        1: "one_view_success",
        2: "two_views_success",
        3: "all_three_success",
    }
    for k in (1, 3, 5):
        counts = per_sample[f"num_views_hit_at_{k}"].value_counts().sort_index()
        for num_success, count in counts.items():
            rows.append(
                {
                    "k": k,
                    "success_type": label_by_count.get(int(num_success), "other"),
                    "num_successful_views": int(num_success),
                    "count": int(count),
                    "ratio": float(count / len(per_sample)),
                }
            )
    return pd.DataFrame(rows)


def build_hit1_margin_by_view(per_sample, views):
    rows = []
    for view in views:
        subset = per_sample[per_sample[f"true_rank_{view}"] == 1].copy()
        if subset.empty:
            rows.append(
                {
                    "view": view,
                    "hit_at_1_count": 0,
                    "focal_only_hit_at_1_count": 0,
                    "focal_only_hit_at_1_ratio": 0.0,
                    "shared_with_one_other_count": 0,
                    "shared_with_two_others_count": 0,
                    "best_other_rank_mean": "",
                    "best_other_rank_median": "",
                    "best_other_margin_mean": "",
                    "best_other_margin_median": "",
                    "best_other_rank_ge_3_count": 0,
                    "best_other_rank_ge_3_ratio": 0.0,
                    "best_other_rank_ge_5_count": 0,
                    "best_other_rank_ge_5_ratio": 0.0,
                    "worst_other_rank_mean": "",
                    "worst_other_rank_median": "",
                }
            )
            continue

        other_rank_cols = [f"true_rank_{other}" for other in views if other != view]
        best_other_rank = subset[other_rank_cols].min(axis=1)
        worst_other_rank = subset[other_rank_cols].max(axis=1)
        best_other_margin = best_other_rank - 1
        hit_view_count = len(subset)

        rows.append(
            {
                "view": view,
                "hit_at_1_count": int(hit_view_count),
                "focal_only_hit_at_1_count": int((subset["num_views_hit_at_1"] == 1).sum()),
                "focal_only_hit_at_1_ratio": float((subset["num_views_hit_at_1"] == 1).mean()),
                "shared_with_one_other_count": int((subset["num_views_hit_at_1"] == 2).sum()),
                "shared_with_two_others_count": int((subset["num_views_hit_at_1"] == 3).sum()),
                "best_other_rank_mean": float(best_other_rank.mean()),
                "best_other_rank_median": float(best_other_rank.median()),
                "best_other_margin_mean": float(best_other_margin.mean()),
                "best_other_margin_median": float(best_other_margin.median()),
                "best_other_rank_ge_3_count": int((best_other_rank >= 3).sum()),
                "best_other_rank_ge_3_ratio": float((best_other_rank >= 3).mean()),
                "best_other_rank_ge_5_count": int((best_other_rank >= 5).sum()),
                "best_other_rank_ge_5_ratio": float((best_other_rank >= 5).mean()),
                "worst_other_rank_mean": float(worst_other_rank.mean()),
                "worst_other_rank_median": float(worst_other_rank.median()),
            }
        )
    return pd.DataFrame(rows)


def build_hit1_large_gap_examples(per_sample, views, limit_per_view):
    examples = []
    for view in views:
        other_rank_cols = [f"true_rank_{other}" for other in views if other != view]
        subset = per_sample[per_sample[f"true_rank_{view}"] == 1].copy()
        if subset.empty:
            continue
        subset["best_other_rank"] = subset[other_rank_cols].min(axis=1)
        subset["worst_other_rank"] = subset[other_rank_cols].max(axis=1)
        subset["best_other_margin"] = subset["best_other_rank"] - 1
        subset = subset.sort_values(["best_other_margin", "worst_other_rank"], ascending=False).head(limit_per_view)
        for _, row in subset.iterrows():
            item = {
                "hit1_view": view,
                "bundle_id": row["bundle_id"],
                "true_indice": row["true_indice"],
                "true_option_char": row["true_option_char"],
                "best_other_rank": row["best_other_rank"],
                "best_other_margin": row["best_other_margin"],
                "worst_other_rank": row["worst_other_rank"],
                "candidate_indices": row.get("candidate_indices", ""),
            }
            for v in views:
                item[f"true_rank_{v}"] = row[f"true_rank_{v}"]
                item[f"ranking_{v}"] = row.get(f"ranking_{v}", "")
                item[f"prediction_{v}"] = row.get(f"prediction_{v}", "")
            examples.append(item)
    return pd.DataFrame(examples)


def build_hit1_conditional_other_ranks(per_sample, views):
    unique_rows = []
    pair_rows = []

    for view in views:
        subset = per_sample[
            (per_sample[f"true_rank_{view}"] == 1)
            & (per_sample["num_views_hit_at_1"] == 1)
        ].copy()
        row = {
            "unique_hit_view": view,
            "count": int(len(subset)),
        }
        for other in views:
            if other == view:
                continue
            ranks = pd.to_numeric(subset[f"true_rank_{other}"], errors="coerce")
            row[f"avg_rank_{other}"] = float(ranks.mean()) if len(ranks) else ""
            row[f"median_rank_{other}"] = float(ranks.median()) if len(ranks) else ""
        unique_rows.append(row)

    for first, second in itertools.combinations(views, 2):
        remaining = [view for view in views if view not in {first, second}][0]
        subset = per_sample[
            (per_sample[f"true_rank_{first}"] == 1)
            & (per_sample[f"true_rank_{second}"] == 1)
            & (per_sample["num_views_hit_at_1"] == 2)
        ].copy()
        ranks = pd.to_numeric(subset[f"true_rank_{remaining}"], errors="coerce")
        pair_rows.append(
            {
                "hit_view_pair": f"{first}|{second}",
                "remaining_view": remaining,
                "count": int(len(subset)),
                "avg_rank_remaining_view": float(ranks.mean()) if len(ranks) else "",
                "median_rank_remaining_view": float(ranks.median()) if len(ranks) else "",
            }
        )

    return pd.DataFrame(unique_rows), pd.DataFrame(pair_rows)


def build_pairwise(merged, views):
    rows = []
    for a, b in itertools.combinations(views, 2):
        ra = merged[f"true_rank_{a}"]
        rb = merged[f"true_rank_{b}"]
        rows.append(
            {
                "view_a": a,
                "view_b": b,
                "a_better_count": int((ra < rb).sum()),
                "b_better_count": int((rb < ra).sum()),
                "tie_count": int((ra == rb).sum()),
                "mean_rank_delta_a_minus_b": float((ra - rb).mean()),
                "mean_abs_rank_delta": float((ra - rb).abs().mean()),
            }
        )
    return pd.DataFrame(rows)


def build_regret_summary(per_sample):
    regret = as_numeric(per_sample["best_fixed_regret"])
    return pd.DataFrame(
        [
            {
                "best_fixed_view": per_sample.attrs["best_fixed_view"],
                "mean_regret": float(regret.mean()),
                "median_regret": float(regret.median()),
                "regret_gt_0_count": int((regret > 0).sum()),
                "regret_gt_0_ratio": float((regret > 0).mean()),
                "regret_ge_3_count": int((regret >= 3).sum()),
                "regret_ge_3_ratio": float((regret >= 3).mean()),
                "max_regret": float(regret.max()),
            }
        ]
    )


def pick_examples(per_sample, views, limit_per_view):
    examples = []
    for view in views:
        subset = per_sample[
            (per_sample["unique_oracle_winner"] == view)
            & (per_sample[f"true_rank_{view}"] <= 3)
            & (per_sample["rank_gap"] >= 3)
        ].copy()
        subset = subset.sort_values(["rank_gap", f"true_rank_{view}"], ascending=[False, True]).head(limit_per_view)
        for _, row in subset.iterrows():
            item = {
                "winner": view,
                "bundle_id": row["bundle_id"],
                "true_indice": row["true_indice"],
                "true_option_char": row["true_option_char"],
                "oracle_true_rank": row["oracle_true_rank"],
                "rank_gap": row["rank_gap"],
                "input_str": row.get(f"input_str_{view}", ""),
                "target_str": row.get(f"target_str_{view}", ""),
                "candidate_indices": row.get("candidate_indices", ""),
            }
            for v in views:
                item[f"true_rank_{v}"] = row[f"true_rank_{v}"]
                item[f"ranking_{v}"] = row.get(f"ranking_{v}", "")
                item[f"prediction_{v}"] = row.get(f"prediction_{v}", "")
            examples.append(item)
    return pd.DataFrame(examples)


def write_markdown_report(
    out_path,
    dataset,
    view_paths,
    summary,
    winner_unique,
    success_overlap,
    success_cardinality,
    hit1_margin_by_view,
    unique_hit_other_ranks,
    two_hit_remaining_ranks,
    pairwise,
    regret_summary,
    best_fixed_view,
    best_tie_cardinality,
    best_set_distribution,
):
    def markdown_table(df, floatfmt=".4f"):
        if df.empty:
            return "(empty)"
        headers = [str(col) for col in df.columns]
        rows = []
        for _, row in df.iterrows():
            values = []
            for value in row.tolist():
                if isinstance(value, float):
                    values.append(format(value, floatfmt))
                else:
                    values.append(str(value))
            rows.append(values)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(values) + " |" for values in rows)
        return "\n".join(lines)

    oracle = summary[summary["method"] == "OracleSelector"].iloc[0]
    best = summary[summary["method"] == best_fixed_view].iloc[0]
    gain_hit1 = oracle["hit_at_1"] - best["hit_at_1"]
    gain_mrr = oracle["mrr"] - best["mrr"]
    lines = []
    lines.append(f"# Ranking View Complementarity Analysis: {dataset}")
    lines.append("")
    lines.append("## Input files")
    for view, path in view_paths.items():
        lines.append(f"- {view}: `{path}`")
    lines.append("")
    lines.append("## Core claim")
    lines.append(
        "The oracle per-sample selector is an upper bound for a future view-selection agent. "
        "If it outperforms the best fixed view, then the three graph-derived views contain complementary evidence."
    )
    lines.append("")
    lines.append(f"- Best fixed view by MRR: `{best_fixed_view}`")
    lines.append(f"- Oracle Hit@1 gain over best fixed view: {gain_hit1:.4f}")
    lines.append(f"- Oracle MRR gain over best fixed view: {gain_mrr:.4f}")
    lines.append("")
    lines.append("## Summary")
    lines.append(markdown_table(summary))
    lines.append("")
    lines.append("## Unique oracle winner distribution")
    lines.append(markdown_table(winner_unique))
    lines.append("")
    lines.append("## Best-view tie cardinality")
    lines.append(markdown_table(best_tie_cardinality))
    lines.append("")
    lines.append("## Exact best-view sets")
    lines.append(markdown_table(best_set_distribution))
    lines.append("")
    lines.append("## Success overlap")
    lines.append(markdown_table(success_overlap))
    lines.append("")
    lines.append("## Success cardinality")
    lines.append(markdown_table(success_cardinality))
    lines.append("")
    lines.append("## Hit@1 margin by focal view")
    lines.append(markdown_table(hit1_margin_by_view))
    lines.append("")
    lines.append("## Unique Hit@1 Other-View Ranks")
    lines.append(markdown_table(unique_hit_other_ranks))
    lines.append("")
    lines.append("## Two-View Hit@1 Remaining-View Ranks")
    lines.append(markdown_table(two_hit_remaining_ranks))
    lines.append("")
    lines.append("## Pairwise rank comparison")
    lines.append(markdown_table(pairwise))
    lines.append("")
    lines.append("## Best-fixed regret")
    lines.append(markdown_table(regret_summary))
    lines.append("")
    lines.append("## Suggested paper/report wording")
    lines.append(
        "A fixed relational view is not uniformly optimal across test instances. "
        "The oracle selector, which chooses the best view per instance using the ground-truth rank, "
        "substantially improves over the best single view. "
        "This gap motivates a final sample-aware agent that decides which relational evidence view "
        "to expose to the LLM before producing the final ranking."
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Analyze complementarity among three RANK result CSVs.")
    parser.add_argument("--dataset", default="pog")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--files", nargs="*", default=None, help="Optional explicit CSV paths.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--example-limit-per-view", type=int, default=5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir) if args.results_dir else repo_root / "results" / args.dataset
    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "analysis" / f"{args.dataset}_ranking_view_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(p) for p in args.files] if args.files else find_latest_rank_files(results_dir, args.dataset)
    merged, view_paths = merge_view_results(paths)
    views = [view for view in ("IBxBI", "IUxUI", "BIxIB") if view in view_paths]

    summary_single = build_view_summary(merged, views, args.num_candidates)
    best_fixed_view = summary_single.sort_values(["mrr", "hit_at_1", "ndcg_at_10"], ascending=False).iloc[0]["method"]

    per_sample = build_per_sample(merged, views, args.num_candidates, best_fixed_view)
    per_sample.attrs["best_fixed_view"] = best_fixed_view
    summary = pd.concat([summary_single, build_oracle_summary(per_sample, args.num_candidates)], ignore_index=True)
    winner_unique, winner_sets = build_winner_distribution(per_sample)
    best_tie_cardinality, best_set_distribution = build_best_tie_distribution(per_sample)
    success_overlap = build_success_overlap(per_sample, views)
    success_cardinality = build_success_cardinality(per_sample, views)
    hit1_margin_by_view = build_hit1_margin_by_view(per_sample, views)
    unique_hit_other_ranks, two_hit_remaining_ranks = build_hit1_conditional_other_ranks(per_sample, views)
    pairwise = build_pairwise(merged, views)
    regret_summary = build_regret_summary(per_sample)
    examples = pick_examples(per_sample, views, args.example_limit_per_view)
    hit1_large_gap_examples = build_hit1_large_gap_examples(per_sample, views, args.example_limit_per_view)

    merged.to_csv(out_dir / "merged_view_results.csv", index=False, encoding="utf-8-sig")
    per_sample.to_csv(out_dir / "per_sample_oracle.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "view_oracle_summary.csv", index=False, encoding="utf-8-sig")
    winner_unique.to_csv(out_dir / "unique_winner_distribution.csv", index=False, encoding="utf-8-sig")
    winner_sets.to_csv(out_dir / "winner_set_distribution.csv", index=False, encoding="utf-8-sig")
    best_tie_cardinality.to_csv(out_dir / "best_tie_cardinality.csv", index=False, encoding="utf-8-sig")
    best_set_distribution.to_csv(out_dir / "best_set_distribution.csv", index=False, encoding="utf-8-sig")
    success_overlap.to_csv(out_dir / "success_overlap.csv", index=False, encoding="utf-8-sig")
    success_cardinality.to_csv(out_dir / "success_cardinality.csv", index=False, encoding="utf-8-sig")
    hit1_margin_by_view.to_csv(out_dir / "hit1_margin_by_view.csv", index=False, encoding="utf-8-sig")
    unique_hit_other_ranks.to_csv(out_dir / "unique_hit_other_ranks.csv", index=False, encoding="utf-8-sig")
    two_hit_remaining_ranks.to_csv(out_dir / "two_hit_remaining_ranks.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(out_dir / "pairwise_rank_comparison.csv", index=False, encoding="utf-8-sig")
    regret_summary.to_csv(out_dir / "best_fixed_regret.csv", index=False, encoding="utf-8-sig")
    examples.to_csv(out_dir / "case_study_examples.csv", index=False, encoding="utf-8-sig")
    hit1_large_gap_examples.to_csv(out_dir / "hit1_large_gap_examples.csv", index=False, encoding="utf-8-sig")

    write_markdown_report(
        out_dir / "summary.md",
        args.dataset,
        view_paths,
        summary,
        winner_unique,
        success_overlap,
        success_cardinality,
        hit1_margin_by_view,
        unique_hit_other_ranks,
        two_hit_remaining_ranks,
        pairwise,
        regret_summary,
        best_fixed_view,
        best_tie_cardinality,
        best_set_distribution,
    )

    print(f"Saved analysis to: {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

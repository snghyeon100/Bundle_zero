import argparse
from pathlib import Path

import pandas as pd


DEFAULT_PAIRS = {
    "pog": [
        (
            "BIxIB",
            "results/pog/results_pog_BGRAPH_HN_C10_T5_20260512_164752.csv",
            "results/pog/results_pog_RANK_BGRAPH_HN_C10_T5_20260527_132228.csv",
        ),
        (
            "IUxUI",
            "results/pog/results_pog_USERPUR_HN_C10_T5_20260512_141640.csv",
            "results/pog/results_pog_RANK_USERPUR_HN_C10_T5_20260527_132119.csv",
        ),
        (
            "IBxBI",
            "results/pog/results_pog_USER_ITEMAFF_HN_C10_T5_20260512_135419.csv",
            "results/pog/results_pog_RANK_ITEMAFF_HN_C10_T5_20260527_132007.csv",
        ),
    ]
}


def read_csv(path):
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def make_key(df):
    return (
        df["bundle_id"].astype(str)
        + "::"
        + df["true_indice"].astype(str)
        + "::"
        + df["true_option_char"].astype(str)
        + "::"
        + df["candidate_indices"].astype(str)
    )


def compare_pair(repo_root, view, choice_path, ranking_path):
    choice_path = repo_root / choice_path
    ranking_path = repo_root / ranking_path
    choice = read_csv(choice_path)
    ranking = read_csv(ranking_path)
    choice["sample_key"] = make_key(choice)
    ranking["sample_key"] = make_key(ranking)

    merged = choice.merge(
        ranking,
        on="sample_key",
        how="inner",
        suffixes=("_choice", "_ranking"),
        validate="one_to_one",
    ).copy()
    merged["view"] = view
    merged["choice_correct"] = merged["hit_choice"].astype(int)
    merged["ranking_top1_correct"] = merged["hit_ranking"].astype(int)
    merged["top1_same_prediction"] = (
        merged["prediction_choice"].astype(str) == merged["prediction_ranking"].astype(str)
    ).astype(int)
    merged["outcome"] = "both_fail"
    merged.loc[(merged["choice_correct"] == 1) & (merged["ranking_top1_correct"] == 1), "outcome"] = "both_hit"
    merged.loc[(merged["choice_correct"] == 1) & (merged["ranking_top1_correct"] == 0), "outcome"] = "choice_only"
    merged.loc[(merged["choice_correct"] == 0) & (merged["ranking_top1_correct"] == 1), "outcome"] = "ranking_only"

    summary = {
        "view": view,
        "choice_file": str(choice_path),
        "ranking_file": str(ranking_path),
        "choice_n": len(choice),
        "ranking_n": len(ranking),
        "common_n": len(merged),
        "common_ratio_vs_choice": len(merged) / len(choice) if len(choice) else 0.0,
        "common_ratio_vs_ranking": len(merged) / len(ranking) if len(ranking) else 0.0,
        "choice_hit_at_1": float(choice["hit"].mean()),
        "ranking_hit_at_1": float(ranking["hit"].mean()),
        "ranking_minus_choice_hit_at_1": float(ranking["hit"].mean() - choice["hit"].mean()),
        "ranking_hit_at_3": float(ranking["hit_at_3"].mean()),
        "ranking_hit_at_5": float(ranking["hit_at_5"].mean()),
        "ranking_mrr": float(ranking["mrr"].mean()),
        "choice_valid_ratio": float(choice["prediction"].astype(str).str.match(r"^[A-J]$").mean()),
        "ranking_valid_ratio": float(ranking["ranking_valid"].mean()),
        "top1_same_prediction_ratio": float(merged["top1_same_prediction"].mean()) if len(merged) else 0.0,
        "both_hit_count": int((merged["outcome"] == "both_hit").sum()),
        "choice_only_count": int((merged["outcome"] == "choice_only").sum()),
        "ranking_only_count": int((merged["outcome"] == "ranking_only").sum()),
        "both_fail_count": int((merged["outcome"] == "both_fail").sum()),
    }
    return summary, merged


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
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare choice and ranking runs on identical candidate sets.")
    parser.add_argument("--dataset", default="pog")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    pairs = DEFAULT_PAIRS.get(args.dataset)
    if not pairs:
        raise ValueError(f"No default file pairs configured for dataset: {args.dataset}")

    out_dir = Path(args.out_dir) if args.out_dir else repo_root / "analysis" / f"{args.dataset}_choice_vs_ranking"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    details = []
    for view, choice_path, ranking_path in pairs:
        summary, detail = compare_pair(repo_root, view, choice_path, ranking_path)
        summaries.append(summary)
        details.append(detail)

    summary_df = pd.DataFrame(summaries)
    detail_df = pd.concat(details, ignore_index=True)
    outcome_df = (
        detail_df.groupby(["view", "outcome"])
        .size()
        .reset_index(name="count")
        .merge(detail_df.groupby("view").size().reset_index(name="view_n"), on="view")
    )
    outcome_df["ratio"] = outcome_df["count"] / outcome_df["view_n"]
    outcome_df = outcome_df[["view", "outcome", "count", "ratio"]]

    summary_df.to_csv(out_dir / "choice_vs_ranking_summary.csv", index=False, encoding="utf-8-sig")
    outcome_df.to_csv(out_dir / "choice_vs_ranking_outcomes.csv", index=False, encoding="utf-8-sig")
    detail_df.to_csv(out_dir / "choice_vs_ranking_detail.csv", index=False, encoding="utf-8-sig")

    report = [
        f"# Choice vs Ranking Comparison: {args.dataset}",
        "",
        "## Summary",
        markdown_table(
            summary_df[
                [
                    "view",
                    "common_n",
                    "choice_hit_at_1",
                    "ranking_hit_at_1",
                    "ranking_minus_choice_hit_at_1",
                    "ranking_hit_at_3",
                    "ranking_mrr",
                    "top1_same_prediction_ratio",
                    "choice_valid_ratio",
                    "ranking_valid_ratio",
                ]
            ]
        ),
        "",
        "## Outcome Counts",
        markdown_table(outcome_df),
        "",
        "## Interpretation",
        "All three pairs share the same 250 sample/candidate sets, so the difference here is attributable to the choice vs ranking task/prompt rather than sample mismatch.",
    ]
    (out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")

    print(f"Saved comparison to: {out_dir}")
    print(summary_df[["view", "common_n", "choice_hit_at_1", "ranking_hit_at_1", "ranking_minus_choice_hit_at_1", "ranking_hit_at_3", "ranking_mrr"]].to_string(index=False))


if __name__ == "__main__":
    main()

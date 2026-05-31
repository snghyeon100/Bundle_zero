import argparse
import ast
import re
from pathlib import Path

import pandas as pd


VIEWS = ("IBxBI", "IUxUI", "BIxIB")
VIEW_CONTEXT_COLUMN = {
    "IBxBI": "target_str_IBxBI",
    "IUxUI": "target_str_IUxUI",
    "BIxIB": "target_str_BIxIB",
}


def read_csv(path):
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def split_options(target_text):
    text = str(target_text)
    pattern = re.compile(r"(?:^|;\s*)([A-J])\.\s*")
    matches = list(pattern.finditer(text))
    options = {}
    for idx, match in enumerate(matches):
        letter = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        options[letter] = text[start:end].strip()
    return options


def clean_text(text):
    return " ".join(str(text).split())


def parse_list(value):
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def split_added_context(option_text):
    option_text = clean_text(option_text)
    context_match = re.search(r"\[Additional context: (.*)\]$", option_text)
    if not context_match:
        return option_text, ""
    return clean_text(option_text[: context_match.start()]), clean_text(context_match.group(1))


def summarize_case(row, dataset, focal_view):
    options = split_options(row[VIEW_CONTEXT_COLUMN[focal_view]])
    true_letter = str(row["true_option_char"])
    true_base, true_context = split_added_context(options.get(true_letter, ""))

    rank_by_view = {view: parse_list(row[f"ranking_{view}"]) for view in VIEWS}
    top_wrong = {}
    for view in VIEWS:
        if view == focal_view:
            continue
        rank = rank_by_view[view]
        top_letter = rank[0] if rank else ""
        top_base, top_context = split_added_context(options.get(top_letter, ""))
        top_wrong[f"{view}_top_wrong_option"] = top_letter
        top_wrong[f"{view}_top_wrong_item"] = top_base
        top_wrong[f"{view}_top_wrong_context"] = top_context

    return {
        "dataset": dataset,
        "focal_view": focal_view,
        "bundle_id": row["bundle_id"],
        "true_option": true_letter,
        "input": clean_text(row.get(f"input_str_{focal_view}", "")),
        "true_item": true_base,
        "true_added_context": true_context,
        "rank_IBxBI": row["true_rank_IBxBI"],
        "rank_IUxUI": row["true_rank_IUxUI"],
        "rank_BIxIB": row["true_rank_BIxIB"],
        "candidate_indices": row["candidate_indices"],
        **top_wrong,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract unique Hit@1 cases for a focal view.")
    parser.add_argument("--view", choices=VIEWS, required=True)
    parser.add_argument("--datasets", nargs="+", default=["pog", "pog_dense"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "analysis" / f"{args.view.lower()}_unique_case_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in args.datasets:
        path = repo_root / "analysis" / f"{dataset}_ranking_view_analysis" / "per_sample_oracle.csv"
        df = read_csv(path)
        mask = df[f"{args.view}_hit_at_1_calc"] == 1
        for view in VIEWS:
            if view != args.view:
                mask &= df[f"{view}_hit_at_1_calc"] == 0
        subset = df[mask].copy()
        for _, row in subset.iterrows():
            rows.append(summarize_case(row, dataset, args.view))

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"{args.view.lower()}_unique_cases.csv", index=False, encoding="utf-8-sig")

    lines = [f"# {args.view} Unique Hit@1 Cases", ""]
    for idx, row in out.iterrows():
        lines.extend(
            [
                f"## {idx + 1}. {row['dataset']} bundle {row['bundle_id']} ({row['true_option']})",
                f"- ranks: IBxBI={row['rank_IBxBI']}, IUxUI={row['rank_IUxUI']}, BIxIB={row['rank_BIxIB']}",
                f"- input: {row['input']}",
                f"- true item: {row['true_item']}",
                f"- {args.view} context: {row['true_added_context']}",
            ]
        )
        for view in VIEWS:
            if view == args.view:
                continue
            lines.append(
                f"- {view} top wrong: {row.get(f'{view}_top_wrong_option', '')}. "
                f"{row.get(f'{view}_top_wrong_item', '')}"
            )
            wrong_context = row.get(f"{view}_top_wrong_context", "")
            if wrong_context:
                lines.append(f"  - {view} wrong context: {wrong_context}")
        lines.append("")

    (out_dir / f"{args.view.lower()}_unique_cases.md").write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Saved {len(out)} {args.view} unique cases to {out_dir}")
    print(out[["dataset", "bundle_id", "true_option", "rank_IBxBI", "rank_IUxUI", "rank_BIxIB"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()

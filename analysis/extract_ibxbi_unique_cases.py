import ast
import re
from pathlib import Path

import pandas as pd


DATASETS = ["pog", "pog_dense"]


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


def summarize_case(row, dataset):
    options = split_options(row["target_str_IBxBI"])
    true_letter = str(row["true_option_char"])
    true_option = clean_text(options.get(true_letter, ""))
    context_match = re.search(r"\[Additional context: (.*)\]$", true_option)
    if context_match:
        true_base = clean_text(true_option[: context_match.start()])
        true_context = clean_text(context_match.group(1))
    else:
        true_base = true_option
        true_context = ""

    ib_rank = parse_list(row["ranking_IBxBI"])
    iu_rank = parse_list(row["ranking_IUxUI"])
    bi_rank = parse_list(row["ranking_BIxIB"])
    iu_top = clean_text(options.get(iu_rank[0], "")) if iu_rank else ""
    bi_top = clean_text(options.get(bi_rank[0], "")) if bi_rank else ""

    return {
        "dataset": dataset,
        "bundle_id": row["bundle_id"],
        "true_option": true_letter,
        "input": clean_text(row["input_str_IBxBI"]),
        "true_item": true_base,
        "ibxbi_added_context": true_context,
        "iu_top_wrong_option": iu_rank[0] if iu_rank else "",
        "iu_top_wrong_item": iu_top,
        "bixib_top_wrong_option": bi_rank[0] if bi_rank else "",
        "bixib_top_wrong_item": bi_top,
        "rank_IBxBI": row["true_rank_IBxBI"],
        "rank_IUxUI": row["true_rank_IUxUI"],
        "rank_BIxIB": row["true_rank_BIxIB"],
        "candidate_indices": row["candidate_indices"],
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "analysis" / "ibxbi_unique_case_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in DATASETS:
        path = repo_root / "analysis" / f"{dataset}_ranking_view_analysis" / "per_sample_oracle.csv"
        df = read_csv(path)
        subset = df[
            (df["IBxBI_hit_at_1_calc"] == 1)
            & (df["IUxUI_hit_at_1_calc"] == 0)
            & (df["BIxIB_hit_at_1_calc"] == 0)
        ].copy()
        for _, row in subset.iterrows():
            rows.append(summarize_case(row, dataset))

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "ibxbi_unique_cases.csv", index=False, encoding="utf-8-sig")

    lines = ["# IBxBI Unique Hit@1 Cases", ""]
    for idx, row in out.iterrows():
        lines.extend(
            [
                f"## {idx + 1}. {row['dataset']} bundle {row['bundle_id']} ({row['true_option']})",
                f"- ranks: IBxBI={row['rank_IBxBI']}, IUxUI={row['rank_IUxUI']}, BIxIB={row['rank_BIxIB']}",
                f"- input: {row['input']}",
                f"- true item: {row['true_item']}",
                f"- IBxBI context: {row['ibxbi_added_context']}",
                f"- IUxUI top wrong: {row['iu_top_wrong_option']}. {row['iu_top_wrong_item']}",
                f"- BIxIB top wrong: {row['bixib_top_wrong_option']}. {row['bixib_top_wrong_item']}",
                "",
            ]
        )
    (out_dir / "ibxbi_unique_cases.md").write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"Saved {len(out)} IBxBI unique cases to {out_dir}")
    print(out[["dataset", "bundle_id", "true_option", "rank_IUxUI", "rank_BIxIB"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()

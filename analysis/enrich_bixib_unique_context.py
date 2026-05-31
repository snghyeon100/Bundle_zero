import ast
import json
from pathlib import Path

import pandas as pd


def read_csv(path):
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def parse_list(value):
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def clean_text(text):
    return " ".join(str(text).split())


def load_train_bundles(path):
    bundles = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [int(x) for x in line.strip().split(", ") if x.strip()]
            if parts:
                bundles[parts[0]] = parts[1:]
    return bundles


def load_item_titles(path):
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)
    return {int(item_id): clean_text(row.get("title", f"Item {item_id}")) for item_id, row in info.items()}


def latest_bgraph_result(repo_root, dataset):
    files = sorted((repo_root / "results" / dataset).glob(f"results_{dataset}_RANK_BGRAPH*.csv"))
    if not files:
        raise FileNotFoundError(f"No RANK_BGRAPH result found for {dataset}")
    return max(files, key=lambda p: p.stat().st_mtime)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "analysis" / "bixib_unique_case_study"
    unique_path = out_dir / "bixib_unique_cases.csv"
    unique = read_csv(unique_path)

    result_rows = []
    bgraph_by_dataset = {}
    for dataset in unique["dataset"].unique():
        result_path = latest_bgraph_result(repo_root, dataset)
        bgraph = read_csv(result_path)
        bgraph["candidate_key"] = bgraph["candidate_indices"].astype(str)
        bgraph_by_dataset[dataset] = bgraph.set_index(["bundle_id", "candidate_key"], drop=False)

    train_cache = {}
    title_cache = {}
    for _, row in unique.iterrows():
        dataset = row["dataset"]
        if dataset not in train_cache:
            train_cache[dataset] = load_train_bundles(repo_root / "datasets" / dataset / "bi_train.txt")
            title_cache[dataset] = load_item_titles(repo_root / "datasets" / dataset / "item_info.json")

        key = (int(row["bundle_id"]), str(row["candidate_indices"]))
        bgraph_row = bgraph_by_dataset[dataset].loc[key]
        context_bundle_ids = [int(x) for x in parse_list(bgraph_row["bundle_graph_context_bundle_ids"])]
        overlap_counts = parse_list(bgraph_row.get("bundle_graph_context_overlap_counts", "[]"))
        idf_scores = parse_list(bgraph_row.get("bundle_graph_context_idf_scores", "[]"))

        context_lines = []
        for idx, bundle_id in enumerate(context_bundle_ids, start=1):
            items = train_cache[dataset].get(bundle_id, [])[:5]
            titles = [title_cache[dataset].get(int(item_id), f"Item {item_id}") for item_id in items]
            overlap = overlap_counts[idx - 1] if idx - 1 < len(overlap_counts) else ""
            idf = idf_scores[idx - 1] if idx - 1 < len(idf_scores) else ""
            context_lines.append(
                f"{idx}. bundle {bundle_id} (overlap={overlap}, idf={float(idf):.3f}): "
                + "; ".join(titles)
            )

        record = row.to_dict()
        record["retrieved_bundle_ids"] = context_bundle_ids
        record["retrieved_outfits"] = " || ".join(context_lines)
        result_rows.append(record)

    enriched = pd.DataFrame(result_rows)
    enriched.to_csv(out_dir / "bixib_unique_cases_with_retrieved_outfits.csv", index=False, encoding="utf-8-sig")

    lines = ["# BIxIB Unique Hit@1 Cases with Retrieved Outfits", ""]
    for idx, row in enriched.iterrows():
        lines.extend(
            [
                f"## {idx + 1}. {row['dataset']} bundle {row['bundle_id']} ({row['true_option']})",
                f"- ranks: IBxBI={row['rank_IBxBI']}, IUxUI={row['rank_IUxUI']}, BIxIB={row['rank_BIxIB']}",
                f"- input: {row['input']}",
                f"- true item: {row['true_item']}",
                "- retrieved past outfits:",
            ]
        )
        for context_line in str(row["retrieved_outfits"]).split(" || "):
            lines.append(f"  - {context_line}")
        lines.extend(
            [
                f"- IBxBI top wrong: {row.get('IBxBI_top_wrong_option', '')}. {row.get('IBxBI_top_wrong_item', '')}",
                f"- IUxUI top wrong: {row.get('IUxUI_top_wrong_option', '')}. {row.get('IUxUI_top_wrong_item', '')}",
                "",
            ]
        )

    (out_dir / "bixib_unique_cases_with_retrieved_outfits.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )
    print(f"Saved {len(enriched)} enriched BIxIB cases to {out_dir}")
    print(enriched[["dataset", "bundle_id", "true_option", "retrieved_bundle_ids"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()

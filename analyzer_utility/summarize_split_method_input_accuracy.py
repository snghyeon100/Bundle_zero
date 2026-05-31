import ast
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


SPECS = [
    ("old", "base", "pog", r"results\pog\results_pog_20260416_142034.csv"),
    ("old", "base", "pog_dense", r"results\pog_dense\results_pog_dense_HN_C10_T5_20260430_172343.csv"),
    ("old", "BGRAPH", "pog", r"results\pog\results_pog_BGRAPH_HN_C10_T5_20260518_122018.csv"),
    ("old", "BGRAPH", "pog_dense", r"results\pog_dense\results_pog_dense_BGRAPH_HN_C10_T5_20260518_122059.csv"),
    ("fixed", "base", "pog", r"results\pog\results_pog_HN_C10_T5_20260526_163534.csv"),
    ("fixed", "base", "pog_dense", r"results\pog_dense\results_pog_dense_HN_C10_T5_20260526_163647.csv"),
    ("fixed", "BGRAPH", "pog", r"results\pog\results_pog_BGRAPH_HN_C10_T5_20260526_173230.csv"),
    ("fixed", "BGRAPH", "pog_dense", r"results\pog_dense\results_pog_dense_BGRAPH_HN_C10_T5_20260526_173132.csv"),
]


def input_count(value):
    text = str(value).strip()
    if not text.startswith("["):
        return 0
    return len(ast.literal_eval(text))


def summarize_file(split_setting, method, dataset, rel_path):
    path = REPO_ROOT / rel_path
    df = pd.read_csv(path)
    df["input_count"] = df["input_indices"].apply(input_count)
    df["hit_num"] = pd.to_numeric(df["hit"], errors="coerce").fillna(0).astype(int)

    rows = [{
        "split_setting": split_setting,
        "method": method,
        "dataset": dataset,
        "input_count": "overall",
        "n": len(df),
        "hit": int(df["hit_num"].sum()),
        "accuracy": float(df["hit_num"].mean()),
        "source_file": rel_path,
    }]

    for count, group in df.groupby("input_count"):
        rows.append({
            "split_setting": split_setting,
            "method": method,
            "dataset": dataset,
            "input_count": int(count),
            "n": len(group),
            "hit": int(group["hit_num"].sum()),
            "accuracy": float(group["hit_num"].mean()),
            "source_file": rel_path,
        })
    return rows


def make_wide_table(long_df):
    display_df = long_df.copy()
    display_df["acc_n"] = display_df.apply(
        lambda row: f"{row['accuracy']:.4f} ({int(row['hit'])}/{int(row['n'])})",
        axis=1,
    )
    wide = display_df.pivot_table(
        index=["split_setting", "method", "dataset"],
        columns="input_count",
        values="acc_n",
        aggfunc="first",
    ).reset_index()

    int_columns = sorted(col for col in wide.columns if isinstance(col, int))
    ordered = ["split_setting", "method", "dataset", "overall", *int_columns]
    return wide[[col for col in ordered if col in wide.columns]]


def dataframe_to_markdown(df):
    columns = [str(col) for col in df.columns]
    rows = []
    for _, row in df.iterrows():
        rows.append([str(row[col]) if pd.notna(row[col]) else "" for col in df.columns])

    widths = []
    for idx, column in enumerate(columns):
        values = [row[idx] for row in rows]
        widths.append(max(len(column), *(len(value) for value in values)))

    def fmt_row(values):
        return "| " + " | ".join(
            str(value).ljust(widths[idx]) for idx, value in enumerate(values)
        ) + " |"

    lines = [
        fmt_row(columns),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def main():
    rows = []
    for spec in SPECS:
        rows.extend(summarize_file(*spec))

    out_dir = REPO_ROOT / "analysis" / "split_method_input_accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = pd.DataFrame(rows)
    long_path = out_dir / "long_summary.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    wide = make_wide_table(long_df)
    wide_path = out_dir / "wide_summary.csv"
    wide.to_csv(wide_path, index=False, encoding="utf-8-sig")

    md_path = out_dir / "wide_summary.md"
    markdown = dataframe_to_markdown(wide)
    md_path.write_text(markdown + "\n", encoding="utf-8")

    print(f"Saved long: {long_path}")
    print(f"Saved wide: {wide_path}")
    print(f"Saved markdown: {md_path}")
    print(markdown)


if __name__ == "__main__":
    main()

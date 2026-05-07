import argparse
import os

import pandas as pd


DATASETS = [
    ("POG", "analysis/pog_cf_method_signal_analysis"),
    ("POG-Dense", "analysis/pog_dense_cf_signal_analysis"),
    ("Spotify", "analysis/spotify_cf_method_signal_analysis"),
    ("Spotify-Sparse", "analysis/spotify_sparse_cf_method_signal_analysis"),
]


METHOD_LABELS = {
    "base": "Base",
    "HN_base": "Base",
    "co_occur": "CoCF",
    "co_occer_cf": "CoCF",
    "HN_co_occur": "CoCF",
    "user_prefer": "UserCF",
    "user_prefer_cf": "UserCF",
    "HN_user_prefer": "UserCF",
}


def pct(value):
    if pd.isna(value):
        return "--"
    return f"{float(value) * 100:.1f}\\%"


def num(value, digits=2):
    if pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def tex_escape(value):
    text = str(value)
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def row(values):
    return " & ".join(str(v) for v in values) + r" \\"


def read_table(folder, name):
    return pd.read_csv(os.path.join(folder, name))


def method_label(method):
    return METHOD_LABELS.get(str(method), str(method))


def signal_label(signal):
    return {"cooccurrence": "Co-occurrence", "user_pref": "User preference"}.get(
        str(signal), str(signal)
    )


def make_signal_quality():
    lines = []
    for dataset, folder in DATASETS:
        df = read_table(folder, "cf_signal_quality.csv")
        for _, r in df.iterrows():
            lines.append(
                row(
                    [
                        dataset,
                        signal_label(r["signal"]),
                        int(r["n"]),
                        pct(r["gt_in_top_tie_rate"]),
                        pct(r["gt_unique_top1_rate"]),
                        pct(r["positive_signal_rate"]),
                        pct(r["all_zero_rate"]),
                        num(r["mean_top_tie_size"]),
                    ]
                )
            )
    body = "\n".join(lines)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Quality of collaborative-filtering signals across datasets. Top-tie allows ties for the highest score, while unique top-1 requires the ground-truth item to be the single highest-scoring candidate.}}
\label{{tab:cf_signal_quality_all}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrr}}
\toprule
Dataset & Signal & $n$ & GT top-tie & GT unique top-1 & Positive signal & All-zero & Mean top-tie size \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}"""


def select_method_rows(df):
    selected = []
    for _, r in df.iterrows():
        label = method_label(r["method"])
        signal = r["signal"]
        if label == "Base":
            selected.append(r)
        elif label == "CoCF" and signal == "cooccurrence":
            selected.append(r)
        elif label == "UserCF" and signal == "user_pref":
            selected.append(r)
    return pd.DataFrame(selected)


def make_method_reliance(signal):
    lines = []
    for dataset, folder in DATASETS:
        df = read_table(folder, "cf_method_reliance.csv")
        df = df[df["signal"] == signal].copy()
        for _, r in df.iterrows():
            lines.append(
                row(
                    [
                        dataset,
                        method_label(r["method"]),
                        int(r["n"]),
                        pct(r["accuracy"]),
                        pct(r["pred_in_top_tie_rate"]),
                        pct(r["pred_unique_top1_rate"]),
                        pct(r["hit_when_pred_in_top_tie"]),
                        pct(r["hit_when_pred_not_in_top_tie"]),
                    ]
                )
            )
    body = "\n".join(lines)
    label = "cooccurrence" if signal == "cooccurrence" else "userpref"
    caption_signal = signal_label(signal)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{LLM reliance on the {caption_signal} signal. Pred top-tie measures whether the model selected a candidate tied for the highest CF score.}}
\label{{tab:cf_reliance_{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrrrr}}
\toprule
Dataset & Method & $n$ & Acc. & Pred top-tie & Pred unique top-1 & Hit if pred top & Hit if pred not top \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}"""


def make_targeted_method_summary():
    lines = []
    for dataset, folder in DATASETS:
        df = read_table(folder, "cf_method_reliance.csv")
        target_rows = []
        for _, r in df.iterrows():
            label = method_label(r["method"])
            if label == "Base" and r["signal"] == "cooccurrence":
                target_rows.append((r, "Co-occurrence"))
            elif label == "CoCF" and r["signal"] == "cooccurrence":
                target_rows.append((r, "Co-occurrence"))
            elif label == "UserCF" and r["signal"] == "user_pref":
                target_rows.append((r, "User preference"))

        for r, inspected in target_rows:
            lines.append(
                row(
                    [
                        dataset,
                        method_label(r["method"]),
                        inspected,
                        pct(r["accuracy"]),
                        pct(r["pred_in_top_tie_rate"]),
                        pct(r["pred_unique_top1_rate"]),
                        num(r["pred_mean_avg_rank"]),
                    ]
                )
            )
    body = "\n".join(lines)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Targeted CF-method behavior. CoCF is evaluated against co-occurrence scores and UserCF against user-preference scores; Base is shown with co-occurrence as the stronger baseline signal.}}
\label{{tab:cf_targeted_methods_all}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lllrrrr}}
\toprule
Dataset & Method & Inspected signal & Acc. & Pred top-tie & Pred unique top-1 & Mean pred avg-rank \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}"""


def make_pairwise_delta():
    lines = []
    wanted = {
        ("Base", "CoCF", "cooccurrence"): "Base vs. CoCF",
        ("Base", "UserCF", "user_pref"): "Base vs. UserCF",
    }
    for dataset, folder in DATASETS:
        df = read_table(folder, "cf_pairwise.csv")
        for _, r in df.iterrows():
            left = method_label(r["left_method"])
            right = method_label(r["right_method"])
            signal = r["signal"]
            if (left, right, signal) not in wanted:
                continue
            lines.append(
                row(
                    [
                        dataset,
                        wanted[(left, right, signal)],
                        signal_label(signal),
                        int(r["n"]),
                        pct(r["left_accuracy"]),
                        pct(r["right_accuracy"]),
                        int(r["left_only"]),
                        int(r["right_only"]),
                        int(r["both_hit"]),
                        int(r["both_fail"]),
                        pct(r["right_only_gt_unique_top1_rate"]),
                    ]
                )
            )
    body = "\n".join(lines)
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{Pairwise method deltas under the intended CF signal. Right-only cases are problems solved only by the CF method, and the last column asks whether those gains occur when the ground-truth item is a unique top-1 under the same CF score.}}
\label{{tab:cf_pairwise_delta_all}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lllrrrrrrrr}}
\toprule
Dataset & Comparison & Signal & $n$ & Left acc. & Right acc. & Left only & Right only & Both hit & Both fail & Right-only GT unique top-1 \\
\midrule
{body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}"""


def make_text():
    return "\n\n".join(
        [
            "% Required packages: booktabs, graphicx, placeins",
            "% Suggested preamble:",
            "% \\usepackage{booktabs}",
            "% \\usepackage{graphicx}",
            "% \\usepackage{placeins}",
            make_signal_quality(),
            "\\FloatBarrier",
            make_targeted_method_summary(),
            "\\FloatBarrier",
            make_method_reliance("cooccurrence"),
            "\\FloatBarrier",
            make_method_reliance("user_pref"),
            "\\FloatBarrier",
            make_pairwise_delta(),
            "\\FloatBarrier",
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="analysis/cf_signal_all_datasets_overleaf.tex",
        help="Output TeX file.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(make_text())
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

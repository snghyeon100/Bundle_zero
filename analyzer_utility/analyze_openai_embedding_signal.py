import argparse
import ast
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULT_FILES = {
    "pog": r"results\pog\results_pog_20260416_142034.csv",
    "pog_dense": r"results\pog_dense\results_pog_dense_HN_C10_T5_20260430_172343.csv",
    "spotify": r"results\spotify\results_spotify_20260411_191411.csv",
    "spotify_sparse": r"results\spotify_sparse\results_spotify_sparse_20260411_195706.csv",
}

DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_CACHE_ROOT = r"analysis\openai_embedding_cache"
DEFAULT_OUTPUT_ROOT = r"analysis\openai_embedding_signal"


def parse_id_list(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    if pd.isna(value):
        return []
    return [int(x) for x in ast.literal_eval(str(value))]


def option_idx_from_prediction(prediction):
    if pd.isna(prediction):
        return None
    text = str(prediction).strip().upper()
    if len(text) == 1 and "A" <= text <= "Z":
        return ord(text) - ord("A")
    return None


def load_embedding_cache(repo_root, cache_root, model, dataset, dtype):
    cache_dir = repo_root / cache_root / model / dataset
    cache_path = cache_dir / f"embeddings_{model}_{dtype}.npz"
    metadata_path = cache_dir / "metadata.json"
    target_path = cache_dir / "target_items.csv"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    with np.load(cache_path, allow_pickle=False) as data:
        ids = data["ids"].astype(np.int64)
        embeddings = data["embeddings"].astype(np.float32)
        text_hashes = data["text_sha256"]
    norms = np.linalg.norm(embeddings, axis=1)
    safe = norms > 0
    embeddings_normed = np.zeros_like(embeddings, dtype=np.float32)
    embeddings_normed[safe] = embeddings[safe] / norms[safe, None]
    id_to_row = {int(item_id): idx for idx, item_id in enumerate(ids.tolist())}

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "cache_dir": cache_dir,
        "cache_path": cache_path,
        "metadata_path": metadata_path,
        "target_path": target_path,
        "ids": ids,
        "embeddings": embeddings_normed,
        "raw_norms": norms,
        "text_hashes": text_hashes,
        "id_to_row": id_to_row,
        "metadata": metadata,
    }


def vectors_for_ids(cache, item_ids, strict=True):
    missing = [int(item_id) for item_id in item_ids if int(item_id) not in cache["id_to_row"]]
    if missing and strict:
        raise KeyError(f"{len(missing)} item ids missing from cache. First ids: {missing[:20]}")
    rows = [cache["id_to_row"][int(item_id)] for item_id in item_ids if int(item_id) in cache["id_to_row"]]
    return cache["embeddings"][rows], rows, missing


def stable_desc_rank(scores, true_idx):
    order = np.argsort(-scores, kind="mergesort")
    return int(np.where(order == true_idx)[0][0] + 1), order


def analyze_one_dataset(repo_root, dataset, result_csv, args):
    result_csv = Path(result_csv)
    if not result_csv.is_absolute():
        result_csv = repo_root / result_csv
    if not result_csv.exists():
        raise FileNotFoundError(result_csv)

    cache = load_embedding_cache(repo_root, Path(args.cache_root), args.model, dataset, args.dtype)
    df = pd.read_csv(result_csv)
    rows = []

    for row_idx, row in df.iterrows():
        input_ids = parse_id_list(row["input_indices"])
        candidate_ids = parse_id_list(row["candidate_indices"])
        true_item = int(row["true_indice"])
        true_option_idx = (
            int(row["true_option_idx"])
            if "true_option_idx" in row and not pd.isna(row["true_option_idx"])
            else candidate_ids.index(true_item)
        )
        pred_option_idx = option_idx_from_prediction(row.get("prediction"))

        input_vecs, _, input_missing = vectors_for_ids(cache, input_ids, strict=False)
        cand_vecs, _, cand_missing = vectors_for_ids(cache, candidate_ids, strict=False)
        if input_missing or cand_missing or len(input_vecs) == 0 or len(cand_vecs) != len(candidate_ids):
            rows.append(
                {
                    "row_idx": row_idx,
                    "bundle_id": int(row["bundle_id"]),
                    "skipped": 1,
                    "skip_reason": f"missing embeddings input={input_missing[:5]} cand={cand_missing[:5]}",
                }
            )
            continue

        bundle_vec = input_vecs.mean(axis=0)
        bundle_norm = np.linalg.norm(bundle_vec)
        if bundle_norm == 0:
            rows.append(
                {
                    "row_idx": row_idx,
                    "bundle_id": int(row["bundle_id"]),
                    "skipped": 1,
                    "skip_reason": "zero input bundle embedding",
                }
            )
            continue
        bundle_vec = bundle_vec / bundle_norm

        input_to_candidates = cand_vecs @ bundle_vec
        input_pair_to_candidates = input_vecs @ cand_vecs.T
        input_pair_mean_to_candidates = input_pair_to_candidates.mean(axis=0)
        input_pair_max_to_candidates = input_pair_to_candidates.max(axis=0)

        gt_sim = float(input_to_candidates[true_option_idx])
        gt_pair_mean = float(input_pair_mean_to_candidates[true_option_idx])
        gt_pair_max = float(input_pair_max_to_candidates[true_option_idx])

        neg_mask = np.ones(len(candidate_ids), dtype=bool)
        neg_mask[true_option_idx] = False
        neg_sims = input_to_candidates[neg_mask]
        neg_pair_means = input_pair_mean_to_candidates[neg_mask]
        neg_pair_maxes = input_pair_max_to_candidates[neg_mask]

        gt_rank, order = stable_desc_rank(input_to_candidates, true_option_idx)
        gt_pair_mean_rank, pair_mean_order = stable_desc_rank(input_pair_mean_to_candidates, true_option_idx)
        gt_pair_max_rank, pair_max_order = stable_desc_rank(input_pair_max_to_candidates, true_option_idx)

        pred_sim = (
            float(input_to_candidates[pred_option_idx])
            if pred_option_idx is not None and 0 <= pred_option_idx < len(candidate_ids)
            else np.nan
        )

        gt_vec = cand_vecs[true_option_idx]
        gt_to_candidates = cand_vecs @ gt_vec
        gt_to_distractors = gt_to_candidates[neg_mask]

        rows.append(
            {
                "row_idx": row_idx,
                "bundle_id": int(row["bundle_id"]),
                "true_indice": true_item,
                "true_option_idx": true_option_idx,
                "true_option_char": row.get("true_option_char"),
                "prediction": row.get("prediction"),
                "pred_option_idx": pred_option_idx,
                "hit": int(row["hit"]) if "hit" in row and not pd.isna(row["hit"]) else np.nan,
                "skipped": 0,
                "skip_reason": "",
                "num_input_items": len(input_ids),
                "num_candidates": len(candidate_ids),
                "input_gt_sim": gt_sim,
                "input_neg_sim_mean": float(neg_sims.mean()),
                "input_neg_sim_max": float(neg_sims.max()),
                "input_neg_sim_min": float(neg_sims.min()),
                "input_gt_margin_vs_neg_mean": float(gt_sim - neg_sims.mean()),
                "input_gt_margin_vs_best_neg": float(gt_sim - neg_sims.max()),
                "input_gt_rank": gt_rank,
                "input_gt_top1": int(gt_rank == 1),
                "input_gt_top3": int(gt_rank <= 3),
                "input_gt_mrr": 1.0 / gt_rank,
                "input_semantic_top_option_idx": int(order[0]),
                "input_semantic_top_item": int(candidate_ids[order[0]]),
                "input_semantic_top_sim": float(input_to_candidates[order[0]]),
                "pred_input_sim": pred_sim,
                "pred_is_input_semantic_top1": int(pred_option_idx == int(order[0])) if pred_option_idx is not None else np.nan,
                "pred_input_rank": int(np.where(order == pred_option_idx)[0][0] + 1)
                if pred_option_idx is not None and 0 <= pred_option_idx < len(candidate_ids)
                else np.nan,
                "pairmean_input_gt_sim": gt_pair_mean,
                "pairmean_input_neg_sim_mean": float(neg_pair_means.mean()),
                "pairmean_input_neg_sim_max": float(neg_pair_means.max()),
                "pairmean_input_gt_margin_vs_best_neg": float(gt_pair_mean - neg_pair_means.max()),
                "pairmean_input_gt_rank": gt_pair_mean_rank,
                "pairmean_input_gt_top1": int(gt_pair_mean_rank == 1),
                "pairmax_input_gt_sim": gt_pair_max,
                "pairmax_input_neg_sim_mean": float(neg_pair_maxes.mean()),
                "pairmax_input_neg_sim_max": float(neg_pair_maxes.max()),
                "pairmax_input_gt_margin_vs_best_neg": float(gt_pair_max - neg_pair_maxes.max()),
                "pairmax_input_gt_rank": gt_pair_max_rank,
                "pairmax_input_gt_top1": int(gt_pair_max_rank == 1),
                "gt_to_distractor_sim_mean": float(gt_to_distractors.mean()),
                "gt_to_distractor_sim_max": float(gt_to_distractors.max()),
                "gt_to_distractor_sim_min": float(gt_to_distractors.min()),
                "candidate_ids": json.dumps(candidate_ids, ensure_ascii=False),
                "input_sims_by_option": json.dumps([float(x) for x in input_to_candidates], ensure_ascii=False),
                "pairmean_sims_by_option": json.dumps([float(x) for x in input_pair_mean_to_candidates], ensure_ascii=False),
                "pairmax_sims_by_option": json.dumps([float(x) for x in input_pair_max_to_candidates], ensure_ascii=False),
            }
        )

    detail = pd.DataFrame(rows)
    usable = detail[detail["skipped"] == 0].copy()
    if usable.empty:
        raise ValueError(f"{dataset}: no usable rows after cache matching")

    summary = make_summary(usable)
    examples = make_examples(usable)

    out_dir = repo_root / args.output_root / args.model / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = result_csv.stem
    detail_path = out_dir / f"{stem}_openai_embedding_detail.csv"
    summary_path = out_dir / f"{stem}_openai_embedding_summary.csv"
    examples_path = out_dir / f"{stem}_openai_embedding_examples.csv"
    sanity_path = out_dir / f"{stem}_openai_embedding_sanity.json"

    detail.to_csv(detail_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    examples.to_csv(examples_path, index=False, quoting=csv.QUOTE_MINIMAL)

    sanity = {
        "dataset": dataset,
        "result_csv": str(result_csv),
        "cache_path": str(cache["cache_path"]),
        "metadata_path": str(cache["metadata_path"]),
        "target_path": str(cache["target_path"]),
        "cache_num_items": int(len(cache["ids"])),
        "cache_embedding_dim": int(cache["embeddings"].shape[1]),
        "result_rows": int(len(df)),
        "usable_rows": int(len(usable)),
        "skipped_rows": int((detail["skipped"] == 1).sum()),
        "embedding_norm_min": float(cache["raw_norms"].min()),
        "embedding_norm_mean": float(cache["raw_norms"].mean()),
        "embedding_norm_max": float(cache["raw_norms"].max()),
    }
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(sanity, f, ensure_ascii=False, indent=2)

    return {
        "dataset": dataset,
        "detail_path": detail_path,
        "summary_path": summary_path,
        "examples_path": examples_path,
        "sanity_path": sanity_path,
        "summary": summary,
        "sanity": sanity,
    }


def make_summary(usable):
    rows = []
    splits = [("all", usable)]
    if "hit" in usable:
        splits.extend([("llm_hit", usable[usable["hit"] == 1]), ("llm_miss", usable[usable["hit"] == 0])])

    for split, sub in splits:
        if sub.empty:
            continue
        rows.append(
            {
                "split": split,
                "n": int(len(sub)),
                "llm_hit_rate": float(sub["hit"].mean()) if "hit" in sub else np.nan,
                "input_gt_top1_rate": float(sub["input_gt_top1"].mean()),
                "input_gt_top3_rate": float(sub["input_gt_top3"].mean()),
                "input_gt_mrr": float(sub["input_gt_mrr"].mean()),
                "input_gt_rank_mean": float(sub["input_gt_rank"].mean()),
                "input_gt_sim_mean": float(sub["input_gt_sim"].mean()),
                "input_neg_sim_mean": float(sub["input_neg_sim_mean"].mean()),
                "input_neg_sim_max_mean": float(sub["input_neg_sim_max"].mean()),
                "input_gt_margin_vs_neg_mean_mean": float(sub["input_gt_margin_vs_neg_mean"].mean()),
                "input_gt_margin_vs_best_neg_mean": float(sub["input_gt_margin_vs_best_neg"].mean()),
                "pred_is_input_semantic_top1_rate": float(sub["pred_is_input_semantic_top1"].dropna().mean()),
                "pred_input_rank_mean": float(sub["pred_input_rank"].dropna().mean()),
                "pairmean_gt_top1_rate": float(sub["pairmean_input_gt_top1"].mean()),
                "pairmean_gt_rank_mean": float(sub["pairmean_input_gt_rank"].mean()),
                "pairmean_gt_margin_vs_best_neg_mean": float(sub["pairmean_input_gt_margin_vs_best_neg"].mean()),
                "pairmax_gt_top1_rate": float(sub["pairmax_input_gt_top1"].mean()),
                "pairmax_gt_rank_mean": float(sub["pairmax_input_gt_rank"].mean()),
                "pairmax_gt_margin_vs_best_neg_mean": float(sub["pairmax_input_gt_margin_vs_best_neg"].mean()),
                "gt_to_distractor_sim_mean": float(sub["gt_to_distractor_sim_mean"].mean()),
                "gt_to_distractor_sim_max_mean": float(sub["gt_to_distractor_sim_max"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_examples(usable):
    example_specs = [
        ("easy_hit_semantic_top1", usable[(usable["hit"] == 1) & (usable["input_gt_top1"] == 1)], "input_gt_margin_vs_best_neg", False),
        ("hard_miss_gt_low_rank", usable[(usable["hit"] == 0) & (usable["input_gt_top1"] == 0)], "input_gt_rank", False),
        ("semantic_top1_but_llm_miss", usable[(usable["hit"] == 0) & (usable["input_gt_top1"] == 1)], "input_gt_margin_vs_best_neg", False),
        ("llm_hit_not_semantic_top1", usable[(usable["hit"] == 1) & (usable["input_gt_top1"] == 0)], "input_gt_margin_vs_best_neg", True),
    ]
    frames = []
    keep_cols = [
        "row_idx",
        "bundle_id",
        "true_indice",
        "true_option_char",
        "prediction",
        "hit",
        "input_gt_rank",
        "input_gt_sim",
        "input_neg_sim_max",
        "input_gt_margin_vs_best_neg",
        "input_semantic_top_option_idx",
        "input_semantic_top_item",
        "pred_input_rank",
        "gt_to_distractor_sim_max",
    ]
    for label, sub, sort_col, ascending in example_specs:
        if sub.empty:
            continue
        sample = sub.sort_values(sort_col, ascending=ascending).head(10).copy()
        sample.insert(0, "example_type", label)
        frames.append(sample[["example_type"] + keep_cols])
    if not frames:
        return pd.DataFrame(columns=["example_type"] + keep_cols)
    return pd.concat(frames, ignore_index=True)


def parse_result_override(values):
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid --result value: {value}. Use dataset=path")
        dataset, path = value.split("=", 1)
        overrides[dataset.strip()] = path.strip()
    return overrides


def write_combined_outputs(repo_root, output_root, model, results):
    all_rows = []
    for result in results:
        summary = result["summary"].copy()
        summary.insert(0, "dataset", result["dataset"])
        all_rows.append(summary)
    combined = pd.concat(all_rows, ignore_index=True)

    out_dir = repo_root / output_root / model
    combined_path = out_dir / "combined_openai_embedding_summary.csv"
    combined.to_csv(combined_path, index=False, quoting=csv.QUOTE_MINIMAL)

    all_split = combined[combined["split"] == "all"].copy()
    md_path = out_dir / "summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# OpenAI Embedding Semantic Signal Summary\n\n")
        f.write("Model: `text-embedding-3-large`\n\n")
        f.write("| dataset | n | LLM hit | GT top1 | GT top3 | MRR | rank mean | GT sim | neg mean | margin vs best neg | pred=semantic top1 |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in all_split.iterrows():
            f.write(
                f"| {row['dataset']} | {int(row['n'])} | {row['llm_hit_rate']:.3f} | "
                f"{row['input_gt_top1_rate']:.3f} | {row['input_gt_top3_rate']:.3f} | "
                f"{row['input_gt_mrr']:.3f} | {row['input_gt_rank_mean']:.2f} | "
                f"{row['input_gt_sim_mean']:.3f} | {row['input_neg_sim_mean']:.3f} | "
                f"{row['input_gt_margin_vs_best_neg_mean']:.3f} | "
                f"{row['pred_is_input_semantic_top1_rate']:.3f} |\n"
            )
    return combined_path, md_path, combined


def main():
    parser = argparse.ArgumentParser(description="Analyze semantic signal with cached OpenAI item embeddings.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_RESULT_FILES), choices=list(DEFAULT_RESULT_FILES))
    parser.add_argument("--result", action="append", default=[], help="Override result CSV as dataset=path.")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    result_files = dict(DEFAULT_RESULT_FILES)
    result_files.update(parse_result_override(args.result))

    results = []
    for dataset in args.datasets:
        result = analyze_one_dataset(repo_root, dataset, result_files[dataset], args)
        results.append(result)
        print(f"\n=== {dataset} ===")
        print(f"detail  : {result['detail_path']}")
        print(f"summary : {result['summary_path']}")
        print(f"examples: {result['examples_path']}")
        print(f"sanity  : {result['sanity_path']}")
        print(result["summary"].to_string(index=False))

    combined_path, md_path, combined = write_combined_outputs(repo_root, Path(args.output_root), args.model, results)
    print("\n=== combined ===")
    print(f"summary csv: {combined_path}")
    print(f"summary md : {md_path}")
    print(combined[combined["split"] == "all"].to_string(index=False))


if __name__ == "__main__":
    main()

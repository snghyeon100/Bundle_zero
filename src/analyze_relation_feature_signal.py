import argparse
import ast
import csv
import io
import json
import math
import os
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULT_FILES = {
    "pog": r"results\pog\results_pog_20260416_142034.csv",
    "pog_dense": r"results\pog_dense\results_pog_dense_HN_C10_T5_20260430_172343.csv",
    "spotify": r"results\spotify\results_spotify_20260411_191411.csv",
    "spotify_sparse": r"results\spotify_sparse\results_spotify_sparse_20260411_195706.csv",
}


FEATURE_FILES = {
    "item_cf": "item_cf_feature.pt",
    "bi_lgcn": "{dataset}_LightGCN_bi_feature.pt",
}


class _FakeStorageType:
    pass


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    return {
        "storage": storage,
        "storage_offset": storage_offset,
        "size": tuple(size),
        "stride": tuple(stride),
        "requires_grad": requires_grad,
    }


class _TorchMetaUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name in {"FloatStorage", "DoubleStorage", "HalfStorage"}:
            return _FakeStorageType
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")

    def persistent_load(self, pid):
        return {
            "kind": pid[0],
            "storage_type": getattr(pid[1], "__name__", str(pid[1])),
            "key": pid[2],
            "location": pid[3],
            "numel": pid[4],
        }


def read_torch_float_tensor(path):
    with zipfile.ZipFile(path) as zf:
        data_pkl = [name for name in zf.namelist() if name.endswith("data.pkl")][0]
        meta = _TorchMetaUnpickler(io.BytesIO(zf.read(data_pkl))).load()
        storage_name = f"{data_pkl.rsplit('/', 1)[0]}/data/{meta['storage']['key']}"
        raw = zf.read(storage_name)
    arr = np.frombuffer(raw, dtype="<f4")
    expected = math.prod(meta["size"])
    offset = int(meta.get("storage_offset", 0))
    size = tuple(meta["size"])
    stride = tuple(meta["stride"])
    if arr.size == expected and offset == 0 and stride == (size[1], 1):
        return arr.reshape(size), meta
    max_index = offset
    for dim, st in zip(size, stride):
        max_index += (dim - 1) * st
    if max_index >= arr.size:
        raise ValueError(f"{path}: tensor view exceeds storage ({max_index} >= {arr.size})")
    view = np.lib.stride_tricks.as_strided(
        arr[offset:],
        shape=size,
        strides=tuple(st * arr.dtype.itemsize for st in stride),
    )
    return np.array(view, copy=True), meta


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


def normalize_rows(features):
    features = features.astype(np.float32, copy=False)
    norms = np.linalg.norm(features, axis=1)
    normalized = np.zeros_like(features, dtype=np.float32)
    safe = norms > 0
    normalized[safe] = features[safe] / norms[safe, None]
    return normalized, norms


def stable_desc_rank(scores, true_idx):
    order = np.argsort(-scores, kind="mergesort")
    return int(np.where(order == true_idx)[0][0] + 1), order


def feature_path_for(dataset_dir, dataset, feature):
    pattern = FEATURE_FILES[feature]
    return dataset_dir / pattern.format(dataset=dataset)


def analyze_dataset(repo_root, dataset, result_csv, feature, args):
    dataset_dir = repo_root / args.data_path / dataset
    feature_path = feature_path_for(dataset_dir, dataset, feature)
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    features, meta = read_torch_float_tensor(feature_path)
    features, norms = normalize_rows(features)

    with open(dataset_dir / "count.json", encoding="utf-8") as f:
        count = json.load(f)
    if features.shape[0] != count["#I"]:
        raise ValueError(f"{dataset}/{feature}: rows {features.shape[0]} != #I {count['#I']}")

    result_csv = Path(result_csv)
    if not result_csv.is_absolute():
        result_csv = repo_root / result_csv
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

        valid_input_ids = [item_id for item_id in input_ids if 0 <= item_id < features.shape[0] and norms[item_id] > 0]
        if len(valid_input_ids) == 0:
            rows.append({"row_idx": row_idx, "bundle_id": int(row["bundle_id"]), "skipped": 1, "skip_reason": "no valid input embeddings"})
            continue
        if any(item_id < 0 or item_id >= features.shape[0] for item_id in candidate_ids):
            rows.append({"row_idx": row_idx, "bundle_id": int(row["bundle_id"]), "skipped": 1, "skip_reason": "candidate id out of range"})
            continue

        input_vecs = features[valid_input_ids]
        cand_vecs = features[candidate_ids]
        bundle_vec = input_vecs.mean(axis=0)
        bundle_norm = np.linalg.norm(bundle_vec)
        if bundle_norm == 0:
            rows.append({"row_idx": row_idx, "bundle_id": int(row["bundle_id"]), "skipped": 1, "skip_reason": "zero input bundle embedding"})
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
        gt_pair_mean_rank, _ = stable_desc_rank(input_pair_mean_to_candidates, true_option_idx)
        gt_pair_max_rank, _ = stable_desc_rank(input_pair_max_to_candidates, true_option_idx)
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
                "num_valid_input_embeddings": len(valid_input_ids),
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
            }
        )

    detail = pd.DataFrame(rows)
    usable = detail[detail["skipped"] == 0].copy()
    if usable.empty:
        raise ValueError(f"{dataset}/{feature}: no usable rows")

    summary = make_summary(usable)
    sanity = {
        "dataset": dataset,
        "feature": feature,
        "feature_path": str(feature_path),
        "result_csv": str(result_csv),
        "shape": list(features.shape),
        "stride": list(meta["stride"]),
        "result_rows": int(len(df)),
        "usable_rows": int(len(usable)),
        "skipped_rows": int((detail["skipped"] == 1).sum()),
        "zero_norm_rows": int((norms == 0).sum()),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
    }

    out_dir = repo_root / args.output_root / feature / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = result_csv.stem
    detail_path = out_dir / f"{stem}_{feature}_detail.csv"
    summary_path = out_dir / f"{stem}_{feature}_summary.csv"
    sanity_path = out_dir / f"{stem}_{feature}_sanity.json"
    detail.to_csv(detail_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(sanity, f, ensure_ascii=False, indent=2)
    return {
        "dataset": dataset,
        "feature": feature,
        "detail_path": detail_path,
        "summary_path": summary_path,
        "sanity_path": sanity_path,
        "summary": summary,
        "sanity": sanity,
    }


def make_summary(usable):
    rows = []
    splits = [("all", usable), ("llm_hit", usable[usable["hit"] == 1]), ("llm_miss", usable[usable["hit"] == 0])]
    for split, sub in splits:
        if sub.empty:
            continue
        rows.append(
            {
                "split": split,
                "n": int(len(sub)),
                "llm_hit_rate": float(sub["hit"].mean()),
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


def parse_result_override(values):
    overrides = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid --result value: {value}. Use dataset=path")
        dataset, path = value.split("=", 1)
        overrides[dataset.strip()] = path.strip()
    return overrides


def write_combined(repo_root, output_root, results):
    rows = []
    for result in results:
        summary = result["summary"].copy()
        summary.insert(0, "feature", result["feature"])
        summary.insert(0, "dataset", result["dataset"])
        rows.append(summary)
    combined = pd.concat(rows, ignore_index=True)
    out_dir = repo_root / output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "combined_relation_feature_summary.csv"
    combined.to_csv(combined_path, index=False, quoting=csv.QUOTE_MINIMAL)

    md_path = out_dir / "summary.md"
    all_rows = combined[combined["split"] == "all"].copy()
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Relation Feature Signal Summary\n\n")
        f.write("| dataset | feature | n | LLM hit | GT top1 | GT top3 | MRR | rank mean | GT sim | neg mean | margin vs best neg | pred=feature top1 |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in all_rows.iterrows():
            f.write(
                f"| {row['dataset']} | {row['feature']} | {int(row['n'])} | {row['llm_hit_rate']:.3f} | "
                f"{row['input_gt_top1_rate']:.3f} | {row['input_gt_top3_rate']:.3f} | "
                f"{row['input_gt_mrr']:.3f} | {row['input_gt_rank_mean']:.2f} | "
                f"{row['input_gt_sim_mean']:.3f} | {row['input_neg_sim_mean']:.3f} | "
                f"{row['input_gt_margin_vs_best_neg_mean']:.3f} | {row['pred_is_input_semantic_top1_rate']:.3f} |\n"
            )
    return combined_path, md_path, combined


def main():
    parser = argparse.ArgumentParser(description="Analyze item_cf_feature.pt and LightGCN BI feature signal.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_RESULT_FILES), choices=list(DEFAULT_RESULT_FILES))
    parser.add_argument("--features", nargs="+", default=list(FEATURE_FILES), choices=list(FEATURE_FILES))
    parser.add_argument("--result", action="append", default=[], help="Override result CSV as dataset=path.")
    parser.add_argument("--data-path", default="datasets")
    parser.add_argument("--output-root", default=r"analysis\relation_feature_signal")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    result_files = dict(DEFAULT_RESULT_FILES)
    result_files.update(parse_result_override(args.result))

    results = []
    for feature in args.features:
        for dataset in args.datasets:
            result = analyze_dataset(repo_root, dataset, result_files[dataset], feature, args)
            results.append(result)
            print(f"\n=== {dataset} / {feature} ===")
            print(f"detail : {result['detail_path']}")
            print(f"summary: {result['summary_path']}")
            print(f"sanity : {result['sanity_path']}")
            print(result["summary"].to_string(index=False))

    combined_path, md_path, combined = write_combined(repo_root, Path(args.output_root), results)
    print("\n=== combined ===")
    print(f"summary csv: {combined_path}")
    print(f"summary md : {md_path}")
    print(combined[combined["split"] == "all"].to_string(index=False))


if __name__ == "__main__":
    main()

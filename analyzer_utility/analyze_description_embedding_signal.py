import argparse
import ast
import csv
import hashlib
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


DATASETS = ("spotify_sparse", "spotify", "pog_dense", "pog_dedup", "pog")


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
        if module == "torch" and name == "FloatStorage":
            return _FakeStorageType
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        raise pickle.UnpicklingError(f"Unsupported pickle global: {module}.{name}")

    def persistent_load(self, pid):
        # PyTorch storage persistent id:
        # ("storage", StorageType, key, location, numel)
        return {
            "kind": pid[0],
            "storage_type": getattr(pid[1], "__name__", str(pid[1])),
            "key": pid[2],
            "location": pid[3],
            "numel": pid[4],
        }


def infer_dataset(path):
    base = os.path.basename(str(path)).lower()
    parent = os.path.basename(os.path.dirname(str(path))).lower()
    joined = f"{parent}/{base}"
    for dataset in DATASETS:
        if dataset in joined:
            return dataset
    return None


def read_torch_float_tensor(path):
    """Read a simple torch.save(float32 Tensor) file without importing torch."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        data_pkl = [name for name in zf.namelist() if name.endswith("data.pkl")][0]
        meta = _TorchMetaUnpickler(io.BytesIO(zf.read(data_pkl))).load()
        storage_key = meta["storage"]["key"]
        storage_name = f"{data_pkl.rsplit('/', 1)[0]}/data/{storage_key}"
        raw = zf.read(storage_name)

    arr = np.frombuffer(raw, dtype="<f4")
    expected = math.prod(meta["size"])
    if arr.size != expected:
        raise ValueError(f"{path}: storage has {arr.size} float32 values, expected {expected}")
    return arr.reshape(meta["size"]), meta


def sha256_storage(path):
    with zipfile.ZipFile(path) as zf:
        data_pkl = [name for name in zf.namelist() if name.endswith("data.pkl")][0]
        meta = _TorchMetaUnpickler(io.BytesIO(zf.read(data_pkl))).load()
        storage_name = f"{data_pkl.rsplit('/', 1)[0]}/data/{meta['storage']['key']}"
        h = hashlib.sha256()
        with zf.open(storage_name) as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()


def parse_id_list(value):
    if isinstance(value, list):
        return [int(x) for x in value]
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    return [int(x) for x in parsed]


def normalize_rows(features):
    features = features.astype(np.float32, copy=False)
    norms = np.linalg.norm(features, axis=1)
    safe = norms > 0
    normalized = np.zeros_like(features, dtype=np.float32)
    normalized[safe] = features[safe] / norms[safe, None]
    return normalized, norms


def option_idx_from_prediction(prediction):
    if pd.isna(prediction):
        return None
    text = str(prediction).strip().upper()
    if len(text) == 1 and "A" <= text <= "Z":
        return ord(text) - ord("A")
    return None


def describe_features(dataset_dir, features, norms, meta):
    adjacent_equal = int(np.all(features[:-1] == features[1:], axis=1).sum())
    norm_q = np.quantile(norms, [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "shape": tuple(features.shape),
        "stride": meta["stride"],
        "finite_ratio": float(np.isfinite(features).mean()),
        "zero_norm_rows": int((norms == 0).sum()),
        "zero_norm_first_ids": np.where(norms == 0)[0][:20].astype(int).tolist(),
        "norm_quantiles_0_1_5_50_95_99_100": [float(x) for x in norm_q],
        "adjacent_exact_duplicate_pairs": adjacent_equal,
        "adjacent_exact_duplicate_ratio": adjacent_equal / max(1, features.shape[0] - 1),
        "same_storage_as_content_feature": (
            sha256_storage(dataset_dir / "description_feature.pt") == sha256_storage(dataset_dir / "content_feature.pt")
            if (dataset_dir / "content_feature.pt").exists()
            else None
        ),
    }


def analyze_results(results_path, dataset, data_path, output_dir):
    dataset_dir = Path(data_path) / dataset
    features, meta = read_torch_float_tensor(dataset_dir / "description_feature.pt")
    normed, norms = normalize_rows(features)

    with open(dataset_dir / "count.json", encoding="utf-8") as f:
        count = json.load(f)
    if features.shape[0] != count["#I"]:
        raise ValueError(f"Feature rows {features.shape[0]} != count #I {count['#I']}")

    df = pd.read_csv(results_path)
    rows = []
    for row_idx, row in df.iterrows():
        input_ids = parse_id_list(row["input_indices"])
        candidate_ids = parse_id_list(row["candidate_indices"])
        true_item = int(row["true_indice"])
        true_pos = int(row["true_option_idx"]) if "true_option_idx" in row and not pd.isna(row["true_option_idx"]) else candidate_ids.index(true_item)

        input_ids_valid = [i for i in input_ids if 0 <= i < normed.shape[0] and norms[i] > 0]
        candidate_ids_valid = [i for i in candidate_ids if 0 <= i < normed.shape[0]]
        if not input_ids_valid or len(candidate_ids_valid) != len(candidate_ids):
            continue

        input_vec = normed[input_ids_valid].mean(axis=0)
        input_norm = np.linalg.norm(input_vec)
        if input_norm == 0:
            continue
        input_vec = input_vec / input_norm

        cand_vecs = normed[candidate_ids]
        input_to_candidates = cand_vecs @ input_vec
        gt_sim = float(input_to_candidates[true_pos])
        neg_mask = np.ones(len(candidate_ids), dtype=bool)
        neg_mask[true_pos] = False
        neg_sims = input_to_candidates[neg_mask]

        # Stable descending rank: rank 1 means highest similarity.
        order = np.argsort(-input_to_candidates, kind="mergesort")
        gt_rank = int(np.where(order == true_pos)[0][0] + 1)
        pred_idx = option_idx_from_prediction(row.get("prediction"))
        pred_sim = float(input_to_candidates[pred_idx]) if pred_idx is not None and pred_idx < len(candidate_ids) else np.nan

        gt_vec = normed[true_item]
        gt_to_candidates = cand_vecs @ gt_vec
        gt_to_neg = gt_to_candidates[neg_mask]

        rows.append(
            {
                "row_idx": row_idx,
                "bundle_id": int(row["bundle_id"]),
                "true_indice": true_item,
                "num_input_items": len(input_ids),
                "num_valid_input_embeddings": len(input_ids_valid),
                "gt_input_sim": gt_sim,
                "neg_input_sim_mean": float(neg_sims.mean()),
                "neg_input_sim_max": float(neg_sims.max()),
                "gt_margin_vs_best_neg": float(gt_sim - neg_sims.max()),
                "gt_margin_vs_neg_mean": float(gt_sim - neg_sims.mean()),
                "gt_semantic_rank": gt_rank,
                "gt_semantic_top1": int(gt_rank == 1),
                "gt_semantic_mrr": 1.0 / gt_rank,
                "semantic_top_option_idx": int(order[0]),
                "semantic_top_item": int(candidate_ids[order[0]]),
                "semantic_top_sim": float(input_to_candidates[order[0]]),
                "pred_option_idx": pred_idx,
                "pred_input_sim": pred_sim,
                "pred_is_semantic_top1": int(pred_idx == int(order[0])) if pred_idx is not None else np.nan,
                "hit": int(row["hit"]) if "hit" in row and not pd.isna(row["hit"]) else np.nan,
                "gt_to_distractor_sim_mean": float(gt_to_neg.mean()),
                "gt_to_distractor_sim_max": float(gt_to_neg.max()),
            }
        )

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError(f"No analyzable rows in {results_path}")

    summary_rows = []
    for label, sub in [("all", detail), ("llm_hit", detail[detail["hit"] == 1]), ("llm_miss", detail[detail["hit"] == 0])]:
        if sub.empty:
            continue
        summary_rows.append(
            {
                "split": label,
                "n": len(sub),
                "llm_hit_rate": float(sub["hit"].mean()) if "hit" in sub else np.nan,
                "semantic_top1_rate": float(sub["gt_semantic_top1"].mean()),
                "semantic_mrr": float(sub["gt_semantic_mrr"].mean()),
                "gt_rank_mean": float(sub["gt_semantic_rank"].mean()),
                "gt_input_sim_mean": float(sub["gt_input_sim"].mean()),
                "neg_input_sim_mean": float(sub["neg_input_sim_mean"].mean()),
                "neg_input_sim_max_mean": float(sub["neg_input_sim_max"].mean()),
                "gt_margin_vs_best_neg_mean": float(sub["gt_margin_vs_best_neg"].mean()),
                "gt_margin_vs_neg_mean_mean": float(sub["gt_margin_vs_neg_mean"].mean()),
                "pred_is_semantic_top1_rate": float(sub["pred_is_semantic_top1"].dropna().mean()),
                "gt_to_distractor_sim_mean": float(sub["gt_to_distractor_sim_mean"].mean()),
                "gt_to_distractor_sim_max_mean": float(sub["gt_to_distractor_sim_max"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(results_path).stem
    detail_path = out_dir / f"{stem}_description_embedding_detail.csv"
    summary_path = out_dir / f"{stem}_description_embedding_summary.csv"
    sanity_path = out_dir / f"{dataset}_description_feature_sanity.json"
    detail.to_csv(detail_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    sanity = describe_features(dataset_dir, features, norms, meta)
    sanity["dataset"] = dataset
    sanity["count_num_items"] = count["#I"]
    sanity["results_path"] = str(results_path)
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(sanity, f, ensure_ascii=False, indent=2)

    return detail_path, summary_path, sanity_path, summary, sanity


def main():
    parser = argparse.ArgumentParser(description="Analyze semantic signal in description_feature.pt for zero-shot bundle results.")
    parser.add_argument("--results", nargs="+", required=True, help="Result CSV file(s) containing input_indices and candidate_indices.")
    parser.add_argument("--dataset", default=None, help="Dataset name. Inferred from each result path when omitted.")
    parser.add_argument("--data-path", default="./datasets")
    parser.add_argument("--output-dir", default=None, help="Default: analysis/<dataset>_description_embedding_signal")
    args = parser.parse_args()

    for results_path in args.results:
        dataset = args.dataset or infer_dataset(results_path)
        if not dataset:
            raise ValueError(f"Could not infer dataset from {results_path}; pass --dataset.")
        output_dir = args.output_dir or os.path.join("analysis", f"{dataset}_description_embedding_signal")
        detail_path, summary_path, sanity_path, summary, sanity = analyze_results(results_path, dataset, args.data_path, output_dir)
        print(f"\n=== {dataset}: {results_path} ===")
        print(f"detail : {detail_path}")
        print(f"summary: {summary_path}")
        print(f"sanity : {sanity_path}")
        print(summary.to_string(index=False))
        print("feature_sanity:", json.dumps(sanity, ensure_ascii=False))


if __name__ == "__main__":
    main()

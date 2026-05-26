import argparse
import csv
import io
import json
import math
import os
import pickle
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


class _FakeStorageType:
    pass


_FAKE_STORAGE_TYPES = {
    name: type(name, (_FakeStorageType,), {})
    for name in ("FloatStorage", "HalfStorage", "DoubleStorage")
}


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
        if module == "torch" and name in _FAKE_STORAGE_TYPES:
            return _FAKE_STORAGE_TYPES[name]
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
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        data_pkl = [name for name in zf.namelist() if name.endswith("data.pkl")][0]
        meta = _TorchMetaUnpickler(io.BytesIO(zf.read(data_pkl))).load()
        storage_key = meta["storage"]["key"]
        storage_name = f"{data_pkl.rsplit('/', 1)[0]}/data/{storage_key}"
        raw = zf.read(storage_name)

    storage_type = meta["storage"]["storage_type"]
    dtype = {
        "FloatStorage": "<f4",
        "HalfStorage": "<f2",
        "DoubleStorage": "<f8",
    }.get(storage_type)
    if dtype is None:
        raise ValueError(f"{path}: unsupported storage type {storage_type}")

    arr = np.frombuffer(raw, dtype=dtype).astype(np.float32, copy=False)
    expected = math.prod(meta["size"])
    if arr.size != expected:
        raise ValueError(f"{path}: storage has {arr.size} values, expected {expected}")
    return arr.reshape(meta["size"])


def normalize_rows(matrix):
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1)
    out = np.zeros_like(matrix, dtype=np.float32)
    safe = norms > 0
    out[safe] = matrix[safe] / norms[safe, None]
    return out


def parse_bundle_file(path):
    bundles = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = [int(x) for x in line.strip().split(", ") if x.strip()]
            if not parts:
                continue
            bundles[parts[0]] = parts[1:]
    return bundles


def detect_category_field(item_info):
    candidates = ("cate", "cate_id", "category", "category_id")
    counts = {field: 0 for field in candidates}
    for item in item_info.values():
        for field in candidates:
            value = item.get(field)
            if value is not None and str(value).strip():
                counts[field] += 1
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise ValueError("No category field found in item_info.json")
    return best


def load_item_categories(dataset_dir, num_items):
    with open(dataset_dir / "item_info.json", encoding="utf-8") as f:
        item_info = json.load(f)
    category_field = detect_category_field(item_info)
    item_to_category = {}
    category_to_items = defaultdict(list)
    for item_id_str, item in item_info.items():
        item_id = int(item_id_str)
        if not 0 <= item_id < num_items:
            continue
        category = item.get(category_field)
        if category is None or not str(category).strip():
            continue
        category = str(category).strip()
        item_to_category[item_id] = category
        category_to_items[category].append(item_id)
    return item_to_category, category_to_items, category_field


def build_samples(dataset_dir, num_items, num_cans, num_token, seed, shuffle_seed, toy_eval):
    input_bundles = parse_bundle_file(dataset_dir / "bi_test_input.txt")
    gt_bundles = parse_bundle_file(dataset_dir / "bi_test_gt.txt")
    all_items = np.arange(num_items)
    samples = []

    for bundle_id in sorted(gt_bundles):
        input_items_full = np.asarray(input_bundles.get(bundle_id, []), dtype=np.int64)
        gt_items = np.asarray(gt_bundles[bundle_id], dtype=np.int64)
        occupied = np.zeros(num_items, dtype=bool)
        occupied[input_items_full] = True
        occupied[gt_items] = True
        false_pool = all_items[~occupied]

        for true_item in gt_items:
            rng_cand = np.random.default_rng(int(bundle_id) + seed)
            false_indices = rng_cand.choice(false_pool, size=num_cans - 1, replace=False)
            candidate_ids = np.concatenate([[int(true_item)], false_indices.astype(np.int64)])
            rng_cand.shuffle(candidate_ids)
            true_pos = int(np.argwhere(candidate_ids == int(true_item))[0][0])

            rng_input = np.random.default_rng(int(bundle_id) + shuffle_seed)
            input_ids = input_items_full.copy()
            rng_input.shuffle(input_ids)
            if num_token > 0 and len(input_ids) > num_token:
                input_ids = input_ids[:num_token]

            samples.append({
                "bundle_id": int(bundle_id),
                "input_ids": input_ids.astype(int).tolist(),
                "candidate_ids": candidate_ids.astype(int).tolist(),
                "true_item": int(true_item),
                "true_pos": true_pos,
            })
            if toy_eval > 0 and len(samples) >= toy_eval:
                return samples
    return samples


def load_existing_category_embeddings(repo_root, dataset, embedding_name, dtype="float32"):
    path = (
        repo_root
        / "analysis"
        / "category_embedding_cache"
        / embedding_name
        / "all_items"
        / dataset
        / f"category_embeddings_{embedding_name}_{dtype}.npz"
    )
    with np.load(path, allow_pickle=True) as data:
        category_ids = [str(x) for x in data["category_ids"].tolist()]
        embeddings = data["embeddings_normed"].astype(np.float32)
    return path, category_ids, embeddings


def build_content_category_embeddings(dataset_dir, category_to_items):
    features = read_torch_float_tensor(dataset_dir / "content_feature.pt")
    category_ids = sorted(category_to_items)
    embeddings = np.zeros((len(category_ids), features.shape[1]), dtype=np.float32)
    counts = np.zeros(len(category_ids), dtype=np.int64)
    for row, category in enumerate(category_ids):
        item_ids = np.asarray(category_to_items[category], dtype=np.int64)
        item_ids = item_ids[(0 <= item_ids) & (item_ids < features.shape[0])]
        counts[row] = len(item_ids)
        if len(item_ids):
            embeddings[row] = features[item_ids].mean(axis=0)
    return category_ids, normalize_rows(embeddings), counts


def align_category_matrix(category_ids, embeddings):
    return {category: idx for idx, category in enumerate(category_ids)}, embeddings


def category_list(item_ids, item_to_category):
    return [item_to_category.get(int(item_id)) for item_id in item_ids]


def analyze_space(dataset, space_name, category_ids, embeddings, samples, item_to_category):
    cat_to_row, emb = align_category_matrix(category_ids, embeddings)
    rows = []
    skipped = 0
    for sample_idx, sample in enumerate(samples):
        input_categories = [c for c in category_list(sample["input_ids"], item_to_category) if c in cat_to_row]
        candidate_categories = category_list(sample["candidate_ids"], item_to_category)
        if not input_categories:
            skipped += 1
            continue

        input_rows = np.asarray([cat_to_row[c] for c in input_categories], dtype=np.int64)
        query = emb[input_rows].mean(axis=0)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            skipped += 1
            continue
        query = query / query_norm

        scores = []
        valid_candidate = []
        for category in candidate_categories:
            if category in cat_to_row:
                scores.append(float(emb[cat_to_row[category]] @ query))
                valid_candidate.append(True)
            else:
                scores.append(float("nan"))
                valid_candidate.append(False)

        true_pos = sample["true_pos"]
        if true_pos >= len(scores) or not valid_candidate[true_pos]:
            skipped += 1
            continue
        gt_cos = scores[true_pos]
        distractor_scores = [
            score for idx, score in enumerate(scores)
            if idx != true_pos and valid_candidate[idx] and np.isfinite(score)
        ]
        if not distractor_scores:
            skipped += 1
            continue

        finite_scores = [score for score, valid in zip(scores, valid_candidate) if valid and np.isfinite(score)]
        gt_rank = 1 + sum(score > gt_cos for score in finite_scores)
        rows.append({
            "dataset": dataset,
            "space": space_name,
            "sample_idx": sample_idx,
            "bundle_id": sample["bundle_id"],
            "true_item": sample["true_item"],
            "true_option_char": chr(ord("A") + true_pos),
            "input_categories": "|".join(str(c) for c in category_list(sample["input_ids"], item_to_category)),
            "gt_category": candidate_categories[true_pos],
            "gt_cos": gt_cos,
            "distractor_mean_cos": float(np.mean(distractor_scores)),
            "distractor_max_cos": float(np.max(distractor_scores)),
            "gt_rank": int(gt_rank),
            "num_valid_candidates": int(sum(valid_candidate)),
        })
    return rows, skipped


def summarize(detail_df):
    grouped = detail_df.groupby(["dataset", "space"], as_index=False)
    summary = grouped.agg(
        samples=("gt_cos", "count"),
        gt_cos_mean=("gt_cos", "mean"),
        gt_cos_std=("gt_cos", "std"),
        distractor_mean_cos_mean=("distractor_mean_cos", "mean"),
        distractor_max_cos_mean=("distractor_max_cos", "mean"),
        gt_minus_distractor_mean=("gt_cos", lambda s: np.nan),
        gt_rank_mean=("gt_rank", "mean"),
        hit_at_1=("gt_rank", lambda s: float((s == 1).mean())),
        hit_at_3=("gt_rank", lambda s: float((s <= 3).mean())),
    )
    for idx, row in summary.iterrows():
        mask = (detail_df["dataset"] == row["dataset"]) & (detail_df["space"] == row["space"])
        subset = detail_df.loc[mask]
        summary.loc[idx, "gt_minus_distractor_mean"] = (
            subset["gt_cos"] - subset["distractor_mean_cos"]
        ).mean()
        summary.loc[idx, "gt_minus_distractor_max"] = (
            subset["gt_cos"] - subset["distractor_max_cos"]
        ).mean()
    return summary


def analyze_dataset(repo_root, dataset, args):
    dataset_dir = repo_root / "datasets" / dataset
    with open(dataset_dir / "count.json", encoding="utf-8") as f:
        count = json.load(f)
    num_items = int(count["#I"])
    item_to_category, category_to_items, category_field = load_item_categories(dataset_dir, num_items)
    samples = build_samples(
        dataset_dir,
        num_items,
        args.num_cans,
        args.num_token,
        args.seed,
        args.shuffle_seed,
        args.toy_eval,
    )

    spaces = []
    text_path, text_categories, text_embeddings = load_existing_category_embeddings(
        repo_root, dataset, "text-embedding-3-large"
    )
    spaces.append(("text", text_categories, text_embeddings, text_path))

    bi_path, bi_categories, bi_embeddings = load_existing_category_embeddings(
        repo_root, dataset, "LightGCN_bi"
    )
    spaces.append(("bi", bi_categories, bi_embeddings, bi_path))

    content_categories, content_embeddings, content_counts = build_content_category_embeddings(
        dataset_dir, category_to_items
    )
    spaces.append(("content", content_categories, content_embeddings, dataset_dir / "content_feature.pt"))

    all_rows = []
    meta_rows = []
    for space_name, category_ids, embeddings, source_path in spaces:
        rows, skipped = analyze_space(dataset, space_name, category_ids, embeddings, samples, item_to_category)
        all_rows.extend(rows)
        meta_rows.append({
            "dataset": dataset,
            "space": space_name,
            "source": str(source_path),
            "category_field": category_field,
            "num_categories": len(category_ids),
            "embedding_dim": embeddings.shape[1],
            "samples_built": len(samples),
            "samples_used": len(rows),
            "samples_skipped": skipped,
        })
    return all_rows, meta_rows


def main():
    parser = argparse.ArgumentParser(
        description="Analyze input-vs-GT category cosine using text, BI, and content category embeddings."
    )
    parser.add_argument("--datasets", nargs="+", default=["pog", "pog_dense"])
    parser.add_argument("--num-cans", type=int, default=10)
    parser.add_argument("--num-token", type=int, default=5)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--shuffle-seed", type=int, default=41)
    parser.add_argument("--toy-eval", type=int, default=-1)
    parser.add_argument("--output-dir", default=os.path.join("analysis", "category_embedding_input_gt"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_meta = []
    for dataset in args.datasets:
        rows, meta = analyze_dataset(repo_root, dataset, args)
        all_rows.extend(rows)
        all_meta.extend(meta)

    detail_df = pd.DataFrame(all_rows)
    summary_df = summarize(detail_df)
    meta_df = pd.DataFrame(all_meta)

    detail_path = output_dir / "category_embedding_input_gt_detail.csv"
    summary_path = output_dir / "category_embedding_input_gt_summary.csv"
    meta_path = output_dir / "category_embedding_input_gt_meta.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")

    print(f"[Done] detail: {detail_path}")
    print(f"[Done] summary: {summary_path}")
    print(f"[Done] meta: {meta_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

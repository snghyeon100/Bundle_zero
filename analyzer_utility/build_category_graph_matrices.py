import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_count(dataset_dir):
    with open(os.path.join(dataset_dir, "count.json"), "r", encoding="utf-8") as f:
        stat = json.load(f)
    return int(stat["#B"]), int(stat["#I"])


def detect_category_field(item_info):
    field_candidates = ["cate_id", "cate", "category"]
    counts = {field: 0 for field in field_candidates}
    for item in item_info.values():
        for field in field_candidates:
            value = item.get(field)
            if value is not None and str(value).strip():
                counts[field] += 1
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise ValueError("No category field found in item_info.json")
    return best


def load_item_info(dataset_dir):
    with open(os.path.join(dataset_dir, "item_info.json"), "r", encoding="utf-8") as f:
        item_info = json.load(f)
    return item_info, detect_category_field(item_info)


def build_ic(item_info, category_field, num_items):
    item_categories = {}
    category_counts = Counter()
    for item_id_str, item in item_info.items():
        item_id = int(item_id_str)
        if not 0 <= item_id < num_items:
            continue
        value = item.get(category_field)
        if value is None or not str(value).strip():
            continue
        category = str(value).strip()
        item_categories[item_id] = category
        category_counts[category] += 1

    category_ids = sorted(category_counts)
    category_to_col = {category: idx for idx, category in enumerate(category_ids)}
    rows = []
    cols = []
    for item_id, category in item_categories.items():
        rows.append(item_id)
        cols.append(category_to_col[category])

    values = np.ones(len(rows), dtype=np.float32)
    ic = sp.csr_matrix(
        (values, (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(num_items, len(category_ids)),
        dtype=np.float32,
    )
    return ic, category_ids, category_counts


def read_bi(path, num_bundles, num_items):
    rows = []
    cols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = [int(v) for v in line.strip().split(", ") if v]
            if len(vals) < 2:
                continue
            bundle_id = vals[0]
            if not 0 <= bundle_id < num_bundles:
                continue
            for item_id in vals[1:]:
                if 0 <= item_id < num_items:
                    rows.append(bundle_id)
                    cols.append(item_id)

    values = np.ones(len(rows), dtype=np.float32)
    bi = sp.csr_matrix(
        (values, (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(num_bundles, num_items),
        dtype=np.float32,
    )
    bi.sum_duplicates()
    return bi


def binarize(matrix):
    binary = matrix.copy().tocsr()
    binary.data[:] = 1.0
    binary.eliminate_zeros()
    return binary


def top_pairs_to_csv(c_matrix, category_ids, path, top_k=200):
    coo = c_matrix.tocoo()
    rows = []
    for row, col, value in zip(coo.row, coo.col, coo.data):
        if row >= col or value <= 0:
            continue
        rows.append((category_ids[int(row)], category_ids[int(col)], float(value)))
    rows.sort(key=lambda x: (-x[2], x[0], x[1]))
    pd.DataFrame(
        rows[:top_k],
        columns=["category_a", "category_b", "cooccur_weight"],
    ).to_csv(path, index=False, encoding="utf-8-sig")


def save_category_mapping(category_ids, category_counts, path):
    rows = [
        {
            "category_col": idx,
            "category_id": category,
            "item_count": int(category_counts[category]),
        }
        for idx, category in enumerate(category_ids)
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_category_frequency(bc_count, bc_binary, category_ids, path):
    count_sum = np.asarray(bc_count.sum(axis=0)).ravel()
    bundle_freq = np.asarray(bc_binary.sum(axis=0)).ravel()
    rows = [
        {
            "category_col": idx,
            "category_id": category,
            "category_item_occurrences": float(count_sum[idx]),
            "category_bundle_frequency": int(bundle_freq[idx]),
        }
        for idx, category in enumerate(category_ids)
    ]
    pd.DataFrame(rows).sort_values(
        ["category_bundle_frequency", "category_id"],
        ascending=[False, True],
    ).to_csv(path, index=False, encoding="utf-8-sig")


def build_for_split(dataset_dir, split_file, split_name, ic, category_ids, output_dir):
    num_bundles, num_items = read_count(dataset_dir)
    split_path = os.path.join(dataset_dir, split_file)
    if not os.path.exists(split_path):
        print(f"[Skip] Missing {split_path}")
        return None

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    bi = read_bi(split_path, num_bundles, num_items)
    bc_count = (bi @ ic).tocsr()
    bc_binary = binarize(bc_count)

    cc_count = (bc_count.T @ bc_count).tocsr()
    cc_binary = (bc_binary.T @ bc_binary).tocsr()

    sp.save_npz(os.path.join(split_dir, "BI_bundle_item.npz"), bi)
    sp.save_npz(os.path.join(split_dir, "BC_bundle_category_count.npz"), bc_count)
    sp.save_npz(os.path.join(split_dir, "BC_bundle_category_binary.npz"), bc_binary)
    sp.save_npz(os.path.join(split_dir, "CC_category_category_count.npz"), cc_count)
    sp.save_npz(os.path.join(split_dir, "CC_category_category_binary.npz"), cc_binary)

    np.savez_compressed(
        os.path.join(split_dir, "CC_category_category_dense.npz"),
        category_ids=np.asarray(category_ids),
        count=cc_count.toarray().astype(np.float32),
        binary=cc_binary.toarray().astype(np.float32),
    )
    top_pairs_to_csv(cc_count, category_ids, os.path.join(split_dir, "CC_top_pairs_count.csv"))
    top_pairs_to_csv(cc_binary, category_ids, os.path.join(split_dir, "CC_top_pairs_binary.csv"))
    save_category_frequency(
        bc_count,
        bc_binary,
        category_ids,
        os.path.join(split_dir, "category_frequency.csv"),
    )

    per_bundle_repeated_cells = np.asarray((bc_count > 1).sum(axis=1)).ravel()
    category_count_per_bundle = np.asarray(bc_binary.sum(axis=1)).ravel()
    item_count_per_bundle = np.asarray(bi.sum(axis=1)).ravel()
    category_bundle_freq = np.asarray(bc_binary.sum(axis=0)).ravel()
    summary = {
        "split": split_name,
        "source_file": split_file,
        "num_bundles": int(num_bundles),
        "num_items": int(num_items),
        "num_categories": int(len(category_ids)),
        "BI_nnz": int(bi.nnz),
        "BC_count_nnz": int(bc_count.nnz),
        "BC_binary_nnz": int(bc_binary.nnz),
        "CC_count_nnz": int(cc_count.nnz),
        "CC_binary_nnz": int(cc_binary.nnz),
        "bundles_with_category": int(np.count_nonzero(category_count_per_bundle)),
        "bundles_with_repeated_category": int(np.count_nonzero(per_bundle_repeated_cells)),
        "repeated_category_bundle_rate": float(
            np.count_nonzero(per_bundle_repeated_cells) / max(1, np.count_nonzero(category_count_per_bundle))
        ),
        "max_category_count_in_bundle": int(bc_count.max()) if bc_count.nnz else 0,
        "avg_items_per_nonempty_bundle": float(
            item_count_per_bundle[item_count_per_bundle > 0].mean()
        ) if np.any(item_count_per_bundle > 0) else 0.0,
        "avg_categories_per_nonempty_bundle": float(
            category_count_per_bundle[category_count_per_bundle > 0].mean()
        ) if np.any(category_count_per_bundle > 0) else 0.0,
        "max_category_bundle_frequency": int(category_bundle_freq.max()) if len(category_bundle_freq) else 0,
        "mean_category_bundle_frequency": float(category_bundle_freq.mean()) if len(category_bundle_freq) else 0.0,
    }
    with open(os.path.join(split_dir, "matrix_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def build_dataset(root, dataset, splits):
    dataset_dir = os.path.join(root, dataset)
    num_bundles, num_items = read_count(dataset_dir)
    item_info, category_field = load_item_info(dataset_dir)
    ic, category_ids, category_counts = build_ic(item_info, category_field, num_items)

    output_dir = os.path.join(dataset_dir, "category_graph")
    os.makedirs(output_dir, exist_ok=True)

    sp.save_npz(os.path.join(output_dir, "IC_item_category.npz"), ic)
    save_category_mapping(
        category_ids,
        category_counts,
        os.path.join(output_dir, "category_mapping.csv"),
    )

    summaries = []
    for split_name, split_file in splits:
        summary = build_for_split(dataset_dir, split_file, split_name, ic, category_ids, output_dir)
        if summary is not None:
            summaries.append(summary)

    metadata = {
        "dataset": dataset,
        "category_field": category_field,
        "num_bundles": num_bundles,
        "num_items": num_items,
        "num_categories": len(category_ids),
        "splits": summaries,
        "notes": {
            "IC": "item x category one-hot matrix",
            "BI": "bundle x item incidence matrix from the split file",
            "BC_count": "BI @ IC, preserving repeated categories within a bundle",
            "BC_binary": "binary version of BC_count",
            "CC_count": "BC_count.T @ BC_count",
            "CC_binary": "BC_binary.T @ BC_binary",
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(
        f"[Done] {dataset}: categories={len(category_ids)} field={category_field} "
        f"output={output_dir}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build BI, IC, BC, and category-category matrices for POG datasets."
    )
    parser.add_argument("--root", default="datasets", help="Dataset root directory")
    parser.add_argument("--datasets", nargs="+", default=["pog", "pog_dense"])
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train:bi_train.txt", "full:bi_full.txt"],
        help="Split specs formatted as name:file, e.g. train:bi_train.txt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    splits = []
    for spec in args.splits:
        if ":" not in spec:
            raise ValueError(f"Invalid split spec {spec}; expected name:file")
        split_name, split_file = spec.split(":", 1)
        splits.append((split_name, split_file))

    for dataset in args.datasets:
        build_dataset(args.root, dataset, splits)


if __name__ == "__main__":
    main()

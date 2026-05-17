import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


DEFAULT_DATASETS = ["pog", "pog_dense"]
DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_CACHE_ROOT = r"analysis\openai_embedding_cache"
DEFAULT_OUTPUT_ROOT = r"analysis\category_embedding_cache"
CATEGORY_FIELD_CANDIDATES = ["cate_id", "cate", "category"]


def load_item_info(repo_root, data_path, dataset):
    item_info_path = repo_root / data_path / dataset / "item_info.json"
    if not item_info_path.exists():
        raise FileNotFoundError(item_info_path)
    with open(item_info_path, encoding="utf-8") as f:
        return json.load(f), item_info_path


def resolve_category(item, category_field):
    if category_field != "auto":
        value = item.get(category_field)
        return str(value).strip() if value is not None and str(value).strip() else None
    for field in CATEGORY_FIELD_CANDIDATES:
        value = item.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def detect_category_field(item_info, category_field):
    if category_field != "auto":
        return category_field
    counts = {field: 0 for field in CATEGORY_FIELD_CANDIDATES}
    for item in item_info.values():
        for field in CATEGORY_FIELD_CANDIDATES:
            value = item.get(field)
            if value is not None and str(value).strip():
                counts[field] += 1
    resolved = max(counts, key=counts.get)
    if counts[resolved] == 0:
        return "auto"
    return resolved


def load_all_item_embeddings(repo_root, cache_root, model, dataset, dtype):
    cache_dir = repo_root / cache_root / model / "all_items" / dataset
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

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "cache_dir": cache_dir,
        "cache_path": cache_path,
        "metadata_path": metadata_path,
        "target_path": target_path,
        "ids": ids,
        "embeddings": embeddings,
        "text_hashes": text_hashes,
        "metadata": metadata,
    }


def build_category_members(item_info, item_ids, category_field):
    members = {}
    missing_items = []
    uncategorized_items = []

    for item_id in item_ids.tolist():
        key = str(int(item_id))
        item = item_info.get(key)
        if item is None:
            missing_items.append(int(item_id))
            continue
        category = resolve_category(item, category_field)
        if category is None:
            uncategorized_items.append(int(item_id))
            continue
        members.setdefault(category, []).append(int(item_id))

    return members, missing_items, uncategorized_items


def mean_category_embeddings(cache, members, dtype):
    id_to_row = {int(item_id): idx for idx, item_id in enumerate(cache["ids"].tolist())}
    categories = sorted(members)
    out_dtype = np.float16 if dtype == "float16" else np.float32

    category_ids = []
    counts = []
    embeddings = []
    mean_raw_norms = []
    normed_embeddings = []

    for category in categories:
        rows = [id_to_row[item_id] for item_id in members[category] if item_id in id_to_row]
        if not rows:
            continue
        mean_vec = cache["embeddings"][rows].mean(axis=0, dtype=np.float32)
        mean_norm = float(np.linalg.norm(mean_vec))
        if mean_norm > 0:
            normed_vec = mean_vec / mean_norm
        else:
            normed_vec = mean_vec

        category_ids.append(category)
        counts.append(len(rows))
        embeddings.append(mean_vec.astype(out_dtype, copy=False))
        normed_embeddings.append(normed_vec.astype(out_dtype, copy=False))
        mean_raw_norms.append(mean_norm)

    return {
        "category_ids": np.asarray(category_ids, dtype="U256"),
        "counts": np.asarray(counts, dtype=np.int64),
        "embeddings": np.vstack(embeddings).astype(out_dtype, copy=False),
        "embeddings_normed": np.vstack(normed_embeddings).astype(out_dtype, copy=False),
        "mean_raw_norms": np.asarray(mean_raw_norms, dtype=np.float32),
    }


def write_summary_csv(path, category_data):
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-category_data["counts"], kind="mergesort")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category_id", "item_count", "mean_embedding_norm"],
        )
        writer.writeheader()
        for idx in order.tolist():
            writer.writerow(
                {
                    "category_id": category_data["category_ids"][idx],
                    "item_count": int(category_data["counts"][idx]),
                    "mean_embedding_norm": float(category_data["mean_raw_norms"][idx]),
                }
            )


def run_dataset(repo_root, dataset, args):
    item_info, item_info_path = load_item_info(repo_root, args.data_path, dataset)
    resolved_category_field = detect_category_field(item_info, args.category_field)
    cache = load_all_item_embeddings(repo_root, Path(args.cache_root), args.model, dataset, args.input_dtype)
    members, missing_items, uncategorized_items = build_category_members(
        item_info=item_info,
        item_ids=cache["ids"],
        category_field=resolved_category_field,
    )
    if not members:
        raise ValueError(f"{dataset}: no categorized items found using category_field={args.category_field}")

    category_data = mean_category_embeddings(cache, members, args.output_dtype)
    out_dir = repo_root / args.output_root / args.model / "all_items" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"category_embeddings_{args.model}_{args.output_dtype}.npz"
    summary_path = out_dir / "category_summary.csv"
    metadata_path = out_dir / "metadata.json"

    np.savez_compressed(
        npz_path,
        category_ids=category_data["category_ids"],
        counts=category_data["counts"],
        embeddings=category_data["embeddings"],
        embeddings_normed=category_data["embeddings_normed"],
        mean_raw_norms=category_data["mean_raw_norms"],
    )
    write_summary_csv(summary_path, category_data)

    counts = category_data["counts"]
    metadata = {
        "dataset": dataset,
        "model": args.model,
        "source_item_info": str(item_info_path),
        "source_embedding_cache": str(cache["cache_path"]),
        "category_field": args.category_field,
        "resolved_category_field": resolved_category_field,
        "num_cache_items": int(len(cache["ids"])),
        "num_categories": int(len(category_data["category_ids"])),
        "embedding_dim": int(category_data["embeddings"].shape[1]),
        "input_dtype": args.input_dtype,
        "output_dtype": args.output_dtype,
        "min_items_per_category": int(counts.min()),
        "mean_items_per_category": float(counts.mean()),
        "median_items_per_category": float(np.median(counts)),
        "max_items_per_category": int(counts.max()),
        "missing_item_info_count": int(len(missing_items)),
        "uncategorized_item_count": int(len(uncategorized_items)),
        "output_path": str(npz_path),
        "summary_path": str(summary_path),
        "created_at_unix": int(time.time()),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata, summary_path, npz_path


def main():
    parser = argparse.ArgumentParser(
        description="Build category embeddings by averaging cached item text embeddings within each item category."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--data-path", default="datasets")
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--output-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--category-field", default="auto")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    for dataset in args.datasets:
        metadata, summary_path, npz_path = run_dataset(repo_root, dataset, args)
        print(f"\n=== {dataset} ===")
        print(f"categories : {metadata['num_categories']}")
        print(f"items      : {metadata['num_cache_items']}")
        print(f"count min/median/mean/max: {metadata['min_items_per_category']} / "
              f"{metadata['median_items_per_category']:.1f} / "
              f"{metadata['mean_items_per_category']:.2f} / "
              f"{metadata['max_items_per_category']}")
        print(f"npz        : {npz_path}")
        print(f"summary    : {summary_path}")


if __name__ == "__main__":
    main()

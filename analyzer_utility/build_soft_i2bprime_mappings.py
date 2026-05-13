import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analyze_relation_feature_signal import read_torch_float_tensor


DEFAULT_CACHE_ROOT = Path("analysis") / "openai_embedding_cache" / "text-embedding-3-large" / "all_items"
DEFAULT_MODEL = "text-embedding-3-large"


def load_train_bundles(dataset_dir):
    bundle_items = {}
    item_to_bundles = {}
    with open(dataset_dir / "bi_train.txt", encoding="utf-8") as f:
        for line in f:
            vals = [int(v) for v in line.strip().split(", ") if v]
            if len(vals) < 2:
                continue
            bundle_id = vals[0]
            items = list(dict.fromkeys(vals[1:]))
            bundle_items[bundle_id] = items
            for item_id in items:
                item_to_bundles.setdefault(item_id, set()).add(bundle_id)
    return bundle_items, item_to_bundles


def load_text_embeddings(repo_root, dataset, model, cache_root):
    cache_path = repo_root / cache_root / dataset / f"embeddings_{model}_float16.npz"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    with np.load(cache_path, allow_pickle=False) as data:
        ids = data["ids"].astype(np.int64)
        emb = data["embeddings"].astype(np.float32)

    order = np.argsort(ids)
    ids = ids[order]
    emb = emb[order]
    if not np.array_equal(ids, np.arange(len(ids), dtype=np.int64)):
        raise ValueError(f"{dataset}: all-items embedding ids are not contiguous 0..N-1")

    norms = np.linalg.norm(emb, axis=1)
    safe = norms > 0
    emb[safe] = emb[safe] / norms[safe, None]
    emb[~safe] = 0.0
    return emb, cache_path


def load_bi_lgcn_embeddings(dataset_dir, dataset):
    feature_path = dataset_dir / f"{dataset}_LightGCN_bi_feature.pt"
    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    emb, _ = read_torch_float_tensor(feature_path)
    emb = emb.astype(np.float32, copy=False)
    norms = np.linalg.norm(emb, axis=1)
    safe = norms > 0
    emb[safe] = emb[safe] / norms[safe, None]
    emb[~safe] = 0.0
    return emb, feature_path


def top1_similar_items(emb, train_active_items, block_size):
    active = np.asarray(sorted(train_active_items), dtype=np.int64)
    active_emb = emb[active]
    top_item = np.empty(emb.shape[0], dtype=np.int64)
    top_sim = np.empty(emb.shape[0], dtype=np.float32)

    active_pos = {int(item_id): pos for pos, item_id in enumerate(active.tolist())}
    for start in range(0, emb.shape[0], block_size):
        end = min(start + block_size, emb.shape[0])
        sims = emb[start:end] @ active_emb.T
        for local_idx, item_id in enumerate(range(start, end)):
            pos = active_pos.get(item_id)
            if pos is not None:
                sims[local_idx, pos] = -np.inf
        best_pos = np.argmax(sims, axis=1)
        top_item[start:end] = active[best_pos]
        top_sim[start:end] = sims[np.arange(end - start), best_pos]
    return top_item, top_sim


def build_item_smooth_mapping(top_item, item_to_bundles):
    mapping = {}
    for item_id, similar_item in enumerate(top_item.tolist()):
        bundles = item_to_bundles.get(int(similar_item), set())
        if bundles:
            mapping[item_id] = sorted(int(bundle_id) for bundle_id in bundles)
    return mapping


def build_bundle_embeddings(emb, bundle_items):
    bundle_ids = np.asarray(sorted(bundle_items), dtype=np.int64)
    bundle_emb = np.zeros((len(bundle_ids), emb.shape[1]), dtype=np.float32)
    for row, bundle_id in enumerate(bundle_ids.tolist()):
        item_ids = [item_id for item_id in bundle_items[bundle_id] if 0 <= item_id < emb.shape[0]]
        if not item_ids:
            continue
        vec = emb[item_ids].mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            bundle_emb[row] = vec / norm
    return bundle_ids, bundle_emb


def top1_similar_bundles(bundle_ids, bundle_emb, block_size):
    top_bundle = np.empty(len(bundle_ids), dtype=np.int64)
    top_sim = np.empty(len(bundle_ids), dtype=np.float32)
    for start in range(0, len(bundle_ids), block_size):
        end = min(start + block_size, len(bundle_ids))
        sims = bundle_emb[start:end] @ bundle_emb.T
        row_idx = np.arange(end - start)
        sims[row_idx, np.arange(start, end)] = -np.inf
        best_pos = np.argmax(sims, axis=1)
        top_bundle[start:end] = bundle_ids[best_pos]
        top_sim[start:end] = sims[row_idx, best_pos]
    return top_bundle, top_sim


def build_bundle_smooth_mapping(item_to_bundles, bundle_to_top1):
    mapping = {}
    for item_id, bundles in item_to_bundles.items():
        smooth_bundles = {
            int(bundle_to_top1[bundle_id])
            for bundle_id in bundles
            if bundle_id in bundle_to_top1
        }
        if smooth_bundles:
            mapping[int(item_id)] = sorted(smooth_bundles)
    return mapping


def write_mapping_txt(path, mapping):
    with open(path, "w", encoding="utf-8", newline="") as f:
        for item_id in sorted(mapping):
            bundles = mapping[item_id]
            f.write(", ".join([str(item_id)] + [str(bundle_id) for bundle_id in bundles]) + "\n")


def write_mapping_json(path, mapping):
    serializable = {str(item_id): bundles for item_id, bundles in sorted(mapping.items())}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)


def write_item_neighbors(path, top_item, top_sim):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "top1_similar_train_item_id", "cosine"])
        for item_id, similar_item in enumerate(top_item.tolist()):
            writer.writerow([item_id, int(similar_item), float(top_sim[item_id])])


def write_bundle_neighbors(path, bundle_ids, top_bundle, top_sim):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bundle_id", "top1_similar_train_bundle_id", "cosine"])
        for row, bundle_id in enumerate(bundle_ids.tolist()):
            writer.writerow([int(bundle_id), int(top_bundle[row]), float(top_sim[row])])


def mapping_stats(mapping, num_items):
    lengths = np.asarray([len(v) for v in mapping.values()], dtype=np.float64)
    if lengths.size == 0:
        return {
            "covered_items": 0,
            "coverage": 0.0,
            "bundle_count_mean": 0.0,
            "bundle_count_median": 0.0,
            "bundle_count_max": 0,
        }
    return {
        "covered_items": int(len(mapping)),
        "coverage": float(len(mapping) / num_items),
        "bundle_count_mean": float(lengths.mean()),
        "bundle_count_median": float(np.median(lengths)),
        "bundle_count_max": int(lengths.max()),
    }


def build_for_dataset(repo_root, dataset, args):
    dataset_dir = repo_root / args.data_path / dataset
    with open(dataset_dir / "count.json", encoding="utf-8") as f:
        count = json.load(f)
    num_items = int(count["#I"])

    bundle_items, item_to_bundles = load_train_bundles(dataset_dir)
    emb, cache_path = load_text_embeddings(repo_root, dataset, args.model, Path(args.cache_root))
    if emb.shape[0] != num_items:
        raise ValueError(f"{dataset}: embedding rows {emb.shape[0]} != #I {num_items}")

    train_active_items = set(item_to_bundles)
    print(f"[{dataset}] items={num_items}, train_bundles={len(bundle_items)}, train_active_items={len(train_active_items)}")

    top_item, top_item_sim = top1_similar_items(emb, train_active_items, args.item_block_size)
    item_smooth = build_item_smooth_mapping(top_item, item_to_bundles)

    bundle_ids, bundle_emb = build_bundle_embeddings(emb, bundle_items)
    top_bundle, top_bundle_sim = top1_similar_bundles(bundle_ids, bundle_emb, args.bundle_block_size)
    bundle_to_top1 = {int(bundle_id): int(top_bundle[row]) for row, bundle_id in enumerate(bundle_ids.tolist())}
    bundle_smooth = build_bundle_smooth_mapping(item_to_bundles, bundle_to_top1)

    bi_emb, bi_feature_path = load_bi_lgcn_embeddings(dataset_dir, dataset)
    if bi_emb.shape[0] != num_items:
        raise ValueError(f"{dataset}: BI-LightGCN rows {bi_emb.shape[0]} != #I {num_items}")
    bi_bundle_ids, bi_bundle_emb = build_bundle_embeddings(bi_emb, bundle_items)
    top_bi_bundle, top_bi_bundle_sim = top1_similar_bundles(
        bi_bundle_ids,
        bi_bundle_emb,
        args.bundle_block_size,
    )
    bi_bundle_to_top1 = {
        int(bundle_id): int(top_bi_bundle[row])
        for row, bundle_id in enumerate(bi_bundle_ids.tolist())
    }
    bundle_bi_smooth = build_bundle_smooth_mapping(item_to_bundles, bi_bundle_to_top1)

    item_json = dataset_dir / "item_smoothing_i2bprime_text_top1.json"
    item_txt = dataset_dir / "item_smoothing_i2bprime_text_top1.txt"
    bundle_json = dataset_dir / "bundle_smoothing_i2bprime_text_top1.json"
    bundle_txt = dataset_dir / "bundle_smoothing_i2bprime_text_top1.txt"
    bundle_bi_json = dataset_dir / "bundle_smoothing_i2bprime_bi_lgcn_top1.json"
    bundle_bi_txt = dataset_dir / "bundle_smoothing_i2bprime_bi_lgcn_top1.txt"
    item_neighbor_csv = dataset_dir / "item_top1_similar_train_item_text.csv"
    bundle_neighbor_csv = dataset_dir / "train_bundle_top1_similar_bundle_text.csv"
    bundle_bi_neighbor_csv = dataset_dir / "train_bundle_top1_similar_bundle_bi_lgcn.csv"
    meta_path = dataset_dir / "soft_i2bprime_text_top1_meta.json"
    meta_bi_path = dataset_dir / "soft_i2bprime_bi_lgcn_top1_meta.json"

    write_mapping_json(item_json, item_smooth)
    write_mapping_txt(item_txt, item_smooth)
    write_mapping_json(bundle_json, bundle_smooth)
    write_mapping_txt(bundle_txt, bundle_smooth)
    write_mapping_json(bundle_bi_json, bundle_bi_smooth)
    write_mapping_txt(bundle_bi_txt, bundle_bi_smooth)
    write_item_neighbors(item_neighbor_csv, top_item, top_item_sim)
    write_bundle_neighbors(bundle_neighbor_csv, bundle_ids, top_bundle, top_bundle_sim)
    write_bundle_neighbors(bundle_bi_neighbor_csv, bi_bundle_ids, top_bi_bundle, top_bi_bundle_sim)

    meta = {
        "dataset": dataset,
        "embedding_cache": str(cache_path),
        "model": args.model,
        "num_items": num_items,
        "num_train_bundles": len(bundle_items),
        "num_train_active_items": len(train_active_items),
        "method": {
            "item_smoothing": "I -> top-1 text-similar train item I' -> train bundles B'",
            "bundle_smoothing": "I -> train bundles B -> top-1 text-similar train bundle B'",
            "bundle_smoothing_bi_lgcn": "I -> train bundles B -> top-1 BI-LightGCN-similar train bundle B'",
            "similarity": "cosine over normalized OpenAI text embeddings; bundle embedding is mean item embedding",
            "bi_lgcn_similarity": "cosine over normalized BI-LightGCN item embeddings; bundle embedding is mean item embedding",
            "exact_I_B_included": False,
        },
        "outputs": {
            "item_smoothing_json": str(item_json),
            "item_smoothing_txt": str(item_txt),
            "bundle_smoothing_json": str(bundle_json),
            "bundle_smoothing_txt": str(bundle_txt),
            "bundle_smoothing_bi_lgcn_json": str(bundle_bi_json),
            "bundle_smoothing_bi_lgcn_txt": str(bundle_bi_txt),
            "item_neighbors_csv": str(item_neighbor_csv),
            "bundle_neighbors_csv": str(bundle_neighbor_csv),
            "bundle_bi_lgcn_neighbors_csv": str(bundle_bi_neighbor_csv),
        },
        "item_smoothing_stats": mapping_stats(item_smooth, num_items),
        "bundle_smoothing_stats": mapping_stats(bundle_smooth, num_items),
        "bundle_smoothing_bi_lgcn_stats": mapping_stats(bundle_bi_smooth, num_items),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[{dataset}] item smoothing: {meta['item_smoothing_stats']}")
    print(f"[{dataset}] bundle smoothing: {meta['bundle_smoothing_stats']}")
    print(f"[{dataset}] bundle smoothing BI-LightGCN: {meta['bundle_smoothing_bi_lgcn_stats']}")
    print(f"[{dataset}] wrote {meta_path}")

    meta_bi = {
        "dataset": dataset,
        "bi_lgcn_feature_path": str(bi_feature_path),
        "num_items": num_items,
        "num_train_bundles": len(bundle_items),
        "num_train_active_items": len(train_active_items),
        "method": {
            "bundle_smoothing": "I -> train bundles B -> top-1 BI-LightGCN-similar train bundle B'",
            "similarity": "cosine over normalized BI-LightGCN item embeddings; bundle embedding is mean item embedding",
            "exact_I_B_included": False,
        },
        "outputs": {
            "bundle_smoothing_bi_lgcn_json": str(bundle_bi_json),
            "bundle_smoothing_bi_lgcn_txt": str(bundle_bi_txt),
            "bundle_bi_lgcn_neighbors_csv": str(bundle_bi_neighbor_csv),
        },
        "bundle_smoothing_bi_lgcn_stats": mapping_stats(bundle_bi_smooth, num_items),
    }
    with open(meta_bi_path, "w", encoding="utf-8") as f:
        json.dump(meta_bi, f, ensure_ascii=False, indent=2)
    print(f"[{dataset}] wrote {meta_bi_path}")


def main():
    parser = argparse.ArgumentParser(description="Build text top-1 soft I->B' mappings for POG datasets.")
    parser.add_argument("--datasets", nargs="+", default=["pog", "pog_dense"])
    parser.add_argument("--data-path", default="datasets")
    parser.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--item-block-size", type=int, default=512)
    parser.add_argument("--bundle-block-size", type=int, default=512)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    for dataset in args.datasets:
        build_for_dataset(repo_root, dataset, args)


if __name__ == "__main__":
    main()

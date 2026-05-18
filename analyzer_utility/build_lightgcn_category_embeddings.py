import argparse
import io
import json
import os
import pickle
import sys
import types
import zipfile

import numpy as np
import pandas as pd


class _FakeFloatStorage:
    pass


def _install_fake_torch_modules():
    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch._utils")

    def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        return {
            "storage": storage,
            "storage_offset": int(storage_offset),
            "size": tuple(int(v) for v in size),
            "stride": tuple(int(v) for v in stride),
            "requires_grad": bool(requires_grad),
        }

    torch_mod.FloatStorage = _FakeFloatStorage
    utils_mod._rebuild_tensor_v2 = rebuild_tensor_v2
    torch_mod._utils = utils_mod
    sys.modules.setdefault("torch", torch_mod)
    sys.modules.setdefault("torch._utils", utils_mod)


class _TorchTensorMetadataUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            _, storage_type, key, location, numel = pid[:5]
            return {
                "storage_type": getattr(storage_type, "__name__", str(storage_type)),
                "key": str(key),
                "location": str(location),
                "numel": int(numel),
            }
        raise pickle.UnpicklingError(f"unsupported persistent id: {pid}")


def load_torch_saved_float_tensor(path):
    _install_fake_torch_modules()
    with zipfile.ZipFile(path, "r") as zf:
        data_pkl_name = next(name for name in zf.namelist() if name.endswith("/data.pkl"))
        root = data_pkl_name.rsplit("/", 1)[0]
        metadata = _TorchTensorMetadataUnpickler(io.BytesIO(zf.read(data_pkl_name))).load()
        storage = metadata["storage"]
        storage_key = storage["key"]
        storage_name = f"{root}/data/{storage_key}"
        byteorder_name = f"{root}/byteorder"
        byteorder = zf.read(byteorder_name).decode("utf-8").strip() if byteorder_name in zf.namelist() else "little"
        if byteorder != "little":
            raise ValueError(f"Unsupported tensor byteorder={byteorder}")
        raw = zf.read(storage_name)

    if storage["storage_type"] not in {"_FakeFloatStorage", "FloatStorage"}:
        raise ValueError(f"Unsupported storage type={storage['storage_type']}")

    full = np.frombuffer(raw, dtype="<f4")
    offset = int(metadata["storage_offset"])
    size = tuple(metadata["size"])
    stride = tuple(metadata["stride"])
    expected = int(np.prod(size))
    if stride != (size[1], 1):
        raise ValueError(f"Only contiguous 2D tensors are supported; got size={size}, stride={stride}")
    tensor = full[offset:offset + expected].reshape(size).copy()
    return tensor, {
        "storage_numel": int(storage["numel"]),
        "storage_location": storage["location"],
        "storage_offset": offset,
        "tensor_shape": list(size),
        "tensor_stride": list(stride),
    }


def detect_category_field(item_info):
    candidates = ["cate_id", "cate", "category"]
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


def build_category_embeddings(dataset_dir, dataset, feature_path, output_root, embedding_name):
    with open(os.path.join(dataset_dir, "item_info.json"), "r", encoding="utf-8") as f:
        item_info = json.load(f)
    category_field = detect_category_field(item_info)

    item_embeddings, tensor_meta = load_torch_saved_float_tensor(feature_path)
    item_count, dim = item_embeddings.shape

    category_to_rows = {}
    for item_id_str, item in item_info.items():
        item_id = int(item_id_str)
        if not 0 <= item_id < item_count:
            continue
        value = item.get(category_field)
        if value is None or not str(value).strip():
            continue
        category = str(value).strip()
        category_to_rows.setdefault(category, []).append(item_id)

    category_ids = sorted(category_to_rows)
    embeddings = np.zeros((len(category_ids), dim), dtype=np.float32)
    counts = np.zeros(len(category_ids), dtype=np.int64)
    for idx, category in enumerate(category_ids):
        rows = np.asarray(category_to_rows[category], dtype=np.int64)
        counts[idx] = len(rows)
        embeddings[idx] = item_embeddings[rows].mean(axis=0)

    norms = np.linalg.norm(embeddings, axis=1)
    embeddings_normed = np.zeros_like(embeddings)
    valid = norms > 0
    embeddings_normed[valid] = embeddings[valid] / norms[valid, None]

    output_dir = os.path.join(output_root, embedding_name, "all_items", dataset)
    os.makedirs(output_dir, exist_ok=True)
    npz_path = os.path.join(output_dir, f"category_embeddings_{embedding_name}_float32.npz")
    np.savez_compressed(
        npz_path,
        category_ids=np.asarray(category_ids),
        counts=counts,
        embeddings=embeddings.astype(np.float32),
        embeddings_normed=embeddings_normed.astype(np.float32),
    )

    summary_path = os.path.join(output_dir, "category_summary.csv")
    pd.DataFrame({
        "category_id": category_ids,
        "item_count": counts,
        "embedding_norm": norms,
    }).to_csv(summary_path, index=False, encoding="utf-8-sig")

    metadata = {
        "dataset": dataset,
        "source_feature_path": os.path.abspath(feature_path),
        "embedding_name": embedding_name,
        "category_field": category_field,
        "num_items_in_feature": int(item_count),
        "embedding_dim": int(dim),
        "num_categories": int(len(category_ids)),
        "output_npz": os.path.abspath(npz_path),
        "tensor_metadata": tensor_meta,
        "pooling": "mean item LightGCN BI embeddings per category",
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[Done] {dataset}: {len(category_ids)} categories, dim={dim}")
    print(f"       npz: {npz_path}")
    print(f"   summary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build category embeddings by pooling BI LightGCN item embeddings.")
    parser.add_argument("--dataset", default="pog_dense")
    parser.add_argument("--data_path", default="datasets")
    parser.add_argument("--feature_path", default="")
    parser.add_argument("--output_root", default=os.path.join("analysis", "category_embedding_cache"))
    parser.add_argument("--embedding_name", default="LightGCN_bi")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = os.path.join(args.data_path, args.dataset)
    feature_path = args.feature_path or os.path.join(dataset_dir, f"{args.dataset}_LightGCN_bi_feature.pt")
    build_category_embeddings(
        dataset_dir=dataset_dir,
        dataset=args.dataset,
        feature_path=feature_path,
        output_root=args.output_root,
        embedding_name=args.embedding_name,
    )


if __name__ == "__main__":
    main()

import ast
import io
import json
import math
import pickle
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


VIEWS = ("IBxBI", "IUxUI", "BIxIB")


def read_csv(path):
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def parse_list(value):
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def normalize_matrix(x):
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1)
    safe = norms > 0
    out = np.zeros_like(x, dtype=np.float32)
    out[safe] = x[safe] / norms[safe, None]
    return out


def normalize_vector(x):
    x = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(x))
    if norm <= 0:
        return None
    return x / norm


def cosine(a, b):
    if a is None or b is None:
        return np.nan
    return float(np.dot(a, b))


def mean_embedding(item_ids, matrix, item_to_row):
    rows = [item_to_row[int(item_id)] for item_id in item_ids if int(item_id) in item_to_row]
    if not rows:
        return None
    return normalize_vector(matrix[rows].mean(axis=0))


def item_embedding(item_id, matrix, item_to_row):
    row = item_to_row.get(int(item_id))
    if row is None:
        return None
    return matrix[row]


class TorchTensorUnpickler(pickle.Unpickler):
    def persistent_load(self, pid):
        typename, storage_type, key, location, size = pid
        if typename != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {pid}")
        return {"key": str(key), "storage_size": int(size)}

    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor_v2
        if module == "torch" and name == "FloatStorage":
            return "FloatStorage"
        return super().find_class(module, name)


def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    return {
        "storage": storage,
        "storage_offset": int(storage_offset),
        "size": tuple(int(x) for x in size),
        "stride": tuple(int(x) for x in stride),
    }


def load_torch_float_tensor(path):
    """Load a simple contiguous float32 tensor saved by torch.save without importing torch."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        data_pkl_name = next(name for name in zf.namelist() if name.endswith("data.pkl"))
        prefix = data_pkl_name.rsplit("/", 1)[0]
        tensor_meta = TorchTensorUnpickler(io.BytesIO(zf.read(data_pkl_name))).load()
        storage_key = tensor_meta["storage"]["key"]
        raw = zf.read(f"{prefix}/data/{storage_key}")

    storage = np.frombuffer(raw, dtype="<f4")
    offset = tensor_meta["storage_offset"]
    shape = tensor_meta["size"]
    stride = tensor_meta["stride"]
    if stride != (shape[1], 1):
        raise ValueError(f"Unsupported non-contiguous tensor stride={stride} shape={shape}")
    size = int(np.prod(shape))
    return storage[offset: offset + size].reshape(shape).copy()


def load_text_embedding(repo_root, dataset):
    path = (
        repo_root
        / "analysis"
        / "openai_embedding_cache"
        / "text-embedding-3-large"
        / "all_items"
        / dataset
        / "embeddings_text-embedding-3-large_float16.npz"
    )
    data = np.load(path)
    ids = data["ids"].astype(np.int64)
    matrix = normalize_matrix(data["embeddings"])
    return matrix, {int(item_id): idx for idx, item_id in enumerate(ids.tolist())}, str(path)


def load_bi_embedding(repo_root, dataset):
    filename = "pog_LightGCN_bi_feature.pt" if dataset == "pog" else "pog_dense_LightGCN_bi_feature.pt"
    path = repo_root / "datasets" / dataset / filename
    matrix = normalize_matrix(load_torch_float_tensor(path))
    return matrix, {idx: idx for idx in range(matrix.shape[0])}, str(path)


def load_train_bundles(path):
    bundles = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [int(x) for x in line.strip().split(", ") if x.strip()]
            if parts:
                bundles[parts[0]] = parts[1:]
    return bundles


def sample_from_counter(counter, item_id, k=3, alpha=0.5, seed=45):
    if not counter:
        return []
    indices = np.array(sorted(counter), dtype=np.int64)
    counts = np.array([counter[int(item)] for item in indices], dtype=np.float64)
    weights = np.power(counts, float(alpha))
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return []
    rng = np.random.default_rng(int(item_id) + int(seed))
    sample_size = min(int(k), len(indices))
    sampled = rng.choice(indices, size=sample_size, replace=False, p=weights / weight_sum)
    return [int(x) for x in sampled.tolist()]


def build_needed_bundle_neighbors(train_bundles, needed_items):
    needed = {int(x) for x in needed_items}
    counters = {item_id: Counter() for item_id in needed}
    for items in train_bundles.values():
        item_set = set(int(x) for x in items)
        hit_items = needed & item_set
        if not hit_items:
            continue
        for item_id in hit_items:
            for other in item_set:
                if other != item_id:
                    counters[item_id][other] += 1
    return counters


def build_needed_user_neighbors(ui_path, needed_items):
    needed = {int(x) for x in needed_items}
    counters = {item_id: Counter() for item_id in needed}
    with open(ui_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [int(x) for x in line.strip().split(", ") if x.strip()]
            if len(parts) <= 2:
                continue
            items = parts[1:]
            item_set = set(items)
            hit_items = needed & item_set
            if not hit_items:
                continue
            for item_id in hit_items:
                for other in item_set:
                    if other != item_id:
                        counters[item_id][other] += 1
    return counters


def latest_bgraph_result(repo_root, dataset):
    files = sorted((repo_root / "results" / dataset).glob(f"results_{dataset}_RANK_BGRAPH*.csv"))
    if not files:
        raise FileNotFoundError(f"No RANK_BGRAPH result found for {dataset}")
    return max(files, key=lambda p: p.stat().st_mtime)


def unique_cases(df):
    rows = []
    for view in VIEWS:
        mask = df[f"{view}_hit_at_1_calc"] == 1
        for other in VIEWS:
            if other != view:
                mask &= df[f"{other}_hit_at_1_calc"] == 0
        subset = df[mask].copy()
        subset["unique_view"] = view
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def bgraph_context_lookup(repo_root, dataset):
    result = read_csv(latest_bgraph_result(repo_root, dataset))
    result["candidate_key"] = result["candidate_indices"].astype(str)
    return result.set_index(["bundle_id", "candidate_key"], drop=False)


def context_items_for_case(row, view, bundle_counters, user_counters, bgraph, train_bundles):
    gt_id = int(row["true_indice"])
    if view == "IBxBI":
        return sample_from_counter(bundle_counters.get(gt_id, Counter()), gt_id)
    if view == "IUxUI":
        return sample_from_counter(user_counters.get(gt_id, Counter()), gt_id)

    key = (int(row["bundle_id"]), str(row["candidate_indices"]))
    if key not in bgraph.index:
        return []
    bgraph_row = bgraph.loc[key]
    bundle_ids = [int(x) for x in parse_list(bgraph_row.get("bundle_graph_context_bundle_ids", "[]"))]
    items = []
    for bundle_id in bundle_ids:
        items.extend(train_bundles.get(bundle_id, [])[:5])
    return [int(x) for x in dict.fromkeys(items)]


def context_metrics(prefix, context_items, input_vec, gt_vec, distractor_vecs, matrix, item_to_row):
    context_vec = mean_embedding(context_items, matrix, item_to_row)
    context_input = cosine(context_vec, input_vec)
    context_gt = cosine(context_vec, gt_vec)
    context_distractor = [x for x in (cosine(context_vec, vec) for vec in distractor_vecs) if not math.isnan(x)]
    return {
        f"{prefix}_input_sim": context_input,
        f"{prefix}_gt_sim": context_gt,
        f"{prefix}_distractor_mean_sim": float(np.nanmean(context_distractor)) if context_distractor else np.nan,
        f"{prefix}_distractor_max_sim": float(np.nanmax(context_distractor)) if context_distractor else np.nan,
        f"{prefix}_gt_margin_vs_distractor_mean": context_gt - float(np.nanmean(context_distractor)) if context_distractor else np.nan,
        f"{prefix}_gt_margin_vs_best_distractor": context_gt - float(np.nanmax(context_distractor)) if context_distractor else np.nan,
        f"num_{prefix}_items": len(context_items),
    }


def compute_metrics(row, context_items, matrix, item_to_row):
    input_ids = [int(x) for x in parse_list(row["input_indices_IBxBI"])]
    candidate_ids = [int(x) for x in parse_list(row["candidate_indices"])]
    gt_id = int(row["true_indice"])
    distractor_ids = [item_id for item_id in candidate_ids if item_id != gt_id]

    input_vec = mean_embedding(input_ids, matrix, item_to_row)
    gt_vec = item_embedding(gt_id, matrix, item_to_row)
    distractor_vecs = [item_embedding(item_id, matrix, item_to_row) for item_id in distractor_ids]
    distractor_vecs = [vec for vec in distractor_vecs if vec is not None]

    input_gt = cosine(input_vec, gt_vec)
    input_distractor = [x for x in (cosine(input_vec, vec) for vec in distractor_vecs) if not math.isnan(x)]
    context_items_excl_input = [item_id for item_id in context_items if item_id not in set(input_ids)]

    input_candidate_sims = []
    for item_id in candidate_ids:
        vec = item_embedding(item_id, matrix, item_to_row)
        input_candidate_sims.append((item_id, cosine(input_vec, vec)))
    sorted_input = sorted(input_candidate_sims, key=lambda x: (-(x[1] if not math.isnan(x[1]) else -999), x[0]))
    input_gt_rank = next((idx + 1 for idx, (item_id, _) in enumerate(sorted_input) if item_id == gt_id), np.nan)

    return {
        "input_gt_sim": input_gt,
        "input_distractor_mean_sim": float(np.nanmean(input_distractor)) if input_distractor else np.nan,
        "input_distractor_max_sim": float(np.nanmax(input_distractor)) if input_distractor else np.nan,
        "input_gt_margin_vs_distractor_mean": input_gt - float(np.nanmean(input_distractor)) if input_distractor else np.nan,
        "input_gt_margin_vs_best_distractor": input_gt - float(np.nanmax(input_distractor)) if input_distractor else np.nan,
        "input_gt_rank_by_embedding": input_gt_rank,
        "num_input_items": len(input_ids),
        "num_context_items": len(context_items),
        "num_context_excl_input_items": len(context_items_excl_input),
        "context_contains_input": int(bool(set(context_items) & set(input_ids))),
        "context_contains_gt": int(gt_id in set(context_items)),
        **context_metrics("context", context_items, input_vec, gt_vec, distractor_vecs, matrix, item_to_row),
        **context_metrics("context_excl_input", context_items_excl_input, input_vec, gt_vec, distractor_vecs, matrix, item_to_row),
    }


def to_markdown_table(df):
    if df.empty:
        return ""
    rendered = df.copy()
    rendered = rendered.astype(str)
    headers = list(rendered.columns)
    rows = rendered.values.tolist()
    widths = [
        max(len(str(headers[idx])), *(len(str(row[idx])) for row in rows))
        for idx in range(len(headers))
    ]
    header_line = "| " + " | ".join(str(headers[idx]).ljust(widths[idx]) for idx in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def summarize(detail):
    metric_cols = [
        "input_gt_sim",
        "input_distractor_mean_sim",
        "input_distractor_max_sim",
        "input_gt_margin_vs_distractor_mean",
        "input_gt_margin_vs_best_distractor",
        "input_gt_rank_by_embedding",
        "context_input_sim",
        "context_gt_sim",
        "context_distractor_mean_sim",
        "context_distractor_max_sim",
        "context_gt_margin_vs_distractor_mean",
        "context_gt_margin_vs_best_distractor",
        "context_excl_input_input_sim",
        "context_excl_input_gt_sim",
        "context_excl_input_distractor_mean_sim",
        "context_excl_input_distractor_max_sim",
        "context_excl_input_gt_margin_vs_distractor_mean",
        "context_excl_input_gt_margin_vs_best_distractor",
        "num_context_items",
        "num_context_excl_input_items",
        "context_contains_input",
        "context_contains_gt",
    ]
    grouped = detail.groupby(["dataset", "embedding_space", "unique_view"], dropna=False)
    rows = []
    for key, group in grouped:
        row = {
            "dataset": key[0],
            "embedding_space": key[1],
            "unique_view": key[2],
            "count": len(group),
            "valid_context_count": int((group["num_context_items"] > 0).sum()),
        }
        for col in metric_cols:
            row[f"{col}_mean"] = group[col].mean()
            row[f"{col}_median"] = group[col].median()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "analysis" / "unique_view_embedding_alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_detail = []
    metadata = []

    for dataset in ("pog", "pog_dense"):
        oracle_path = repo_root / "analysis" / f"{dataset}_ranking_view_analysis" / "per_sample_oracle.csv"
        cases = unique_cases(read_csv(oracle_path))
        train_bundles = load_train_bundles(repo_root / "datasets" / dataset / "bi_train.txt")
        needed_ib = cases.loc[cases["unique_view"] == "IBxBI", "true_indice"].astype(int).tolist()
        needed_iu = cases.loc[cases["unique_view"] == "IUxUI", "true_indice"].astype(int).tolist()
        bundle_counters = build_needed_bundle_neighbors(train_bundles, needed_ib)
        user_counters = build_needed_user_neighbors(repo_root / "datasets" / dataset / "ui_full.txt", needed_iu)
        bgraph = bgraph_context_lookup(repo_root, dataset)

        context_by_row = []
        for _, row in cases.iterrows():
            view = row["unique_view"]
            context_items = context_items_for_case(row, view, bundle_counters, user_counters, bgraph, train_bundles)
            context_by_row.append(context_items)
        cases = cases.copy()
        cases["view_context_items"] = context_by_row

        for space, loader in (("text", load_text_embedding), ("bi", load_bi_embedding)):
            matrix, item_to_row, source = loader(repo_root, dataset)
            metadata.append({
                "dataset": dataset,
                "embedding_space": space,
                "source": source,
                "num_items": matrix.shape[0],
                "dim": matrix.shape[1],
            })
            for _, row in cases.iterrows():
                context_items = [int(x) for x in row["view_context_items"]]
                metrics = compute_metrics(row, context_items, matrix, item_to_row)
                all_detail.append({
                    "dataset": dataset,
                    "embedding_space": space,
                    "unique_view": row["unique_view"],
                    "bundle_id": int(row["bundle_id"]),
                    "true_indice": int(row["true_indice"]),
                    "true_option_char": row["true_option_char"],
                    "candidate_indices": row["candidate_indices"],
                    "input_indices": row["input_indices_IBxBI"],
                    "view_context_items": json.dumps(context_items, ensure_ascii=False),
                    "rank_IBxBI": row["true_rank_IBxBI"],
                    "rank_IUxUI": row["true_rank_IUxUI"],
                    "rank_BIxIB": row["true_rank_BIxIB"],
                    **metrics,
                })

    detail = pd.DataFrame(all_detail)
    summary = summarize(detail)
    metadata_df = pd.DataFrame(metadata)

    detail_path = out_dir / "unique_view_embedding_alignment_detail.csv"
    summary_path = out_dir / "unique_view_embedding_alignment_summary.csv"
    metadata_path = out_dir / "unique_view_embedding_alignment_metadata.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    metadata_df.to_csv(metadata_path, index=False, encoding="utf-8-sig")

    compact_cols = [
        "dataset",
        "embedding_space",
        "unique_view",
        "count",
        "valid_context_count",
        "input_gt_sim_mean",
        "input_distractor_mean_sim_mean",
        "input_gt_margin_vs_distractor_mean_mean",
        "context_input_sim_mean",
        "context_gt_sim_mean",
        "context_gt_margin_vs_distractor_mean_mean",
        "context_excl_input_input_sim_mean",
        "context_excl_input_gt_sim_mean",
        "context_excl_input_gt_margin_vs_distractor_mean_mean",
        "context_contains_input_mean",
        "context_contains_gt_mean",
    ]
    compact = summary[compact_cols].copy()
    compact.to_csv(out_dir / "unique_view_embedding_alignment_summary_compact.csv", index=False, encoding="utf-8-sig")

    lines = ["# Unique View Embedding Alignment", ""]
    lines.append("Mean cosine similarities by dataset, embedding space, and unique-hit view.")
    lines.append("")
    lines.append(to_markdown_table(compact.round(4)))
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved detail to {detail_path}")
    print(f"Saved summary to {summary_path}")
    print(compact.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

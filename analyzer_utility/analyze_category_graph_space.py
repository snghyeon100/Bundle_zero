import argparse
import json
import math
import os
from collections import Counter, defaultdict, deque
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_count(dataset_dir):
    stat = read_json(os.path.join(dataset_dir, "count.json"))
    return int(stat["#B"]), int(stat["#I"])


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


def load_item_info(dataset_dir):
    item_info = read_json(os.path.join(dataset_dir, "item_info.json"))
    return item_info, detect_category_field(item_info)


def item_title(dataset, item_info, item_id):
    item = item_info.get(str(int(item_id)), {})
    if "pog" in dataset:
        return " ".join(str(item.get("title", f"Item {item_id}")).split())
    if "spotify" in dataset:
        parts = [
            item.get("track_name", ""),
            item.get("artist_name", ""),
            item.get("album_name", ""),
        ]
        text = " - ".join(str(p) for p in parts if p)
        return " ".join((text or f"Track {item_id}").split())
    return f"Item {item_id}"


def build_item_category_maps(dataset_dir, category_ids):
    item_info, category_field = load_item_info(dataset_dir)
    category_set = set(str(c) for c in category_ids)
    item_to_category = {}
    category_to_items = defaultdict(list)
    for item_id_str, item in item_info.items():
        value = item.get(category_field)
        if value is None or not str(value).strip():
            continue
        category = str(value).strip()
        if category not in category_set:
            continue
        item_id = int(item_id_str)
        item_to_category[item_id] = category
        category_to_items[category].append(item_id)
    for category in category_to_items:
        category_to_items[category].sort()
    return item_info, category_field, item_to_category, category_to_items


def representative_texts(dataset, item_info, category_to_items, category, k=3):
    texts = []
    for item_id in category_to_items.get(str(category), [])[:k]:
        texts.append(item_title(dataset, item_info, item_id))
    return texts


def load_category_ids(dataset_dir):
    mapping_path = os.path.join(dataset_dir, "category_graph", "category_mapping.csv")
    df = pd.read_csv(mapping_path, encoding="utf-8-sig")
    df = df.sort_values("category_col")
    return [str(c) for c in df["category_id"].tolist()]


def load_cc(dataset_dir, split, mode):
    path = os.path.join(
        dataset_dir,
        "category_graph",
        split,
        f"CC_category_category_{mode}.npz",
    )
    return sp.load_npz(path).astype(np.float64).tocsr()


def load_category_embeddings(repo_root, dataset, category_ids, model, dtype):
    path = os.path.join(
        repo_root,
        "analysis",
        "category_embedding_cache",
        model,
        "all_items",
        dataset,
        f"category_embeddings_{model}_{dtype}.npz",
    )
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as data:
        emb_category_ids = [str(c) for c in data["category_ids"].astype(str).tolist()]
        embeddings = data["embeddings_normed"].astype(np.float64)
    row_by_category = {category: idx for idx, category in enumerate(emb_category_ids)}
    aligned = np.zeros((len(category_ids), embeddings.shape[1]), dtype=np.float64)
    present = np.zeros(len(category_ids), dtype=bool)
    for idx, category in enumerate(category_ids):
        row = row_by_category.get(category)
        if row is not None:
            aligned[idx] = embeddings[row]
            present[idx] = True
    return aligned, present


def matrix_graph_stats(dataset, split, mode, matrix, category_ids):
    dense = matrix.toarray().astype(np.float64)
    n = dense.shape[0]
    offdiag = dense.copy()
    np.fill_diagonal(offdiag, 0.0)
    edge_mask = offdiag > 0
    undirected_edges = int(np.count_nonzero(np.triu(edge_mask, k=1)))
    possible_edges = n * (n - 1) / 2
    weighted_degree = offdiag.sum(axis=1)
    degree = edge_mask.sum(axis=1)

    visited = np.zeros(n, dtype=bool)
    components = []
    adjacency = [np.flatnonzero(edge_mask[i]).tolist() for i in range(n)]
    for start in range(n):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for nxt in adjacency[node]:
                if not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)
        components.append(size)

    return {
        "dataset": dataset,
        "split": split,
        "mode": mode,
        "num_categories": n,
        "nonzero_diag": int(np.count_nonzero(np.diag(dense))),
        "undirected_edges": undirected_edges,
        "density": float(undirected_edges / possible_edges) if possible_edges else 0.0,
        "total_offdiag_weight": float(np.triu(offdiag, k=1).sum()),
        "mean_degree": float(degree.mean()) if n else 0.0,
        "median_degree": float(np.median(degree)) if n else 0.0,
        "max_degree": int(degree.max()) if n else 0,
        "mean_weighted_degree": float(weighted_degree.mean()) if n else 0.0,
        "max_weighted_degree": float(weighted_degree.max()) if n else 0.0,
        "num_components": len(components),
        "largest_component_size": int(max(components)) if components else 0,
        "isolated_categories": int(sum(1 for c in components if c == 1)),
    }


def graph_pair_rows(dataset, split, count_matrix, binary_matrix, category_ids, top_k=300):
    count_dense = count_matrix.toarray().astype(np.float64)
    binary_dense = binary_matrix.toarray().astype(np.float64)
    n = len(category_ids)
    freq = np.diag(binary_dense).copy()
    total_bundles = max(float(freq.max()), 1.0)
    # Prefer the actual number of nonempty bundles when available through diagonal frequencies.
    # For lift/PMI ranking, the same scalar normalization is enough for relative comparison.
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            binary_w = binary_dense[i, j]
            count_w = count_dense[i, j]
            if binary_w <= 0 and count_w <= 0:
                continue
            p_i = freq[i] / total_bundles if total_bundles else 0.0
            p_j = freq[j] / total_bundles if total_bundles else 0.0
            p_ij = binary_w / total_bundles if total_bundles else 0.0
            lift = p_ij / (p_i * p_j) if p_i > 0 and p_j > 0 else 0.0
            pmi = math.log(lift) if lift > 0 else 0.0
            rows.append({
                "dataset": dataset,
                "split": split,
                "category_a": category_ids[i],
                "category_b": category_ids[j],
                "count_weight": float(count_w),
                "binary_weight": float(binary_w),
                "freq_a": float(freq[i]),
                "freq_b": float(freq[j]),
                "p_b_given_a": float(binary_w / freq[i]) if freq[i] > 0 else 0.0,
                "p_a_given_b": float(binary_w / freq[j]) if freq[j] > 0 else 0.0,
                "lift_binary": float(lift),
                "pmi_binary": float(pmi),
                "ppmi_binary": float(max(pmi, 0.0)),
            })
    rows.sort(key=lambda r: (-r["binary_weight"], -r["count_weight"], r["category_a"], r["category_b"]))
    return rows[:top_k]


def ppmi_matrix(matrix):
    dense = matrix.toarray().astype(np.float64)
    np.fill_diagonal(dense, 0.0)
    total = dense.sum()
    if total <= 0:
        return dense
    row_sum = dense.sum(axis=1)
    col_sum = dense.sum(axis=0)
    expected = np.outer(row_sum, col_sum) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((dense * total) / expected)
    pmi[~np.isfinite(pmi)] = 0.0
    pmi[dense <= 0] = 0.0
    pmi[pmi < 0] = 0.0
    return pmi


def row_normalize_by_diag(matrix):
    dense = matrix.toarray().astype(np.float64)
    diag = np.diag(dense).copy()
    np.fill_diagonal(dense, 0.0)
    out = np.zeros_like(dense)
    valid = diag > 0
    out[valid] = dense[valid] / diag[valid, None]
    return out


def row_raw(matrix):
    dense = matrix.toarray().astype(np.float64)
    np.fill_diagonal(dense, 0.0)
    return dense


def top_indices(scores, exclude=None, valid=None):
    exclude = set(exclude or [])
    order = []
    for idx, score in enumerate(scores):
        if idx in exclude:
            continue
        if valid is not None and not valid[idx]:
            continue
        order.append((idx, float(score)))
    order.sort(key=lambda x: (-x[1], x[0]))
    return order


def format_neighbors(order, category_ids, item_info, category_to_items, dataset, k=5):
    parts = []
    for idx, score in order[:k]:
        reps = representative_texts(dataset, item_info, category_to_items, category_ids[idx], k=1)
        rep = reps[0] if reps else ""
        parts.append({
            "category_id": category_ids[idx],
            "score": float(score),
            "representative_item": rep,
        })
    return json.dumps(parts, ensure_ascii=False)


def category_space_neighbors(dataset, dataset_dir, category_ids, item_info, category_to_items, text_emb, lightgcn_emb, graph_mats):
    rows = []
    text_embeddings, text_present = text_emb if text_emb is not None else (None, None)
    lightgcn_embeddings, lightgcn_present = lightgcn_emb if lightgcn_emb is not None else (None, None)
    graph_cond_binary = graph_mats["graph_conditional_binary"]
    graph_raw_binary = graph_mats["graph_raw_binary"]
    graph_ppmi_binary = graph_mats["graph_ppmi_binary"]

    for idx, category in enumerate(category_ids):
        rep_items = representative_texts(dataset, item_info, category_to_items, category, k=3)
        if text_embeddings is not None and text_present[idx]:
            text_scores = text_embeddings @ text_embeddings[idx]
            text_order = top_indices(text_scores, exclude={idx}, valid=text_present)
        else:
            text_order = []
        if lightgcn_embeddings is not None and lightgcn_present[idx]:
            lightgcn_scores = lightgcn_embeddings @ lightgcn_embeddings[idx]
            lightgcn_order = top_indices(lightgcn_scores, exclude={idx}, valid=lightgcn_present)
        else:
            lightgcn_order = []
        raw_order = top_indices(graph_raw_binary[idx], exclude={idx})
        cond_order = top_indices(graph_cond_binary[idx], exclude={idx})
        ppmi_order = top_indices(graph_ppmi_binary[idx], exclude={idx})
        text_top = {i for i, _ in text_order[:5]}
        lightgcn_top = {i for i, _ in lightgcn_order[:5]}
        cond_top = {i for i, _ in cond_order[:5]}
        rows.append({
            "dataset": dataset,
            "category_id": category,
            "representative_items": json.dumps(rep_items, ensure_ascii=False),
            "nearest_text_categories": format_neighbors(text_order, category_ids, item_info, category_to_items, dataset),
            "nearest_lightgcn_bi_categories": format_neighbors(lightgcn_order, category_ids, item_info, category_to_items, dataset),
            "nearest_graph_raw_binary_categories": format_neighbors(raw_order, category_ids, item_info, category_to_items, dataset),
            "nearest_graph_conditional_binary_categories": format_neighbors(cond_order, category_ids, item_info, category_to_items, dataset),
            "nearest_graph_ppmi_binary_categories": format_neighbors(ppmi_order, category_ids, item_info, category_to_items, dataset),
            "text_graph_conditional_overlap_at5": len(text_top & cond_top),
            "lightgcn_graph_conditional_overlap_at5": len(lightgcn_top & cond_top),
            "text_lightgcn_overlap_at5": len(text_top & lightgcn_top),
        })
    return rows


def parse_bundle_file(path):
    bundle_items = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = [int(v) for v in line.strip().split(", ") if v]
            if len(vals) < 2:
                continue
            bundle_items[int(vals[0])] = vals[1:]
    return bundle_items


def unique_categories(items, item_to_category):
    out = []
    seen = set()
    for item_id in items:
        category = item_to_category.get(int(item_id))
        if category and category not in seen:
            out.append(category)
            seen.add(category)
    return sorted(out)


def category_key(categories):
    return "|".join(sorted(str(c) for c in categories))


def build_itemset_counts(train_bundle_items, item_to_category, max_size=6):
    counts = defaultdict(Counter)
    for items in train_bundle_items.values():
        categories = unique_categories(items, item_to_category)
        for size in range(1, min(max_size, len(categories)) + 1):
            for combo in combinations(categories, size):
                counts[size][category_key(combo)] += 1
    return counts


def direct_confidence_scores(observed_indices, category_ids, itemset_counts, max_size=6):
    observed_categories = [category_ids[i] for i in observed_indices]
    observed_size = len(observed_categories)
    if observed_size == 0 or observed_size + 1 > max_size:
        return None, 0
    observed_count = itemset_counts[observed_size].get(category_key(observed_categories), 0)
    if observed_count <= 0:
        return None, observed_count
    scores = np.zeros(len(category_ids), dtype=np.float64)
    observed_set = set(observed_categories)
    for idx, category in enumerate(category_ids):
        if category in observed_set:
            continue
        joint_count = itemset_counts[observed_size + 1].get(
            category_key(observed_categories + [category]),
            0,
        )
        scores[idx] = joint_count / observed_count
    return scores, observed_count


def score_text_centroid(observed_indices, text_embeddings, text_present):
    if text_embeddings is None or not observed_indices:
        return None
    valid_obs = [i for i in observed_indices if text_present[i]]
    if not valid_obs:
        return None
    query = text_embeddings[valid_obs].mean(axis=0)
    norm = np.linalg.norm(query)
    if norm <= 0:
        return None
    query = query / norm
    return text_embeddings @ query


def score_graph_profile(observed_indices, profile_matrix):
    if not observed_indices:
        return None
    return profile_matrix[observed_indices].mean(axis=0)


def rank_heldout(scores, heldout_indices, observed_indices, valid=None):
    if scores is None:
        return None, None, None
    order = top_indices(scores, exclude=set(observed_indices), valid=valid)
    if not order:
        return None, None, None
    rank_by_idx = {idx: rank + 1 for rank, (idx, _) in enumerate(order)}
    ranks = [rank_by_idx[idx] for idx in heldout_indices if idx in rank_by_idx]
    rank = min(ranks) if ranks else None
    heldout_scores = [
        float(scores[idx])
        for idx in heldout_indices
        if 0 <= idx < len(scores)
    ]
    best_score = max(heldout_scores) if heldout_scores else None
    all_zero = int(all(abs(score) <= 1e-12 for _, score in order))
    return rank, best_score, all_zero


def completion_analysis(dataset, dataset_dir, category_ids, item_to_category, text_emb, lightgcn_emb, graph_mats):
    category_to_idx = {category: idx for idx, category in enumerate(category_ids)}
    input_bundles = parse_bundle_file(os.path.join(dataset_dir, "bi_test_input.txt"))
    gt_bundles = parse_bundle_file(os.path.join(dataset_dir, "bi_test_gt.txt"))
    train_bundles = parse_bundle_file(os.path.join(dataset_dir, "bi_train.txt"))
    itemset_counts = build_itemset_counts(train_bundles, item_to_category, max_size=6)
    text_embeddings, text_present = text_emb if text_emb is not None else (None, None)
    lightgcn_embeddings, lightgcn_present = lightgcn_emb if lightgcn_emb is not None else (None, None)

    methods = {
        "text_centroid_similarity": lambda obs: score_text_centroid(obs, text_embeddings, text_present),
        "graph_raw_count": lambda obs: score_graph_profile(obs, graph_mats["graph_raw_count"]),
        "graph_raw_binary": lambda obs: score_graph_profile(obs, graph_mats["graph_raw_binary"]),
        "graph_conditional_count": lambda obs: score_graph_profile(obs, graph_mats["graph_conditional_count"]),
        "graph_conditional_binary": lambda obs: score_graph_profile(obs, graph_mats["graph_conditional_binary"]),
        "graph_ppmi_count": lambda obs: score_graph_profile(obs, graph_mats["graph_ppmi_count"]),
        "graph_ppmi_binary": lambda obs: score_graph_profile(obs, graph_mats["graph_ppmi_binary"]),
        "set_direct_confidence": lambda obs: direct_confidence_scores(obs, category_ids, itemset_counts, max_size=6)[0],
    }
    valid_by_method = {
        "text_centroid_similarity": text_present,
    }
    if lightgcn_embeddings is not None:
        methods["lightgcn_bi_category_similarity"] = lambda obs: score_text_centroid(
            obs,
            lightgcn_embeddings,
            lightgcn_present,
        )
        valid_by_method["lightgcn_bi_category_similarity"] = lightgcn_present

    detail_rows = []
    common_bundle_ids = sorted(set(input_bundles) & set(gt_bundles))
    for sample_idx, bundle_id in enumerate(common_bundle_ids):
        input_categories = unique_categories(input_bundles.get(bundle_id, []), item_to_category)
        heldout_categories = unique_categories(gt_bundles.get(bundle_id, []), item_to_category)
        heldout_indices = [category_to_idx[c] for c in heldout_categories if c in category_to_idx]
        if not input_categories or not heldout_indices:
            continue
        observed_indices = [category_to_idx[c] for c in input_categories if c in category_to_idx]
        heldout_in_observed = int(any(idx in set(observed_indices) for idx in heldout_indices))
        direct_scores, observed_support = direct_confidence_scores(
            observed_indices,
            category_ids,
            itemset_counts,
            max_size=6,
        )
        for method, scorer in methods.items():
            if method == "set_direct_confidence":
                scores = direct_scores
            else:
                scores = scorer(observed_indices)
            rank, heldout_score, all_zero = rank_heldout(
                scores,
                heldout_indices,
                observed_indices,
                valid=valid_by_method.get(method),
            )
            detail_rows.append({
                "dataset": dataset,
                "sample_idx": sample_idx,
                "bundle_id": bundle_id,
                "input_categories": json.dumps(input_categories, ensure_ascii=False),
                "heldout_categories": json.dumps(heldout_categories, ensure_ascii=False),
                "heldout_size": len(heldout_indices),
                "heldout_in_observed": heldout_in_observed,
                "observed_category_count": len(observed_indices),
                "observed_support": int(observed_support),
                "method": method,
                "gt_rank": rank if rank is not None else "",
                "heldout_best_score": heldout_score if heldout_score is not None else "",
                "all_zero_scores": all_zero if all_zero is not None else "",
                "hit_at_1": int(rank == 1) if rank is not None else 0,
                "hit_at_3": int(rank is not None and rank <= 3),
                "hit_at_5": int(rank is not None and rank <= 5),
                "reciprocal_rank": float(1.0 / rank) if rank else 0.0,
                "covered": int(rank is not None),
            })
    return detail_rows


def summarize_completion(detail_df):
    rows = []
    for (dataset, method), group in detail_df.groupby(["dataset", "method"]):
        covered = group[group["covered"] == 1]
        rows.append({
            "dataset": dataset,
            "method": method,
            "n": int(len(group)),
            "coverage": float(group["covered"].mean()) if len(group) else 0.0,
            "all_zero_rate": float(pd.to_numeric(group["all_zero_scores"], errors="coerce").fillna(0).mean()) if len(group) else 0.0,
            "hit_at_1": float(covered["hit_at_1"].mean()) if len(covered) else 0.0,
            "hit_at_3": float(covered["hit_at_3"].mean()) if len(covered) else 0.0,
            "hit_at_5": float(covered["hit_at_5"].mean()) if len(covered) else 0.0,
            "mrr": float(covered["reciprocal_rank"].mean()) if len(covered) else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["dataset", "hit_at_3", "mrr"], ascending=[True, False, False])


def quadrant_analysis(detail_df, graph_method="graph_conditional_binary", text_method="text_centroid_similarity"):
    pivot = detail_df.pivot_table(
        index=["dataset", "sample_idx"],
        columns="method",
        values="hit_at_3",
        aggfunc="first",
    ).reset_index()
    rows = []
    for dataset, group in pivot.groupby("dataset"):
        if text_method not in group.columns or graph_method not in group.columns:
            continue
        text_hit = group[text_method].fillna(0).astype(int)
        graph_hit = group[graph_method].fillna(0).astype(int)
        labels = np.select(
            [
                (text_hit == 1) & (graph_hit == 1),
                (text_hit == 1) & (graph_hit == 0),
                (text_hit == 0) & (graph_hit == 1),
            ],
            ["both_hit", "text_only", "graph_only"],
            default="both_miss",
        )
        counts = Counter(labels)
        total = len(group)
        for label in ["both_hit", "text_only", "graph_only", "both_miss"]:
            rows.append({
                "dataset": dataset,
                "text_method": text_method,
                "graph_method": graph_method,
                "quadrant": label,
                "count": int(counts[label]),
                "ratio": float(counts[label] / total) if total else 0.0,
            })
    return pd.DataFrame(rows)


def write_summary(path, graph_stats_df, completion_summary_df, quadrant_df):
    lines = ["# Category Graph Space Analysis", ""]
    lines.append("## Graph Statistics")
    for _, row in graph_stats_df.iterrows():
        if row["split"] != "train":
            continue
        lines.append(
            f"- {row['dataset']} {row['mode']} train: "
            f"edges={int(row['undirected_edges'])}, density={row['density']:.3f}, "
            f"largest_component={int(row['largest_component_size'])}/{int(row['num_categories'])}"
        )
    lines.append("")
    lines.append("## Completion Summary")
    for dataset, group in completion_summary_df.groupby("dataset"):
        lines.append(f"### {dataset}")
        top = group.sort_values(["hit_at_3", "mrr"], ascending=[False, False]).head(8)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['method']}: hit@1={row['hit_at_1']:.3f}, "
                f"hit@3={row['hit_at_3']:.3f}, hit@5={row['hit_at_5']:.3f}, "
                f"MRR={row['mrr']:.3f}, coverage={row['coverage']:.3f}"
            )
    lines.append("")
    lines.append("## Text vs Graph Hit@3 Quadrants")
    for _, row in quadrant_df.iterrows():
        lines.append(
            f"- {row['dataset']} {row['quadrant']}: "
            f"{int(row['count'])} ({row['ratio']:.3f})"
        )
    lines.append("")
    lines.append("Interpretation: text centroid similarity measures semantic category similarity, while category graph profiles measure bundle complementarity. A large graph-only quadrant supports the claim that category completion is structural rather than semantic similarity.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_method_quadrants(detail_df, output_dir):
    specs = [
        ("text_vs_graph_conditional", "text_centroid_similarity", "graph_conditional_binary"),
        ("text_vs_lightgcn_bi", "text_centroid_similarity", "lightgcn_bi_category_similarity"),
        ("lightgcn_bi_vs_graph_conditional", "lightgcn_bi_category_similarity", "graph_conditional_binary"),
    ]
    frames = []
    for name, left, right in specs:
        q = quadrant_analysis(detail_df, graph_method=right, text_method=left)
        if q.empty:
            continue
        q["comparison"] = name
        frames.append(q)
        q.to_csv(os.path.join(output_dir, f"{name}_quadrants.csv"), index=False, encoding="utf-8-sig")
    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame()
    out.to_csv(os.path.join(output_dir, "category_method_quadrants.csv"), index=False, encoding="utf-8-sig")
    return out


def analyze_dataset(repo_root, dataset, output_root, model, category_dtype):
    dataset_dir = os.path.join(repo_root, "datasets", dataset)
    category_ids = load_category_ids(dataset_dir)
    item_info, category_field, item_to_category, category_to_items = build_item_category_maps(
        dataset_dir,
        category_ids,
    )

    matrices = {}
    graph_stats = []
    pair_rows = []
    for split in ["train", "full"]:
        count_matrix = load_cc(dataset_dir, split, "count")
        binary_matrix = load_cc(dataset_dir, split, "binary")
        for mode, matrix in [("count", count_matrix), ("binary", binary_matrix)]:
            graph_stats.append(matrix_graph_stats(dataset, split, mode, matrix, category_ids))
        pair_rows.extend(graph_pair_rows(dataset, split, count_matrix, binary_matrix, category_ids))
        if split == "train":
            matrices["graph_raw_count"] = row_raw(count_matrix)
            matrices["graph_raw_binary"] = row_raw(binary_matrix)
            matrices["graph_conditional_count"] = row_normalize_by_diag(count_matrix)
            matrices["graph_conditional_binary"] = row_normalize_by_diag(binary_matrix)
            matrices["graph_ppmi_count"] = ppmi_matrix(count_matrix)
            matrices["graph_ppmi_binary"] = ppmi_matrix(binary_matrix)

    text_emb = load_category_embeddings(repo_root, dataset, category_ids, model, category_dtype)
    lightgcn_emb = load_category_embeddings(repo_root, dataset, category_ids, "LightGCN_bi", "float32")
    neighbor_rows = category_space_neighbors(
        dataset,
        dataset_dir,
        category_ids,
        item_info,
        category_to_items,
        text_emb,
        lightgcn_emb,
        matrices,
    )
    detail_rows = completion_analysis(
        dataset,
        dataset_dir,
        category_ids,
        item_to_category,
        text_emb,
        lightgcn_emb,
        matrices,
    )

    dataset_output = os.path.join(output_root, dataset)
    os.makedirs(dataset_output, exist_ok=True)
    pd.DataFrame(graph_stats).to_csv(os.path.join(dataset_output, "category_graph_stats.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(pair_rows).to_csv(os.path.join(dataset_output, "category_graph_top_pairs.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(neighbor_rows).to_csv(os.path.join(dataset_output, "category_space_neighbors.csv"), index=False, encoding="utf-8-sig")
    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(os.path.join(dataset_output, "category_graph_completion_detail.csv"), index=False, encoding="utf-8-sig")
    summary_df = summarize_completion(detail_df)
    summary_df.to_csv(os.path.join(dataset_output, "category_graph_completion_summary.csv"), index=False, encoding="utf-8-sig")
    quadrant_df = quadrant_analysis(detail_df)
    quadrant_df.to_csv(os.path.join(dataset_output, "category_text_graph_quadrants.csv"), index=False, encoding="utf-8-sig")
    method_quadrant_df = write_method_quadrants(detail_df, dataset_output)
    write_summary(
        os.path.join(dataset_output, "summary.md"),
        pd.DataFrame(graph_stats),
        summary_df,
        quadrant_df,
    )

    return {
        "graph_stats": pd.DataFrame(graph_stats),
        "pairs": pd.DataFrame(pair_rows),
        "neighbors": pd.DataFrame(neighbor_rows),
        "detail": detail_df,
        "summary": summary_df,
        "quadrants": quadrant_df,
        "method_quadrants": method_quadrant_df,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze category graph spaces and completion signal.")
    parser.add_argument("--datasets", nargs="+", default=["pog", "pog_dense"])
    parser.add_argument("--output_dir", default=os.path.join("analysis", "category_graph_space"))
    parser.add_argument("--embedding_model", default="text-embedding-3-large")
    parser.add_argument("--category_dtype", default="float32")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_root = os.path.join(repo_root, args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    combined = defaultdict(list)
    for dataset in args.datasets:
        result = analyze_dataset(
            repo_root,
            dataset,
            output_root,
            args.embedding_model,
            args.category_dtype,
        )
        for key, df in result.items():
            combined[key].append(df)

    combined_graph_stats = pd.concat(combined["graph_stats"], ignore_index=True)
    combined_summary = pd.concat(combined["summary"], ignore_index=True)
    combined_quadrants = pd.concat(combined["quadrants"], ignore_index=True)
    combined_method_quadrants = pd.concat(combined["method_quadrants"], ignore_index=True)
    combined_graph_stats.to_csv(os.path.join(output_root, "category_graph_stats_all.csv"), index=False, encoding="utf-8-sig")
    combined_summary.to_csv(os.path.join(output_root, "category_graph_completion_summary_all.csv"), index=False, encoding="utf-8-sig")
    combined_quadrants.to_csv(os.path.join(output_root, "category_text_graph_quadrants_all.csv"), index=False, encoding="utf-8-sig")
    combined_method_quadrants.to_csv(os.path.join(output_root, "category_method_quadrants_all.csv"), index=False, encoding="utf-8-sig")
    write_summary(
        os.path.join(output_root, "summary.md"),
        combined_graph_stats,
        combined_summary,
        combined_quadrants,
    )
    print(f"[Done] Wrote category graph space analysis to {output_root}")


if __name__ == "__main__":
    main()

import json
import os
from pathlib import Path

import numpy as np


def read_bundle_file(path):
    bundles = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = [int(v) for v in line.strip().split(", ") if v]
            if len(vals) >= 3:
                bundles.append((vals[0], vals[1:]))
    return bundles


def option_char(index):
    return chr(ord("A") + int(index))


class InputEmbeddingICLRetriever:
    def __init__(self, conf, dataset):
        self.conf = conf
        self.dataset = dataset
        self.dataset_name = conf["dataset"]
        self.num_cans = int(conf.get("icl_example_num_cans", conf.get("num_cans", 10)))
        self.num_token = int(conf.get("icl_example_num_token", conf.get("num_token", 5)))
        self.seed = int(conf.get("icl_example_seed", conf.get("seed", 45)))
        self.max_train_examples = int(conf.get("icl_max_train_examples", -1))
        self.exclude_query_candidate_items = bool(conf.get("icl_exclude_query_candidate_items", True))

        self.cache = self._load_embedding_cache()
        self.examples = self._build_train_examples()
        self.example_matrix = self._build_example_matrix()
        print(
            f">>> ICL retrieval ready: {len(self.examples)} train examples, "
            f"embedding dim {self.example_matrix.shape[1]}"
        )

    def _load_embedding_cache(self):
        cache_root = Path(
            self.conf.get(
                "icl_embedding_cache_root",
                os.path.join(
                    "analysis",
                    "openai_embedding_cache",
                    "text-embedding-3-large",
                    "all_items",
                ),
            )
        )
        if not cache_root.is_absolute():
            cache_root = Path.cwd() / cache_root

        model = self.conf.get("icl_embedding_model", "text-embedding-3-large")
        dtype = self.conf.get("icl_embedding_dtype", "float16")
        cache_path = cache_root / self.dataset_name / f"embeddings_{model}_{dtype}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"ICL embedding cache not found: {cache_path}. "
                "Run src/build_openai_embedding_cache.py --mode all-items first."
            )

        with np.load(cache_path, allow_pickle=False) as data:
            ids = data["ids"].astype(np.int64)
            embeddings = data["embeddings"].astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1)
        safe = norms > 0
        embeddings_normed = np.zeros_like(embeddings, dtype=np.float32)
        embeddings_normed[safe] = embeddings[safe] / norms[safe, None]

        contiguous = bool(len(ids) > 0 and ids[0] == 0 and np.all(ids == np.arange(len(ids))))
        id_to_row = None if contiguous else {int(item_id): idx for idx, item_id in enumerate(ids.tolist())}
        return {
            "path": cache_path,
            "ids": ids,
            "embeddings": embeddings_normed,
            "id_to_row": id_to_row,
            "contiguous": contiguous,
            "all_item_ids": ids.tolist(),
        }

    def _rows_for_ids(self, item_ids):
        if self.cache["contiguous"]:
            rows = np.asarray(item_ids, dtype=np.int64)
            if rows.min(initial=0) < 0 or rows.max(initial=0) >= len(self.cache["ids"]):
                raise KeyError(f"Item id out of cache range: {item_ids}")
            return rows
        missing = [int(item_id) for item_id in item_ids if int(item_id) not in self.cache["id_to_row"]]
        if missing:
            raise KeyError(f"Missing item ids from ICL cache: {missing[:20]}")
        return np.asarray([self.cache["id_to_row"][int(item_id)] for item_id in item_ids], dtype=np.int64)

    def _mean_embedding(self, item_ids):
        rows = self._rows_for_ids(item_ids)
        vec = self.cache["embeddings"][rows].mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _build_train_examples(self):
        train_path = Path(self.conf.get("data_path", "./datasets")) / self.dataset_name / "bi_train.txt"
        bundles = read_bundle_file(train_path)
        if self.max_train_examples > 0:
            bundles = bundles[: self.max_train_examples]

        examples = []
        for bundle_id, items in bundles:
            if len(items) < 2:
                continue
            rng = np.random.default_rng(bundle_id + self.seed)
            shuffled = np.asarray(items, dtype=np.int64).copy()
            rng.shuffle(shuffled)
            gt_item = int(shuffled[0])
            input_items = shuffled[1:].astype(int).tolist()
            if self.num_token > 0 and len(input_items) > self.num_token:
                input_items = input_items[: self.num_token]
            if not input_items:
                continue
            examples.append(
                {
                    "bundle_id": int(bundle_id),
                    "input_indices": input_items,
                    "true_indice": gt_item,
                    "bundle_items": [int(x) for x in items],
                }
            )
        if not examples:
            raise ValueError(f"No train examples built from {train_path}")
        return examples

    def _build_example_matrix(self):
        vectors = [self._mean_embedding(ex["input_indices"]) for ex in self.examples]
        return np.vstack(vectors).astype(np.float32)

    def _sample_candidates(self, example, query_candidate_ids):
        excluded = set(int(x) for x in example["bundle_items"])
        if self.exclude_query_candidate_items:
            excluded.update(int(x) for x in query_candidate_ids)
        excluded.discard(int(example["true_indice"]))

        rng = np.random.default_rng(
            int(example["bundle_id"]) + int(self.seed) + 1000003
        )
        negatives = []
        seen = set([int(example["true_indice"])])
        all_ids = self.cache["all_item_ids"]
        while len(negatives) < self.num_cans - 1:
            cand = int(all_ids[int(rng.integers(0, len(all_ids)))])
            if cand in excluded or cand in seen:
                continue
            seen.add(cand)
            negatives.append(cand)

        candidates = np.asarray([int(example["true_indice"])] + negatives, dtype=np.int64)
        rng.shuffle(candidates)
        candidates = candidates.astype(int).tolist()
        true_option_idx = candidates.index(int(example["true_indice"]))
        return candidates, true_option_idx

    def _format_options(self, candidate_ids):
        return "; ".join(
            [
                f"{option_char(idx)}. {self.dataset.get_item_text(item_id)}"
                for idx, item_id in enumerate(candidate_ids)
            ]
        )

    def _format_inputs(self, input_ids):
        return "; ".join(
            [
                f"{idx + 1}. {self.dataset.get_item_text(item_id)}"
                for idx, item_id in enumerate(input_ids)
            ]
        )

    def retrieve(self, sample):
        query_input_ids = [int(x) for x in sample.get("input_indices", [])]
        query_candidate_ids = [int(x) for x in sample.get("candidate_indices", [])]
        query_vec = self._mean_embedding(query_input_ids)
        scores = self.example_matrix @ query_vec
        order = np.argsort(-scores, kind="mergesort")
        query_candidate_set = set(query_candidate_ids)

        selected_rank = None
        selected_index = None
        for rank, ex_idx in enumerate(order, start=1):
            example = self.examples[int(ex_idx)]
            if self.exclude_query_candidate_items and int(example["true_indice"]) in query_candidate_set:
                continue
            selected_rank = rank
            selected_index = int(ex_idx)
            break

        if selected_index is None:
            raise ValueError("No eligible ICL example found after applying leakage filters")

        example = dict(self.examples[selected_index])
        candidates, true_option_idx = self._sample_candidates(example, query_candidate_ids)
        true_option_char = option_char(true_option_idx)
        input_overlap = len(set(query_input_ids) & set(example["input_indices"]))
        candidate_overlap = len(set(query_candidate_ids) & set(candidates))

        example.update(
            {
                "candidate_indices": candidates,
                "true_option_idx": true_option_idx,
                "true_option_char": true_option_char,
                "input_str": self._format_inputs(example["input_indices"]),
                "target_str": self._format_options(candidates),
                "true_item_text": self.dataset.get_item_text(example["true_indice"]),
                "retrieval_score": float(scores[selected_index]),
                "retrieval_rank_after_filter": int(selected_rank),
                "retrieval_pool_index": int(selected_index),
                "query_input_overlap_count": int(input_overlap),
                "query_candidate_overlap_count": int(candidate_overlap),
                "example_gt_in_query_candidates": int(example["true_indice"] in query_candidate_set),
            }
        )
        return example

    def metadata_for_csv(self, example):
        return {
            "icl_example_bundle_id": example["bundle_id"],
            "icl_example_true_indice": example["true_indice"],
            "icl_example_true_option_idx": example["true_option_idx"],
            "icl_example_true_option_char": example["true_option_char"],
            "icl_retrieval_score": example["retrieval_score"],
            "icl_retrieval_rank_after_filter": example["retrieval_rank_after_filter"],
            "icl_retrieval_pool_index": example["retrieval_pool_index"],
            "icl_example_input_indices": json.dumps(example["input_indices"], ensure_ascii=False),
            "icl_example_candidate_indices": json.dumps(example["candidate_indices"], ensure_ascii=False),
            "icl_query_input_overlap_count": example["query_input_overlap_count"],
            "icl_query_candidate_overlap_count": example["query_candidate_overlap_count"],
            "icl_example_gt_in_query_candidates": example["example_gt_in_query_candidates"],
        }

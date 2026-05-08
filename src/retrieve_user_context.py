import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_user_item_file(path):
    users = {}
    item_to_users = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = [int(v) for v in line.strip().split(", ") if v]
            if len(vals) < 2:
                continue
            user_id = vals[0]
            items = vals[1:]
            users[user_id] = items
            for item_id in items:
                item_to_users[item_id].append(user_id)
    return users, item_to_users


class UserContextRetriever:
    def __init__(self, conf, dataset):
        self.conf = conf
        self.dataset = dataset
        self.dataset_name = conf["dataset"]
        self.top_k = int(conf.get("user_context_top_k", 5))
        self.seed = int(conf.get("user_context_seed", conf.get("seed", 45)))
        self.exclude_candidates = bool(conf.get("user_context_exclude_candidate_items", True))
        self.min_overlap = int(conf.get("user_context_min_input_overlap", 1))

        self.cache = self._load_embedding_cache()
        ui_path = Path(conf.get("data_path", "./datasets")) / self.dataset_name / "ui_full.txt"
        self.users, self.item_to_users = read_user_item_file(ui_path)
        print(
            f">>> User context ready: {len(self.users)} users, "
            f"{len(self.item_to_users)} indexed items, top_k={self.top_k}"
        )

    def _load_embedding_cache(self):
        cache_root = Path(
            self.conf.get(
                "user_context_embedding_cache_root",
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
        model = self.conf.get("user_context_embedding_model", "text-embedding-3-large")
        dtype = self.conf.get("user_context_embedding_dtype", "float16")
        cache_path = cache_root / self.dataset_name / f"embeddings_{model}_{dtype}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"User context embedding cache not found: {cache_path}. "
                "Run src/build_openai_embedding_cache.py --mode all-items first."
            )

        with np.load(cache_path, allow_pickle=False) as data:
            ids = data["ids"].astype(np.int64)
            embeddings = data["embeddings"].astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1)
        normalized = np.zeros_like(embeddings, dtype=np.float32)
        safe = norms > 0
        normalized[safe] = embeddings[safe] / norms[safe, None]
        contiguous = bool(len(ids) > 0 and ids[0] == 0 and np.all(ids == np.arange(len(ids))))
        id_to_row = None if contiguous else {int(item_id): idx for idx, item_id in enumerate(ids.tolist())}
        return {
            "path": cache_path,
            "ids": ids,
            "embeddings": normalized,
            "contiguous": contiguous,
            "id_to_row": id_to_row,
        }

    def _rows_for_ids(self, item_ids):
        if self.cache["contiguous"]:
            rows = np.asarray(item_ids, dtype=np.int64)
            if rows.min(initial=0) < 0 or rows.max(initial=0) >= len(self.cache["ids"]):
                raise KeyError(f"Item id out of cache range: {item_ids}")
            return rows
        missing = [int(item_id) for item_id in item_ids if int(item_id) not in self.cache["id_to_row"]]
        if missing:
            raise KeyError(f"Missing item ids from user context cache: {missing[:20]}")
        return np.asarray([self.cache["id_to_row"][int(item_id)] for item_id in item_ids], dtype=np.int64)

    def _mean_embedding(self, item_ids):
        rows = self._rows_for_ids(item_ids)
        vec = self.cache["embeddings"][rows].mean(axis=0)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _select_user(self, input_ids, bundle_id):
        counts = defaultdict(int)
        for item_id in input_ids:
            for user_id in self.item_to_users.get(int(item_id), []):
                counts[user_id] += 1
        if not counts:
            return None, 0, 0
        max_overlap = max(counts.values())
        if max_overlap < self.min_overlap:
            return None, max_overlap, 0
        candidate_users = [user_id for user_id, overlap in counts.items() if overlap == max_overlap]
        rng = np.random.default_rng(int(bundle_id) + self.seed)
        selected = int(candidate_users[int(rng.integers(0, len(candidate_users)))])
        return selected, int(max_overlap), len(candidate_users)

    def _select_history_items(self, user_items, input_ids, candidate_ids):
        excluded = set(int(x) for x in input_ids)
        if self.exclude_candidates:
            excluded.update(int(x) for x in candidate_ids)
        pool = [int(item_id) for item_id in user_items if int(item_id) not in excluded]
        if not pool:
            return [], []

        query_vec = self._mean_embedding([int(x) for x in input_ids])
        rows = self._rows_for_ids(pool)
        scores = self.cache["embeddings"][rows] @ query_vec
        order = np.argsort(-scores, kind="mergesort")[: self.top_k]
        selected_items = [pool[int(i)] for i in order]
        selected_scores = [float(scores[int(i)]) for i in order]
        return selected_items, selected_scores

    def retrieve(self, sample):
        input_ids = [int(x) for x in sample.get("input_indices", [])]
        candidate_ids = [int(x) for x in sample.get("candidate_indices", [])]
        user_id, input_overlap, tie_pool_size = self._select_user(input_ids, sample.get("bundle_id", 0))
        if user_id is None:
            return None

        user_items = [int(x) for x in self.users[user_id]]
        selected_items, selected_scores = self._select_history_items(user_items, input_ids, candidate_ids)
        context = {
            "user_id": int(user_id),
            "input_overlap_count": int(input_overlap),
            "tie_pool_size": int(tie_pool_size),
            "user_history_size": int(len(user_items)),
            "selected_item_indices": selected_items,
            "selected_item_scores": selected_scores,
            "selected_item_texts": [self.dataset.get_item_text(item_id) for item_id in selected_items],
            "excluded_candidate_items": int(self.exclude_candidates),
        }
        return context

    def format_context(self, context):
        if not context or not context["selected_item_indices"]:
            return ""
        if "spotify" in self.dataset_name:
            entity = "listener"
            item_name = "songs"
        else:
            entity = "shopper"
            item_name = "fashion items"

        lines = [
            (
                f"Historical user context: A {entity} who interacted with "
                f"{context['input_overlap_count']} of the given input {item_name} "
                f"also interacted with these related {item_name}:"
            )
        ]
        for idx, text in enumerate(context["selected_item_texts"], start=1):
            lines.append(f"{idx}. {text}")
        lines.append("Use this only as supplementary evidence about user preference.")
        return "\n".join(lines) + "\n\n"

    def metadata_for_csv(self, context):
        if not context:
            return {
                "user_context_user_id": "",
                "user_context_input_overlap_count": 0,
                "user_context_tie_pool_size": 0,
                "user_context_history_size": 0,
                "user_context_selected_item_indices": "[]",
                "user_context_selected_item_scores": "[]",
            }
        return {
            "user_context_user_id": context["user_id"],
            "user_context_input_overlap_count": context["input_overlap_count"],
            "user_context_tie_pool_size": context["tie_pool_size"],
            "user_context_history_size": context["user_history_size"],
            "user_context_selected_item_indices": json.dumps(context["selected_item_indices"], ensure_ascii=False),
            "user_context_selected_item_scores": json.dumps(context["selected_item_scores"], ensure_ascii=False),
        }

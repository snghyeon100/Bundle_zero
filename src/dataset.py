import os
import json
import random
import numpy as np
import scipy.sparse as sp
from collections import defaultdict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def list2pairs(file):
    pairs = []
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            l = [int(i) for i in line.strip().split(", ")]
            b_id = l[0]
            for i_id in l[1:]:
                pairs.append([b_id, i_id])
    return np.array(pairs)

def pairs2csr(pairs, shape):
    indice = np.array(pairs, dtype=np.int32)
    values = np.ones(len(pairs), dtype=np.float32)
    return sp.csr_matrix((values, (indice[:, 0], indice[:, 1])), shape=shape)

class BundleZeroShotDataset:
    def __init__(self, conf):
        self.conf = conf
        self.path = conf["data_path"]
        self.name = conf["dataset"]
        self.num_cans = conf["num_cans"]
        self.toy_eval = conf["toy_eval"]
        self.num_token = conf["num_token"]
        self.seed = conf.get("seed", 45)
        self.shuffle_seed = conf.get("shuffle_seed", 45)
        self.use_item_bundle_affiliation_desc = conf.get("use_item_bundle_affiliation_desc", False)
        self.item_bundle_affiliation_k = conf.get("item_bundle_affiliation_k", 3)
        self.item_bundle_affiliation_alpha = conf.get("item_bundle_affiliation_alpha", 0.5)
        self.item_bundle_affiliation_seed = conf.get("item_bundle_affiliation_seed", self.seed)
        self.item_bundle_affiliation_exclude_query_items = conf.get("item_bundle_affiliation_exclude_query_items", False)
        self.item_bundle_affiliation_matrix = None
        self.use_item_user_copurchase_desc = conf.get("use_item_user_copurchase_desc", False)
        self.item_user_copurchase_k = conf.get("item_user_copurchase_k", 3)
        self.item_user_copurchase_alpha = conf.get("item_user_copurchase_alpha", 0.5)
        self.item_user_copurchase_seed = conf.get("item_user_copurchase_seed", self.seed)
        self.item_user_copurchase_exclude_query_items = conf.get("item_user_copurchase_exclude_query_items", False)
        self.item_user_copurchase_matrix = None
        self.use_cooccurrence = conf.get("use_cooccurrence", False)
        self.item_to_bundles = None
        self.use_soft_cooccurrence = conf.get("use_soft_cooccurrence", False)
        self.soft_cooccurrence_source = conf.get("soft_cooccurrence_source", "item_smoothing_text")
        self.soft_item_to_bundles = None
        self.use_bundle_graph_context = conf.get("use_bundle_graph_context", False)
        self.bundle_graph_context_k = conf.get("bundle_graph_context_k", 1)
        self.bundle_graph_context_max_items = conf.get("bundle_graph_context_max_items", 5)
        self.bundle_graph_context_seed = conf.get("bundle_graph_context_seed", self.seed)
        self.bundle_graph_train_matrix = None
        self.bundle_graph_train_items = {}
        self.bundle_graph_train_bundle_ids = np.array([], dtype=np.int32)
        self.bundle_graph_item_idf = None
        
        # Load counts
        count_path = os.path.join(self.path, self.name, 'count.json')
        with open(count_path, 'r') as f:
            stat = json.loads(f.read())
        self.num_bundles, self.num_items = stat["#B"], stat["#I"]
        self.num_users = stat.get("#U", 0)

        # Load Item info for text
        info_path = os.path.join(self.path, self.name, 'item_info.json')
        with open(info_path, 'r', encoding='utf-8') as f:
            self.item_info = json.loads(f.read())

        # Load Test Graphics
        self.b_i_pairs_i = list2pairs(os.path.join(self.path, self.name, 'bi_test_input.txt'))
        self.b_i_pairs_gt = list2pairs(os.path.join(self.path, self.name, 'bi_test_gt.txt'))
        np.random.shuffle(self.b_i_pairs_gt) # Shuffle GT for testing sequence
        
        self.b_i_graph_i = pairs2csr(self.b_i_pairs_i, (self.num_bundles, self.num_items))
        self.b_i_graph_gt = pairs2csr(self.b_i_pairs_gt, (self.num_bundles, self.num_items))

        self.len_max = int(self.b_i_graph_i.sum(axis=1).max())
        if self.num_token > 0 and self.len_max > self.num_token:
            self.len_max = self.num_token

        if self.use_item_bundle_affiliation_desc:
            self.item_bundle_affiliation_matrix = self._build_item_bundle_affiliation_matrix()
            print(f"[Item Bundle Affiliation] Loaded BI^T BI from bi_train.txt with {self.item_bundle_affiliation_matrix.nnz} item-item links")
        if self.use_item_user_copurchase_desc:
            self.item_user_copurchase_matrix = self._build_item_user_copurchase_matrix()
            print(f"[Item User Co-purchase] Loaded UI^T UI from ui_full.txt with {self.item_user_copurchase_matrix.nnz} item-item links")
        if self.use_cooccurrence or self.use_soft_cooccurrence:
            self.item_to_bundles = self._build_item_to_bundles()
            print(f"[Co-occurrence] Loaded {len(self.item_to_bundles)} items from bi_train.txt")
        if self.use_soft_cooccurrence:
            self.soft_item_to_bundles = self._load_soft_item_to_bundles()
            print(f"[Soft Co-occurrence] Loaded {len(self.soft_item_to_bundles)} items from {self.soft_cooccurrence_source}")
        if self.use_bundle_graph_context:
            self._build_bundle_graph_context_index()
            print(f"[Bundle Graph Context] Loaded {len(self.bundle_graph_train_items)} train bundles from bi_train.txt")

        # Apply toy_eval truncating
        if self.toy_eval > 0:
            self.b_i_pairs_gt = self.b_i_pairs_gt[:self.toy_eval]

    @staticmethod
    def _clean_inline_text(text):
        return " ".join(str(text).split())

    def _build_item_bundle_affiliation_matrix(self):
        """Build item-item co-affiliation counts from train bundles only."""
        train_path = os.path.join(self.path, self.name, 'bi_train.txt')
        if not os.path.exists(train_path):
            print(f"[Warning] bi_train.txt not found: {train_path}")
            return sp.csr_matrix((self.num_items, self.num_items), dtype=np.float32)

        train_pairs = list2pairs(train_path)
        b_i_graph_train = pairs2csr(train_pairs, (self.num_bundles, self.num_items))
        item_item = (b_i_graph_train.T @ b_i_graph_train).tocsr()
        item_item.setdiag(0)
        item_item.eliminate_zeros()
        return item_item

    def _load_train_bundle_items(self):
        train_path = os.path.join(self.path, self.name, 'bi_train.txt')
        if not os.path.exists(train_path):
            print(f"[Warning] bi_train.txt not found: {train_path}")
            return {}

        bundle_items = {}
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                vals = [int(v) for v in line.strip().split(", ") if v]
                if len(vals) < 2:
                    continue
                bundle_id = vals[0]
                seen = set()
                items = []
                for item_id in vals[1:]:
                    if item_id not in seen:
                        items.append(item_id)
                        seen.add(item_id)
                if items:
                    bundle_items[bundle_id] = items
        return bundle_items

    def _build_item_to_bundles(self):
        """Build item -> train bundle set for candidate co-occurrence scores."""
        item_to_bundles = defaultdict(set)
        for bundle_id, items in self._load_train_bundle_items().items():
            for item_id in items:
                item_to_bundles[item_id].add(bundle_id)
        return item_to_bundles

    def _soft_cooccurrence_path(self):
        file_by_source = {
            "item_smoothing_text": "item_smoothing_i2bprime_text_top1.json",
            "bundle_smoothing_text": "bundle_smoothing_i2bprime_text_top1.json",
            "bundle_smoothing_bi_lgcn": "bundle_smoothing_i2bprime_bi_lgcn_top1.json",
        }
        if self.soft_cooccurrence_source not in file_by_source:
            allowed = ", ".join(sorted(file_by_source))
            raise ValueError(f"Unknown soft_cooccurrence_source={self.soft_cooccurrence_source}. Allowed: {allowed}")
        return os.path.join(self.path, self.name, file_by_source[self.soft_cooccurrence_source])

    def _load_soft_item_to_bundles(self):
        """Load precomputed item -> soft train bundle B' mapping."""
        soft_path = self._soft_cooccurrence_path()
        if not os.path.exists(soft_path):
            raise FileNotFoundError(
                f"Soft co-occurrence mapping not found: {soft_path}. "
                "Run analyzer_utility/build_soft_i2bprime_mappings.py first."
            )
        with open(soft_path, "r", encoding="utf-8") as f:
            raw_mapping = json.load(f)
        return {
            int(item_id): {int(bundle_id) for bundle_id in bundle_ids}
            for item_id, bundle_ids in raw_mapping.items()
        }

    def _build_bundle_graph_context_index(self):
        """Build a train-only bundle-item index for similar bundle retrieval."""
        self.bundle_graph_train_items = self._load_train_bundle_items()
        if not self.bundle_graph_train_items:
            self.bundle_graph_train_matrix = sp.csr_matrix((self.num_bundles, self.num_items), dtype=np.float32)
            self.bundle_graph_train_bundle_ids = np.array([], dtype=np.int32)
            self.bundle_graph_item_idf = np.zeros(self.num_items, dtype=np.float64)
            return

        self.bundle_graph_train_bundle_ids = np.array(
            list(self.bundle_graph_train_items.keys()),
            dtype=np.int32
        )

        pairs = [
            (bundle_id, item_id)
            for bundle_id, items in self.bundle_graph_train_items.items()
            for item_id in items
        ]
        self.bundle_graph_train_matrix = pairs2csr(pairs, (self.num_bundles, self.num_items))
        self.bundle_graph_train_matrix.sum_duplicates()
        self.bundle_graph_train_matrix.data[:] = 1.0
        self.bundle_graph_train_matrix.eliminate_zeros()

        item_freq = np.asarray(self.bundle_graph_train_matrix.sum(axis=0)).ravel()
        num_train_bundles = len(self.bundle_graph_train_items)
        self.bundle_graph_item_idf = np.log((num_train_bundles + 1.0) / (item_freq + 1.0))

    def _build_item_user_copurchase_matrix(self):
        """Build item-item co-purchase counts from user-item interactions."""
        ui_path = os.path.join(self.path, self.name, 'ui_full.txt')
        if not os.path.exists(ui_path):
            print(f"[Warning] ui_full.txt not found: {ui_path}")
            return sp.csr_matrix((self.num_items, self.num_items), dtype=np.float32)

        ui_pairs = list2pairs(ui_path)
        num_users = self.num_users or int(ui_pairs[:, 0].max()) + 1
        u_i_graph = pairs2csr(ui_pairs, (num_users, self.num_items))
        u_i_graph.sum_duplicates()
        u_i_graph.data[:] = 1.0
        u_i_graph.eliminate_zeros()
        item_item = (u_i_graph.T @ u_i_graph).tocsr()
        item_item.setdiag(0)
        item_item.eliminate_zeros()
        return item_item

    def _sample_item_neighbors(self, matrix, item_id, k, alpha, seed, exclude_indices=None):
        if matrix is None or k <= 0:
            return []

        row = matrix.getrow(int(item_id))
        if row.nnz == 0:
            return []

        indices = row.indices
        counts = row.data.astype(np.float64, copy=False)
        if exclude_indices:
            exclude_set = {int(i) for i in exclude_indices}
            keep_mask = np.array([int(i) not in exclude_set for i in indices], dtype=bool)
            indices = indices[keep_mask]
            counts = counts[keep_mask]

        if len(indices) == 0:
            return []

        weights = np.power(counts, alpha)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            return []

        rng = np.random.default_rng(int(item_id) + int(seed))
        sample_size = min(int(k), len(indices))
        sampled = rng.choice(indices, size=sample_size, replace=False, p=weights / weight_sum)
        return [int(i) for i in sampled]

    def _sample_item_bundle_affiliates(self, item_id, exclude_indices=None):
        return self._sample_item_neighbors(
            self.item_bundle_affiliation_matrix,
            item_id,
            self.item_bundle_affiliation_k,
            self.item_bundle_affiliation_alpha,
            self.item_bundle_affiliation_seed,
            exclude_indices=exclude_indices
        )

    def _sample_item_user_copurchases(self, item_id, exclude_indices=None):
        return self._sample_item_neighbors(
            self.item_user_copurchase_matrix,
            item_id,
            self.item_user_copurchase_k,
            self.item_user_copurchase_alpha,
            self.item_user_copurchase_seed,
            exclude_indices=exclude_indices
        )

    def get_item_text_with_contexts(self, item_id, bundle_exclude_indices=None, user_exclude_indices=None):
        item_text = self._clean_inline_text(self.get_item_text(item_id))
        context_sentences = []

        if self.use_item_bundle_affiliation_desc:
            sampled_items = self._sample_item_bundle_affiliates(item_id, exclude_indices=bundle_exclude_indices)
            if sampled_items:
                sampled_text = "; ".join(self._clean_inline_text(self.get_item_text(j)) for j in sampled_items)
                if "spotify" in self.name:
                    context_sentences.append(f"This song is frequently included in playlists with: {sampled_text}.")
                else:
                    context_sentences.append(f"This item is frequently included in outfits with: {sampled_text}.")

        if self.use_item_user_copurchase_desc:
            sampled_items = self._sample_item_user_copurchases(item_id, exclude_indices=user_exclude_indices)
            if sampled_items:
                sampled_text = "; ".join(self._clean_inline_text(self.get_item_text(j)) for j in sampled_items)
                if "spotify" in self.name:
                    context_sentences.append(f"This song frequently appears in user histories with: {sampled_text}.")
                else:
                    context_sentences.append(f"This item is frequently co-purchased with: {sampled_text}.")

        if not context_sentences:
            return item_text

        return f"{item_text} [Additional context: {' '.join(context_sentences)}]"

    def retrieve_bundle_graph_context(self, sample):
        if (
            not self.use_bundle_graph_context
            or self.bundle_graph_train_matrix is None
            or self.bundle_graph_context_k <= 0
        ):
            return None

        input_indices = [int(i) for i in sample.get("input_indices", [])]
        input_indices = [i for i in dict.fromkeys(input_indices) if 0 <= i < self.num_items]
        if not input_indices:
            return None

        overlap = np.asarray(self.bundle_graph_train_matrix[:, input_indices].sum(axis=1)).ravel()
        candidate_bundle_ids = np.intersect1d(
            self.bundle_graph_train_bundle_ids,
            np.flatnonzero(overlap > 0),
            assume_unique=False
        )
        if len(candidate_bundle_ids) == 0:
            return None

        idf_weights = self.bundle_graph_item_idf[input_indices]
        idf_overlap = np.asarray(
            self.bundle_graph_train_matrix[:, input_indices].multiply(idf_weights).sum(axis=1)
        ).ravel()

        rng = np.random.default_rng(
            int(sample.get("bundle_id", 0)) + int(self.bundle_graph_context_seed)
        )
        tie_break = rng.random(len(candidate_bundle_ids))
        order = np.lexsort((
            tie_break,
            -idf_overlap[candidate_bundle_ids],
            -overlap[candidate_bundle_ids],
        ))
        selected_bundle_ids = [int(i) for i in candidate_bundle_ids[order[:int(self.bundle_graph_context_k)]]]

        examples = []
        metadata = {
            "bundle_graph_context_bundle_ids": selected_bundle_ids,
            "bundle_graph_context_overlap_counts": [int(overlap[i]) for i in selected_bundle_ids],
            "bundle_graph_context_idf_scores": [float(idf_overlap[i]) for i in selected_bundle_ids],
        }
        for bundle_id in selected_bundle_ids:
            items = self.bundle_graph_train_items.get(bundle_id, [])
            if not items:
                continue
            selected_items = items[:int(self.bundle_graph_context_max_items)]
            item_texts = [self._clean_inline_text(self.get_item_text(item_id)) for item_id in selected_items]
            examples.append({
                "bundle_id": bundle_id,
                "item_indices": selected_items,
                "item_texts": item_texts,
            })

        if not examples:
            return None

        if "spotify" in self.name:
            header = "Additional context: Similar historical playlists based on shared input songs:"
        else:
            header = "Additional context: Similar historical outfits based on shared input items:"

        lines = [header]
        for idx, example in enumerate(examples, start=1):
            lines.append(f"{idx}. {'; '.join(example['item_texts'])}.")
        return {
            "context_block": "\n".join(lines) + "\n\n",
            "examples": examples,
            "metadata": metadata,
        }

    def get_cooccurrence_stats(self, input_indices, candidate_indices):
        """Return train-only co-bundled stats normalized by candidate bundle frequency."""
        if self.item_to_bundles is None:
            self.item_to_bundles = self._build_item_to_bundles()

        input_bundle_union = set()
        for inp_id in input_indices:
            input_bundle_union.update(self.item_to_bundles.get(int(inp_id), set()))

        stats = []
        for cand_id in candidate_indices:
            cand_bundles = self.item_to_bundles.get(int(cand_id), set())
            shared_count = len(cand_bundles & input_bundle_union)
            candidate_bundle_count = len(cand_bundles)
            stats.append({
                "shared_train_bundles": shared_count,
                "candidate_train_bundles": candidate_bundle_count,
            })
        return stats

    def get_cooccurrence_scores(self, input_indices, candidate_indices):
        """Backward-compatible raw shared train bundle counts."""
        return [
            stat["shared_train_bundles"]
            for stat in self.get_cooccurrence_stats(input_indices, candidate_indices)
        ]

    def get_soft_cooccurrence_stats(self, input_indices, candidate_indices):
        """Return co-bundled stats using a precomputed soft I -> B' mapping for input items."""
        if self.item_to_bundles is None:
            self.item_to_bundles = self._build_item_to_bundles()
        if self.soft_item_to_bundles is None:
            self.soft_item_to_bundles = self._load_soft_item_to_bundles()

        soft_input_bundle_union = set()
        for inp_id in input_indices:
            soft_input_bundle_union.update(self.soft_item_to_bundles.get(int(inp_id), set()))

        stats = []
        for cand_id in candidate_indices:
            cand_bundles = self.item_to_bundles.get(int(cand_id), set())
            shared_count = len(cand_bundles & soft_input_bundle_union)
            candidate_bundle_count = len(cand_bundles)
            stats.append({
                "shared_train_bundles": shared_count,
                "candidate_train_bundles": candidate_bundle_count,
                "soft_input_train_bundles": len(soft_input_bundle_union),
                "source": self.soft_cooccurrence_source,
            })
        return stats

    def get_item_text(self, item_id):
        item_id_str = str(int(item_id))
        if "pog" in self.name:
            return self.item_info[item_id_str].get("title", f"Item {item_id_str}")
        elif "spotify" in self.name:
            info = self.item_info[item_id_str]
            # Combination of track, artist, album
            desc = []
            if "track_name" in info: desc.append(info["track_name"])
            if "artist_name" in info: desc.append(info["artist_name"])
            if "album_name" in info: desc.append(info["album_name"])
            return " - ".join(desc) if desc else f"Track {item_id_str}"
        return f"Item {item_id_str}"

    def get_eval_samples(self):
        samples = []
        for b_idx, true_indice in self.b_i_pairs_gt:
            # Current bundle contents
            b_i_i_np = self.b_i_graph_i[b_idx].toarray().squeeze()
            b_i_gt_np = self.b_i_graph_gt[b_idx].toarray().squeeze()

            # 1. 후보군(A-J)용 고정 시드 생성
            rng_cand = np.random.default_rng(int(b_idx) + self.seed)
            
            # Find false indices
            false_indices = np.argwhere((b_i_i_np + b_i_gt_np) == 0).reshape(-1)
            false_indices = rng_cand.choice(false_indices, size=self.num_cans - 1, replace=False)
            
            # Form candidates list and shuffle using candidate-rng
            indices = np.concatenate([[true_indice], false_indices])
            rng_cand.shuffle(indices)
            true_idx = int(np.argwhere(indices == true_indice)[0][0])
            
            # 2. 인풋 아이템용 고정 시드 생성 (후보군과 분리)
            rng_input = np.random.default_rng(int(b_idx) + self.shuffle_seed)
            
            input_indices = np.argwhere(b_i_i_np > 0).reshape(-1)
            # 인풋 아이템만 이 시드에 따라 고정된 무작위 순서로 섞음
            rng_input.shuffle(input_indices)
            if self.num_token > 0 and len(input_indices) > self.num_token:
                input_indices = input_indices[:self.num_token]
            
            # Build string inputs
            query_indices = set(input_indices.tolist()) | set(indices.tolist())
            bundle_exclude_indices = query_indices if self.item_bundle_affiliation_exclude_query_items else None
            user_exclude_indices = query_indices if self.item_user_copurchase_exclude_query_items else None
            if self.use_item_bundle_affiliation_desc or self.use_item_user_copurchase_desc:
                input_str = "; ".join([
                    f"{idx + 1}. {self.get_item_text_with_contexts(j, bundle_exclude_indices=bundle_exclude_indices, user_exclude_indices=user_exclude_indices)}"
                    for idx, j in enumerate(input_indices)
                ])
                target_str = "; ".join([
                    f"{chr(ord('A') + idx)}. {self.get_item_text_with_contexts(j, bundle_exclude_indices=bundle_exclude_indices, user_exclude_indices=user_exclude_indices)}"
                    for idx, j in enumerate(indices)
                ])
            else:
                input_str = "; ".join([f"{idx + 1}. {self.get_item_text(j)}" for idx, j in enumerate(input_indices)])
                target_str = "; ".join([f"{chr(ord('A') + idx)}. {self.get_item_text(j)}" for idx, j in enumerate(indices)])
            
            samples.append({
                "bundle_id": int(b_idx),
                "true_indice": int(true_indice),
                "true_option_idx": true_idx,
                "true_option_char": chr(ord('A') + true_idx),
                "input_indices": input_indices.tolist(),
                "candidate_indices": indices.tolist(),
                "input_str": input_str,
                "target_str": target_str
            })
            
        return samples

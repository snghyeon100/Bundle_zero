import os
import json
import random
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from collections import Counter
from itertools import combinations

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
        if conf.get("use_fixed_test_split", False):
            self.test_input_file = conf.get("test_input_file", "bi_fix_test_input.txt")
            self.test_gt_file = conf.get("test_gt_file", "bi_fix_test_gt.txt")
        else:
            self.test_input_file = conf.get("test_input_file", "bi_test_input.txt")
            self.test_gt_file = conf.get("test_gt_file", "bi_test_gt.txt")
        self.use_item_bundle_affiliation_desc = conf.get("use_item_bundle_affiliation_desc", False)
        self.item_bundle_affiliation_k = conf.get("item_bundle_affiliation_k", 3)
        self.item_bundle_affiliation_alpha = conf.get("item_bundle_affiliation_alpha", 0.5)
        self.item_bundle_affiliation_seed = conf.get("item_bundle_affiliation_seed", self.seed)
        self.item_bundle_affiliation_exclude_query_items = conf.get("item_bundle_affiliation_exclude_query_items", False)
        self.item_bundle_affiliation_use_soft = conf.get("item_bundle_affiliation_use_soft", False)
        self.item_bundle_affiliation_soft_source = conf.get("item_bundle_affiliation_soft_source", "bundle_smoothing_bi_lgcn")
        self.item_bundle_affiliation_soft_alpha = conf.get("item_bundle_affiliation_soft_alpha", 1.0)
        self.item_bundle_affiliation_soft_item_to_bundles = None
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
        self.bundle_graph_context_use_soft = conf.get("bundle_graph_context_use_soft", False)
        self.bundle_graph_context_soft_source = conf.get("bundle_graph_context_soft_source", "bundle_smoothing_bi_lgcn")
        self.bundle_graph_context_soft_alpha = conf.get("bundle_graph_context_soft_alpha", 1.0)
        self.bundle_graph_context_soft_item_to_bundles = None
        self.bundle_graph_train_matrix = None
        self.bundle_graph_train_items = {}
        self.bundle_graph_train_bundle_ids = np.array([], dtype=np.int32)
        self.bundle_graph_item_idf = None
        self.use_ui_category_purchase_prior = conf.get("use_ui_category_purchase_prior", False)
        self.ui_category_purchase_prior_top_k = conf.get("ui_category_purchase_prior_top_k", 5)
        self.ui_category_purchase_prior_min_support = conf.get("ui_category_purchase_prior_min_support", 1)
        self.ui_category_purchase_category_counts = Counter()
        self.ui_category_purchase_pair_counts = defaultdict(Counter)
        self.ui_category_purchase_num_users = 0
        self.use_input_item_description_aug = conf.get("use_input_item_description_aug", False)
        self.input_item_description_cache_root = conf.get(
            "input_item_description_cache_root",
            os.path.join("analysis", "input_item_descriptions", "gemini")
        )
        self.input_item_description_field = conf.get("input_item_description_field", "description")
        self.input_item_descriptions_by_item = {}
        self.use_category_evidence_summary = conf.get("use_category_evidence_summary", False)
        self.category_evidence_summary_k = conf.get("category_evidence_summary_k", 5)
        self.category_evidence_summary_include_evidence = conf.get("category_evidence_summary_include_evidence", False)
        self.use_cc_retrieval_context = conf.get("use_cc_retrieval_context", False)
        self.cc_retrieval_context_k = conf.get("cc_retrieval_context_k", 1)
        self.cc_retrieval_context_seed = conf.get("cc_retrieval_context_seed", self.seed)
        self.cc_retrieval_overlap_weight = conf.get("cc_retrieval_overlap_weight", 1.0)
        self.cc_retrieval_extra_weight = conf.get("cc_retrieval_extra_weight", 1.0)
        self.cc_retrieval_train_bundle_items = {}
        self.cc_retrieval_train_categories = {}
        self.use_category_completion_prior_desc = conf.get("use_category_completion_prior_desc", False)
        self.use_category_item_text_aug = conf.get("use_category_item_text_aug", False)
        self.category_item_aug_apply_to = conf.get("category_item_aug_apply_to", "both")
        self.category_item_aug_rep_items_per_category = conf.get("category_item_aug_rep_items_per_category", 2)
        self.use_category_name_aug = conf.get("use_category_name_aug", False)
        self.category_name_aug_apply_to = conf.get("category_name_aug_apply_to", "both")
        self.category_name_field = conf.get("category_name_field", "category_name_en")
        self.category_name_root = conf.get(
            "category_name_root",
            os.path.join("analysis", "category_names", "gemini")
        )
        self.input_category_co_occur = conf.get("input_category_co_occur", False)
        self.input_category_co_occur_apply_to = conf.get("input_category_co_occur_apply_to", "inputs")
        self.input_category_co_occur_verbalization = conf.get(
            "input_category_co_occur_verbalization",
            "representative_items"
        )
        self.input_category_co_occur_top_k = conf.get("input_category_co_occur_top_k", 3)
        self.input_category_co_occur_rep_items_per_category = conf.get("input_category_co_occur_rep_items_per_category", 1)
        self.category_prior_top_k = conf.get("category_prior_top_k", 3)
        self.category_prior_verbalization = conf.get("category_prior_verbalization", "representative_items")
        self.category_prior_rep_items_per_category = conf.get("category_prior_rep_items_per_category", 3)
        self.category_prior_min_support = conf.get("category_prior_min_support", 3)
        self.category_prior_max_itemset_size = conf.get("category_prior_max_itemset_size", 6)
        self.category_prior_embedding_model = conf.get("category_prior_embedding_model", "text-embedding-3-large")
        self.category_prior_embedding_dtype = conf.get("category_prior_embedding_dtype", "float16")
        self.category_prior_category_dtype = conf.get("category_prior_category_dtype", "float32")
        self.category_prior_item_embedding_cache_root = conf.get(
            "category_prior_item_embedding_cache_root",
            os.path.join("analysis", "openai_embedding_cache")
        )
        self.category_prior_category_embedding_cache_root = conf.get(
            "category_prior_category_embedding_cache_root",
            os.path.join("analysis", "category_embedding_cache")
        )
        self.category_prior_itemset_counts = defaultdict(Counter)
        self.category_prior_train_category_counts = Counter()
        self.category_pair_cooccur_counts = defaultdict(Counter)
        self.category_prior_categories = []
        self.category_prior_item_category_field = None
        self.category_prior_train_items_by_category = defaultdict(list)
        self.category_prior_ranked_rep_items_by_category = {}
        self.category_names_by_category = {}
        
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

        # Load test split. Fixed split files hold out exactly one GT item per bundle.
        test_input_path = os.path.join(self.path, self.name, self.test_input_file)
        test_gt_path = os.path.join(self.path, self.name, self.test_gt_file)
        if not os.path.exists(test_input_path):
            raise FileNotFoundError(f"Test input file not found: {test_input_path}")
        if not os.path.exists(test_gt_path):
            raise FileNotFoundError(f"Test GT file not found: {test_gt_path}")
        print(f"[Test Split] input={self.test_input_file}, gt={self.test_gt_file}")
        self.b_i_pairs_i = list2pairs(test_input_path)
        self.b_i_pairs_gt = list2pairs(test_gt_path)
        np.random.shuffle(self.b_i_pairs_gt) # Shuffle GT for testing sequence
        
        self.b_i_graph_i = pairs2csr(self.b_i_pairs_i, (self.num_bundles, self.num_items))
        self.b_i_graph_gt = pairs2csr(self.b_i_pairs_gt, (self.num_bundles, self.num_items))

        self.len_max = int(self.b_i_graph_i.sum(axis=1).max())
        if self.num_token > 0 and self.len_max > self.num_token:
            self.len_max = self.num_token

        if self.use_item_bundle_affiliation_desc:
            self.item_bundle_affiliation_matrix = self._build_item_bundle_affiliation_matrix()
            print(f"[Item Bundle Affiliation] Loaded BI^T BI from bi_train.txt with {self.item_bundle_affiliation_matrix.nnz} item-item links")
            if self.item_bundle_affiliation_use_soft:
                soft_matrix = self._build_soft_item_bundle_affiliation_matrix(
                    self.item_bundle_affiliation_soft_source
                )
                if self.item_bundle_affiliation_soft_alpha != 1.0:
                    soft_matrix = soft_matrix.copy()
                    soft_matrix.data *= float(self.item_bundle_affiliation_soft_alpha)
                self.item_bundle_affiliation_matrix = (
                    self.item_bundle_affiliation_matrix + soft_matrix
                ).tocsr()
                self.item_bundle_affiliation_matrix.setdiag(0)
                self.item_bundle_affiliation_matrix.eliminate_zeros()
                print(
                    f"[Item Bundle Affiliation] Added soft I-B' x BI from "
                    f"{self.item_bundle_affiliation_soft_source} with {soft_matrix.nnz} item-item links "
                    f"(alpha={self.item_bundle_affiliation_soft_alpha})"
                )
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
            if self.bundle_graph_context_use_soft:
                self.bundle_graph_context_soft_item_to_bundles = self._load_soft_item_to_bundles(
                    self.bundle_graph_context_soft_source
                )
                print(
                    f"[Bundle Graph Context] Loaded soft I-B' mapping for "
                    f"{len(self.bundle_graph_context_soft_item_to_bundles)} items from "
                    f"{self.bundle_graph_context_soft_source}"
                )

        if self.use_input_item_description_aug:
            self._load_input_item_descriptions()
            print(
                f"[Input Item Description] Loaded {len(self.input_item_descriptions_by_item)} "
                f"cached item descriptions"
            )

        if self.use_cc_retrieval_context or self.use_category_evidence_summary:
            self._build_cc_retrieval_context_index()
            print(f"[C-C Retrieval Context] Loaded {len(self.cc_retrieval_train_bundle_items)} train bundles from bi_train.txt")

        if (
            self.use_category_evidence_summary
            or self.use_category_completion_prior_desc
            or self.use_category_item_text_aug
            or self.use_category_name_aug
            or self.input_category_co_occur
            or self.use_cc_retrieval_context
            or self.use_ui_category_purchase_prior
        ):
            self._build_category_completion_prior()
            if self.use_ui_category_purchase_prior:
                self._build_ui_category_purchase_prior()
            if (
                (
                    self.use_category_completion_prior_desc
                    and self._category_prior_uses_representative_items()
                )
                or self.use_category_item_text_aug
                or (
                    self.input_category_co_occur
                    and self._input_category_co_occur_uses_representative_items()
                )
            ):
                self._build_category_representative_index()
            if (
                self.use_category_name_aug
                or self.use_category_evidence_summary
                or self.use_ui_category_purchase_prior
                or (
                    self.use_category_completion_prior_desc
                    and self._category_prior_uses_category_names()
                )
                or (
                    self.input_category_co_occur
                    and self._input_category_co_occur_uses_category_names()
                )
            ):
                self._load_category_names()
            print(
                f"[Category Context] Loaded {len(self.category_prior_categories)} categories "
                f"from bi_train.txt using field={self.category_prior_item_category_field}"
            )

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

    def _build_soft_item_bundle_affiliation_matrix(self, source):
        """Build item-item co-affiliation counts through soft item -> train bundle B' links."""
        train_bundle_items = self._load_train_bundle_items()
        if not train_bundle_items:
            return sp.csr_matrix((self.num_items, self.num_items), dtype=np.float32)

        self.item_bundle_affiliation_soft_item_to_bundles = self._load_soft_item_to_bundles(source)
        rows = []
        cols = []
        for item_id, bundle_ids in self.item_bundle_affiliation_soft_item_to_bundles.items():
            if not 0 <= int(item_id) < self.num_items:
                continue
            for bundle_id in bundle_ids:
                for co_item_id in train_bundle_items.get(int(bundle_id), []):
                    if 0 <= int(co_item_id) < self.num_items and int(co_item_id) != int(item_id):
                        rows.append(int(item_id))
                        cols.append(int(co_item_id))

        if not rows:
            return sp.csr_matrix((self.num_items, self.num_items), dtype=np.float32)

        values = np.ones(len(rows), dtype=np.float32)
        item_item = sp.csr_matrix(
            (values, (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
            shape=(self.num_items, self.num_items)
        )
        item_item.sum_duplicates()
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

    def _detect_item_category_field(self):
        field_candidates = ["cate_id", "cate", "category"]
        counts = {field: 0 for field in field_candidates}
        for item in self.item_info.values():
            for field in field_candidates:
                value = item.get(field)
                if value is not None and str(value).strip():
                    counts[field] += 1
        best = max(counts, key=counts.get)
        if counts[best] == 0:
            return None
        return best

    def _get_item_category(self, item_id):
        if self.category_prior_item_category_field is None:
            self.category_prior_item_category_field = self._detect_item_category_field()
        if self.category_prior_item_category_field is None:
            return None
        item = self.item_info.get(str(int(item_id)), {})
        value = item.get(self.category_prior_item_category_field)
        if value is None or not str(value).strip():
            return None
        return str(value).strip()

    @staticmethod
    def _category_key(categories):
        return "|".join(sorted(str(c) for c in categories))

    def _items_to_unique_categories(self, item_ids):
        categories = []
        seen = set()
        for item_id in item_ids:
            category = self._get_item_category(item_id)
            if category and category not in seen:
                categories.append(category)
                seen.add(category)
        return sorted(categories)

    def _build_category_completion_prior(self):
        train_bundle_items = self._load_train_bundle_items()
        max_size = max(1, int(self.category_prior_max_itemset_size))
        for items in train_bundle_items.values():
            categories = self._items_to_unique_categories(items)
            if not categories:
                continue
            for item_id in items:
                category = self._get_item_category(item_id)
                if category:
                    self.category_prior_train_items_by_category[category].append(int(item_id))
            for category in categories:
                self.category_prior_train_category_counts[category] += 1
            for cat_a, cat_b in combinations(categories, 2):
                self.category_pair_cooccur_counts[cat_a][cat_b] += 1
                self.category_pair_cooccur_counts[cat_b][cat_a] += 1
            for size in range(1, min(max_size, len(categories)) + 1):
                for combo in combinations(categories, size):
                    self.category_prior_itemset_counts[size][self._category_key(combo)] += 1
        self.category_prior_categories = sorted(self.category_prior_train_category_counts)

    def _build_ui_category_purchase_prior(self):
        ui_path = os.path.join(self.path, self.name, "ui_full.txt")
        if not os.path.exists(ui_path):
            print(f"[Warning] ui_full.txt not found: {ui_path}")
            return

        with open(ui_path, "r", encoding="utf-8") as f:
            for line in f:
                vals = [int(v) for v in line.strip().split(", ") if v]
                if len(vals) < 2:
                    continue
                unique_items = list(dict.fromkeys(vals[1:]))
                categories = self._items_to_unique_categories(unique_items)
                if not categories:
                    continue
                self.ui_category_purchase_num_users += 1
                for category in categories:
                    self.ui_category_purchase_category_counts[category] += 1
                for cat_a, cat_b in combinations(categories, 2):
                    self.ui_category_purchase_pair_counts[cat_a][cat_b] += 1
                    self.ui_category_purchase_pair_counts[cat_b][cat_a] += 1

    def _ui_category_purchase_support(self, input_categories, candidate_category):
        scores = []
        pair_counts = []
        for input_category in input_categories:
            denom = self.ui_category_purchase_category_counts.get(input_category, 0)
            pair_count = self.ui_category_purchase_pair_counts[input_category].get(candidate_category, 0)
            if denom <= 0:
                continue
            scores.append(pair_count / denom)
            pair_counts.append(pair_count)
        if not scores:
            return 0.0, 0
        return float(np.mean(scores)), int(sum(pair_counts))

    def retrieve_ui_category_purchase_prior_context(self, sample):
        if (
            not self.use_ui_category_purchase_prior
            or "spotify" in self.name
            or int(self.ui_category_purchase_prior_top_k) <= 0
        ):
            return None

        input_indices = [int(i) for i in sample.get("input_indices", [])]
        input_categories = self._items_to_unique_categories(input_indices)
        if not input_categories:
            return None

        input_set = set(input_categories)
        min_support = int(self.ui_category_purchase_prior_min_support)
        scored = []
        for category in sorted(self.ui_category_purchase_category_counts):
            if category in input_set:
                continue
            score, pair_support = self._ui_category_purchase_support(input_categories, category)
            if pair_support < min_support or score <= 0:
                continue
            scored.append((category, score, pair_support, self.ui_category_purchase_category_counts[category]))

        scored.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
        top = scored[:int(self.ui_category_purchase_prior_top_k)]
        if not top:
            return None

        input_category_names = [self._category_display_name(category) for category in input_categories]
        top_category_names = [self._category_display_name(category) for category, _, _, _ in top]
        lines = [
            "Aggregate user-purchase category prior:",
            (
                "Users who purchased items in the input categories frequently also purchased "
                "items from these categories:"
            ),
        ]
        for idx, (category, score, pair_support, category_support) in enumerate(top, start=1):
            lines.append(f"{idx}. {self._category_display_name(category)}")
        lines.append(
            "Use this only as a soft category-level signal from aggregate user behavior. "
            "The selected item should still complete the outfit without duplicating an input category."
        )

        return {
            "context_block": "\n".join(lines) + "\n\n",
            "metadata": {
                "ui_category_purchase_prior_input_categories": json.dumps(input_categories, ensure_ascii=False),
                "ui_category_purchase_prior_input_category_names": json.dumps(input_category_names, ensure_ascii=False),
                "ui_category_purchase_prior_top_categories": json.dumps([category for category, _, _, _ in top], ensure_ascii=False),
                "ui_category_purchase_prior_top_category_names": json.dumps(top_category_names, ensure_ascii=False),
                "ui_category_purchase_prior_scores": json.dumps([score for _, score, _, _ in top], ensure_ascii=False),
                "ui_category_purchase_prior_pair_supports": json.dumps([pair_support for _, _, pair_support, _ in top], ensure_ascii=False),
                "ui_category_purchase_prior_category_supports": json.dumps([category_support for _, _, _, category_support in top], ensure_ascii=False),
                "ui_category_purchase_prior_selected_count": int(len(top)),
            },
        }

    def _resolve_repo_relative_path(self, path):
        if os.path.isabs(path):
            return path
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(repo_root, path)

    def _load_input_item_descriptions(self):
        desc_path = os.path.join(
            self._resolve_repo_relative_path(self.input_item_description_cache_root),
            self.name,
            "input_item_descriptions.json"
        )
        if not os.path.exists(desc_path):
            raise FileNotFoundError(
                f"Input item description cache not found: {desc_path}. "
                "Run analyzer_utility/generate_input_item_descriptions_gemini.py first."
            )

        with open(desc_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = raw.get("items", raw)
        descriptions = {}
        for item_id, record in records.items():
            if isinstance(record, dict):
                text = record.get(self.input_item_description_field, "")
            else:
                text = record
            text = self._clean_inline_text(text)
            if text:
                descriptions[int(item_id)] = text
        self.input_item_descriptions_by_item = descriptions

    def _input_item_description_aug_text(self, item_id):
        description = self.input_item_descriptions_by_item.get(int(item_id), "")
        if not description:
            return ""
        label = "Generated item summary"
        if "description" in str(self.input_item_description_field).lower():
            label = "Generated item description"
        return f"{label}: {description}."

    def _load_category_names(self):
        names_path = os.path.join(
            self._resolve_repo_relative_path(self.category_name_root),
            self.name,
            "category_names.json"
        )
        if not os.path.exists(names_path):
            raise FileNotFoundError(
                f"Category name file not found: {names_path}. "
                "Run analyzer_utility/generate_category_names_gemini.py first, "
                "or disable use_category_name_aug / category-name co-occurrence verbalization."
            )

        with open(names_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        records = payload.get("categories", payload if isinstance(payload, list) else [])
        self.category_names_by_category = {
            str(record.get("category_id", "")).strip(): record
            for record in records
            if str(record.get("category_id", "")).strip()
        }
        print(
            f"[Category Names] Loaded {len(self.category_names_by_category)} names "
            f"from {names_path} using field={self.category_name_field}"
        )

    def _category_display_name(self, category):
        record = self.category_names_by_category.get(str(category))
        if not record:
            return ""
        fields = [
            self.category_name_field,
            "category_name_en",
            "category_name_ko",
            "short_description_en",
        ]
        for field in fields:
            value = record.get(field)
            if value is not None and str(value).strip():
                return self._clean_inline_text(value)
        return ""

    def _load_category_prior_item_embeddings(self):
        cache_root = self._resolve_repo_relative_path(self.category_prior_item_embedding_cache_root)
        cache_dir = os.path.join(cache_root, self.category_prior_embedding_model, "all_items", self.name)
        cache_path = os.path.join(
            cache_dir,
            f"embeddings_{self.category_prior_embedding_model}_{self.category_prior_embedding_dtype}.npz"
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Category prior item embedding cache not found: {cache_path}. "
                "Run analyzer_utility/build_openai_embedding_cache.py --mode all-items first."
            )
        with np.load(cache_path, allow_pickle=False) as data:
            ids = data["ids"].astype(np.int64)
            embeddings = data["embeddings"].astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1)
        safe = norms > 0
        embeddings_normed = np.zeros_like(embeddings, dtype=np.float32)
        embeddings_normed[safe] = embeddings[safe] / norms[safe, None]
        return ids, embeddings_normed, {int(item_id): idx for idx, item_id in enumerate(ids.tolist())}

    def _load_category_prior_category_embeddings(self):
        cache_root = self._resolve_repo_relative_path(self.category_prior_category_embedding_cache_root)
        cache_dir = os.path.join(cache_root, self.category_prior_embedding_model, "all_items", self.name)
        cache_path = os.path.join(
            cache_dir,
            f"category_embeddings_{self.category_prior_embedding_model}_{self.category_prior_category_dtype}.npz"
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Category embedding cache not found: {cache_path}. "
                "Run analyzer_utility/build_category_embedding_cache.py first."
            )
        with np.load(cache_path, allow_pickle=False) as data:
            category_ids = data["category_ids"].astype(str)
            embeddings_normed = data["embeddings_normed"].astype(np.float32)
        return {
            str(category_id): idx
            for idx, category_id in enumerate(category_ids.tolist())
        }, embeddings_normed

    def _build_category_representative_index(self):
        item_ids, item_embeddings, item_to_row = self._load_category_prior_item_embeddings()
        category_to_row, category_embeddings = self._load_category_prior_category_embeddings()
        del item_ids

        for category, item_list in self.category_prior_train_items_by_category.items():
            if category not in category_to_row:
                continue
            unique_items = sorted(set(int(item_id) for item_id in item_list if int(item_id) in item_to_row))
            if not unique_items:
                continue
            rows = [item_to_row[item_id] for item_id in unique_items]
            centroid = category_embeddings[category_to_row[category]]
            scores = item_embeddings[rows] @ centroid
            order = np.argsort(-scores, kind="mergesort")
            self.category_prior_ranked_rep_items_by_category[category] = [
                int(unique_items[int(i)]) for i in order.tolist()
            ]

    def _category_prior_scores(self, observed_categories):
        observed = sorted(set(observed_categories))
        observed_size = len(observed)
        if observed_size == 0:
            return {}, 0
        if observed_size + 1 > int(self.category_prior_max_itemset_size):
            return {}, 0
        observed_count = self.category_prior_itemset_counts[observed_size].get(
            self._category_key(observed),
            0
        )
        if observed_count < int(self.category_prior_min_support):
            return {}, observed_count
        scores = {}
        observed_set = set(observed)
        for category in self.category_prior_categories:
            if category in observed_set:
                continue
            joint_count = self.category_prior_itemset_counts[observed_size + 1].get(
                self._category_key(observed + [category]),
                0
            )
            scores[category] = joint_count / observed_count
        return scores, observed_count

    def _representative_items_for_category(self, category, exclude_items, max_items=None):
        reps = []
        exclude = {int(item_id) for item_id in exclude_items}
        limit = int(max_items if max_items is not None else self.category_prior_rep_items_per_category)
        if limit <= 0:
            return reps
        for item_id in self.category_prior_ranked_rep_items_by_category.get(category, []):
            if int(item_id) in exclude:
                continue
            reps.append(int(item_id))
            if len(reps) >= limit:
                break
        return reps

    def _category_item_aug_enabled_for_role(self, role):
        if not self.use_category_item_text_aug:
            return False
        apply_to = str(self.category_item_aug_apply_to).strip().lower()
        if apply_to == "both":
            return role in {"input", "candidate"}
        if apply_to in {"inputs", "input"}:
            return role == "input"
        if apply_to in {"candidates", "candidate"}:
            return role == "candidate"
        return False

    def _category_item_aug_text(self, item_id, exclude_items):
        if "spotify" in self.name:
            return ""
        category = self._get_item_category(item_id)
        if not category:
            return ""

        rep_items = self._representative_items_for_category(
            category,
            exclude_items,
            max_items=self.category_item_aug_rep_items_per_category
        )
        if not rep_items:
            return ""

        rep_text = "; ".join(self._clean_inline_text(self.get_item_text(j)) for j in rep_items)
        return f"Item Category examples: {rep_text}."

    def _category_name_aug_enabled_for_role(self, role):
        if not self.use_category_name_aug:
            return False
        apply_to = str(self.category_name_aug_apply_to).strip().lower()
        if apply_to == "both":
            return role in {"input", "candidate"}
        if apply_to in {"inputs", "input"}:
            return role == "input"
        if apply_to in {"candidates", "candidate"}:
            return role == "candidate"
        return False

    def _category_name_aug_text(self, item_id):
        if "spotify" in self.name:
            return ""
        category = self._get_item_category(item_id)
        if not category:
            return ""
        name = self._category_display_name(category)
        if not name:
            return ""
        return f"Item category: {name}."

    def _input_category_co_occur_uses_category_names(self):
        mode = str(self.input_category_co_occur_verbalization).strip().lower()
        return mode in {"category_names", "category_name", "names", "name"}

    def _input_category_co_occur_uses_representative_items(self):
        return not self._input_category_co_occur_uses_category_names()

    def _category_prior_uses_category_names(self):
        mode = str(self.category_prior_verbalization).strip().lower()
        return mode in {"category_names", "category_name", "names", "name"}

    def _category_prior_uses_representative_items(self):
        return not self._category_prior_uses_category_names()

    def _input_category_co_occur_text(self, item_id, exclude_items):
        if "spotify" in self.name:
            return ""
        category = self._get_item_category(item_id)
        if not category:
            return ""
        top_k = int(self.input_category_co_occur_top_k)
        if top_k <= 0:
            return ""

        neighbors = sorted(
            self.category_pair_cooccur_counts.get(category, {}).items(),
            key=lambda x: (-x[1], x[0])
        )
        if not neighbors:
            return ""

        if self._input_category_co_occur_uses_category_names():
            names = []
            selected_categories = 0
            for neighbor_category, _ in neighbors:
                name = self._category_display_name(neighbor_category)
                if not name:
                    continue
                names.append(name)
                selected_categories += 1
                if selected_categories >= top_k:
                    break

            if not names:
                return ""
            return (
                "Category context: this item's category often appears with these "
                f"other categories: {'; '.join(names)}."
            )

        rep_texts = []
        selected_categories = 0
        for neighbor_category, _ in neighbors:
            rep_items = self._representative_items_for_category(
                neighbor_category,
                exclude_items,
                max_items=self.input_category_co_occur_rep_items_per_category
            )
            if not rep_items:
                continue
            for rep_item in rep_items:
                rep_texts.append(self._clean_inline_text(self.get_item_text(rep_item)))
            selected_categories += 1
            if selected_categories >= top_k:
                break

        if not rep_texts:
            return ""
        return (
            "Category context: this item's category often appears with the following "
            f"other categories, each represented by one example item: {'; '.join(rep_texts)}."
        )

    def _input_category_co_occur_enabled_for_role(self, role):
        if not self.input_category_co_occur:
            return False
        apply_to = str(self.input_category_co_occur_apply_to).strip().lower()
        if apply_to == "both":
            return role in {"input", "candidate"}
        if apply_to in {"inputs", "input"}:
            return role == "input"
        if apply_to in {"candidates", "candidate"}:
            return role == "candidate"
        return False

    def get_item_text_for_prompt(
        self,
        item_id,
        role,
        bundle_exclude_indices=None,
        user_exclude_indices=None,
        category_exclude_indices=None
    ):
        item_text = self.get_item_text_with_contexts(
            item_id,
            bundle_exclude_indices=bundle_exclude_indices,
            user_exclude_indices=user_exclude_indices
        )

        context_sentences = []
        if role == "input" and self.use_input_item_description_aug:
            description_text = self._input_item_description_aug_text(item_id)
            if description_text:
                context_sentences.append(description_text)
        if self._category_name_aug_enabled_for_role(role):
            category_name_text = self._category_name_aug_text(item_id)
            if category_name_text:
                context_sentences.append(category_name_text)
        if self._category_item_aug_enabled_for_role(role):
            category_text = self._category_item_aug_text(
                item_id,
                category_exclude_indices or []
            )
            if category_text:
                context_sentences.append(category_text)
        if self._input_category_co_occur_enabled_for_role(role):
            co_occur_text = self._input_category_co_occur_text(
                item_id,
                category_exclude_indices or []
            )
            if co_occur_text:
                context_sentences.append(co_occur_text)

        if not context_sentences:
            return item_text

        if item_text.endswith("]") and " [Additional context: " in item_text:
            return item_text[:-1] + " " + " ".join(context_sentences) + "]"
        return f"{item_text} [Additional context: {' '.join(context_sentences)}]"

    def retrieve_category_completion_prior_context(self, sample):
        if not self.use_category_completion_prior_desc:
            return None
        if "spotify" in self.name:
            return None

        input_indices = [int(i) for i in sample.get("input_indices", [])]
        candidate_indices = [int(i) for i in sample.get("candidate_indices", [])]
        observed_categories = self._items_to_unique_categories(input_indices)
        scores, observed_support = self._category_prior_scores(observed_categories)
        if not scores:
            return {
                "context_block": "",
                "metadata": {
                    "category_prior_observed_categories": json.dumps(observed_categories, ensure_ascii=False),
                    "category_prior_observed_support": int(observed_support),
                    "category_prior_top_categories": json.dumps([], ensure_ascii=False),
                    "category_prior_top_scores": json.dumps([], ensure_ascii=False),
                    "category_prior_top_category_names": json.dumps([], ensure_ascii=False),
                    "category_prior_rep_item_ids": json.dumps([], ensure_ascii=False),
                    "category_prior_rep_item_texts": json.dumps([], ensure_ascii=False),
                    "category_prior_verbalization": self.category_prior_verbalization,
                    "category_prior_coverage": 0,
                }
            }

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        top = ranked[:int(self.category_prior_top_k)]
        exclude_items = set(input_indices) | set(candidate_indices)
        if self._category_prior_uses_category_names():
            lines = [
                "Additional outfit context:",
                "The current outfit is commonly completed by the following item categories.",
            ]
        else:
            lines = [
                "Additional outfit context:",
                "The current outfit is commonly completed by item categories similar to the examples below.",
            ]
        top_categories = []
        top_scores = []
        top_category_names = []
        rep_item_ids_by_category = []
        rep_item_texts_by_category = []

        for idx, (category, score) in enumerate(top, start=1):
            category_name = self._category_display_name(category)
            if self._category_prior_uses_category_names():
                if not category_name:
                    continue
                lines.append(f"{idx}. {category_name}.")
                rep_items = []
                rep_texts = []
            else:
                rep_items = self._representative_items_for_category(category, exclude_items)
                if not rep_items:
                    continue
                rep_texts = [self._clean_inline_text(self.get_item_text(item_id)) for item_id in rep_items]
                lines.append(f"{idx}. Similar category examples: {'; '.join(rep_texts)}.")
            top_categories.append(category)
            top_scores.append(float(score))
            top_category_names.append(category_name)
            rep_item_ids_by_category.append(rep_items)
            rep_item_texts_by_category.append(rep_texts)

        if not top_categories:
            return None
        if self._category_prior_uses_category_names():
            lines.append("Use these categories only as a soft hint about plausible missing item types.")
        else:
            lines.append("Use these examples only as a soft hint about plausible missing item types.")
        return {
            "context_block": "\n".join(lines) + "\n\n",
            "metadata": {
                "category_prior_observed_categories": json.dumps(observed_categories, ensure_ascii=False),
                "category_prior_observed_support": int(observed_support),
                "category_prior_top_categories": json.dumps(top_categories, ensure_ascii=False),
                "category_prior_top_scores": json.dumps(top_scores, ensure_ascii=False),
                "category_prior_top_category_names": json.dumps(top_category_names, ensure_ascii=False),
                "category_prior_rep_item_ids": json.dumps(rep_item_ids_by_category, ensure_ascii=False),
                "category_prior_rep_item_texts": json.dumps(rep_item_texts_by_category, ensure_ascii=False),
                "category_prior_verbalization": self.category_prior_verbalization,
                "category_prior_coverage": 1,
            }
        }

    def retrieve_category_evidence_summary_context(self, sample):
        if (
            not self.use_category_evidence_summary
            or "spotify" in self.name
            or not self.cc_retrieval_train_categories
            or int(self.category_evidence_summary_k) <= 0
        ):
            return None

        input_indices = [int(i) for i in sample.get("input_indices", [])]
        input_categories = self._items_to_unique_categories(input_indices)
        if not input_categories:
            return None

        input_category_set = set(input_categories)
        input_category_names = [
            self._category_display_name(category) or str(category)
            for category in input_categories
        ]

        subset_specs = []
        max_size = len(input_categories)
        for size in range(max_size, 0, -1):
            for combo in combinations(input_categories, size):
                combo_set = set(combo)
                if size == max_size:
                    level = "full-match"
                elif size == 2:
                    level = "pair-match"
                elif size == 1:
                    level = "single-match"
                else:
                    level = f"{size}-category-match"
                subset_specs.append((level, combo_set))

        selected = []
        selected_bundle_ids = set()
        selected_category_keys = set()
        rng = np.random.default_rng(
            int(sample.get("bundle_id", 0)) + int(self.cc_retrieval_context_seed)
        )

        for level, subset in subset_specs:
            candidates = []
            for bundle_id, categories in self.cc_retrieval_train_categories.items():
                if int(bundle_id) in selected_bundle_ids:
                    continue
                if not subset.issubset(categories):
                    continue
                extra_categories = sorted(categories - input_category_set)
                if not extra_categories:
                    continue
                extra_prior = sum(
                    self._cc_completion_prior(sorted(subset), category)
                    for category in extra_categories
                )
                union_count = len(input_category_set | categories)
                jaccard = len(input_category_set & categories) / union_count if union_count else 0.0
                candidates.append((bundle_id, extra_prior, jaccard, extra_categories))

            if not candidates:
                continue

            tie_breaks = rng.random(len(candidates))
            ranked = sorted(
                range(len(candidates)),
                key=lambda idx: (
                    -candidates[idx][1],
                    -candidates[idx][2],
                    tie_breaks[idx],
                )
            )
            for idx in ranked:
                bundle_id, extra_prior, jaccard, extra_categories = candidates[idx]
                categories = sorted(self.cc_retrieval_train_categories.get(bundle_id, []))
                category_key = self._category_key(categories)
                if category_key in selected_category_keys:
                    continue
                category_names = [
                    self._category_display_name(category) or str(category)
                    for category in categories
                ]
                matched_categories = sorted(subset)
                matched_names = [
                    self._category_display_name(category) or str(category)
                    for category in matched_categories
                ]
                extra_names = [
                    self._category_display_name(category) or str(category)
                    for category in extra_categories
                ]
                selected.append({
                    "bundle_id": int(bundle_id),
                    "match_level": level,
                    "matched_categories": matched_categories,
                    "matched_category_names": matched_names,
                    "extra_categories": extra_categories,
                    "extra_category_names": extra_names,
                    "bundle_categories": categories,
                    "bundle_category_names": category_names,
                    "extra_prior": float(extra_prior),
                    "jaccard": float(jaccard),
                })
                selected_bundle_ids.add(int(bundle_id))
                selected_category_keys.add(category_key)
                if len(selected) >= int(self.category_evidence_summary_k):
                    break
            if len(selected) >= int(self.category_evidence_summary_k):
                break

        if not selected:
            return None

        lines = [
            "Input category names:",
            *[f"- {name}" for name in input_category_names],
            "",
            "Retrieved historical outfit category evidence:",
        ]
        for idx, example in enumerate(selected, start=1):
            lines.append(
                f"{idx}. [{example['match_level']}] "
                f"matched input categories: {'; '.join(example['matched_category_names'])}."
            )
            lines.append(
                f"   completed outfit categories: {'; '.join(example['bundle_category_names'])}."
            )
        evidence_block = "\n".join(lines)

        return {
            "evidence_block": evidence_block,
            "metadata": {
                "category_evidence_summary_input_categories": json.dumps(input_categories, ensure_ascii=False),
                "category_evidence_summary_input_category_names": json.dumps(input_category_names, ensure_ascii=False),
                "category_evidence_summary_bundle_ids": json.dumps([row["bundle_id"] for row in selected], ensure_ascii=False),
                "category_evidence_summary_match_levels": json.dumps([row["match_level"] for row in selected], ensure_ascii=False),
                "category_evidence_summary_matched_category_names": json.dumps([row["matched_category_names"] for row in selected], ensure_ascii=False),
                "category_evidence_summary_extra_category_names": json.dumps([row["extra_category_names"] for row in selected], ensure_ascii=False),
                "category_evidence_summary_bundle_category_names": json.dumps([row["bundle_category_names"] for row in selected], ensure_ascii=False),
                "category_evidence_summary_extra_priors": json.dumps([row["extra_prior"] for row in selected], ensure_ascii=False),
                "category_evidence_summary_jaccards": json.dumps([row["jaccard"] for row in selected], ensure_ascii=False),
                "category_evidence_summary_selected_count": int(len(selected)),
            },
        }

    def _soft_mapping_path(self, source):
        file_by_source = {
            "item_smoothing_text": "item_smoothing_i2bprime_text_top1.json",
            "item_smoothing_text_input_desc": "item_smoothing_i2bprime_text_input_desc_top1.json",
            "item_smoothing_bi_lgcn": "item_smoothing_i2bprime_bi_lgcn_top1.json",
            "bundle_smoothing_text": "bundle_smoothing_i2bprime_text_top1.json",
            "bundle_smoothing_text_input_desc": "bundle_smoothing_i2bprime_text_input_desc_top1.json",
            "bundle_smoothing_bi_lgcn": "bundle_smoothing_i2bprime_bi_lgcn_top1.json",
        }
        if source not in file_by_source:
            allowed = ", ".join(sorted(file_by_source))
            raise ValueError(f"Unknown soft mapping source={source}. Allowed: {allowed}")
        return os.path.join(self.path, self.name, file_by_source[source])

    def _soft_cooccurrence_path(self):
        return self._soft_mapping_path(self.soft_cooccurrence_source)

    def _load_soft_item_to_bundles(self, source=None):
        """Load precomputed item -> soft train bundle B' mapping."""
        source = source or self.soft_cooccurrence_source
        soft_path = self._soft_mapping_path(source)
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

    def _build_cc_retrieval_context_index(self):
        """Build train-only bundle category sets for C-C completion retrieval."""
        self.cc_retrieval_train_bundle_items = self._load_train_bundle_items()
        self.cc_retrieval_train_categories = {}
        for bundle_id, items in self.cc_retrieval_train_bundle_items.items():
            categories = self._items_to_unique_categories(items)
            if categories:
                self.cc_retrieval_train_categories[int(bundle_id)] = set(categories)

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
        candidate_bundle_set = set(int(i) for i in np.flatnonzero(overlap > 0))

        soft_hit = np.zeros(self.num_bundles, dtype=np.float64)
        if self.bundle_graph_context_use_soft:
            if self.bundle_graph_context_soft_item_to_bundles is None:
                self.bundle_graph_context_soft_item_to_bundles = self._load_soft_item_to_bundles(
                    self.bundle_graph_context_soft_source
                )
            for item_id in input_indices:
                for bundle_id in self.bundle_graph_context_soft_item_to_bundles.get(int(item_id), set()):
                    if 0 <= bundle_id < self.num_bundles:
                        soft_hit[bundle_id] += 1.0
                        candidate_bundle_set.add(int(bundle_id))

        candidate_bundle_ids = np.intersect1d(
            self.bundle_graph_train_bundle_ids,
            np.fromiter(candidate_bundle_set, dtype=np.int32),
            assume_unique=False
        )
        if len(candidate_bundle_ids) == 0:
            return None

        idf_weights = self.bundle_graph_item_idf[input_indices]
        idf_overlap = np.asarray(
            self.bundle_graph_train_matrix[:, input_indices].multiply(idf_weights).sum(axis=1)
        ).ravel()
        score = overlap + (float(self.bundle_graph_context_soft_alpha) * soft_hit)

        rng = np.random.default_rng(
            int(sample.get("bundle_id", 0)) + int(self.bundle_graph_context_seed)
        )
        tie_break = rng.random(len(candidate_bundle_ids))
        order = np.lexsort((
            tie_break,
            -idf_overlap[candidate_bundle_ids],
            -soft_hit[candidate_bundle_ids],
            -overlap[candidate_bundle_ids],
            -score[candidate_bundle_ids],
        ))
        if not self.bundle_graph_context_use_soft:
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
            "bundle_graph_context_soft_hit_counts": [float(soft_hit[i]) for i in selected_bundle_ids],
            "bundle_graph_context_scores": [float(score[i]) for i in selected_bundle_ids],
            "bundle_graph_context_idf_scores": [float(idf_overlap[i]) for i in selected_bundle_ids],
            "bundle_graph_context_use_soft": int(self.bundle_graph_context_use_soft),
            "bundle_graph_context_soft_source": self.bundle_graph_context_soft_source if self.bundle_graph_context_use_soft else "",
            "bundle_graph_context_soft_alpha": float(self.bundle_graph_context_soft_alpha),
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
            header = "Additional context: Related past playlists:"
        else:
            header = "Additional context: Related past outfits:"

        lines = [header]
        for idx, example in enumerate(examples, start=1):
            lines.append(f"{idx}. {'; '.join(example['item_texts'])}.")
        return {
            "context_block": "\n".join(lines) + "\n\n",
            "examples": examples,
            "metadata": metadata,
        }

    def _cc_completion_prior(self, input_categories, category):
        scores = []
        for input_category in input_categories:
            denom = self.category_prior_train_category_counts.get(input_category, 0)
            if denom <= 0:
                continue
            scores.append(self.category_pair_cooccur_counts[input_category].get(category, 0) / denom)
        if not scores:
            return 0.0
        return float(np.mean(scores))

    def retrieve_cc_retrieval_context(self, sample):
        if (
            not self.use_cc_retrieval_context
            or "spotify" in self.name
            or not self.cc_retrieval_train_categories
            or int(self.cc_retrieval_context_k) <= 0
        ):
            return None

        input_indices = [int(i) for i in sample.get("input_indices", [])]
        input_categories = set(self._items_to_unique_categories(input_indices))
        if not input_categories:
            return None

        scored = []
        input_categories_sorted = sorted(input_categories)
        for bundle_id, categories in self.cc_retrieval_train_categories.items():
            overlap_count = len(input_categories & categories)
            extra_categories = sorted(categories - input_categories)
            if overlap_count < 1 or not extra_categories:
                continue
            extra_prior = sum(
                self._cc_completion_prior(input_categories_sorted, category)
                for category in extra_categories
            )
            union_count = len(input_categories | categories)
            jaccard = overlap_count / union_count if union_count else 0.0
            score = (
                float(self.cc_retrieval_overlap_weight) * overlap_count
                + float(self.cc_retrieval_extra_weight) * extra_prior
            )
            if score <= 0:
                continue
            scored.append((bundle_id, score, overlap_count, extra_prior, jaccard, extra_categories))

        if not scored:
            return None

        rng = np.random.default_rng(
            int(sample.get("bundle_id", 0)) + int(self.cc_retrieval_context_seed)
        )
        tie_breaks = rng.random(len(scored))
        ranked_indices = sorted(
            range(len(scored)),
            key=lambda idx: (
                -scored[idx][1],
                -scored[idx][3],
                -scored[idx][2],
                -scored[idx][4],
                tie_breaks[idx],
            )
        )

        selected = []
        for idx in ranked_indices:
            bundle_id, score, overlap_count, extra_prior, jaccard, extra_categories = scored[idx]
            items = self.cc_retrieval_train_bundle_items.get(bundle_id, [])
            item_texts = [self._clean_inline_text(self.get_item_text(item_id)) for item_id in items]
            if not item_texts:
                continue
            selected.append({
                "bundle_id": int(bundle_id),
                "score": float(score),
                "overlap_count": int(overlap_count),
                "extra_prior": float(extra_prior),
                "jaccard": float(jaccard),
                "extra_categories": extra_categories,
                "bundle_categories": sorted(self.cc_retrieval_train_categories.get(bundle_id, [])),
                "items": [int(i) for i in items],
                "item_texts": item_texts,
            })
            if len(selected) >= int(self.cc_retrieval_context_k):
                break

        if not selected:
            return None

        header = (
            "\nHistorical examples of completed outfits with similar category patterns:"
            if len(selected) > 1
            else "\nHistorical example of a completed outfit with a similar category pattern:"
        )
        lines = [header]
        for idx, example in enumerate(selected, start=1):
            prefix = f"{idx}. " if len(selected) > 1 else ""
            lines.append(f"{prefix}{'; '.join(example['item_texts'])}")
        context_block = "\n".join(lines) + "\n"

        first = selected[0]
        best = scored[ranked_indices[0]]
        tie_count = sum(
            1
            for row in scored
            if row[1] == best[1]
            and row[3] == best[3]
            and row[2] == best[2]
            and row[4] == best[4]
        )
        return {
            "context_block": context_block,
            "metadata": {
                "cc_retrieval_context_bundle_id": int(first["bundle_id"]),
                "cc_retrieval_context_score": float(first["score"]),
                "cc_retrieval_context_overlap_count": int(first["overlap_count"]),
                "cc_retrieval_context_extra_prior": float(first["extra_prior"]),
                "cc_retrieval_context_jaccard": float(first["jaccard"]),
                "cc_retrieval_context_bundle_ids": json.dumps([row["bundle_id"] for row in selected], ensure_ascii=False),
                "cc_retrieval_context_scores": json.dumps([row["score"] for row in selected], ensure_ascii=False),
                "cc_retrieval_context_overlap_counts": json.dumps([row["overlap_count"] for row in selected], ensure_ascii=False),
                "cc_retrieval_context_extra_priors": json.dumps([row["extra_prior"] for row in selected], ensure_ascii=False),
                "cc_retrieval_context_jaccards": json.dumps([row["jaccard"] for row in selected], ensure_ascii=False),
                "cc_retrieval_context_input_categories": json.dumps(sorted(input_categories), ensure_ascii=False),
                "cc_retrieval_context_extra_categories": json.dumps(first["extra_categories"], ensure_ascii=False),
                "cc_retrieval_context_extra_categories_by_bundle": json.dumps([row["extra_categories"] for row in selected], ensure_ascii=False),
                "cc_retrieval_context_bundle_categories": json.dumps(
                    first["bundle_categories"],
                    ensure_ascii=False
                ),
                "cc_retrieval_context_bundle_categories_by_bundle": json.dumps([row["bundle_categories"] for row in selected], ensure_ascii=False),
                "cc_retrieval_context_item_ids": json.dumps(first["items"], ensure_ascii=False),
                "cc_retrieval_context_item_texts": json.dumps(first["item_texts"], ensure_ascii=False),
                "cc_retrieval_context_item_ids_by_bundle": json.dumps([row["items"] for row in selected], ensure_ascii=False),
                "cc_retrieval_context_item_texts_by_bundle": json.dumps([row["item_texts"] for row in selected], ensure_ascii=False),
                "cc_retrieval_context_selected_count": int(len(selected)),
                "cc_retrieval_context_tie_count": int(tie_count),
            },
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
            input_str = "; ".join([
                f"{idx + 1}. {self.get_item_text_for_prompt(j, role='input', bundle_exclude_indices=bundle_exclude_indices, user_exclude_indices=user_exclude_indices, category_exclude_indices=query_indices)}"
                for idx, j in enumerate(input_indices)
            ])
            target_str = "; ".join([
                f"{chr(ord('A') + idx)}. {self.get_item_text_for_prompt(j, role='candidate', bundle_exclude_indices=bundle_exclude_indices, user_exclude_indices=user_exclude_indices, category_exclude_indices=query_indices)}"
                for idx, j in enumerate(indices)
            ])
            
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

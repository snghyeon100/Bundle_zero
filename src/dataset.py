import os
import json
import random
import numpy as np
import scipy.sparse as sp

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
        
        # Load counts
        count_path = os.path.join(self.path, self.name, 'count.json')
        with open(count_path, 'r') as f:
            stat = json.loads(f.read())
        self.num_bundles, self.num_items = stat["#B"], stat["#I"]

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

        # Apply toy_eval truncating
        if self.toy_eval > 0:
            self.b_i_pairs_gt = self.b_i_pairs_gt[:self.toy_eval]

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

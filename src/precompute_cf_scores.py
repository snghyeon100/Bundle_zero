"""
precompute_cf_scores.py

Precomputes co-occurrence and user-preference scores for all evaluation samples
and saves them to a JSON cache file:
  datasets/<dataset>/cf_scores_<dataset>.json

Run once per dataset before main.py:
  python src/precompute_cf_scores.py

The cached scores are keyed by "<bundle_id>_<true_indice>" so main.py can do
an O(1) lookup instead of recomputing per sample at evaluation time.
"""

import os
import sys
import json
import yaml
from tqdm import tqdm

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))
from dataset import BundleZeroShotDataset, set_seed


def build_cache_key(sample):
    return f"{sample['bundle_id']}_{sample['true_indice']}"


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    # Force both CF options on for precomputation
    conf["use_cooccurrence"] = True
    conf["use_user_pref"] = True

    set_seed(conf["seed"])

    print(f">>> Loading dataset: {conf['dataset']}")
    dataset = BundleZeroShotDataset(conf)

    # Load samples: hard negative JSON if available, else generate from dataset
    hard_negative_path = os.path.join(
        conf.get("data_path", "./datasets"),
        conf["dataset"],
        f"hard_negative_samples_{conf['dataset']}.json"
    )
    if conf.get("use_hard_negative", False) and os.path.exists(hard_negative_path):
        print(f">>> Using hard negative samples from {hard_negative_path}")
        with open(hard_negative_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        print(">>> Generating eval samples from dataset...")
        samples = dataset.get_eval_samples()

    print(f">>> Precomputing CF scores for {len(samples)} samples...")

    cf_cache = {}
    for sample in tqdm(samples):
        input_ids  = sample.get("input_indices", [])
        cand_ids   = sample.get("candidate_indices", [])

        entry = {}
        if hasattr(dataset, "get_cooccurrence_scores"):
            entry["cooccurrence"] = dataset.get_cooccurrence_scores(input_ids, cand_ids)
        if hasattr(dataset, "get_user_pref_scores"):
            entry["user_pref"] = dataset.get_user_pref_scores(input_ids, cand_ids)

        key = build_cache_key(sample)
        cf_cache[key] = entry

    out_path = os.path.join(
        conf.get("data_path", "./datasets"),
        conf["dataset"],
        f"cf_scores_{conf['dataset']}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cf_cache, f)

    print(f">>> Saved CF score cache ({len(cf_cache)} entries) to: {out_path}")


if __name__ == "__main__":
    main()

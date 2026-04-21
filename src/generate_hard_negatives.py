import os
import yaml
import json
import random
import numpy as np
import pandas as pd
from dataset import BundleZeroShotDataset, set_seed

def generate_hard_negatives():
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    
    if "spotify" not in conf["dataset"].lower():
        print("This script is currently designed for the Spotify dataset (using same-artist hard negatives).")
        return

    set_seed(conf["seed"])
    dataset = BundleZeroShotDataset(conf)
    
    print(">>> 1. Loading items and building Artist -> Items mapping (Pandas)...")
    # Convert item_info dict to a pandas DataFrame
    df_items = pd.DataFrame.from_dict(dataset.item_info, orient='index')
    
    # Ensure index is treated as integer since indices in model are ints
    df_items.index = df_items.index.astype(int) 
    
    if "artist_name" not in df_items.columns:
        print("Error: 'artist_name' column not found in item_info.json")
        return

    # Filter out items with empty artist names
    valid_artist_items = df_items[df_items['artist_name'].notna() & (df_items['artist_name'] != "")]
    
    # Create mapping: Artist -> list of integer item ids
    artist_track_map = valid_artist_items.groupby('artist_name').groups
    
    print(">>> 2. Generating Hard Negative Samples...")
    hard_samples = []
    total_bundles = dataset.b_i_pairs_gt.shape[0]
    
    for idx_counter, (b_idx, true_indice) in enumerate(dataset.b_i_pairs_gt):
        if (idx_counter + 1) % 500 == 0:
            print(f"Processing {idx_counter + 1}/{total_bundles}...")
            
        b_idx_int = int(b_idx)
        true_indice_int = int(true_indice)
        
        b_i_i_np = dataset.b_i_graph_i[b_idx_int].toarray().squeeze()
        b_i_gt_np = dataset.b_i_graph_gt[b_idx_int].toarray().squeeze()
        
        # Determine Artist of Target string
        target_info = dataset.item_info.get(str(true_indice_int), {})
        target_artist = target_info.get("artist_name", "")
        
        # 1. Gather Candidate Indices (Hard Negative Logic)
        rng_cand = np.random.default_rng(b_idx_int + dataset.seed)
        false_indices = []
        num_neg_needed = dataset.num_cans - 1
        
        if target_artist and target_artist in artist_track_map:
            artist_tracks = artist_track_map[target_artist].tolist()
            valid_artist_tracks = [t for t in artist_tracks if (b_i_i_np[t] + b_i_gt_np[t]) == 0]
            
            if len(valid_artist_tracks) >= num_neg_needed:
                false_indices = rng_cand.choice(valid_artist_tracks, size=num_neg_needed, replace=False).tolist()
                hn_count = num_neg_needed
            else:
                false_indices = list(valid_artist_tracks)
                hn_count = len(false_indices)
                remaining_needed = num_neg_needed - len(false_indices)
                
                all_false_pool = np.argwhere((b_i_i_np + b_i_gt_np) == 0).reshape(-1)
                all_false_pool_set = set(all_false_pool) - set(false_indices)
                fallback_choices = rng_cand.choice(list(all_false_pool_set), size=remaining_needed, replace=False).tolist()
                false_indices.extend(fallback_choices)
        else:
            hn_count = 0
            all_false_pool = np.argwhere((b_i_i_np + b_i_gt_np) == 0).reshape(-1)
            false_indices = rng_cand.choice(all_false_pool, size=num_neg_needed, replace=False).tolist()

        indices = [true_indice_int] + false_indices
        indices = np.array(indices)
        rng_cand.shuffle(indices)
        true_idx = int(np.argwhere(indices == true_indice_int)[0][0])

        rng_input = np.random.default_rng(b_idx_int + dataset.shuffle_seed)
        input_indices = np.argwhere(b_i_i_np > 0).reshape(-1)
        rng_input.shuffle(input_indices)
        
        if dataset.num_token > 0 and len(input_indices) > dataset.num_token:
            input_indices = input_indices[:dataset.num_token]
            
        input_str = "; ".join([f"{idx + 1}. {dataset.get_item_text(j)}" for idx, j in enumerate(input_indices)])
        target_str = "; ".join([f"{chr(ord('A') + idx)}. {dataset.get_item_text(j)}" for idx, j in enumerate(indices)])

        hard_samples.append({
            "bundle_id": b_idx_int,
            "true_indice": true_indice_int,
            "true_option_idx": true_idx,
            "true_option_char": chr(ord('A') + true_idx),
            "input_indices": input_indices.tolist(),
            "candidate_indices": indices.tolist(),
            "input_str": input_str,
            "target_str": target_str,
            "hard_neg_count": hn_count
        })
        
    # Save the output file
    dataset_dir = os.path.join(conf["data_path"], conf["dataset"])
    out_path = os.path.join(dataset_dir, f"hard_negative_samples_{conf['dataset']}.json")
    
    print(f">>> 3. Saving {len(hard_samples)} Hard Negative samples to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hard_samples, f, indent=4, ensure_ascii=False)
        
    print("Done! You can now optionally load this file in main.py instead of dataset.get_eval_samples()")

if __name__ == "__main__":
    generate_hard_negatives()

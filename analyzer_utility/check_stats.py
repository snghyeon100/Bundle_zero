import json
import os
import statistics

dataset_name = "spotify_sparse"
json_path = f"datasets/{dataset_name}/hard_negative_samples_{dataset_name}.json"
item_info_path = f"datasets/{dataset_name}/item_info.json"

if not os.path.exists(json_path):
    print(f"File not found: {json_path}")
    exit()

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open(item_info_path, 'r', encoding='utf-8') as f:
    item_info = json.load(f)

total_samples = len(data)
hard_neg_counts = [] # 번들당 가짜 후보(9개) 중 하드 네거티브 개수 저장

for row in data:
    true_idx = str(row['true_indice'])
    target_artist = item_info.get(true_idx, {}).get('artist_name', '')
    
    current_bundle_hard_neg = 0
    for cand_idx in row['candidate_indices']:
        cand_str = str(cand_idx)
        
        # 정답 아이템은 검사에서 제외 (항상 본인과 가수가 같으므로)
        if cand_str == true_idx:
            continue
            
        cand_artist = item_info.get(cand_str, {}).get('artist_name', '')
        if target_artist and cand_artist == target_artist:  # 아티스트가 같으면 1 추가
            current_bundle_hard_neg += 1
            
    hard_neg_counts.append(current_bundle_hard_neg)

avg_hard_neg = sum(hard_neg_counts) / len(hard_neg_counts) if hard_neg_counts else 0
zero_hard_neg_bundles = sum(1 for x in hard_neg_counts if x == 0)
perfect_hard_neg_bundles = sum(1 for x in hard_neg_counts if x == 9)

print("="*50)
print("[ HARD NEGATIVE DETAILED STATS (Per Bundle) ]")
print("="*50)
print(f"Total Bundles Evaluated : {total_samples}")
print(f"Avg Hard Negatives / 9  : {avg_hard_neg:.2f} items per bundle")
print(f"Bundles with 9/9 Hard   : {perfect_hard_neg_bundles} ({(perfect_hard_neg_bundles/total_samples)*100:.1f}%)")
print(f"Bundles with 0/9 Hard   : {zero_hard_neg_bundles} ({(zero_hard_neg_bundles/total_samples)*100:.1f}%)")
print("="*50)

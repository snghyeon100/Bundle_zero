import os
import json
import ast
import pandas as pd
import argparse

def analyze_artist_bias(csv_path, dataset="spotify"):
    print(f"Loading results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    item_info_path = f"./datasets/{dataset}/item_info.json"
    print(f"Loading item info from: {item_info_path}")
    with open(item_info_path, "r", encoding="utf-8") as f:
        item_info = json.load(f)

    total_samples = len(df)
    artist_match_count = 0
    hit_with_artist_match = 0
    total_hits = 0
    gt_artist_match_count = 0

    for idx, row in df.iterrows():
        # Parse list strings back to python lists
        input_indices = ast.literal_eval(str(row['input_indices']))
        cand_indices = ast.literal_eval(str(row['candidate_indices']))
        true_idx = str(row['true_indice'])
        
        # Check GT overlap
        true_artist = item_info.get(true_idx, {}).get("artist_name", "").strip().lower()
        if true_artist:
            for i_id in input_indices:
                input_artist = item_info.get(str(i_id), {}).get("artist_name", "").strip().lower()
                if input_artist and input_artist == true_artist:
                    gt_artist_match_count += 1
                    break
        
        
        pred_char = str(row['prediction']).strip()
        if not pred_char or pred_char not in "ABCDEFGHIJ":
            continue
            
        # Get the chosen candidate's item id
        pred_idx_in_list = ord(pred_char) - ord('A')
        if pred_idx_in_list >= len(cand_indices):
            continue
            
        chosen_item_id = str(cand_indices[pred_idx_in_list])
        
        # Check if the chosen item shares an artist with ANY of the input items
        chosen_artist = item_info.get(chosen_item_id, {}).get("artist_name", "").strip().lower()
        
        has_overlap = False
        if chosen_artist:
            for i_id in input_indices:
                input_artist = item_info.get(str(i_id), {}).get("artist_name", "").strip().lower()
                if input_artist and input_artist == chosen_artist:
                    has_overlap = True
                    break
                    
        if has_overlap:
            artist_match_count += 1
            if row['hit'] == 1:
                hit_with_artist_match += 1
                
        if row['hit'] == 1:
            total_hits += 1

    print("-" * 50)
    print(f"총 분석 샘플 수: {total_samples}")
    print(f"[데이터셋 자체 특성] 정답(Ground Truth) 자체가 인풋과 가수가 겹치는 비율: {(gt_artist_match_count/total_samples)*100:.1f}% ({gt_artist_match_count}개)")
    print(f"정답을 맞춘 수 (Hit): {total_hits} ({(total_hits/total_samples)*100:.1f}%)")
    print(f"모델이 선택한 옵션이 Input 아이템과 가수가 겹치는 경우: {artist_match_count} ({(artist_match_count/total_samples)*100:.1f}%)")
    
    if total_hits > 0:
        print(f"정답을 맞춘 케이스 중, 가수가 겹쳐서 맞춘 비율: {(hit_with_artist_match/total_hits)*100:.1f}%")
        
    print("-" * 50)
    print("결론: 만약 '모델이 선택한 옵션이 가수가 겹치는 경우'가 매우 높다면,")
    print("모델은 문맥이나 추천 로직보다 단순히 '텍스트에 겹치는 가수 이름'을 찾아 찍고 있을 확률이 높습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Artist Match Bias")
    parser.add_argument("--csv", type=str, required=True, help="Path to the results CSV file")
    parser.add_argument("--dataset", type=str, default="spotify", help="Dataset name")
    args = parser.parse_args()
    
    analyze_artist_bias(args.csv, args.dataset)

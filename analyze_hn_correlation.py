import pandas as pd
import json
import os

# ==========================================
# 💡 돌리신 CSV 결과 파일과 데이터셋 이름을 넣어주세요!
# ==========================================
result_csv_path = r"results\spotify_sparse\여기에_파일_이름_넣기.csv" 
dataset_name = "spotify_sparse"
# ==========================================

json_path = f"datasets/{dataset_name}/hard_negative_samples_{dataset_name}.json"
item_info_path = f"datasets/{dataset_name}/item_info.json"

if not os.path.exists(result_csv_path):
    print(f"👉 파일을 아직 돌리지 않으셨거나 CSV 경로가 다릅니다. \n   (현재 등록된 경로: {result_csv_path})")
else:
    try:
        df_results = pd.read_csv(result_csv_path)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            hn_data = json.load(f)
            
        with open(item_info_path, 'r', encoding='utf-8') as f:
            item_info = json.load(f)
            
        # 1. 번들별 Hard Negative 개수 딕셔너리화 (이제 JSON 안에 이미 저장되어 있음!)
        hn_count_dict = {int(row['bundle_id']): int(row['hard_neg_count']) for row in hn_data}

        # 2. 결과 데이터프레임에 하드 네거티브 개수 매핑
        df_results['hard_neg_count'] = df_results['bundle_id'].map(hn_count_dict)

        # 3. 통계 계산
        hit_df = df_results[df_results['hit'] == 1]
        miss_df = df_results[df_results['hit'] == 0]
        
        hit_mean = hit_df['hard_neg_count'].mean() if len(hit_df) > 0 else 0
        miss_mean = miss_df['hard_neg_count'].mean() if len(miss_df) > 0 else 0

        print("="*60)
        print("🎯 Hard Negative 난이도 vs 모델 정답률 상관관계 분석")
        print("="*60)
        print(f"Total Evaluated : {len(df_results)} 문제")
        print(f"✅ 정답 처리된 문제들({len(hit_df)}개)의 평균 Hard Negative 갯수 : {hit_mean:.2f}개 / 9개")
        print(f"❌ 오답 처리된 문제들({len(miss_df)}개)의 평균 Hard Negative 갯수 : {miss_mean:.2f}개 / 9개")
        print("="*60)
        print("💡 결과 해석:")
        print("- 오답의 수치가 정답보다 높다면, '진짜로 Hard Negative가 많아서 모델을 속이는 데 성공했다'는 좋은 디자인 증거입니다.")

    except Exception as e:
        print(f"Error: {e}")

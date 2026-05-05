import pandas as pd
import os
import json
import argparse
'''
python compare_results.py --fileA results\pog\원본결과.csv --fileB results\pog\새로운결과.csv --name pog_base_vs_hn
'''
import yaml

def main():
    parser = argparse.ArgumentParser(description="비교할 두 방법론의 결과를 병합합니다.")
    parser.add_argument("--fileA", required=False, help="방법론 A의 결과 CSV 파일 경로 (기준점)")
    parser.add_argument("--fileB", required=False, help="방법론 B의 결과 CSV 파일 경로 (새로운 프롬프트 등)")
    parser.add_argument("--name", required=False, help="출력 파일명 접두사 (예: pog_base_vs_hn)")
    args = parser.parse_args()

    # Load from config.yaml if available
    if os.path.exists("config.yaml"):
        with open("config.yaml", "r", encoding="utf-8") as f:
            conf = yaml.safe_load(f)
    else:
        conf = {}

    fileA = args.fileA if args.fileA else conf.get("fileA")
    fileB = args.fileB if args.fileB else conf.get("fileB")
    name = args.name if args.name else conf.get("compare_name")

    if not fileA or not fileB or not name:
        print("[오류] fileA, fileB, name 값이 모두 필요합니다. (명령어 인수 또는 config.yaml을 통해 설정하세요)")
        return

    # Load dataframes
    print(f">>> Loading File A: {fileA}")
    df1 = pd.read_csv(fileA)
    print(f">>> Loading File B: {fileB}")
    df2 = pd.read_csv(fileB)

    # Create a sample index since problem_difficulty_meta.csv uses row index
    df1['sample_idx'] = df1.index
    df2['sample_idx'] = df2.index

    # Define columns to keep from fileA (base information)
    base_cols = [
        'bundle_id', 'sample_idx', 'true_option_char', 'input_indices', 'candidate_indices', 
        'input_str', 'target_str'
    ]

    # Select and rename columns for A and B
    df1_sub = df1[base_cols + ['prediction', 'raw_response', 'hit']].rename(
        columns={'prediction': 'prediction_A', 'raw_response': 'raw_response_A', 'hit': 'hit_A'}
    )
    df2_sub = df2[['sample_idx', 'prediction', 'raw_response', 'hit']].rename(
        columns={'prediction': 'prediction_B', 'raw_response': 'raw_response_B', 'hit': 'hit_B'}
    )

    # Merge predictions
    df_merged = pd.merge(df1_sub, df2_sub, on='sample_idx')

    # Load difficulty metadata if exists
    # Infer dataset folder from fileA
    dataset_folder = os.path.dirname(fileA)
    diff_file = os.path.join(dataset_folder, "problem_difficulty_meta.csv")
    
    if os.path.exists(diff_file):
        print(f">>> Loading difficulty metadata from {diff_file}")
        df_diff = pd.read_csv(diff_file)
        if 'index' in df_diff.columns:
            df_diff = df_diff.rename(columns={'index': 'sample_idx'})
        df_merged = pd.merge(df_merged, df_diff[['sample_idx', 'difficulty', 'reason']], on='sample_idx', how='left')
    else:
        print(f">>> [경고] 난이도 메타데이터 파일을 찾을 수 없습니다: {diff_file}")

    # Categorize into groups
    def categorize(row):
        if row['hit_A'] == 1 and row['hit_B'] == 1:
            return 'Both_Hit'
        elif row['hit_A'] == 0 and row['hit_B'] == 0:
            return 'Both_Fail'
        elif row['hit_A'] == 1 and row['hit_B'] == 0:
            return 'A_Hit_Only'
        else:
            return 'B_Hit_Only'

    df_merged['group'] = df_merged.apply(categorize, axis=1)

    # Summary Stats
    total = len(df_merged)
    both_hit = len(df_merged[df_merged['group'] == 'Both_Hit'])
    both_miss = len(df_merged[df_merged['group'] == 'Both_Fail'])
    a_hit_only = len(df_merged[df_merged['group'] == 'A_Hit_Only'])
    b_hit_only = len(df_merged[df_merged['group'] == 'B_Hit_Only'])

    print("="*50)
    print("📊 PROMPT COMPARISON & ERROR ANALYSIS PREP")
    print("="*50)
    print(f"Total Evaluated Bundles : {total}")
    print(f"File A Hits             : {df_merged['hit_A'].sum()} / {total} ({(df_merged['hit_A'].sum()/total)*100:.2f}%)")
    print(f"File B Hits             : {df_merged['hit_B'].sum()} / {total} ({(df_merged['hit_B'].sum()/total)*100:.2f}%)")
    print("-" * 50)
    print(f"✅ Both Correct       : {both_hit} ({(both_hit/total)*100:.2f}%)")
    print(f"❌ Both Incorrect     : {both_miss} ({(both_miss/total)*100:.2f}%)")
    print(f"⬆️ File A Only Correct: {a_hit_only} ({(a_hit_only/total)*100:.2f}%)")
    print(f"⭐ File B Only Correct: {b_hit_only} ({(b_hit_only/total)*100:.2f}%)")
    print("="*50)

    # Save rich dataset for LLM Analysis
    out_dir = "analysis"
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, f"{name}_analysis_ready.csv")
    df_merged.to_csv(out_csv, index=False)
    print(f"💾 Comprehensive analysis dataset saved to: {out_csv}")
    print(f"   -> 다음 명령어 실행: python analyze_error_patterns.py --data {out_csv}")

if __name__ == "__main__":
    main()

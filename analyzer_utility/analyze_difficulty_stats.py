import pandas as pd
import os
import yaml
import argparse

def evaluate_single_file(df, file_name):
    print(f"\n========================================================")
    print(f"📊 난이도별 정답률 통계: {os.path.basename(file_name)}")
    print(f"========================================================")
    
    # 난이도별로 그룹화하여 hit 분포 계산
    stats = df.groupby('difficulty')['hit'].agg(['count', 'sum']).reset_index()
    stats.columns = ['난이도', '총문제수', '정답수']
    stats['오답수'] = stats['총문제수'] - stats['정답수']
    stats['정답률(%)'] = (stats['정답수'] / stats['총문제수']) * 100
    
    # 소수점 1자리로 반올림
    stats['정답률(%)'] = stats['정답률(%)'].round(1)
    
    # 출력
    for _, row in stats.iterrows():
        diff = int(row['난이도'])
        total = int(row['총문제수'])
        correct = int(row['정답수'])
        wrong = int(row['오답수'])
        acc = row['정답률(%)']
        print(f"⭐ 난이도 {diff}점 ({total:3d}문제): 정답 {correct:3d}개 | 오답 {wrong:3d}개  👉 [ 정답률: {acc:5.1f}% ]")
        
    avg_acc = (stats['정답수'].sum() / stats['총문제수'].sum()) * 100
    print(f"--------------------------------------------------------")
    print(f"✅ 전체 평균 정답률: {avg_acc:.1f}%")
    print(f"========================================================\n")


def evaluate_compare_files(df1, df2, name1, name2):
    print(f"\n========================================================")
    print(f"⚔️ 2개 모델 난이도별 정답률 비교 결과")
    print(f"[A] {os.path.basename(name1)}")
    print(f"[B] {os.path.basename(name2)}")
    print(f"========================================================")
    
    stats1 = df1.groupby('difficulty')['hit'].agg(['count', 'sum']).reset_index()
    stats1.columns = ['난이도', '총문제수', '정답수_A']
    stats1['정답률_A(%)'] = (stats1['정답수_A'] / stats1['총문제수']) * 100
    
    stats2 = df2.groupby('difficulty')['hit'].agg(['sum']).reset_index()
    stats2.columns = ['난이도', '정답수_B']
    stats2['정답률_B(%)'] = (stats2['정답수_B'] / stats1['총문제수']) * 100 # 총문제수는 동일
    
    stats_merged = pd.merge(stats1, stats2, on='난이도')
    
    for _, row in stats_merged.iterrows():
        diff = int(row['난이도'])
        total = int(row['총문제수'])
        acc_a = row['정답률_A(%)']
        acc_b = row['정답률_B(%)']
        
        diff_str = ""
        gap = acc_b - acc_a
        if gap > 0: diff_str = f" (+{gap:.1f}%)"
        elif gap < 0: diff_str = f" ({gap:.1f}%)"
        
        print(f"⭐ 난이도 {diff}점 ({total:3d}문제): [A] {acc_a:5.1f}%  vs  [B] {acc_b:5.1f}% {diff_str}")
        
    total_a = (stats_merged['정답수_A'].sum() / stats_merged['총문제수'].sum()) * 100
    total_b = (stats_merged['정답수_B'].sum() / stats_merged['총문제수'].sum()) * 100
    print(f"--------------------------------------------------------")
    print(f"✅ 전체 평균 정답률: [A] {total_a:.1f}%  vs  [B] {total_b:.1f}%")
    print(f"========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="난이도 기반 평가 결과 분석기")
    parser.add_argument("--file1", type=str, help="분석할 첫 번째 CSV 결과 파일 경로", required=True)
    parser.add_argument("--file2", type=str, help="비교할 두 번째 CSV 결과 파일 경로 (선택사항)", default=None)
    args = parser.parse_args()

    # config 설정 로드
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    dataset_name = conf.get("dataset", "").lower()
    actual_output_dir = os.path.join(conf.get("output_dir", "./results"), dataset_name)
    meta_path = os.path.join(actual_output_dir, "problem_difficulty_meta.csv")

    if not os.path.exists(meta_path):
        print(f"❌ [에러] 난이도 메타데이터 파일을 찾을 수 없습니다: {meta_path}")
        print("evaluate_difficulty.py를 먼저 실행하여 난이도 점수를 생성해주세요.")
        return

    # meta 파일 로드 및 결측치 제거
    meta_df = pd.read_csv(meta_path)
    meta_df = meta_df.dropna(subset=['difficulty']) 

    # File 1 처리
    if not os.path.exists(args.file1):
        print(f"❌ [에러] 결과 파일 1을 찾을 수 없습니다: {args.file1}")
        return
    
    df1 = pd.read_csv(args.file1)
    df1['index'] = df1.index # Merge를 위해 원본의 index 컬럼 생성
    merged_df1 = pd.merge(meta_df, df1, on='index')

    if args.file2:
        # File 2 처리 (비교 모드)
        if not os.path.exists(args.file2):
            print(f"❌ [에러] 결과 파일 2를 찾을 수 없습니다: {args.file2}")
            return
            
        df2 = pd.read_csv(args.file2)
        df2['index'] = df2.index
        merged_df2 = pd.merge(meta_df, df2, on='index')
        
        evaluate_compare_files(merged_df1, merged_df2, args.file1, args.file2)
    else:
        # 단일 파일 (통계 모드)
        evaluate_single_file(merged_df1, args.file1)

if __name__ == "__main__":
    main()

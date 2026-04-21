import os
import pandas as pd
import yaml
import glob
from deep_translator import GoogleTranslator

def batch_translate(series, translator, batch_size=10):
    texts = series.tolist()
    translated = []
    print(f">>> 총 {len(texts)}개 행 번역 시작 (Batch Size: {batch_size})")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            res = translator.translate_batch(batch)
            translated.extend(res)
            print(f"    ㄴ [{i+len(batch)}/{len(texts)}] 번역 완료...")
        except Exception as e:
            print(f"    ㄴ [경고] 배치 {i//batch_size + 1} 실패: {e}. 원본 유지.")
            translated.extend(batch)
    return translated

def main():
    # 1. 설정 로드
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    
    dataset = conf["dataset"]
    output_dir = conf["output_dir"]
    
    # 2. 해당 데이터셋 폴더 내에서 가장 최근의 CSV 파일 찾기 (kor 제외)
    actual_output_dir = os.path.join(output_dir, dataset)
    if not os.path.exists(actual_output_dir):
        print(f">>> [오류] {actual_output_dir} 폴더가 존재하지 않습니다.")
        return
        
    pattern = os.path.join(actual_output_dir, f"results_{dataset}_*.csv")
    files = [f for f in glob.glob(pattern) if "_kor_" not in f and "_analyzed" not in f]
    
    if not files:
        print(f">>> [오류] {actual_output_dir} 폴더에서 분석할 CSV 파일을 찾지 못했습니다.")
        return
        
    latest_file = max(files, key=os.path.getctime)
    print(f">>> 타겟 파일 발견: {latest_file}")
    
    # 3. 데이터 로드
    df = pd.read_csv(latest_file)
    
    # 4. 번역 수행
    translator = GoogleTranslator(source='auto', target='ko')
    
    print(">>> 'input_str' 컬럼 번역 중...")
    df['input_str'] = batch_translate(df['input_str'], translator)
    
    print(">>> 'target_str' 컬럼 번역 중...")
    df['target_str'] = batch_translate(df['target_str'], translator)
    
    # 5. 저장
    save_path = latest_file.replace(".csv", f"_kor_retranslated.csv")
    df.to_csv(save_path, index=False)
    print(f"\n>>> [성공] 번역 완료! 파일이 저장되었습니다: \n{save_path}")

if __name__ == "__main__":
    main()

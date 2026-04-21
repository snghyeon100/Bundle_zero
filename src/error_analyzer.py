import os
import yaml
import json
import time
import pandas as pd
import argparse
from dotenv import load_dotenv
from google import genai

# Load Env
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path, encoding='utf-8-sig')

def analyze_errors_batch(client, model, rows_list, conf):
    """
    여러 개의 오답 사례를 묶어서 한 번의 API 호출로 분석합니다. (25:1 최적화)
    """
    dataset_name = conf.get("dataset", "").lower()
    if "spotify" in dataset_name:
        t_name = "Playlist Continuation (음악 플레이리스트 연장)"
        b_name = "플레이리스트 (playlist)"
        i_name = "노래 (song)"
    else:
        t_name = "Bundle Construction (패션 번들 구성)"
        b_name = "패션 번들 (outfit)"
        i_name = "패션 아이템 (item)"

    cases = []
    for idx, row in enumerate(rows_list):
        case_info = (
            f"Case #{idx+1}\n"
            f"- 상황 (기존 {b_name}에 포함된 것들): {row['input_str']}\n"
            f"- 후보 {i_name}들: {row['target_str']}\n"
            f"- 정답: {row['true_option_char']}, 모델이 고른 오답: {row['prediction']}\n"
            f"- 모델의 추론(있는 경우): {row.get('raw_response', 'N/A')}\n"
        )
        cases.append(case_info)

    all_cases_str = "\n".join(cases)
    
    prompt = (
        f"너는 추천 시스템 전문가야. 다음은 '{t_name}' 태스크에 대한 {len(rows_list)}개의 오답 사례들이야.\n"
        f"이 태스크의 목표는 주어진 {b_name}의 맥락을 보고, 10개의 후보 {i_name} 중 가장 잘 어울리는 하나를 고르는 것이야.\n\n"
        f"{all_cases_str}\n"
        f"### 임무\n"
        f"각 Case별로 **정답 아이템과 모델이 고른 오답 아이템의 특징을 서로 비교**하여, 모델이 왜 정답보다 오답이 더 적절하다고 오판했는지 그 논리적 원인을 분석해줘.\n"
        f"분석 내용을 작성할 때 언급되는 모든 상품명이나 아이템 특징은 반드시 한국어로 번역하여 자연스럽게 서술해야 해.\n"
        f"분석 결과는 아래 JSON 리스트 형식으로만 답변해줘. 다른 설명은 절대 하지 마.\n"
        f"[\n"
        f"  {{\"reason\": \"정답 vs 오답 비교를 통한 상세 분석 내용 (상품명은 한국어로)\", \"tag\": \"오류_태그\"}},\n"
        f"  {{\"reason\": \"...\", \"tag\": \"...\"}},\n"
        f"  ... {len(rows_list)}개 순서대로\n"
        f"]"
    )

    try:
        res = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.0, "response_mime_type": "application/json"}
        )
        # AI 응답에서 JSON 부분만 추출 (마크다운 기호 등 제거)
        clean_text = res.text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[-1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[-1].split("```")[0].strip()
            
        # 가끔 앞뒤에 이상한 문자가 붙는 경우 방지 ( [ ] 추출 )
        start_idx = clean_text.find("[")
        end_idx = clean_text.rfind("]")
        if start_idx != -1 and end_idx != -1:
            clean_text = clean_text[start_idx:end_idx+1]
            
        results = json.loads(clean_text)
        # 결과 개수 검증
        if len(results) != len(rows_list):
            print(f">>> [경고] 응답 개수 불일치 ({len(results)} vs {len(rows_list)})")
        return results
    except Exception as e:
        print(f">>> [오류] API 호출 중 치명적인 에러 발생 (프로그램을 중단합니다): {str(e)}")
        # 예외를 던져서 main 루프에서 저장되지 않게 함 (나중에 재질의 가능)
        raise e

def main():
    parser = argparse.ArgumentParser(description="Zero-shot 추천 오답 분석기 (25개 묶음 배치 모드)")
    parser.add_argument("--path", type=str, help="분석할 CSV 파일 경로 (미지정 시 최신 파일)")
    parser.add_argument("--batch_size", type=int, default=25, help="한 번에 분석할 문항 수 (quota 설정 시 무시됨)")
    parser.add_argument("--quota", type=int, default=20, help="이번 분석에 사용할 최대 API 호출 횟수 (기본 20)")
    parser.add_argument("--num_cans", type=int, default=10, help="후보 개수 (기본 10)")
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    # 파악할 파일 선정 (데이터셋별 하위 폴더에서 탐색)
    actual_output_dir = os.path.join(conf["output_dir"], conf["dataset"])
    csv_path = args.path
    if not csv_path:
        if not os.path.exists(actual_output_dir):
            print(f">>> [오류] {actual_output_dir} 폴더가 존재하지 않습니다.")
            return
            
        files = [os.path.join(actual_output_dir, f) for f in os.listdir(actual_output_dir) if f.endswith(".csv") and "analyzed" not in f]
        if not files:
            print(f">>> [오류] {actual_output_dir} 폴더에서 분석할 CSV 파일을 찾을 수 없습니다.")
            return
        csv_path = max(files, key=os.path.getctime)

    print(f">>> 분석 대상 파일 시도: {csv_path}")
    
    # 이어하기 로직: 이미 분석된 파일(_analyzed.csv)이 있으면 그걸 불러와서 이어서 진행
    save_path = csv_path.replace(".csv", "_analyzed.csv") if "_analyzed" not in csv_path else csv_path
    
    if os.path.exists(save_path) and save_path != csv_path:
        print(f">>> 이전에 분석 중이던 파일을 발견했습니다: {save_path}")
        print(">>> 해당 파일에서 이어서 분석을 진행합니다.")
        df = pd.read_csv(save_path)
    else:
        print(f">>> 새 분석을 시작합니다: {csv_path}")
        df = pd.read_csv(csv_path)
        save_path = csv_path.replace(".csv", "_analyzed.csv") if "_analyzed" not in csv_path else csv_path
    
    # 필요한 컬럼이 없으면 생성
    if 'failure_reason' not in df.columns: df['failure_reason'] = ""
    if 'failure_tag' not in df.columns: df['failure_tag'] = ""
    
    # 진짜 오답만 필터링: hit가 0이면서, prediction이 A-J 중 하나인 경우
    # (비어있거나, 혹은 이전에 '분석 실패'라고 기록된 것들도 포함해서 다시 시도)
    valid_options = [chr(65 + i) for i in range(args.num_cans)]
    unprocessed_errors = df[
        (df['hit'] == 0) & 
        (df['prediction'].isin(valid_options)) &
        (df['failure_reason'].isna() | (df['failure_reason'] == "") | df['failure_reason'].str.contains("분석 실패", na=False))
    ].index.tolist()
    
    if not unprocessed_errors:
        print(">>> 분석할 '진짜' 오답이 없습니다! (API 에러 등은 제외됨)")
        return

    # 할당된 Quota 내에서 소화하기 위해 배치 사이즈 자동 계산
    total_count = len(unprocessed_errors)
    # quota 횟수 내외로 나누어 처리 (최소 1, 나머지는 올림 처리)
    auto_batch_size = (total_count + (args.quota - 1)) // args.quota 
    current_batch_size = max(auto_batch_size, 1)
    
    print(f">>> 총 {total_count}개의 오답을 발견했습니다.")
    print(f">>> 설정된 Quota({args.quota}회)에 맞추어 배치 사이즈를 [{current_batch_size}]로 자동 조정합니다.")
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    analysis_model = "gemini-3-flash-preview" # 분석은 똑똑한 2.0 Flash 권장
    save_path = csv_path.replace(".csv", "_analyzed.csv")

    # 자동 계산된 배치 사이즈로 루프 실행
    i = 0
    retry_count = 0
    max_retries = 3

    while i < len(unprocessed_errors):
        batch_indices = unprocessed_errors[i:i + current_batch_size]
        
        if retry_count > 0:
            print(f">>> 배치 [{i // current_batch_size + 1}] 분석 중... ({len(batch_indices)}개 문항) - 재시도 {retry_count}/{max_retries}")
        else:
            print(f">>> 배치 [{i // current_batch_size + 1}] 분석 중... ({len(batch_indices)}개 문항)")
        
        batch_rows = [df.loc[idx] for idx in batch_indices]
        
        try:
            batch_results = analyze_errors_batch(client, analysis_model, batch_rows, conf)
            
            # 결과 매칭 및 저장
            for row_idx, result in zip(batch_indices, batch_results):
                df.at[row_idx, 'failure_reason'] = result.get('reason', '분석 실패')
                df.at[row_idx, 'failure_tag'] = result.get('tag', '알수없음')
            
            df.to_csv(save_path, index=False)
            print(f"    ㄴ 저장 완료! (누적 분석 건수: {i + len(batch_indices)})")
            
            # 2.5 Flash 무료 티어 RPM 고려 (15초 대기)
            if i + current_batch_size < len(unprocessed_errors):
                time.sleep(15)
                
            # 성공 시에만 다음 배치로 전진
            i += current_batch_size
            retry_count = 0
            
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f">>> [치명적 에러] 동일 배치에서 {max_retries}회 연속 실패하여 작업을 중단합니다. ({e})")
                break
                
            print(f">>> 해당 배치에서 오류가 발생하여 10초 후 제자리 재시도합니다. ({e})")
            time.sleep(10)

    print(f"\n>>> 분석 완료! 최종 파일: {save_path}")
    
    analyzed_df = df[df['failure_tag'] != ""].copy()
    print("\n" + "="*40)
    print("      📢 오답 분석 요약 리포트 (누적)")
    print("="*40)
    tag_counts = analyzed_df['failure_tag'].value_counts().head(10)
    for tag, count in tag_counts.items():
        percentage = (count / len(analyzed_df)) * 100
        print(f"- {tag:15}: {count}건 ({percentage:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    main()


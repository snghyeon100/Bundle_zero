import os
import yaml
import json
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from google import genai

# 환경 변수 로드 (.env에서 API 키 가져오기)
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, encoding='utf-8-sig')

def evaluate_difficulty_batch(client, model, rows_list, dataset_name):
    """
    여러 개의 문제를 묶어서 한 번의 API 호출로 난이도를 평가합니다.
    """
    if "spotify" in dataset_name:
        t_name = "Playlist Continuation (음악 플레이리스트 연장)"
        b_name = "플레이리스트 (playlist)"
        i_name = "노래 (song)"
    else:
        t_name = "Bundle Construction (패션 번들 구성)"
        b_name = "패션 번들 (outfit)"
        i_name = "패션 아이템 (item)"

    cases = []
    for idx, row in rows_list:
        # 모델의 예측값은 제외하고, 문제 상황과 후보, 정답만 제공
        case_info = (
            f"Case Index: {idx}\n"
            f"- 상황 (기존 {b_name}에 포함된 것들): {row['input_str']}\n"
            f"- 후보 {i_name}들:\n{row['target_str']}\n"
            f"- 정답: {row['true_option_char']}\n"
        )
        cases.append(case_info)

    all_cases_str = "\n".join(cases)
    
    prompt = (
        f"너는 추천 시스템 문제 난이도 평가 전문가야. 다음은 '{t_name}' 태스크에 대한 {len(rows_list)}개의 문제입니다.\n"
        f"이 태스크는 주어진 {b_name}의 맥락을 보고, 10개의 후보 {i_name} 중 정답을 고르는 것입니다.\n\n"
        f"{all_cases_str}\n"
        f"### 임무\n"
        f"정답과 나머지 9개의 오답들을 비교하여, 정답을 맞추기가 얼마나 어려운지 1점부터 5점까지 난이도를 평가해줘.\n"
        f"평가 기준은 다음과 같아:\n"
        f"- 1점 (매우 쉬움): 오답들이 완전히 엉뚱한 장르나 스타일이어서 헷갈릴 여지가 전혀 없고, 정답이 너무 뚜렷함.\n"
        f"- 3점 (보통): 오답 중 2~3개 정도가 주어진 상황과 비슷한 속성(동일 아티스트/태그 등)을 가져서 고민이 필요함.\n"
        f"- 5점 (매우 어려움): 오답 대다수가 정답과 매우 유사한 속성(Hard Negatives)을 띄고 있어서 전문가도 맞추기 힘듦.\n\n"
        f"평가 이유는 한국어로 구체적이고 간결하게 설명해줘 (예: 오답 중 A, B가 정답과 같은 아티스트라서 헷갈림).\n"
        f"결과는 반드시 아래 JSON 리스트 형식으로만 답변하고 다른 설명은 절대 하지 마.\n"
        f"[\n"
        f"  {{\"index\": 정수형태의_Case_Index, \"difficulty\": 1~5사이의_정수, \"reason\": \"이유 설명\"}},\n"
        f"  ... {len(rows_list)}개 순서대로\n"
        f"]"
    )

    try:
        res = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0.0, "response_mime_type": "application/json"}
        )
        
        clean_text = res.text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[-1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[-1].split("```")[0].strip()
            
        start_idx = clean_text.find("[")
        end_idx = clean_text.rfind("]")
        if start_idx != -1 and end_idx != -1:
            clean_text = clean_text[start_idx:end_idx+1]
            
        results = json.loads(clean_text)
        
        if len(results) != len(rows_list):
            print(f">>> [경고] 응답 개수 불일치 ({len(results)} vs {len(rows_list)})")
            
        return results
    except Exception as e:
        print(f">>> [오류] API 호출 에러 발생: {str(e)}")
        raise e

def main():
    parser = argparse.ArgumentParser(description="Zero-shot 문제 난이도 측정기 (Batch 모드)")
    parser.add_argument("--batch_size", type=int, default=None, help="한 번에 분석할 문항 수 (지정 시 quota 무시)")
    parser.add_argument("--quota", type=int, default=20, help="사용할 최대 API 호출 횟수 (기본 20)")
    parser.add_argument("--path", type=str, default=None, help="기준으로 사용할 결과 CSV 파일 경로 (미지정 시 최신 파일 자동 선택)")
    args = parser.parse_args()

    # config 설정 로드
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)


    dataset_name = conf.get("dataset", "").lower()
    actual_output_dir = os.path.join(conf.get("output_dir", "./results"), dataset_name)
    
    if not os.path.exists(actual_output_dir):
        print(f">>> [오류] {actual_output_dir} 폴더가 존재하지 않습니다.")
        return

    # --path 인자가 있으면 직접 지정, 없으면 최신 CSV 자동 선택
    if args.path:
        if not os.path.exists(args.path):
            print(f">>> [오류] 지정한 파일을 찾을 수 없습니다: {args.path}")
            return
        base_csv_path = args.path
    else:
        csv_files = [os.path.join(actual_output_dir, f) for f in os.listdir(actual_output_dir) if f.endswith(".csv") and "meta" not in f]
        if not csv_files:
            print(f">>> [오류] 기준 데이터로 사용할 CSV 파일을 찾을 수 없습니다.")
            return
        base_csv_path = max(csv_files, key=os.path.getctime)
    
    print(f">>> 기준 데이터 로드 중: {os.path.basename(base_csv_path)}")
    df = pd.read_csv(base_csv_path)
    
    # 평가 메타데이터 저장 경로
    meta_path = os.path.join(actual_output_dir, "problem_difficulty_meta.csv")
    
    # 이미 분석된 내용이 있으면 불러와서 이어서 진행
    if os.path.exists(meta_path):
        print(f">>> 이전에 분석 중이던 파일 발견: {meta_path}")
        meta_df = pd.read_csv(meta_path)
        # 난이도가 비어있지 않은 인덱스 확인
        analyzed_indices = meta_df[~meta_df['difficulty'].isna()]['index'].tolist()
    else:
        print(">>> 새 난이도 분석을 시작합니다.")
        meta_df = pd.DataFrame({'index': df.index, 'difficulty': [pd.NA]*len(df), 'reason': [None]*len(df)})
        meta_df['difficulty'] = meta_df['difficulty'].astype('Int64')
        analyzed_indices = []

    # 전체 문제 중 아직 분석 안 된 문제 추출
    unprocessed_indices = [i for i in df.index.tolist() if i not in analyzed_indices]
    
    if not unprocessed_indices:
        print(">>> 모든 문제의 난이도 평가가 완료되어 있습니다!")
        return

    print(f">>> 총 {len(df)}개 문제 중 {len(unprocessed_indices)}개 문제 평가 진행 예정...")
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    eval_model = "gemma-4-31b-it" # 명시하신 대로 2.5 flash 사용
    
    total_count = len(unprocessed_indices)
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = max((total_count + (args.quota - 1)) // args.quota, 1)
        print(f">>> 설정된 Quota({args.quota}회)에 맞추어 배치 사이즈를 [{batch_size}]로 자동 조정합니다.")
    
    i = 0
    retry_count = 0
    max_retries = 99
    
    while i < len(unprocessed_indices):
        batch_indices = unprocessed_indices[i:i + batch_size]
        
        if retry_count > 0:
            print(f">>> 배치 [{i // batch_size + 1}] 난이도 평가 중... ({len(batch_indices)}개 문제) - 재시도 {retry_count}/{max_retries}")
        else:
            print(f">>> 배치 [{i // batch_size + 1}] 난이도 평가 중... ({len(batch_indices)}개 문제)")
        
        batch_rows = [(idx, df.loc[idx]) for idx in batch_indices]
        
        try:
            batch_results = evaluate_difficulty_batch(client, eval_model, batch_rows, dataset_name)
            
            # 결과 저장
            for result in batch_results:
                q_idx = result.get('index')
                if q_idx is not None and q_idx in meta_df['index'].values:
                    meta_df.loc[meta_df['index'] == q_idx, 'difficulty'] = result.get('difficulty')
                    meta_df.loc[meta_df['index'] == q_idx, 'reason'] = result.get('reason')
            
            # 진행 중간중간 저장
            meta_df.to_csv(meta_path, index=False, encoding='utf-8-sig')
            print(f"    ㄴ 저장 완료! (누적 완료 건수: {len(analyzed_indices) + i + len(batch_indices)})")
            
            # RPM 제한(15RPM) 등을 고려하여 한 번 수행 후 15초 대기
            if i + batch_size < len(unprocessed_indices):
                time.sleep(5) 
                
            # 성공 시에만 다음 배치로 인덱스 전진 및 재시도 카운트 초기화
            i += batch_size
            retry_count = 0
            
        except Exception as e:
            retry_count += 1
            if retry_count > max_retries:
                print(f">>> [치명적 에러] 동일 배치에서 {max_retries}회 연속 실패하여 작업을 중단합니다. ({e})")
                break
                
            print(f">>> 해당 배치에서 오류가 발생하여 건너뛰지 않고 30초 후 재시도합니다. ({e})")
            time.sleep(10)
            
    print(f"\n>>> 모든 난이도 평가 완료! 결과 파일: {meta_path}")

    # 최종 요약 출력
    completed_df = meta_df[~meta_df['difficulty'].isna()]
    print("\n" + "="*40)
    print("      📢 난이도 분석 요약 리포트")
    print("="*40)
    diff_counts = completed_df['difficulty'].value_counts().sort_index()
    for diff, count in diff_counts.items():
         percentage = (count / len(completed_df)) * 100
         print(f"- {diff}점 난이도: {count}건 ({percentage:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    main()

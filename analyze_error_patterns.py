import os
import yaml
import json
import pandas as pd
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

import argparse

def generate_full_report():
    parser = argparse.ArgumentParser(description="에러 패턴 메타 분석기")
    parser.add_argument("--data", required=True, help="비교 병합이 완료된 CSV 파일 경로 (compare_results.py 의 결과물)")
    args = parser.parse_args()

    # Load env for API key
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path=env_path, encoding='utf-8-sig')
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("[Error] GEMINI_API_KEY not found in .env")
        return

    client = genai.Client(api_key=api_key)
    
    csv_path = args.data
    if not os.path.exists(csv_path):
        print(f"[Error] Data file not found: {csv_path}")
        return

    print(f">>> Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # 1. 전체 250문제를 그룹(섹터)별로 나누어 분석합니다.
    # 그룹: Both_Hit, Both_Fail, A_Hit_Only, B_Hit_Only
    groups = ['Both_Hit', 'Both_Fail', 'A_Hit_Only', 'B_Hit_Only']
    
    chunk_size = 15  # 한 번 호출할 때마다 분석할 문제 수 (250 / 15 ≈ 17 API calls)
    
    chunks_with_meta = []
    
    for g in groups:
        sub_df = df[df['group'] == g].copy()
        if len(sub_df) == 0:
            continue
            
        # 난이도가 있는 경우, 난이도가 높은 것부터
        if 'difficulty' in sub_df.columns:
            sub_df = sub_df.sort_values(by='difficulty', ascending=False)
            
        sub_chunks = [sub_df[i:i + chunk_size] for i in range(0, len(sub_df), chunk_size)]
        for c in sub_chunks:
            chunks_with_meta.append({'group': g, 'data': c})
    
    # 동적 출력 파일명 생성 (입력 파일명 기반)
    base_name = os.path.basename(csv_path).replace("_analysis_ready.csv", "")
    out_file = os.path.join("analysis", f"{base_name}_meta_report.md")
    
    os.makedirs("analysis", exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"# Meta-Analysis Report: {base_name}\n\n")

    model_name = 'gemini-2.5-flash'
    print(f">>> 총 {len(df)}개의 케이스를 {len(chunks_with_meta)}번의 API 호출로 나누어 섹터별로 분석합니다 (모델: {model_name})...")
    
    for idx, chunk_meta in enumerate(chunks_with_meta):
        g_name = chunk_meta['group']
        chunk_data = chunk_meta['data']
        
        print(f"\n--- Batch {idx+1}/{len(chunks_with_meta)} [{g_name} 섹터] 진행 중 ---")
        
        prompt = f"You are a Senior AI Recommendation System Researcher. This is Batch {idx+1} of our analysis.\n"
        prompt += f"We are currently analyzing the sector: **{g_name}**.\n"
        prompt += "Method A: Base instruction. Method B: Process-of-elimination instruction.\n\n"
        
        if g_name == 'Both_Hit':
            prompt += "Task: Analyze why BOTH models succeeded here. What makes these items easier? Are there strong obvious signals? What is the LLM doing well?\n\n"
        elif g_name == 'Both_Fail':
            prompt += "Task: Analyze why BOTH models FAILED here. What is the fundamental bottleneck? Are they falling for the same distractor? Is the ground truth counter-intuitive?\n\n"
        elif g_name == 'A_Hit_Only':
            prompt += "Task: Method A succeeded but Method B FAILED. Why did the new prompt (Method B) cause a regression? Did it overthink or get confused by the elimination process?\n\n"
        else: # B_Hit_Only
            prompt += "Task: Method B succeeded while Method A FAILED. What specific improvement did Method B bring? Did the elimination process help bypass a distractor that trapped Method A?\n\n"

        for _, row in chunk_data.iterrows():
            prompt += f"=== Group: {row['group']} | Bundle ID: {row['bundle_id']} ===\n"
            if 'difficulty' in row and pd.notna(row['difficulty']):
                prompt += f"Difficulty: {row['difficulty']}/5 (Reason: {row['reason']})\n"
            
            input_text = str(row['input_str']).replace(';', '\n   ')
            target_text = str(row['target_str']).replace(';', '\n   ')
            
            prompt += f"Input Items:\n   {input_text}\n"
            prompt += f"Candidates:\n   {target_text}\n"
            prompt += f"Ground Truth: {row['true_option_char']}\n"
            
            raw_a = str(row.get('raw_response_A', '')).replace('\n', ' ')[:100]
            raw_b = str(row.get('raw_response_B', '')).replace('\n', ' ')[:100]
            prompt += f"Method A Prediction: {row['prediction_A']} (Raw: {raw_a}...)\n"
            prompt += f"Method B Prediction: {row['prediction_B']} (Raw: {raw_b}...)\n\n"
        
        prompt += f"""
        Provide a brief meta-analysis for this {g_name} batch.
        1. Core Patterns: What are the main characteristics of these {g_name} cases?
        2. Specific Insights: Highlight 1-2 interesting specific cases from this batch that perfectly illustrate this pattern.
        3. Strategic Takeaway: Based *only* on this batch, what have we learned about the LLM's behavior or our prompt?
        """
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                res = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=0.2)
                )
                
                with open(out_file, "a", encoding="utf-8") as f:
                    f.write(f"## Batch {idx+1} Analysis: {g_name} Sector\n")
                    f.write(res.text + "\n\n---\n\n")
                
                print(f"✅ Batch {idx+1} ({g_name}) 성공적으로 작성 및 저장됨!")
                
                # API 호출 간격 조절 (Rate limit 방지)
                if idx < len(chunks_with_meta) - 1:
                    time.sleep(6)  # 분당 10회 요청 제한 고려
                break
            except Exception as e:
                print(f"[Attempt {attempt+1}/{max_retries}] API Error: {e}")
                if attempt < max_retries - 1:
                    sleep_time = 15 * (attempt + 1)
                    print(f"    Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    print(f"❌ Failed Batch {idx+1} ({g_name}) after multiple retries.")
                    with open(out_file, "a", encoding="utf-8") as f:
                        f.write(f"## Batch {idx+1} Analysis: {g_name} Sector\n[Error] Failed to generate due to API errors.\n\n---\n\n")

    print(f"\n>>> ✨ All batches processed! Full report saved to {out_file}")

if __name__ == "__main__":
    generate_full_report()

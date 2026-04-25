import os
import re
import yaml
import json
import time
import asyncio
import pandas as pd
from dotenv import load_dotenv
from google import genai
from dataset import BundleZeroShotDataset, set_seed

# Load Env 
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=env_path, encoding='utf-8-sig')

def parse_model_response(raw_text):
    if not raw_text:
        return "ERR_EM"
    # Remove common prefixes like 'Choice:', 'Option:', 'Answer:' (case-insensitive)
    clean_text = re.sub(r'(?i)\b(choice|option|answer)\b[\s]*[:=]*[\s]*', '', raw_text.strip())
    # Extract the first uppercase letter (A-Z)
    match = re.search(r'([A-Z])', clean_text.upper())
    return match.group(1) if match else raw_text.strip()[0].upper()

def generate_prompt(dataset_name, input_str, target_str):
    if "spotify" in dataset_name:
        t_name = "playlist continuation"
        b_name = "music playlist"
        i_name = "song"
    else:
        t_name = "bundle construction"
        b_name = "fashion outfit"
        i_name = "fashion item"

    extra_instruction = (
        f"First infer the intent of the given {b_name}. Then, use the process of elimination: "
        f"evaluate each option, identify why the incorrect options do not fit the intent, "
        f"and eliminate them one by one until you find the best candidate {i_name}.\n"
    )

    prompt = (
        f"You are a helpful and honest assistant. The following are multiple choice questions about {t_name}. "
        f"You should directly answer the question by choosing the letter of the correct option. "
        f"Only provide the letter of your answer, without any explanation or mentioning the option content.\n"
        f"Question: Given the partial {b_name}: {input_str}, which candidate {i_name} should be included into this {b_name}?\n"
        f"Options: {target_str}\n"
        f"{extra_instruction}"
        f"Output MUST be exactly one uppercase letter (e.g., A, B, C).\nAnswer: "
    )
    return prompt

async def process_sync_samples(client, model, samples, conf):
    results = []
    
    print(f">>> Processing {len(samples)} samples sequentially to avoid rate limits (13s sleep per item)...")

    for idx, sample in enumerate(samples):
        prompt = generate_prompt(conf["dataset"], sample["input_str"], sample["target_str"])
        max_retries = 10
        base_delay = 20
        
        for attempt in range(max_retries):
            try:
                res = await client.aio.models.generate_content(
                    model=model, 
                    contents=prompt,
                    config={"temperature": conf["temperature"], "max_output_tokens": 10}
                )
                raw_text = res.text if res.text else ""
                pred_text = parse_model_response(raw_text)
                break  # Success! Break out of the retry loop
            except Exception as e:
                err_str = str(e).lower()
                # Check for rate limits (429) or high demand/server errors (503)
                if "429" in err_str or "503" in err_str or "quota" in err_str or "demand" in err_str or "overloaded" in err_str:
                    if attempt < max_retries - 1:
                        wait_time = base_delay * (attempt + 1)
                        print(f"[{idx+1}/{len(samples)}] API Error/High Demand. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                raw_text = str(e)
                pred_text = "ERR_EX"
                break
        
        sample['prediction'] = pred_text
        sample['raw_response'] = raw_text # Save verbatim output or error trace
        sample['hit'] = int(pred_text == sample['true_option_char'])
        results.append(sample)
        
        print(f"[{idx+1}/{len(samples)}] True: {sample['true_option_char']} | Pred: {pred_text}")
        
        # Enforce rate limit (Dynamic based on model Free Tier limits)
        # Gemini 2.5 Flash / Pro -> 5 requests / min = 12s interval. (Using 13s)
        # Gemini 1.5 Flash / 3.x Lite -> 15 requests / min = 4s interval. (Using 4.5s)
        sleep_time = 15.0
        if "gemma" in model or "lite" in model.lower():
            sleep_time = 5
            
        if idx < len(samples) - 1:
            await asyncio.sleep(sleep_time)
            
    return results

def process_batch_samples(client, model, samples, conf):
    print(">>> 1. Creating JSONL for Batch API...")
    jsonl_path = os.path.join(conf["output_dir"], f"batch_requests_{conf['dataset']}.jsonl")
    
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            prompt = generate_prompt(conf["dataset"], sample["input_str"], sample["target_str"])
            req_obj = {
                "id": str(idx),
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": conf["temperature"], "maxOutputTokens": 10}
                }
            }
            f.write(json.dumps(req_obj) + "\n")
            
    print(f">>> 2. Uploading file to Gemini API...")
    uploaded_file = client.files.upload(
        file=jsonl_path,
        config={'mime_type': 'application/jsonl'}
    )
    
    print(f">>> 3. Submitting Batch Job...")
    batch_job = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={"display_name": f"{conf['batch_display_name']}_{conf['dataset']}"}
    )
    
    print(f">>> Batch Job Submitted! ID: {batch_job.name}")
    print(">>> 4. Polling until complete...")
    
    while True:
        job = client.batches.get(name=batch_job.name)
        state_str = str(job.state)
        print(f"[{time.strftime('%X')}] Status: {state_str}")
        if "SUCCEEDED" in state_str or "FAILED" in state_str or "CANCELLED" in state_str:
            break
        time.sleep(conf.get("poll_interval", 60))
        
    if "SUCCEEDED" in str(job.state):
        out_file_name = job.dest.file_name
        print(f">>> 5. Complete! Results file format: {out_file_name}")
        print(">>> 6. Downloading and parsing batch results...")
        
        file_bytes = client.files.download(file=out_file_name)
        out_jsonl_path = os.path.join(conf["output_dir"], f"batch_response_{conf['dataset']}.jsonl")
        with open(out_jsonl_path, "wb") as f:
            f.write(file_bytes)
            
        print(">>> 7. Calculating Hit Rate...")
        # Parse the JSONL results 
        result_map = {}
        for line in file_bytes.decode("utf-8").splitlines():
            if not line.strip(): continue
            try:
                resp_obj = json.loads(line)
                req_id = resp_obj.get("id")
                # Try to extract the generated text
                if "response" in resp_obj:
                    raw_text = resp_obj["response"].get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    pred_text = parse_model_response(raw_text)
                elif "error" in resp_obj:
                    raw_text = str(resp_obj["error"])
                    pred_text = "ERR_API"
                else:
                    raw_text = "UNKNOWN_FORMAT"
                    pred_text = "ERR_API"
                result_map[int(req_id)] = (pred_text, raw_text)
            except Exception as e:
                continue

        # Merge with samples and evaluate
        results = []
        for idx, sample in enumerate(samples):
            pred_info = result_map.get(idx, ("ERR_MISSING", "Not found in batch response"))
            sample['prediction'] = pred_info[0]
            sample['raw_response'] = pred_info[1]
            sample['hit'] = int(pred_info[0] == sample['true_option_char'])
            results.append(sample)

        # Calculate metrics and prepare DataFrame
        df = pd.DataFrame(results)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        hit_rate = df['hit'].mean()
        valid_options = [chr(ord('A')+i) for i in range(conf["num_cans"])]
        valid_mask = df['prediction'].isin(valid_options)
        valid_ratio = valid_mask.mean()
        
        valid_only_hit_rate = df.loc[valid_mask, 'hit'].mean() if valid_mask.sum() > 0 else 0.0
        
        df['overall_hit_rate'] = hit_rate
        df['overall_valid_ratio'] = valid_ratio
        df['valid_only_hit_rate'] = valid_only_hit_rate
        
        # Insert experiment configurations
        df['cfg_num_cans'] = conf.get("num_cans", "")
        df['cfg_num_token'] = conf.get("num_token", "")
        df['cfg_toy_eval'] = conf.get("toy_eval", "")
        df['cfg_seed'] = conf.get("seed", "")
        df['cfg_shuffle_seed'] = conf.get("shuffle_seed", "")
        df['cfg_use_hard_negative'] = conf.get("use_hard_negative", False)

        # Save results in dataset-specific subfolder
        actual_output_dir = os.path.join(conf["output_dir"], conf["dataset"])
        os.makedirs(actual_output_dir, exist_ok=True)
        
        hn_str = "HN_" if conf.get("use_hard_negative", False) else ""
        save_path = os.path.join(actual_output_dir, f"results_{conf['dataset']}_batch_{hn_str}C{conf.get('num_cans', '')}_T{conf.get('num_token', '')}_{timestamp}.csv")
        df.to_csv(save_path, index=False)
        
        print("-" * 30)
        print(f"Batch Dataset: {conf['dataset']}")
        print(f"Saved to: {save_path}")
        print(f"Overall Hit Rate: {hit_rate:.4f}")
        print(f"Valid-Only Hit Rate: {valid_only_hit_rate:.4f} (from {valid_mask.sum()} samples without errors)")
        print(f"Valid Ratio: {valid_ratio:.4f}")
        print("-" * 30)
        
        save_translated_csv(df, conf, timestamp, mode_suffix="_batch", actual_output_dir=actual_output_dir)
    else:
        print(">>> Batch job did not succeed.")

def save_translated_csv(df, conf, base_timestamp, mode_suffix="", actual_output_dir=None):
    if "spotify" in conf["dataset"].lower():
        return
        
    print(">>> Translating input/target columns to Korean (via Google Translate Batch Mode - 10 per batch)...")
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='auto', target='ko')
        
        df_kor = df.copy()
        
        # 안전한 배치 번역 로직 (10개씩 묶기)
        def batch_translate(series):
            texts = series.tolist()
            translated = []
            batch_size = 10 
            try:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    try:
                        translated.extend(translator.translate_batch(batch))
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"[경고] 배치 {i//batch_size + 1} 번역 실패: {e}. 원본 유지.")
                        translated.extend(batch)
            except KeyboardInterrupt:
                print("\n>>> [중단] 사용자에 의해 번역이 중지되었습니다.")
                exit(1)
            return translated

        df_kor['input_str'] = batch_translate(df_kor['input_str'])
        df_kor['target_str'] = batch_translate(df_kor['target_str'])
        
        if not actual_output_dir:
            actual_output_dir = os.path.join(conf["output_dir"], conf["dataset"])
            
        save_path_kor = os.path.join(actual_output_dir, f"results_{conf['dataset']}{mode_suffix}_kor_{base_timestamp}.csv")
        df_kor.to_csv(save_path_kor, index=False)
        print(f">>> Saved translated file: {save_path_kor}")
    except ImportError:
        print("[경고] deep-translator 모듈이 필요합니다.")
        
if __name__ == "__main__":
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    os.makedirs(conf["output_dir"], exist_ok=True)
    set_seed(conf["seed"])
    
    dataset = BundleZeroShotDataset(conf)
    
    # NEW LOGIC: optionally load offline-generated hard negatives
    hard_negative_path = os.path.join(conf.get("data_path", "./datasets"), conf["dataset"], f"hard_negative_samples_{conf['dataset']}.json")
    if conf.get("use_hard_negative", False) and os.path.exists(hard_negative_path):
        print(f">>> Loading PRE-GENERATED HARD NEGATIVE samples from {hard_negative_path}")
        with open(hard_negative_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        samples = dataset.get_eval_samples()
        
    print(f"Total test samples prepared: {len(samples)}")
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[오류] .env 파일 안에 GEMINI_API_KEY (또는 GOOGLE_API_KEY)가 설정되지 않았습니다.")
        exit(1)
        
    client = genai.Client(api_key=api_key)
    
    if conf["mode"] == "sync":
        print(">>> Running in Sync mode...")
        import asyncio
        results = asyncio.run(process_sync_samples(client, conf["model"], samples, conf))
        
        # Save results
        df = pd.DataFrame(results)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        hit_rate = df['hit'].mean()
        valid_options = [chr(ord('A')+i) for i in range(conf["num_cans"])]
        valid_mask = df['prediction'].isin(valid_options)
        valid_ratio = valid_mask.mean()
        
        valid_only_hit_rate = df.loc[valid_mask, 'hit'].mean() if valid_mask.sum() > 0 else 0.0
        
        df['overall_hit_rate'] = hit_rate
        df['overall_valid_ratio'] = valid_ratio
        df['valid_only_hit_rate'] = valid_only_hit_rate
        
        # Insert experiment configurations
        df['cfg_num_cans'] = conf.get("num_cans", "")
        df['cfg_num_token'] = conf.get("num_token", "")
        df['cfg_toy_eval'] = conf.get("toy_eval", "")
        df['cfg_seed'] = conf.get("seed", "")
        df['cfg_shuffle_seed'] = conf.get("shuffle_seed", "")
        df['cfg_use_hard_negative'] = conf.get("use_hard_negative", False)
        
        # Save results in dataset-specific subfolder
        actual_output_dir = os.path.join(conf["output_dir"], conf["dataset"])
        os.makedirs(actual_output_dir, exist_ok=True)
        
        hn_str = "HN_" if conf.get("use_hard_negative", False) else ""
        save_path = os.path.join(actual_output_dir, f"results_{conf['dataset']}_{hn_str}C{conf.get('num_cans', '')}_T{conf.get('num_token', '')}_{timestamp}.csv")
        df.to_csv(save_path, index=False)
        
        print("-" * 30)
        print(f"Dataset: {conf['dataset']}")
        print(f"Saved to: {save_path}")
        print(f"Overall Hit Rate: {hit_rate:.4f}")
        print(f"Valid-Only Hit Rate: {valid_only_hit_rate:.4f} (from {valid_mask.sum()} samples without errors)")
        print(f"Valid Ratio: {valid_ratio:.4f}")
        print("-" * 30)
        
        save_translated_csv(df, conf, timestamp, mode_suffix="", actual_output_dir=actual_output_dir)

    elif conf["mode"] == "batch":
        print(">>> Running in Batch API mode...")
        process_batch_samples(client, conf["model"], samples, conf)

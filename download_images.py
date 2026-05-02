import os
import yaml
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.dataset import BundleZeroShotDataset

def download_image(item_id, url, save_dir):
    if not url:
        return item_id, False, "No URL"
        
    # 기본 확장자는 jpg, URL에서 식별 가능하면 사용
    ext = url.split('.')[-1].split('?')[0] # 쿼리 파라미터 제거
    if ext.lower() not in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
        ext = 'jpg'
        
    save_path = os.path.join(save_dir, f"{item_id}.{ext}")
    
    # 이미 다운로드 된 파일이 있으면 스킵
    if os.path.exists(save_path):
        return item_id, True, "Already exists"
        
    # '//gw.alicdn.com...' 형태로 시작하는 URL 처리
    if url.startswith("//"):
        url = "http:" + url
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'http://www.taobao.com/'
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return item_id, True, "Downloaded"
    except Exception as e:
        return item_id, False, str(e)

def main():
    print(">>> Loading configuration...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    dataset_name = conf["dataset"]
    dataset = BundleZeroShotDataset(conf)
    
    # Hard negative 설정에 따라 샘플 로드 방식 결정
    hard_negative_path = os.path.join(conf.get("data_path", "./datasets"), dataset_name, f"hard_negative_samples_{dataset_name}.json")
    if conf.get("use_hard_negative", False) and os.path.exists(hard_negative_path):
        print(f">>> Loading samples from {hard_negative_path}")
        with open(hard_negative_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
    else:
        print(">>> Generating evaluation samples...")
        samples = dataset.get_eval_samples()
        
    print(f"Total evaluated samples (bundles): {len(samples)}")
    
    # 평가에 사용되는 모든 item_id 수집 (중복 제거)
    unique_item_ids = set()
    for s in samples:
        unique_item_ids.update(s.get("input_indices", []))
        unique_item_ids.update(s.get("candidate_indices", []))
        
    print(f"Total unique items required: {len(unique_item_ids)}")
    
    # item_info.json 로드하여 이미지 URL 확보
    info_path = os.path.join(conf.get("data_path", "./datasets"), dataset_name, 'item_info.json')
    with open(info_path, 'r', encoding='utf-8') as f:
        item_info = json.load(f)
        
    save_dir = os.path.join(conf.get("data_path", "./datasets"), dataset_name, "images")
    os.makedirs(save_dir, exist_ok=True)
    print(f">>> Image save directory: {save_dir}")
    
    # 다운로드할 작업 리스트 생성
    download_tasks = []
    for item_id in unique_item_ids:
        item_id_str = str(item_id)
        if item_id_str in item_info:
            # pog는 "pic", pog_dense는 "pic_url"을 키로 사용
            pic_url = item_info[item_id_str].get("pic") or item_info[item_id_str].get("pic_url")
            if pic_url:
                download_tasks.append((item_id_str, pic_url))
                
    print(f"Found valid URLs for {len(download_tasks)} items. Starting concurrent download (max 20 workers)...")
    
    # 멀티스레딩으로 고속 다운로드
    success_count = 0
    exist_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(download_image, item_id, url, save_dir) for item_id, url in download_tasks]
        
        for i, future in enumerate(as_completed(futures)):
            item_id, success, msg = future.result()
            if success:
                if msg == "Already exists":
                    exist_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
                
            # 진행 상황 출력
            if (i + 1) % 200 == 0 or (i + 1) == len(futures):
                print(f"Progress: [{i+1}/{len(futures)}] | Downloaded: {success_count} | Skipped(Exists): {exist_count} | Failed: {fail_count}")

    print(">>> All download tasks completed!")

if __name__ == "__main__":
    main()

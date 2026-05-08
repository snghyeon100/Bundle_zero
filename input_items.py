import yaml
from collections import Counter
from src.dataset import BundleZeroShotDataset, set_seed

def check_original_input_lengths():
    # 1. config.yaml 로드
    with open("config.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)

    # 2. 인풋 아이템 수 제한 해제 (-1)
    original_num_token = conf.get("num_token", 5)
    conf["num_token"] = -1
    print(f"[*] 기존 제한(num_token): {original_num_token} -> 무제한(-1)으로 임시 변경하여 확인합니다.\n")

    # 3. 설정된 시드로 고정 (현재 45)
    set_seed(conf["seed"])

    # 4. 데이터셋 객체 생성 및 문제(샘플) 추출
    # (주의: 하드 네거티브가 아닌 원본 데이터셋을 직접 통과시켜야 원래 길이를 알 수 있습니다)
    dataset = BundleZeroShotDataset(conf)
    samples = dataset.get_eval_samples()

    # 5. 길이 계산
    input_lengths = [len(sample["input_indices"]) for sample in samples]
    
    if not input_lengths:
        print("생성된 문제가 없습니다.")
        return

    # 6. 통계 출력
    print("=" * 40)
    print(f"[ {conf['dataset']} ] 원본 인풋 아이템 통계")
    print("=" * 40)
    print(f"총 평가 문제 수 : {len(input_lengths)} 개")
    print(f"평균 아이템 수   : {sum(input_lengths)/len(input_lengths):.2f} 개")
    print(f"최소 아이템 수   : {min(input_lengths)} 개")
    print(f"최대 아이템 수   : {max(input_lengths)} 개")
    print("-" * 40)
    
    print("[ 아이템 개수별 분포 ]")
    counts = Counter(input_lengths)
    for length in sorted(counts.keys()):
        print(f"  {length:2d}개: {counts[length]} 문제")
    print("=" * 40)

if __name__ == "__main__":
    check_original_input_lengths()

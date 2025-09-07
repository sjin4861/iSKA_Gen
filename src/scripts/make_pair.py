import os
import json
import random

# 프롬프트 및 추론 함수는 변경 없음
PROMPTS = {
    "image_caption": "아래 [이미지 설명/상황 제시]와 [문항 세트]가 주어집니다.\n   [평가 기준]\n- (난이도) 어휘 수준·문장 길이·구문 복잡도가 과도하지 않은가? 관용구·은어·한자어 남용은 감점.\n- (명료성) 암묵적 배경지식 없이도 시각 단서가 이해되도록 설명되는가? 필요한 경우 간단한 정의/예시로 보완되는가?\n- (문항 적합성) 각 stem이 **명확하고 과도한 추론을 요구하지 않으며**, 텍스트 근거로 답할 수 있는가?\n\n위 기준에 따라 [이미지 설명/상황 제시]와 [문항 세트]를 L2 한국어 학습자를 가정하여 적절한 난이도·표현·구조인지 평가하세요.",
    "passage": "아래 [지문]과 [문항 세트]가 주어집니다.\n\n[평가 기준]\n- (난이도) 어휘 수준·문장 길이·구문 복잡도가 과도하지 않은가?\n- (명료성) 전문 용어·암묵적 배경지식 의존을 피하고, 필요한 경우 간단한 정의/예시로 해소되는가?\n- (문항 적합성) 각 stem이 **명확하고 과도한 추론을 요구하지 않으며**, 지문 근거로 답할 수 있는가?\n\n위 기준에 따라 [지문]과 [문항 세트]를 L2 한국어 학습자를 가정하여 적절한 난이도·표현·구조인지 평가하세요.",
    "audio_script": "아래 [대화]와 [문항 세트]가 주어집니다.\n\n[평가 기준]\n- (난이도) 어휘 수준·문장 길이·구문 복잡도가 과도하지 않은가? 구어체 축약·속담·은어 남용은 감점.\n- (명료성) 암묵적 배경지식 의존 없이 의미가 전달되는가? 필요한 경우 간단한 정의·예시로 해소되는가?\n- (문항 적합성) 각 stem이 **명확하고 과도한 추론을 요구하지 않으며**, 발화 근거로 답할 수 있는가?\n\n위 기준에 따라 [대화]와 [문항 세트]를 L2 한국어 학습자를 가정하여 적절한 난이도·표현·구조인지 평가하세요."
}

def infer_content_type(benchmark_id: int) -> str or None:
    if benchmark_id in [1, 2]: return 'passage'
    if benchmark_id in [3, 4]: return 'audio_script'
    if benchmark_id == 5: return 'image_caption'
    return None

# ⭐ 신규: JSONL 파일을 단순 리스트로 읽어오는 함수
def load_jsonl_as_list(file_path: str) -> list:
    """JSONL 파일을 읽어 객체 리스트로 반환합니다."""
    data_list = []
    print(f"파일 로딩 중: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))
    except FileNotFoundError:
        print(f"🚨 오류: 파일을 찾을 수 없습니다 - {file_path}")
    except json.JSONDecodeError as e:
        print(f"🚨 오류: JSON 파싱 중 오류가 발생했습니다 - {file_path}, {e}")
    return data_list

def write_jsonl(data: list, file_path: str):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ {len(data)}개의 항목을 '{file_path}'에 저장했습니다.")
    except Exception as e:
        print(f"🚨 오류: 파일 저장 중 문제가 발생했습니다 - {file_path}, {e}")

def main():
    base_dir = '/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/'
    output_dir = os.path.join(base_dir, 'rm_pair/ablation/')
    os.makedirs(output_dir, exist_ok=True)

    pairing_groups = [
        {
            "chosen_path": os.path.join(base_dir, 'chosen/chosen_dataset.jsonl'),
            "rejected_path": os.path.join(base_dir, 'rejected/rejected_dataset_l2.jsonl')
        },
        # {
        #     "chosen_path": os.path.join(base_dir, 'chosen/chosen_empg_l2.jsonl'),
        #     "rejected_path": os.path.join(base_dir, 'rejected/rejected_empg_l2.jsonl')
        # }
    ]

    all_pairs = []
    
    # ⭐ 수정: 각 그룹별로 순차적 페어링 진행
    for i, group in enumerate(pairing_groups):
        print(f"\n--- [페어링 그룹 {i+1}] 작업을 시작합니다 ---")
        chosen_list = load_jsonl_as_list(group["chosen_path"])
        rejected_list = load_jsonl_as_list(group["rejected_path"])
        
        # 두 파일의 줄 수가 다를 경우를 대비해 짧은 쪽 길이를 기준으로 함
        num_items = min(len(chosen_list), len(rejected_list))
        if len(chosen_list) != len(rejected_list):
            print(f"⚠️ 경고: 파일 라인 수가 다릅니다. Chosen({len(chosen_list)}) vs Rejected({len(rejected_list)}). {num_items}개만 페어링됩니다.")

        group_pairs_count = 0
        for j in range(num_items):
            chosen_item = chosen_list[j]
            rejected_item = rejected_list[j]

            # 프롬프트 생성을 위해 benchmark_id 추출
            meta = chosen_item.get('meta', {})
            benchmark_id = meta.get('benchmark_id') or chosen_item.get('benchmark_id')
            
            if benchmark_id is None:
                print(f"⚠️ 경고: Benchmark ID를 찾을 수 없어 {j+1}번째 항목을 건너뜁니다.")
                continue

            content_type = infer_content_type(int(benchmark_id))
            prompt = PROMPTS.get(content_type)
            
            if not prompt:
                print(f"⚠️ 경고: 유효한 content_type을 결정할 수 없어 {j+1}번째 항목을 건너뜁니다.")
                continue
                
            pair = {
                "prompt": prompt,
                "chosen": chosen_item.get('chosen'),
                "rejected": rejected_item.get('rejected')
            }
            all_pairs.append(pair)
            group_pairs_count += 1
        print(f"--- [페어링 그룹 {i+1}]에서 {group_pairs_count}개의 페어를 생성했습니다 ---")

    print(f"\n\n총 {len(all_pairs)}개의 페어를 생성했습니다.")

    random.shuffle(all_pairs)
    total_size = len(all_pairs)
    train_split_idx = int(total_size * 0.90)
    eval_split_idx = int(total_size * 0.95)

    train_data = all_pairs[:train_split_idx]
    eval_data = all_pairs[train_split_idx:eval_split_idx]
    test_data = all_pairs[eval_split_idx:]

    print(f"데이터 분할 완료: Train({len(train_data)}), Eval({len(eval_data)}), Test({len(test_data)})")

    write_jsonl(train_data, os.path.join(output_dir, 'l2_train.jsonl'))
    write_jsonl(eval_data, os.path.join(output_dir, 'l2_eval.jsonl'))
    write_jsonl(test_data, os.path.join(output_dir, 'l2_test.jsonl'))
    
    print("\n🎉 모든 작업이 성공적으로 완료되었습니다!")

if __name__ == '__main__':
    main()
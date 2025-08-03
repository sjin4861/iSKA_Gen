import json
from pathlib import Path
import random

def convert_to_reward_format(input_file: str, train_output_file: str, eval_output_file: str, eval_split_size: int = 50):
    """
    기존 데이터 형식을 'prompt', 'chosen', 'rejected' 키를 가진 JSONL 형식으로 변환하고,
    훈련(train)과 검증(eval) 데이터셋으로 분리하여 저장합니다.
    """
    input_path = Path(input_file)
    train_path = Path(train_output_file)
    eval_path = Path(eval_output_file)
    
    all_records = []

    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            source_data = json.load(infile) 
            
            for record in source_data:
                # 'conversations' 리스트의 첫 번째 항목에서 'value'를 prompt로 추출
                try:
                    prompt_text = record.get("conversations", [{}])[0].get("value", "")
                except (IndexError, AttributeError):
                    prompt_text = ""

                # 'chosen'과 'rejected' 딕셔너리에서 'value'를 텍스트로 추출
                chosen_response = record.get("chosen", {}).get("value", "")
                rejected_response = record.get("rejected", {}).get("value", "")
                
                # 모든 값이 존재할 경우에만 리스트에 추가
                if prompt_text and chosen_response and rejected_response:
                    new_record = {
                        "prompt": prompt_text.strip(),
                        "chosen": chosen_response.strip(),
                        "rejected": rejected_response.strip()
                    }
                    all_records.append(new_record)
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파일을 읽는 중 오류가 발생했습니다: {e}")
        return
    except FileNotFoundError:
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_path}")
        return

    # 데이터를 랜덤하게 섞음
    random.shuffle(all_records)
    
    # 훈련 데이터와 검증 데이터로 분리
    if len(all_records) < eval_split_size:
        print(f"⚠️ 데이터가 {eval_split_size}개보다 적어서 검증 세트를 만들 수 없습니다.")
        train_records = all_records
        eval_records = []
    else:
        eval_records = all_records[:eval_split_size]
        train_records = all_records[eval_split_size:]

    # 훈련 데이터셋 파일 작성
    with open(train_path, 'w', encoding='utf-8') as outfile:
        for record in train_records:
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print(f"✅ 총 {len(train_records)}개의 훈련 데이터 변환 완료!")
    print(f"   결과가 '{train_path}'에 저장되었습니다.")

    # 검증 데이터셋 파일 작성
    if eval_records:
        with open(eval_path, 'w', encoding='utf-8') as outfile:
            for record in eval_records:
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        print(f"✅ 총 {len(eval_records)}개의 검증 데이터 변환 완료!")
        print(f"   결과가 '{eval_path}'에 저장되었습니다.")


# --- 실행 예시 ---
if __name__ == "__main__":
    # 변환할 원본 데이터 파일 경로
    source_dataset_path = "src/data/completeness_for_guidelines_rm.json" # 실제 파일 경로로 수정하세요
    
    # 저장될 훈련 데이터 파일 경로
    train_dataset_path = "src/data/completeness_for_guidelines_rm_train.jsonl"
    # 저장될 검증 데이터 파일 경로
    eval_dataset_path = "src/data/completeness_for_guidelines_rm_eval.jsonl"

    # 함수 실행 (검증 데이터 50개 분리)
    convert_to_reward_format(source_dataset_path, train_dataset_path, eval_dataset_path, eval_split_size=50)
import json
from pathlib import Path

def convert_to_reward_format_with_margin(input_file: str, output_file: str, margin_value: float = 1.0):
    """
    기존 데이터 형식을 RewardTrainer에 맞는 chosen/rejected 쌍으로 변환하고,
    고정된 margin 값을 추가합니다.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    converted_count = 0
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        # 원본 파일이 JSON 배열 형식이라고 가정
        source_data = json.load(infile) 
        
        for record in source_data:
            chosen_response = record.get("chosen", {}).get("value", "")
            rejected_response = record.get("rejected", {}).get("value", "")
            
            if chosen_response and rejected_response:
                new_record = {
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                    "margin": margin_value  # ✨ 여기에 고정 마진 값 추가
                }
                outfile.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                converted_count += 1
                
    print(f"✅ 총 {converted_count}개의 데이터에 margin({margin_value}) 추가 완료!")
    print(f"   결과가 '{output_path}'에 저장되었습니다.")


# --- 실행 예시 ---
if __name__ == "__main__":
    # 원본 데이터셋 경로
    source_dataset_path = "/home/sjin4861/25-1/HCLT/iSKA_Gen/src/data/korean_rm_chat_test.json" 
    # Margin이 추가된 새로운 데이터 파일 경로
    rm_training_dataset_path = "/home/sjin4861/25-1/HCLT/iSKA_Gen/src/data/korean_rm_chat_test_with_margin.jsonl"

    # 함수를 실행하여 margin 추가 (마진값을 1.0으로 설정)
    convert_to_reward_format_with_margin(source_dataset_path, rm_training_dataset_path)


# import json
# from pathlib import Path

# def convert_to_reward_format(input_file: str, output_file: str):
#     """
#     기존 데이터 형식을 RewardTrainer에 맞는 chosen/rejected 쌍으로 변환합니다.
#     """
#     input_path = Path(input_file)
#     output_path = Path(output_file)
    
#     converted_count = 0
#     with open(input_path, 'r', encoding='utf-8') as infile, \
#          open(output_path, 'w', encoding='utf-8') as outfile:
        
#         # 원본 파일이 JSON 배열 형식이라고 가정
#         source_data = json.load(infile) 
        
#         for record in source_data:
#             # ✨ 주요 변경 사항: conversations에서 prompt 텍스트 추출 ✨
#             try:
#                 prompt_text = record.get("conversations", [{}])[0].get("value", "")
#             except (IndexError, AttributeError):
#                 prompt_text = ""

#             chosen_response = record.get("chosen", {}).get("value", "")
#             rejected_response = record.get("rejected", {}).get("value", "")
            
#             # 두 값이 모두 존재할 경우에만 저장
#             if chosen_response and rejected_response:
#                 new_record = {
#                     "chosen": chosen_response,
#                     "rejected": rejected_response
#                 }
#                 outfile.write(json.dumps(new_record, ensure_ascii=False) + '\n')
#                 converted_count += 1
                
#     print(f"✅ 총 {converted_count}개의 데이터 변환 완료!")
#     print(f"   결과가 '{output_path}'에 저장되었습니다.")


# # --- 실행 예시 ---
# # 원본 데이터 파일 경로
# if __name__ == "__main__":
#     # 원본 데이터셋 경로
#     source_dataset_path = "/home/sjin4861/25-1/HCLT/iSKA_Gen/src/data/korean_quality_rm.json" 
#     # RewardTrainer가 사용할 최종 데이터 파일 경로
#     rm_training_dataset_path = "/home/sjin4861/25-1/HCLT/iSKA_Gen/src/data/simple_korean_quality_rm.jsonl"

#     convert_to_reward_format(source_dataset_path, rm_training_dataset_path)

#     #!/usr/bin/env python3

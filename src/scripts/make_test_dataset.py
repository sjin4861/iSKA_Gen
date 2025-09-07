import os
import json

# 1. content_type에 따른 프롬프트 템플릿 정의
PROMPTS = {
    "image_caption": "아래 [이미지 설명/상황 제시]와 [문항 세트]가 주어집니다.\n   [평가 기준]\n- (난이도) 어휘 수준·문장 길이·구문 복잡도가 과도하지 않은가? 관용구·은어·한자어 남용은 감점.\n- (명료성) 암묵적 배경지식 없이도 시각 단서가 이해되도록 설명되는가? 필요한 경우 간단한 정의/예시로 보완되는가?\n- (문항 적합성) 각 stem이 **명확하고 과도한 추론을 요구하지 않으며**, 텍스트 근거로 답할 수 있는가?\n\n위 기준에 따라 [이미지 설명/상황 제시]와 [문항 세트]를 L2 한국어 학습자를 가정하여 적절한 난이도·표현·구조인지 평가하세요.",
    "passage": "아래 [지문]과 [문항 세트]가 주어집니다.\n\n[평가 기준]\n- (난이도) 어휘 수준·문장 길이·구문 복잡도가 과도하지 않은가?\n- (명료성) 전문 용어·암묵적 배경지식 의존을 피하고, 필요한 경우 간단한 정의/예시로 해소되는가?\n- (문항 적합성) 각 stem이 **명확하고 과도한 추론을 요구하지 않으며**, 지문 근거로 답할 수 있는가?\n\n위 기준에 따라 [지문]과 [문항 세트]를 L2 한국어 학습자를 가정하여 적절한 난이도·표현·구조인지 평가하세요.",
    "audio_script": "아래 [대화]와 [문항 세트]가 주어집니다.\n\n[평가 기준]\n- (난이도) 어휘 수준·문장 길이·구문 복잡도가 과도하지 않은가? 구어체 축약·속담·은어 남용은 감점.\n- (명료성) 암묵적 배경지식 의존 없이 의미가 전달되는가? 필요한 경우 간단한 정의·예시로 해소되는가?\n- (문항 적합성) 각 stem이 **명확하고 과도한 추론을 요구하지 않으며**, 발화 근거로 답할 수 있는가?\n\n위 기준에 따라 [대화]와 [문항 세트]를 L2 한국어 학습자를 가정하여 적절한 난이도·표현·구조인지 평가하세요."
}

def infer_content_type(benchmark_id: int) -> str or None:
    """benchmark_id를 기반으로 content_type을 추론합니다."""
    if benchmark_id in [1, 2]:
        return 'passage'
    elif benchmark_id in [3, 4]:
        return 'audio_script'
    elif benchmark_id == 5:
        return 'image_caption'
    else:
        return None

def main():
    """
    메인 실행 함수: rejected 데이터셋에 프롬프트를 추가합니다.
    """
    # 1. 입출력 경로 설정
    base_dir = '/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/'
    input_file_path = os.path.join(base_dir, 'rejected/rejected_dataset_test_1.jsonl')
    output_file_path = os.path.join(base_dir, 'final_prompted_dataset_rejected_1.jsonl') # 최종 결과 파일명

    processed_data = []

    print(f"입력 파일 처리 중: {input_file_path}")
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                meta = item.get('meta', {})
                rejected_text = item.get('rejected')

                # benchmark_id와 rejected 텍스트가 있는지 확인
                benchmark_id = meta.get('benchmark_id')
                if benchmark_id is None or rejected_text is None:
                    print(f"⚠️ 경고: benchmark_id 또는 rejected 내용이 없어 라인을 건너뜁니다: {line.strip()}")
                    continue

                # content_type 추론 및 프롬프트 선택
                content_type = infer_content_type(int(benchmark_id))
                prompt = PROMPTS.get(content_type)
                
                if not prompt:
                    print(f"⚠️ 경고: 유효한 프롬프트를 찾을 수 없어 라인을 건너뜁니다 (benchmark_id: {benchmark_id})")
                    continue

                # 새로운 데이터 형식 생성
                new_item = {
                    "prompt": prompt,
                    "rejected": rejected_text
                }
                processed_data.append(new_item)

        print("\n✅ 파일 읽기 및 처리 완료.")

        # 2. 최종 파일 저장
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for entry in processed_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"총 {len(processed_data)}개의 항목을 '{output_file_path}' 파일로 저장했습니다.")
        print("🎉 모든 작업이 성공적으로 완료되었습니다!")

    except FileNotFoundError:
        print(f"🚨 오류: 입력 파일 '{input_file_path}'를 찾을 수 없습니다.")
    except Exception as e:
        print(f"🚨 오류: 처리 중 문제가 발생했습니다 - {e}")

if __name__ == '__main__':
    main()
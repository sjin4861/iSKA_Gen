import os
import json
import re
from typing import Dict, List

def load_stems_as_list(stems_dir: str) -> Dict[str, List[str]]:
    """
    stems 디렉터리에서 모든 benchmark_{id}.jsonl 파일을 읽어
    benchmark_id를 key로, question_set 리스트를 value로 갖는 딕셔너리를 생성합니다.
    """
    stems_lookup_table = {}
    print(f"'{stems_dir}'에서 문항 세트(stems)를 로드합니다...")
    
    try:
        filenames = sorted(os.listdir(stems_dir))
        for filename in filenames:
            if filename.startswith("benchmark_") and filename.endswith(".jsonl"):
                match = re.search(r'benchmark_(\d+)', filename)
                if not match: continue
                
                benchmark_id = match.group(1)
                stems_lookup_table[benchmark_id] = []
                
                file_path = os.path.join(stems_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        question_set = data.get("question_set")
                        if question_set:
                            stems_lookup_table[benchmark_id].append(question_set)
                            
        print("✅ 문항 세트 로드 완료.")
        return stems_lookup_table
    except FileNotFoundError:
        print(f"🚨 오류: Stems 디렉터리 '{stems_dir}'를 찾을 수 없습니다.")
        return None

def get_content_label(content_type: str) -> str:
    """content_type에 따라 적절한 한글 레이블을 반환합니다."""
    if content_type == 'passage': return '[지문]'
    if content_type == 'audio_script': return '[대화문]'
    if content_type == 'image_caption': return '[이미지 설명]'
    return '[콘텐츠]'

def clean_content(text: str) -> str:
    """콘텐츠에서 '**...**'와 같은 마크다운 헤더를 제거합니다."""
    if not isinstance(text, str): return ""
    cleaned_text = re.sub(r'^\*\*.*\*\*\n?', '', text, flags=re.MULTILINE).strip()
    return cleaned_text

def write_jsonl(data: list, file_path: str):
    """주어진 데이터를 JSONL 형식으로 파일에 씁니다."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ {len(data)}개의 항목을 '{file_path}'에 저장했습니다.")
    except Exception as e:
        print(f"🚨 오류: 파일 저장 중 문제가 발생했습니다 - {file_path}, {e}")

def main():
    """
    메인 실행 함수: raw_outputs 데이터를 처리하여 rejected 데이터셋을 생성합니다.
    """
    # 1. 경로 설정
    raw_outputs_dir = '/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/raw_outputs/2025-08-23/'
    stems_dir = '/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/stems/'
    output_file_path = '/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/rejected_dataset_test.jsonl'

    # 2. Stems 데이터 로드
    stems_data = load_stems_as_list(stems_dir)
    if not stems_data:
        return

    all_rejected_data = []
    
    print(f"\n'{raw_outputs_dir}'에서 생성된 결과물 처리를 시작합니다...")
    
    # 3. Raw Outputs 디렉터리 순회
    for root, dirs, files in os.walk(raw_outputs_dir):
        for filename in files:
            if not filename.endswith('.json'):
                continue
            
            file_path = os.path.join(root, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    generated_data = json.load(f)

                for item in generated_data:
                    # 4. source_id 기반으로 문항 세트 매칭
                    source_id = item.get("source_id")
                    if not source_id: continue
                    
                    match = re.search(r'bench_(\d+)_item_(\d+)', source_id)
                    if not match: continue
                    
                    benchmark_id, item_index = match.groups()
                    item_index = int(item_index)
                    
                    question_set = None
                    if benchmark_id in stems_data and item_index < len(stems_data[benchmark_id]):
                        question_set = stems_data[benchmark_id][item_index]
                    else:
                        print(f"⚠️ 경고: 일치하는 문항 세트를 찾을 수 없습니다. (Source ID: {source_id})")
                        continue

                    # 5. content 파싱 및 정리
                    content_to_process = item.get('content') or item.get('generated_passage', '')
                    if not content_to_process: continue
                    
                    cleaned_content = clean_content(content_to_process)
                    
                    # 6. 최종 데이터 형식 구성
                    meta = {
                        "source_id": source_id,
                        "benchmark_id": int(benchmark_id),
                        "model_name": item.get("model_name"),
                        "content_type": item.get("content_type"),
                        "template_key": item.get("meta", {}).get("template_key"),
                        "generated_at": item.get("generated_at")
                    }
                    
                    content_label = get_content_label(item.get("content_type"))
                    rejected_text = f"{content_label}\n{cleaned_content}\n\n[문항 세트]\n{question_set}"
                    
                    all_rejected_data.append({
                        "meta": meta,
                        "rejected": rejected_text
                    })
            
            except (IndexError, json.JSONDecodeError) as e:
                print(f"🚨 오류: 파일 처리 중 문제가 발생했습니다. ({file_path}) - {e}")
                continue
    
    print("\n✅ 모든 파일 처리가 완료되었습니다.")
    
    # 7. 최종 파일 저장
    if all_rejected_data:
        write_jsonl(all_rejected_data, output_file_path)
        print("🎉 작업이 성공적으로 완료되었습니다!")
    else:
        print("처리할 데이터를 찾지 못했습니다.")

if __name__ == '__main__':
    main()
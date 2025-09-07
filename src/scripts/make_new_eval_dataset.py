import os
import json
import re
from pathlib import Path
from tqdm import tqdm

# --- 0. 경로 설정 ---
# 벤치마크 원본 정의 파일 (problem_types, eval_goals 정보 포함)
BENCHMARK_DEF_FILE = "/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/benchmarks/v1/iSKA-Gen_Benchmark_v1.1.0_20250808.json"
# 원본 콘텐츠가 있는 루트 디렉터리
SOURCE_CONTENT_ROOT = "/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/raw_outputs/2025-08-08"
# 덧붙일 Stems가 있는 디렉터리
STEMS_ROOT = "/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/stems/train"
# 최종 결과물을 저장할 루트 디렉터리
OUTPUT_ROOT = "/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/raw_outputs/2025-08-24/stem"


# --- 1. 데이터 로딩 함수 ---

def load_benchmark_definitions(file_path: str) -> dict:
    """벤치마크 정의 파일(JSON)을 로드하여 ID를 키로 하는 딕셔너리를 반환합니다."""
    print(f"🔄 벤치마크 정의 로딩 중: {file_path}")
    definitions = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data["benchmark"]
            for benchmark in data:
                definitions[str(benchmark['id'])] = {
                    "problem_types": benchmark.get("problem_types", []),
                    "eval_goals": benchmark.get("eval_goals", [])
                }
        print("✅ 벤치마크 정의 로드 완료.")
        return definitions
    except FileNotFoundError:
        print(f"🚨 오류: 벤치마크 정의 파일을 찾을 수 없습니다 - {file_path}")
        return {}

def parse_stems_from_question_set(question_set: str) -> list:
    """'1) ...\n2) ...' 형식의 문자열을 파싱하여 stem 리스트를 반환합니다."""
    if not question_set:
        return []
    # 각 줄을 분리하고, '1) ', '2) ' 등의 앞부분을 제거
    stems = [re.sub(r'^\d+\)\s*', '', line).strip() for line in question_set.strip().split('\n')]
    return stems

def load_stems(stems_dir: str) -> dict:
    """Stems 디렉터리의 모든 .jsonl 파일을 읽어 딕셔너리로 반환합니다."""
    print(f"🔄 Stems 데이터 로딩 중: {stems_dir}")
    stems_map = {}
    try:
        for filename in sorted(os.listdir(stems_dir)):
            if filename.startswith("benchmark_") and filename.endswith(".jsonl"):
                match = re.search(r'benchmark_(\d+)', filename)
                if not match: continue
                
                benchmark_id = match.group(1)
                stems_map[benchmark_id] = []
                
                file_path = os.path.join(stems_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        question_set = data.get("question_set", "")
                        parsed_stems = parse_stems_from_question_set(question_set)
                        stems_map[benchmark_id].append(parsed_stems)
        print("✅ Stems 데이터 로드 완료.")
        return stems_map
    except FileNotFoundError:
        print(f"🚨 오류: Stems 디렉터리를 찾을 수 없습니다 - {stems_dir}")
        return {}


# --- 2. 메인 실행 블록 ---

def main():
    # 데이터 로드
    benchmark_defs = load_benchmark_definitions(BENCHMARK_DEF_FILE)
    stems_data = load_stems(STEMS_ROOT)

    if not benchmark_defs or not stems_data:
        print("🚨 필수 데이터가 없어 작업을 중단합니다.")
        return

    print(f"\n🚀 콘텐츠와 Stem 통합 작업을 시작합니다...")
    
    # 원본 콘텐츠 디렉터리 순회
    source_files = list(Path(SOURCE_CONTENT_ROOT).rglob("*.json"))
    for source_file_path in tqdm(source_files, desc="파일 처리 중"):
        try:
            # 벤치마크 ID 추출
            match = re.search(r'benchmark_(\d+)', source_file_path.name)
            if not match: continue
            benchmark_id = match.group(1)

            # 필요한 데이터가 있는지 확인
            if benchmark_id not in benchmark_defs or benchmark_id not in stems_data:
                continue

            # 원본 콘텐츠 로드
            with open(source_file_path, 'r', encoding='utf-8') as f:
                source_contents = json.load(f)

            # 새로운 형식의 데이터 생성
            integrated_data = []
            for i, content_item in enumerate(source_contents):
                new_item = {}
                
                # 'generated_passage' 또는 'content'를 'source_passage'로 매핑
                new_item["source_passage"] = content_item.get("generated_passage") or content_item.get("content", "")
                
                # 벤치마크 정의 및 stems 데이터 결합
                defs = benchmark_defs[benchmark_id]
                stems = stems_data[benchmark_id][i] if i < len(stems_data[benchmark_id]) else []
                
                for j in range(len(defs["problem_types"])):
                    new_item[f"problem_type_{j+1}"] = defs["problem_types"][j] if j < len(defs["problem_types"]) else ""
                    new_item[f"eval_goal_{j+1}"] = defs["eval_goals"][j] if j < len(defs["eval_goals"]) else ""
                    new_item[f"stem_{j+1}"] = stems[j] if j < len(stems) else ""
                
                new_item["source_item"] = content_item.get("source_item")
                integrated_data.append(new_item)

            # 결과 파일 저장 경로 생성
            relative_path = source_file_path.relative_to(SOURCE_CONTENT_ROOT)
            # 중간 경로에서 content_type (첫 번째 디렉터리) 제거
            output_relative_path = Path(*relative_path.parts[1:]) 
            
            output_path = Path(OUTPUT_ROOT) / output_relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 결과 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(integrated_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"🚨 파일 처리 중 오류 발생: {source_file_path} - {e}")

    print("\n\n🎉 모든 작업이 성공적으로 완료되었습니다!")
    print(f"결과는 '{OUTPUT_ROOT}' 디렉터리에 저장되었습니다.")

if __name__ == "__main__":
    main()

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path.cwd().parent.parent))

# --- 프로젝트의 루트 디렉토리를 기준으로 기본 저장 경로 설정 ---
# 이 파일(output_saver.py)이 src/utils/에 있다고 가정합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_OUTPUT_DIR = PROJECT_ROOT / "src" / "data" / "raw_outputs"
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "src" / "data" / "evaluations"
DEFAULT_PAIRWISE_DATA_DIR = PROJECT_ROOT / "src" / "data" / "pairwise_data"


def save_model_output(
    model_name: str,
    benchmark_id: int,
    benchmark_version: str,
    template_key: str,
    data: List[Dict[str, Any]],
    base_dir: Path = DEFAULT_RAW_OUTPUT_DIR
) -> Path:
    """
    모델이 생성한 원본 결과물을 지정된 디렉토리 구조에 맞춰 저장합니다.

    Args:
        model_name (str): 결과를 생성한 모델의 이름.
        benchmark_id (int): 실행된 벤치마크의 ID.
        benchmark_version (str): 벤치마크 버전 (예: "v1.0.0").
        template_key (str): 사용된 템플릿 키 (예: "passage", "stem", "options").
        data (List[Dict[str, Any]]): 저장할 데이터 (JSON으로 직렬화 가능해야 함).
        base_dir (Path): 모든 결과물이 저장될 최상위 디렉토리.

    Returns:
        Path: 최종적으로 저장된 파일의 경로 객체.
        
    Raises:
        IOError: 파일 쓰기 중 오류가 발생할 경우.
    """
    try:
        # 1. 날짜와 모델 이름으로 하위 폴더 경로 생성
        # 예: data/raw_outputs/2025-07-26_Qwen3-8B/
        date_str = datetime.now().strftime("%Y-%m-%d")
        model_run_dir = base_dir / f"{date_str}_{model_name}"
        
        # 2. 폴더가 존재하지 않으면 생성
        model_run_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. 최종 파일 경로 생성
        # 예: benchmark_1_v1.0.0_passage_20250725_143022.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"benchmark_{benchmark_id}_{benchmark_version}_{template_key}_{timestamp}.json"
        output_path = model_run_dir / file_name
        
        # 4. JSON 파일로 저장
        print(f"  💾 '{output_path}'에 결과를 저장합니다...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        print(f"  ✅ 저장 완료.")
        return output_path

    except (IOError, TypeError) as e:
        print(f"❌ 파일 저장 중 오류 발생: {e}")
        # 오류를 다시 발생시켜 상위 호출자가 처리할 수 있도록 함
        raise IOError(f"Failed to save data for model {model_name} to {output_path}") from e


# --- 실행 예시 ---
if __name__ == "__main__":
    # 이 스크립트를 직접 실행할 때 아래 코드가 동작합니다.
    
    # 1. 테스트용 더미 데이터
    sample_model_name = "EXAONE-3.5B_test"
    sample_benchmark_id = 1
    sample_benchmark_version = "v1.0.0"
    sample_template_key = "passage"
    sample_data = [
        {"source_item": {"korean_topic": "회식"}, "generated_passage": "회식은..."},
        {"source_item": {"korean_topic": "김장"}, "generated_passage": "김장은..."}
    ]

    print("--- save_model_output 함수 테스트 ---")
    try:
        saved_file = save_model_output(
            model_name=sample_model_name,
            benchmark_id=sample_benchmark_id,
            benchmark_version=sample_benchmark_version,
            template_key=sample_template_key,
            data=sample_data
        )
        print(f"\n테스트 결과가 다음 파일에 성공적으로 저장되었습니다:\n{saved_file}")
        
        # 저장된 파일 내용 확인 (옵션)
        with open(saved_file, 'r', encoding='utf-8') as f:
            read_data = json.load(f)
        print("\n저장된 파일 내용 확인 (일부):")
        print(json.dumps(read_data[0], ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")
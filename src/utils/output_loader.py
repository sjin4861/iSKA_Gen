import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path.cwd().parent.parent))

# --- 프로젝트의 루트 디렉토리를 기준으로 기본 로드 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_OUTPUT_DIR = PROJECT_ROOT / "src" / "data" / "raw_outputs"
DEFAULT_EVALUATION_DIR = PROJECT_ROOT / "src" / "data" / "evaluations"


def list_available_outputs(base_dir: Path = DEFAULT_RAW_OUTPUT_DIR) -> Dict[str, List[Path]]:
    """
    사용 가능한 모든 출력 파일을 모델별로 정리하여 반환합니다.
    
    Args:
        base_dir (Path): 출력 파일들이 저장된 기본 디렉토리
        
    Returns:
        Dict[str, List[Path]]: 모델명을 키로 하고 파일 경로 리스트를 값으로 하는 딕셔너리
    """
    if not base_dir.exists():
        return {}
        
    available_outputs = {}
    
    # 날짜_모델명 형태의 디렉토리들을 탐색
    for model_dir in base_dir.iterdir():
        if model_dir.is_dir() and "_" in model_dir.name:
            # 디렉토리명에서 모델명 추출 (날짜_모델명 형태에서 모델명 부분)
            parts = model_dir.name.split("_", 1)
            if len(parts) == 2:
                date_part, model_name = parts
                
                if model_name not in available_outputs:
                    available_outputs[model_name] = []
                
                # 해당 디렉토리의 모든 JSON 파일 수집
                json_files = list(model_dir.glob("*.json"))
                available_outputs[model_name].extend(json_files)
    
    # 각 모델의 파일들을 시간순으로 정렬
    for model_name in available_outputs:
        available_outputs[model_name].sort()
    
    return available_outputs


def parse_filename(file_path: Path) -> Optional[Dict[str, str]]:
    """
    파일명에서 메타데이터를 추출합니다.
    
    Args:
        file_path (Path): 분석할 파일 경로
        
    Returns:
        Optional[Dict[str, str]]: 추출된 메타데이터 또는 None
    """
    filename = file_path.stem  # 확장자 제거
    
    # benchmark_1_v1.0.0_passage_20250725_143022 형태의 파일명 파싱
    # 템플릿 키 부분이 여러 단어일 수 있으므로 더 유연한 패턴 사용
    pattern = r"benchmark_(\d+)_([^_]+)_(.+?)_(\d{8}_\d{6})$"
    match = re.match(pattern, filename)
    
    if match:
        benchmark_id, version, template_key, timestamp = match.groups()
        return {
            "benchmark_id": benchmark_id,
            "benchmark_version": version,
            "template_key": template_key,
            "timestamp": timestamp,
            "model_name": file_path.parent.name.split("_", 1)[1] if "_" in file_path.parent.name else "unknown"
        }
    return None


def load_model_outputs(
    model_name: str,
    benchmark_id: Optional[int] = None,
    benchmark_version: Optional[str] = None,
    template_key: Optional[str] = None,
    base_dir: Path = DEFAULT_RAW_OUTPUT_DIR,
    latest_only: bool = True
) -> List[Tuple[Dict[str, str], List[Dict[str, Any]]]]:
    """
    지정된 조건에 맞는 모델 출력 데이터를 로드합니다.
    
    Args:
        model_name (str): 로드할 모델의 이름
        benchmark_id (Optional[int]): 특정 벤치마크 ID (None이면 모든 ID)
        benchmark_version (Optional[str]): 특정 벤치마크 버전 (None이면 모든 버전)
        template_key (Optional[str]): 특정 템플릿 키 (None이면 모든 템플릿)
        base_dir (Path): 출력 파일들이 저장된 기본 디렉토리
        latest_only (bool): True면 가장 최근 파일만, False면 모든 매칭 파일
        
    Returns:
        List[Tuple[Dict[str, str], List[Dict[str, Any]]]]: (메타데이터, 데이터) 튜플 리스트
    """
    available_outputs = list_available_outputs(base_dir)
    
    if model_name not in available_outputs:
        print(f"Warning: No outputs found for model '{model_name}'")
        return []
    
    matching_files = []
    
    for file_path in available_outputs[model_name]:
        metadata = parse_filename(file_path)
        if not metadata:
            continue
            
        # 조건 필터링
        if benchmark_id is not None and int(metadata["benchmark_id"]) != benchmark_id:
            continue
        if benchmark_version is not None and metadata["benchmark_version"] != benchmark_version:
            continue
        if template_key is not None:
            # 정확한 매칭을 먼저 시도하고, 없으면 포함 관계로 검색
            if metadata["template_key"] != template_key:
                continue
            
        matching_files.append((file_path, metadata))
    
    if not matching_files:
        print(f"Warning: No files match the specified criteria")
        return []
    
    # 시간순으로 정렬
    matching_files.sort(key=lambda x: x[1]["timestamp"])
    
    if latest_only:
        # 각 고유한 (benchmark_id, template_key) 조합에 대해 가장 최근 파일만 선택
        unique_files = {}
        for file_path, metadata in matching_files:
            key = (metadata["benchmark_id"], metadata["template_key"])
            unique_files[key] = (file_path, metadata)
        matching_files = list(unique_files.values())
    
    # 데이터 로드
    results = []
    for file_path, metadata in matching_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            results.append((metadata, data))
            print(f"✅ Loaded: {file_path.name}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ Failed to load {file_path}: {e}")
            continue
    
    return results


def load_passages(
    model_name: str,
    benchmark_id: Optional[int] = None,
    benchmark_version: str = "v1.0.0",
    template_key: Optional[str] = None,
    base_dir: Path = DEFAULT_RAW_OUTPUT_DIR
) -> List[Dict[str, Any]]:
    """
    특정 모델의 passage 데이터를 로드합니다.
    
    Args:
        model_name (str): 모델 이름
        benchmark_id (Optional[int]): 벤치마크 ID
        benchmark_version (str): 벤치마크 버전
        template_key (Optional[str]): 특정 템플릿 키 (None이면 자동으로 최신 passage 파일 검색)
        base_dir (Path): 기본 디렉토리
        
    Returns:
        List[Dict[str, Any]]: passage 데이터 리스트
    """
    
    # 1. 특정 템플릿 키가 지정된 경우 정확한 매칭으로 검색
    if template_key is not None:
        results = load_model_outputs(
            model_name=model_name,
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
            template_key=template_key,
            base_dir=base_dir,
            latest_only=True
        )
        
        if results:
            print(f"✅ Found passage file with exact template: {template_key}")
            return results[0][1]
        else:
            print(f"❌ No passage file found with template: {template_key}")
            return []
    
    # 2. 템플릿 키가 지정되지 않은 경우 자동으로 passage 파일 검색
    print("🔍 템플릿 키가 지정되지 않음. 자동으로 passage 파일을 검색합니다...")
    
    # 모든 파일을 가져와서 passage가 포함된 파일 중 가장 최근 것 선택
    all_results = load_model_outputs(
        model_name=model_name,
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        template_key=None,  # 모든 템플릿 키 검색
        base_dir=base_dir,
        latest_only=False  # 모든 파일 검색
    )
    
    # passage가 포함되고 stem이나 options가 포함되지 않은 파일 찾기
    passage_files = []
    for metadata, data in all_results:
        template_key_lower = metadata["template_key"].lower()
        if ("passage" in template_key_lower and 
            "stem" not in template_key_lower and 
            "options" not in template_key_lower and
            "eval" not in template_key_lower):
            passage_files.append((metadata, data))
    
    if passage_files:
        # 시간순으로 정렬하여 가장 최근 파일 선택
        passage_files.sort(key=lambda x: x[0]["timestamp"])
        latest_metadata, latest_data = passage_files[-1]
        print(f"✅ Found passage file with template: {latest_metadata['template_key']}")
        return latest_data
    
    return []


def load_stems(
    model_name: str,
    benchmark_id: Optional[int] = None,
    benchmark_version: str = "v1.0.0",
    template_key: str = "passage_stem",
    base_dir: Path = DEFAULT_RAW_OUTPUT_DIR
) -> List[Dict[str, Any]]:
    """
    특정 모델의 stem 데이터를 로드합니다.
    
    Args:
        model_name (str): 모델 이름
        benchmark_id (Optional[int]): 벤치마크 ID
        benchmark_version (str): 벤치마크 버전
        template_key (str): 템플릿 키 (기본: "passage_stem")
        base_dir (Path): 기본 디렉토리
        
    Returns:
        List[Dict[str, Any]]: stem 데이터 리스트
    """
    results = load_model_outputs(
        model_name=model_name,
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        template_key=template_key,  # stem 파일을 찾기 위해 올바른 템플릿 키 사용
        base_dir=base_dir,
        latest_only=True
    )
    
    if results:
        return results[0][1]  # 첫 번째 결과의 데이터 부분
    return []


def load_options(
    model_name: str,
    benchmark_id: Optional[int] = None,
    benchmark_version: str = "v1.0.0",
    template_key: str = "passage_options",
    base_dir: Path = DEFAULT_RAW_OUTPUT_DIR
) -> List[Dict[str, Any]]:
    """
    특정 모델의 options 데이터를 로드합니다.
    
    Args:
        model_name (str): 모델 이름
        benchmark_id (Optional[int]): 벤치마크 ID
        benchmark_version (str): 벤치마크 버전
        template_key (str): 템플릿 키 (보통 "{원본_템플릿}_options" 형태)
        base_dir (Path): 기본 디렉토리
        
    Returns:
        List[Dict[str, Any]]: options 데이터 리스트
    """
    results = load_model_outputs(
        model_name=model_name,
        benchmark_id=benchmark_id,
        benchmark_version=benchmark_version,
        template_key=template_key,
        base_dir=base_dir,
        latest_only=True
    )
    
    if results:
        return results[0][1]  # 첫 번째 결과의 데이터 부분
    return []


def find_latest_file(
    model_name: str,
    benchmark_id: int,
    template_key: str,
    benchmark_version: str = "v1.0.0",
    base_dir: Path = DEFAULT_RAW_OUTPUT_DIR
) -> Optional[Path]:
    """
    특정 조건에 맞는 가장 최근 파일의 경로를 찾습니다.
    
    Args:
        model_name (str): 모델 이름
        benchmark_id (int): 벤치마크 ID
        template_key (str): 템플릿 키
        benchmark_version (str): 벤치마크 버전
        base_dir (Path): 기본 디렉토리
        
    Returns:
        Optional[Path]: 가장 최근 파일의 경로 또는 None
    """
    # 패턴 매칭으로 파일 찾기
    pattern = f"*_{model_name}/benchmark_{benchmark_id}_{benchmark_version}_{template_key}_*.json"
    matching_files = list(base_dir.glob(pattern))
    
    if matching_files:
        # 파일명의 타임스탬프 기준으로 정렬하여 가장 최근 파일 반환
        return sorted(matching_files)[-1]
    
    return None


def debug_available_files(model_name: str, base_dir: Path = DEFAULT_RAW_OUTPUT_DIR) -> None:
    """
    특정 모델의 사용 가능한 파일들을 디버깅 목적으로 출력합니다.
    
    Args:
        model_name (str): 확인할 모델 이름
        base_dir (Path): 기본 디렉토리
    """
    print(f"\n🔍 모델 '{model_name}'의 사용 가능한 파일들:")
    
    available_outputs = list_available_outputs(base_dir)
    
    if model_name not in available_outputs:
        print(f"❌ 모델 '{model_name}'에 대한 파일이 없습니다.")
        
        # 사용 가능한 모든 모델 표시
        print("\n📂 사용 가능한 모델들:")
        for available_model in available_outputs.keys():
            print(f"  - {available_model}")
        return
    
    files = available_outputs[model_name]
    print(f"✅ 총 {len(files)}개 파일 발견:")
    
    for file_path in files:
        metadata = parse_filename(file_path)
        if metadata:
            print(f"  📄 {file_path.name}")
            print(f"     └─ 벤치마크 ID: {metadata['benchmark_id']}")
            print(f"     └─ 버전: {metadata['benchmark_version']}")
            print(f"     └─ 템플릿: {metadata['template_key']}")
            print(f"     └─ 시간: {metadata['timestamp']}")
        else:
            print(f"  ❓ {file_path.name} (파싱 실패)")
        print()


# --- 실행 예시 ---
if __name__ == "__main__":
    print("📂 사용 가능한 출력 파일들:")
    available = list_available_outputs()
    
    for model_name, files in available.items():
        print(f"\n🤖 {model_name}:")
        for file_path in files[:3]:  # 처음 3개만 표시
            metadata = parse_filename(file_path)
            if metadata:
                print(f"  - {file_path.name}")
                print(f"    └─ 벤치마크 ID: {metadata['benchmark_id']}, 템플릿: {metadata['template_key']}, 시간: {metadata['timestamp']}")
        if len(files) > 3:
            print(f"  ... 총 {len(files)}개 파일")
    
    if available:
        print("\n" + "="*60)
        
        # 첫 번째 모델의 passage 데이터 로드 테스트
        first_model = list(available.keys())[0]
        print(f"🧪 '{first_model}' 모델의 passage 데이터 로드 테스트:")
        
        passages = load_passages(first_model, benchmark_id=1)
        if passages:
            print(f"✅ {len(passages)}개의 passage 로드 완료")
            print("첫 번째 passage 예시:")
            print(json.dumps(passages[0], ensure_ascii=False, indent=2)[:200] + "...")
        else:
            print("❌ passage 데이터를 찾을 수 없습니다.")

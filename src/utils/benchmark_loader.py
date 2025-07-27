# src/utils/benchmark_loader.py
"""벤치마크 데이터셋(.json)을 로드하고 관리하는 유틸리티"""

from __future__ import annotations

import functools
import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))

# --- 설정 ---
_DEFAULT_DIR = Path("src/data/benchmarks")

# --- 내부 헬퍼 ---

def _resolve_benchmark_path(file_name: Optional[str] = None) -> Path:
    """벤치마크 파일 또는 디렉토리의 절대 경로를 확인합니다."""
    repo_root = Path(__file__).resolve().parents[2]
    base_dir = repo_root / _DEFAULT_DIR
    
    if file_name:
        # 직접 경로 시도
        resolved_path = (base_dir / file_name).expanduser()
        if resolved_path.exists():
            return resolved_path
        
        # 하위 디렉토리에서 재귀적으로 검색
        for json_file in base_dir.rglob(file_name):
            if json_file.is_file():
                return json_file
    
    if base_dir.exists() and base_dir.is_dir():
        return base_dir
        
    raise FileNotFoundError(f"기본 벤치마크 디렉토리를 찾을 수 없습니다: {base_dir}")

@functools.lru_cache(maxsize=4)
def _load_json(abs_path: Path) -> List[Dict[str, Any]]:
    """지정된 경로의 JSON 파일을 로드하고 캐시에 저장합니다."""
    if not abs_path.is_file():
        raise FileNotFoundError(f"벤치마크 파일이 존재하지 않습니다: {abs_path}")
    
    try:
        with abs_path.open(encoding="utf-8") as f:
            return json.load(f) or []
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 오류: {abs_path}")
        raise e

# --- 공개 API ---

def load_benchmarks(file_name: str) -> List[Dict[str, Any]]:
    """
    지정된 벤치마크 파일을 로드합니다.

    Args:
        file_name (str): 'src/data/benchmarks/' 디렉토리 내의 .json 파일 이름.

    Returns:
        List[Dict[str, Any]]: 벤치마크 데이터 리스트.
    """
    abs_path = _resolve_benchmark_path(file_name)
    return _load_json(abs_path)

def get_benchmark_by_id(file_name: str, benchmark_id: int) -> Optional[Dict[str, Any]]:
    """
    벤치마크 파일에서 특정 ID를 가진 벤치마크 세트를 찾습니다.

    Args:
        file_name (str): 벤치마크 파일 이름.
        benchmark_id (int): 찾고자 하는 벤치마크의 ID.

    Returns:
        Optional[Dict[str, Any]]: 해당 ID의 벤치마크 딕셔너리 또는 None.
    """
    benchmarks = load_benchmarks(file_name)
    for bench in benchmarks:
        if bench.get("id") == benchmark_id:
            return bench
    return None

def list_available_benchmarks() -> List[str]:
    """
    사용 가능한 벤치마크 파일 목록을 반환합니다.
    하위 디렉토리까지 재귀적으로 검색합니다.
    """
    try:
        benchmark_dir = _resolve_benchmark_path()
        if benchmark_dir.is_dir():
            # 재귀적으로 모든 .json 파일 찾기
            json_files = []
            for json_file in benchmark_dir.rglob("*.json"):
                # 상대 경로로 저장 (benchmarks 디렉토리 기준)
                relative_path = json_file.relative_to(benchmark_dir)
                json_files.append(str(relative_path))
            return sorted(json_files)
        return []
    except FileNotFoundError:
        return []

# --- 실행 예시 ---
if __name__ == "__main__":
    print("📋 사용 가능한 벤치마크 파일:")
    print(list_available_benchmarks())
    
    print("\n" + "="*50)
    
    # 특정 벤치마크 파일 로드 및 검증
    try:
        available_files = list_available_benchmarks()
        if available_files:
            B_FILE = available_files[0]  # 첫 번째 벤치마크 파일 사용
            print(f"📂 '{B_FILE}' 로딩 테스트...")
            all_benchmarks = load_benchmarks(B_FILE)
            print(f"✅ 총 {len(all_benchmarks)}개의 벤치마크 세트 로드 완료.")
            
            # 첫 번째 벤치마크 정보 가져오기
            if all_benchmarks:
                first_bench = all_benchmarks[0]
                bench_id = first_bench.get('id', 'N/A')
                print(f"\n🔬 첫 번째 벤치마크 (ID: {bench_id}) 정보:")
                if 'problem_types' in first_bench and first_bench['problem_types']:
                    print(f"  - 문제 유형: {first_bench['problem_types'][0]}, ...")
                if 'items' in first_bench:
                    print(f"  - 아이템 개수: {len(first_bench['items'])}")
        else:
            print("❌ 사용 가능한 벤치마크 파일이 없습니다.")
    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
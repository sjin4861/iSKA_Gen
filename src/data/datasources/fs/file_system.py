from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable, Optional

def ensure_dir(p: Path) -> None:
    """디렉토리가 존재하지 않으면 생성"""
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Any:
    """JSON 파일을 읽어서 파싱된 데이터 반환. 파일이 없으면 None 반환"""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

def write_json_atomic(path: Path, data: Any) -> Path:
    """데이터를 JSON 파일로 원자적으로 저장 (임시 파일 사용)"""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return path

def list_files(root: Path, pattern: str = "*.json") -> Iterable[Path]:
    """디렉토리에서 패턴에 맞는 파일들을 재귀적으로 검색"""
    if not root.exists():
        return []
    return root.rglob(pattern)

def merge_list_by_indices(original: list, patch: dict[int, Any]) -> list:
    """
    원본 리스트에 인덱스별 패치를 적용
    
    Args:
        original: 기존 리스트 (None이면 빈 리스트로 간주)
        patch: {index: value} 형태의 패치 데이터
        
    Returns:
        패치가 적용된 리스트 (최대 index+1 까지 패딩)
    """
    if original is None:
        original = []
    
    max_len = max(len(original), (max(patch.keys()) + 1) if patch else 0)
    out = list(original) + [None] * (max_len - len(original))
    
    for idx, val in patch.items():
        if idx >= len(out):
            out += [None] * (idx - len(out) + 1)
        out[idx] = val
    
    return out

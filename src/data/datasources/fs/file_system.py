from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable, Optional

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json_atomic(path: Path, data: Any) -> Path:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return path

def list_files(root: Path, pattern: str = "*.json") -> Iterable[Path]:
    if not root.exists():
        return []
    return root.rglob(pattern)

def merge_list_by_indices(original: list, patch: dict[int, Any]) -> list:
    """
    original: 기존 리스트(길이가 맞다고 가정하지 않음. None이면 빈 리스트로 간주)
    patch: {index: value}
    반환: indices 적용된 리스트 (최대 index+1 까지 패딩)
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

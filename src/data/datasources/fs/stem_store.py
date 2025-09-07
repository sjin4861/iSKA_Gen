from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from .file_system import read_json, write_json_atomic, merge_list_by_indices
from .data_store_fs import DataStoreFSDataSource

class StemStoreFSDataSource:
    """
    stems 파일 I/O.
    DataStoreFSDataSource를 기반으로 한 레거시 호환성 래퍼
    """
    def __init__(self, data_store: Optional[DataStoreFSDataSource] = None):
        self.data_store = data_store or DataStoreFSDataSource()

    def resolve_path(self, model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str] = None) -> Path:
        """파일 경로 해석 - stem 타입으로 조정"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        # stem용 경로 생성 (passage_processed 대신 stem 사용)
        dir_path = self.data_store.raw_outputs_path / date_str / "stem" / model
        filename = self.data_store.get_file_name_pattern(benchmark_id, version, template_key)
        return dir_path / filename

    def load_list(self, model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        path = self.resolve_path(model, benchmark_id, version, template_key, date_str)
        return read_json(path)

    def save_list(self, data: List[Dict[str, Any]], model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str] = None) -> Path:
        path = self.resolve_path(model, benchmark_id, version, template_key, date_str)
        # ✅ 상위 폴더 보장
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, data)
        return path

    def patch_by_indices(self, model: str, benchmark_id: int, version: str, template_key: str, patch: Dict[int, Dict[str, Any]], date_str: Optional[str] = None) -> Path:
        orig = self.load_list(model, benchmark_id, version, template_key, date_str) or []
        merged = merge_list_by_indices(orig, patch)
        return self.save_list(merged, model, benchmark_id, version, template_key, date_str)

    def find_null_indices(self, items: List[Dict[str, Any]]) -> List[int]:
        """stem이 누락된 인덱스들을 찾음"""
        idxs: List[int] = []
        for i, row in enumerate(items or []):
            if row is None:
                idxs.append(i)
                continue
            # stem_1, stem_2, stem_3 중 하나라도 None이거나 "문항 생성 실패"이면 누락으로 간주
            has_missing_stem = False
            for j in range(1, 4):  # stem_1, stem_2, stem_3
                stem_field = f'stem_{j}'
                stem_value = row.get(stem_field)
                if stem_value is None or stem_value == "문항 생성 실패":
                    has_missing_stem = True
                    break
            if has_missing_stem:
                idxs.append(i)
        return idxs

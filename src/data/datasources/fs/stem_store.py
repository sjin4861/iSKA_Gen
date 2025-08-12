from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from .file_system import read_json, write_json_atomic, merge_list_by_indices
from ...settings import RAW_OUTPUTS_DIR, stem_file_name

class StemFSStore:
    """
    stems 파일 I/O.
    data_store/raw_outputs/{date}/stem/{model}/{template}/benchmark_{id}_{ver}_{template}.json
    """
    def _dir_for(self, date_str: Optional[str], model: str, template_key: str) -> Path:
        eff_date = date_str or datetime.now().strftime("%Y-%m-%d")
        return Path(RAW_OUTPUTS_DIR) / eff_date / "stem" / model / template_key

    def resolve_path(self, model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str] = None) -> Path:
        d = self._dir_for(date_str, model, template_key)
        return d / stem_file_name(benchmark_id, version, template_key)

    def load_list(self, model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        return read_json(self.resolve_path(model, benchmark_id, version, template_key, date_str))

    def save_list(self, data: List[Dict[str, Any]], model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str] = None) -> Path:
        return write_json_atomic(self.resolve_path(model, benchmark_id, version, template_key, date_str), data)

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

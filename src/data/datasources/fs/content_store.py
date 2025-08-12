from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from .file_system import read_json, write_json_atomic, merge_list_by_indices, ensure_dir
from ...settings import RAW_OUTPUTS_DIR, passage_file_name

class ContentFSStore:
    """
    passage(텍스트 산출물) 파일 I/O.
    data_store/raw_outputs/{date}/passage/{model}/{template}/benchmark_{id}_{ver}_{template}.json
    구조를 표준화.
    """
    def _dir_for(self, date_str: Optional[str], model: str, template_key: str) -> Path:
        eff_date = date_str or datetime.now().strftime("%Y-%m-%d")
        return Path(RAW_OUTPUTS_DIR) / eff_date / "passage" / model / template_key

    def resolve_path(
        self,
        model: str,
        benchmark_id: int,
        version: str,
        template_key: str,
        date_str: Optional[str] = None,
    ) -> Path:
        d = self._dir_for(date_str, model, template_key)
        return d / passage_file_name(benchmark_id, version, template_key)

    # -------- 읽기 --------
    def load_passage_list(
        self, model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        path = self.resolve_path(model, benchmark_id, version, template_key, date_str)
        return read_json(path)

    # -------- 쓰기(전체 덮어쓰기) --------
    def save_passage_list(
        self, data: List[Dict[str, Any]], model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str] = None
    ) -> Path:
        path = self.resolve_path(model, benchmark_id, version, template_key, date_str)
        return write_json_atomic(path, data)

    # -------- 널 인덱스 탐색 --------
    def find_null_indices(self, items: List[Dict[str, Any]]) -> List[int]:
        idxs: List[int] = []
        for i, row in enumerate(items or []):
            if row is None:
                idxs.append(i)
                continue
            if row.get("generated_passage") is None:
                idxs.append(i)
        return idxs

    # -------- 부분 업데이트(인덱스 교체) --------
    def patch_by_indices(
        self,
        model: str,
        benchmark_id: int,
        version: str,
        template_key: str,
        patch: Dict[int, Dict[str, Any]],
        date_str: Optional[str] = None,
    ) -> Path:
        path = self.resolve_path(model, benchmark_id, version, template_key, date_str)
        original = read_json(path) or []
        merged = merge_list_by_indices(original, patch)
        return write_json_atomic(path, merged)

    # -------- 빈 파일 골격 만들기(선택) --------
    def init_skeleton(
        self,
        model: str, benchmark_id: int, version: str, template_key: str,
        total_items: int, source_rows: List[Dict[str, Any]], date_str: Optional[str] = None
    ) -> Path:
        """
        benchmark items 개수와 동일한 길이의 리스트를 만들고
        {"source_item": {...}, "generated_passage": None} 형태로 초기화.
        """
        skeleton: List[Dict[str, Any]] = []
        for src in source_rows:
            skeleton.append({"source_item": src, "generated_passage": None})
        path = self.resolve_path(model, benchmark_id, version, template_key, date_str)
        ensure_dir(path.parent)
        return write_json_atomic(path, skeleton)

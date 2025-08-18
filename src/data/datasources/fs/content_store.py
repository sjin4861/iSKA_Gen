from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from .file_system import read_json, write_json_atomic, merge_list_by_indices, ensure_dir
from .data_store_fs import DataStoreFSDataSource

class ContentStoreFSDataSource:
    """
    passage(텍스트 산출물) 파일 I/O.
    DataStoreFSDataSource를 기반으로 한 레거시 호환성 래퍼
    """
    def __init__(self, data_store: Optional[DataStoreFSDataSource] = None):
        self.data_store = data_store or DataStoreFSDataSource()

    def resolve_path(
        self,
        model: str,
        benchmark_id: int,
        version: str,
        template_key: str,
        date_str: Optional[str] = None,
    ) -> Path:
        """파일 경로 해석 - DataStoreFSDataSource의 경로 생성 로직 사용"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        return self.data_store._get_passage_list_path(model, benchmark_id, version, template_key, date_str)

    # -------- 읽기 --------
    def load_passage_list(
        self, model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        return self.data_store.load_passage_list(model, benchmark_id, version, template_key, date_str)

    # -------- 쓰기(전체 덮어쓰기) --------
    def save_passage_list(
        self, data: List[Dict[str, Any]], model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str] = None
    ) -> Path:
        self.data_store.save_passage_list(data, model, benchmark_id, version, template_key, date_str)
        return self.resolve_path(model, benchmark_id, version, template_key, date_str)

    # -------- 널 인덱스 탐색 --------
    def find_null_indices(self, items: List[Dict[str, Any]]) -> List[int]:
        return self.data_store.find_null_indices(items)

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
        self.data_store.patch_by_indices(model, benchmark_id, version, template_key, patch, date_str)
        return self.resolve_path(model, benchmark_id, version, template_key, date_str)

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
        
        self.data_store.save_passage_list(skeleton, model, benchmark_id, version, template_key, date_str)
        return self.resolve_path(model, benchmark_id, version, template_key, date_str)

from __future__ import annotations
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class ContentRepository(ABC):
    """passage 산출물의 영속 계층 접근 (파일/DB 등 구현체 분리)"""

    @abstractmethod
    def load_passage_rows(
        self, *, model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        ...

    @abstractmethod
    def save_passage_rows(
        self, *, data: List[Dict[str, Any]], model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str]
    ) -> None:
        ...

    @abstractmethod
    def patch_passage_rows_by_indices(
        self, *, model: str, benchmark_id: int, version: str, template_key: str,
        patch: Dict[int, Dict[str, Any]], date_str: Optional[str]
    ) -> None:
        ...

    @abstractmethod
    def find_null_indices(self, items: List[Dict[str, Any]]) -> List[int]:
        ...

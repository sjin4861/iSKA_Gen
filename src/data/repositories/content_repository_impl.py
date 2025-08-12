from __future__ import annotations
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.data.datasources.fs.content_store import ContentFSStore
from src.domain.repositories.content_repository import ContentRepository

class ContentRepositoryImpl(ContentRepository):
    def __init__(self, store: Optional[ContentFSStore] = None) -> None:
        self.store = store or ContentFSStore()

    def load_passage_rows(
        self, *, model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str]
    ) -> Optional[List[Dict[str, Any]]]:
        return self.store.load_passage_list(model, benchmark_id, version, template_key, date_str)

    def save_passage_rows(
        self, *, data: List[Dict[str, Any]], model: str, benchmark_id: int, version: str, template_key: str, date_str: Optional[str]
    ) -> None:
        self.store.save_passage_list(data, model, benchmark_id, version, template_key, date_str)

    def patch_passage_rows_by_indices(
        self, *, model: str, benchmark_id: int, version: str, template_key: str,
        patch: Dict[int, Dict[str, Any]], date_str: Optional[str]
    ) -> None:
        self.store.patch_by_indices(model, benchmark_id, version, template_key, patch, date_str)

    def find_null_indices(self, items: List[Dict[str, Any]]) -> List[int]:
        return self.store.find_null_indices(items)

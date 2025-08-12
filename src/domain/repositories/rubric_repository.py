from __future__ import annotations
from typing import Protocol
from src.domain.entities.rubrics import RubricCatalog

class RubricRepository(Protocol):
    """루브릭 카탈로그 로드/저장(필요 시)"""
    def load_catalog(self) -> RubricCatalog: ...

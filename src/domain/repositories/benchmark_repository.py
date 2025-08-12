from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, List

from src.domain.entities.benchmark import BenchmarkSet, BenchmarkItemFlat

class BenchmarkRepository(ABC):
    @abstractmethod
    def get_set_by_id(self, set_id: int) -> BenchmarkSet: ...
    # 평탄화된 아이템 스트림(topic/context 기준; foreign_*는 선택)
    @abstractmethod
    def iter_items(self, set_id: int) -> Iterable[BenchmarkItemFlat]: ...

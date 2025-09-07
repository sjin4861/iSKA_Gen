# src/domain/usecases/benchmark/count_flat_items.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from src.domain.repositories.benchmark_repository import BenchmarkRepository, ItemFilter
from .iter_flat_items import IterFlatItemsUseCase, IterFlatItemsInput

@dataclass(frozen=True)
class CountFlatItemsInput:
    set_id: int
    flt: Optional[ItemFilter] = None

@dataclass(frozen=True)
class CountFlatItemsOutput:
    count: int

class CountFlatItemsUseCase:
    """UC-11(편의): 스트림 소비 비용을 감수하고 개수 산출 (리포가 카운터 제공 시 override 가능)"""
    def __init__(self, repo: BenchmarkRepository):
        self.repo = repo
        self._iter_uc = IterFlatItemsUseCase(repo)

    def execute(self, inp: CountFlatItemsInput) -> CountFlatItemsOutput:
        # 리포가 count_items 제공하면 우선 사용
        if hasattr(self.repo, "count_items"):
            try:
                cnt = self.repo.count_items(inp.set_id, flt=inp.flt)  # type: ignore[arg-type]
                return CountFlatItemsOutput(count=cnt)
            except TypeError:
                pass  # 레거시 시그니처면 폴백

        # 폴백: 스트림 순회 카운트
        it = self._iter_uc.execute(IterFlatItemsInput(set_id=inp.set_id, flt=inp.flt)).items
        cnt = sum(1 for _ in it)
        return CountFlatItemsOutput(count=cnt)

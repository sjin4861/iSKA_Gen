# src/domain/usecases/benchmark/iter_flat_items.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from src.domain.repositories.benchmark_repository import BenchmarkRepository, ItemFilter
from src.domain.entities.benchmark import BenchmarkItemFlat

@dataclass(frozen=True)
class IterFlatItemsInput:
    set_id: int
    # 신규 리포와 정합: ItemFilter 그대로 사용
    flt: Optional[ItemFilter] = None

@dataclass(frozen=True)
class IterFlatItemsOutput:
    items: Iterable[BenchmarkItemFlat]  # generator/iterable

class IterFlatItemsUseCase:
    """
    UC-09/10: 평탄화 아이템을 **스트리밍**으로 제공.
    - 신규 리포( iter_items(..., flt=ItemFilter) )이면 그대로 위임
    - 레거시 리포( iter_items(set_id)->List[...] )이면 유스케이스에서 필터/슬라이스 적용
    """
    def __init__(self, repo: BenchmarkRepository):
        self.repo = repo

    def execute(self, inp: IterFlatItemsInput) -> IterFlatItemsOutput:
        # 시그니처가 flt를 받으면 그대로 사용
        try:
            items = self.repo.iter_items(inp.set_id, flt=inp.flt)  # type: ignore[arg-type]
            return IterFlatItemsOutput(items=items)
        except TypeError:
            # 레거시 구현: iter_items(set_id) -> List[BenchmarkItemFlat]
            legacy_items = self.repo.iter_items(inp.set_id)  # type: ignore[call-arg]
            items = self._apply_filter_stream(legacy_items, inp.flt)
            return IterFlatItemsOutput(items=items)

    # --- 내부: 레거시용 필터/슬라이스 스트리밍 ---
    def _apply_filter_stream(
        self,
        base: Iterable[BenchmarkItemFlat],
        flt: Optional[ItemFilter],
    ) -> Iterable[BenchmarkItemFlat]:
        kind = (flt.kind if flt else "any")
        offset = (flt.offset if flt else 0) or 0
        limit = (flt.limit if flt else None)

        def is_compare_flat(x: BenchmarkItemFlat) -> bool:
            # 플랫된 비교형은 foreign_* 중 하나라도 값이 존재한다고 가정
            return (x.foreign_topic is not None) or (x.foreign_context is not None)

        def gen() -> Iterator[BenchmarkItemFlat]:
            skipped = 0
            emitted = 0
            for x in base:
                if kind != "any":
                    if kind == "compare" and not is_compare_flat(x):
                        continue
                    if kind == "single" and is_compare_flat(x):
                        continue

                if skipped < offset:
                    skipped += 1
                    continue

                yield x
                emitted += 1
                if limit is not None and emitted >= limit:
                    break

        return gen()

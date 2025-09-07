from __future__ import annotations

from curses import raw
import json
from pathlib import Path
from typing import List

from pydantic import TypeAdapter

from src.domain.repositories.benchmark_repository import BenchmarkRepository
from src.domain.entities.benchmark import (
    BenchmarkSet,
    BenchmarkCollection,
    BenchmarkItemFlat,
)


class BenchmarkRepositoryImpl(BenchmarkRepository):
    """
    파일 시스템(JSON)에서 벤치마크를 읽어오는 구현체.

    - benchmarks_root: 벤치마크 JSON 디렉터리 (예: data_store/benchmarks/v1)
    - benchmark_filename: 파일명 (예: iSKA-Gen_Benchmark_v1.1.0_20250808.json)
    """

    def __init__(self, benchmarks_root: Path, benchmark_filename: str):
        self.benchmarks_path = (benchmarks_root / benchmark_filename).resolve()
        if not self.benchmarks_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {self.benchmarks_path}")

    def load_collection(self) -> BenchmarkCollection:
        raw = json.loads(self.benchmarks_path.read_text(encoding="utf-8"))
        return BenchmarkCollection.model_validate(raw)
    
    def get_set_by_id(self, bench_id: int) -> BenchmarkSet:
        coll = self.load_collection()
        for s in coll.benchmarks:
            if s.id == bench_id:
                return s
        raise ValueError(
            f"BenchmarkSet id={bench_id} not found in {self.benchmarks_path.name}"
        )

    def list_items_as_flat(self, bench_id: int) -> List[BenchmarkItemFlat]:
        """
        단일/복합 아이템을 평탄화해서 반환.
        - 비교형: korean_* 기준으로 topic/context, 외국어는 foreign_* 채움
        - 단일형: topic/context 그대로 사용
        """
        s = self.get_set_by_id(bench_id)
        out: List[BenchmarkItemFlat] = []

        for it in s.items:
            if it.korean_topic is not None or it.korean_context is not None:
                out.append(
                    BenchmarkItemFlat(
                        topic=it.korean_topic or "",
                        context=it.korean_context or "",
                        foreign_topic=it.foreign_topic,
                        foreign_context=it.foreign_context,
                    )
                )
            else:
                out.append(
                    BenchmarkItemFlat(
                        topic=it.topic or "",
                        context=it.context or "",
                        foreign_topic=None,
                        foreign_context=None,
                    )
                )
        return out

    def iter_items(self, set_id: int) -> List[BenchmarkItemFlat]:
        """
        평탄화된 아이템 스트림 반환 (추상 메서드 구현)
        """
        return self.list_items_as_flat(set_id)

# src/domain/usecases/benchmark/get_set_by_id.py
from __future__ import annotations
from dataclasses import dataclass
from src.domain.repositories.benchmark_repository import BenchmarkRepository
from src.domain.entities.benchmark import BenchmarkSet

@dataclass(frozen=True)
class GetSetByIdInput:
    set_id: int

@dataclass(frozen=True)
class GetSetByIdOutput:
    benchmark_set: BenchmarkSet

class GetSetByIdUseCase:
    """UC-07: 세트 단위 조회"""
    def __init__(self, repo: BenchmarkRepository):
        self.repo = repo

    def execute(self, inp: GetSetByIdInput) -> GetSetByIdOutput:
        s = self.repo.get_set_by_id(inp.set_id)
        return GetSetByIdOutput(benchmark_set=s)

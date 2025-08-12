from __future__ import annotations
from dataclasses import dataclass
from src.domain.repositories.benchmark_repository import BenchmarkRepository
from src.domain.entities.benchmark import BenchmarkSet

@dataclass(frozen=True)
class GetBenchmarkSetByIdInput:
    benchmark_id: int

@dataclass(frozen=True)
class GetBenchmarkSetByIdOutput:
    benchmark_set: BenchmarkSet

class GetBenchmarkSetByIdUseCase:
    def __init__(self, repo: BenchmarkRepository) -> None:
        self.repo = repo

    def execute(self, i: GetBenchmarkSetByIdInput) -> GetBenchmarkSetByIdOutput:
        s = self.repo.get_set_by_id(i.benchmark_id)
        return GetBenchmarkSetByIdOutput(benchmark_set=s)

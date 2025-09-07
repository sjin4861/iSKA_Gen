# src/domain/usecases/benchmark/load_collection.py
from __future__ import annotations
from dataclasses import dataclass
from src.domain.repositories.benchmark_repository import BenchmarkRepository
from src.domain.entities.benchmark import BenchmarkCollection

@dataclass(frozen=True)
class LoadCollectionOutput:
    collection: BenchmarkCollection

class LoadCollectionUseCase:
    """UC-06: 전체 컬렉션 로드/검증"""
    def __init__(self, repo: BenchmarkRepository):
        self.repo = repo

    def execute(self) -> LoadCollectionOutput:
        coll = self.repo.load_collection()
        return LoadCollectionOutput(collection=coll)

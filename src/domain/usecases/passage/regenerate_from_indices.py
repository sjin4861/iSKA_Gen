# src/domain/usecases/passage/regenerate_from_indices.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from ...repositories.passage_repository import PassageRepository

@dataclass
class RegenerateFromIndicesInput:
    benchmark_file: str
    benchmark_id: int
    indices: List[int]
    template_key: str

@dataclass
class RegenerateFromIndicesOutput:
    regenerated: Dict[int, Dict[str, Any]]

class RegenerateFromIndicesUseCase:
    def __init__(self, repo: PassageRepository):
        self.repo = repo

    def execute(self, inp: RegenerateFromIndicesInput) -> RegenerateFromIndicesOutput:
        result = self.repo.regenerate_from_benchmark_indices(
            benchmark_file=inp.benchmark_file,
            benchmark_id=inp.benchmark_id,
            indices=inp.indices,
            template_key=inp.template_key,
        )
        return RegenerateFromIndicesOutput(regenerated=result)

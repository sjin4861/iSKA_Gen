from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from src.domain.repositories.passage_repository import PassageRepository

@dataclass(frozen=True)
class FillNullPassagesInput:
    model_name: str
    template_key: str
    benchmark_id: int
    benchmark_version: str
    problem_types: List[str]
    eval_goals: List[str]
    sources: List[Dict[str, Any]]
    date_str: Optional[str] = None
    min_length: int = 300
    max_length: int = 800
    max_retries: int = 10

class FillNullPassagesUseCase:
    def __init__(self, repo: PassageRepository) -> None:
        self.repo = repo

    def execute(self, i: FillNullPassagesInput) -> dict:
        return self.repo.generate_and_fill_missing(
            model_name=i.model_name,
            template_key=i.template_key,
            benchmark_id=i.benchmark_id,
            benchmark_version=i.benchmark_version,
            problem_types=i.problem_types,
            eval_goals=i.eval_goals,
            sources=i.sources,
            date_str=i.date_str,
            min_length=i.min_length,
            max_length=i.max_length,
            max_retries=i.max_retries,
        )

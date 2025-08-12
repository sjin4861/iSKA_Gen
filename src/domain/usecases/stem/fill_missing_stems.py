# src/domain/usecases/stem/fill_missing_stems.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.domain.repositories.stem_repository import StemRepository


@dataclass
class FillMissingStemsInput:
    """누락된 stem 채우기 입력"""
    model_name: str
    template_key: str
    benchmark_id: int
    benchmark_version: str
    problem_types: List[str]
    eval_goals: List[str]
    passages: List[Dict[str, Any]]
    date_str: Optional[str] = None
    max_retries: int = 3
    passage_model_name: Optional[str] = None


@dataclass
class FillMissingStemsOutput:
    """누락된 stem 채우기 출력"""
    filled_indices: List[int]
    failed_indices: List[int]
    total_after: int
    success: bool


class FillMissingStemsUseCase:
    """누락된 stem 채우기 유스케이스"""
    
    def __init__(self, stem_repository: StemRepository):
        self.stem_repository = stem_repository
    
    def execute(self, input_data: FillMissingStemsInput) -> FillMissingStemsOutput:
        """누락된 stem 채우기 실행"""
        result = self.stem_repository.generate_and_fill_missing(
            model_name=input_data.model_name,
            template_key=input_data.template_key,
            benchmark_id=input_data.benchmark_id,
            benchmark_version=input_data.benchmark_version,
            problem_types=input_data.problem_types,
            eval_goals=input_data.eval_goals,
            passages=input_data.passages,
            date_str=input_data.date_str,
            max_retries=input_data.max_retries,
            passage_model_name=input_data.passage_model_name,
        )
        
        return FillMissingStemsOutput(
            filled_indices=result["filled"],
            failed_indices=result["failed"],
            total_after=result["total"],
            success=len(result["failed"]) == 0
        )

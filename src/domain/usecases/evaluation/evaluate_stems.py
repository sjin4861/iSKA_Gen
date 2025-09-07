# src/domain/usecases/evaluation/evaluate_stems.py
from __future__ import annotations
from typing import List
from dataclasses import dataclass

from src.domain.repositories.evaluation_repository import EvaluationRepository
from src.domain.entities.outputs import CandidateOutput
from src.domain.entities.rubrics import RubricID
from src.domain.entities.enums import EvaluatorType
from src.domain.usecases.evaluation.evaluate import EvaluateOutput


@dataclass
class EvaluateStemsInput:
    """Stem 평가 입력"""
    stem_candidates: List[CandidateOutput]
    evaluator_model: str
    rubric_ids: List[RubricID]
    run_id: str
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class EvaluateStemsOutput:
    """Stem 평가 출력"""
    evaluations: List[EvaluateOutput]
    total_success: int
    total_failed: int
    total_count: int


class EvaluateStemsUseCase:
    """Stem 평가 유스케이스"""
    
    def __init__(self, evaluation_repo: EvaluationRepository):
        self.evaluation_repo = evaluation_repo
    
    def execute(self, inp: EvaluateStemsInput) -> EvaluateStemsOutput:
        """
        여러 루브릭에 대해 stem 평가를 수행
        """
        # Repository에 위임
        return self.evaluation_repo.evaluate_stems(inp)
    
    def execute_with_shared_client(self, inp: EvaluateStemsInput, shared_client) -> EvaluateStemsOutput:
        """
        공유 클라이언트를 사용하여 여러 루브릭에 대해 stem 평가를 수행 (CUDA 재초기화 방지)
        """
        # Repository에 위임
        return self.evaluation_repo.evaluate_stems_with_shared_client(inp, shared_client)

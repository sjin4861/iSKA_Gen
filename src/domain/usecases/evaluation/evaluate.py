# src/domain/usecases/evaluate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict
from src.domain.entities.base import DomainModel
from src.domain.entities.evaluation import EvaluationRecord
from src.domain.entities.outputs import CandidateOutput
from src.domain.entities.enums import EvalMethod, EvaluatorType
from src.domain.entities.rubrics import RubricID
from src.domain.repositories.evaluation_repository import EvaluationRepository


class EvaluateOptions(DomainModel):
    """
    채점 정책/옵션.
    - rubrics를 지정하지 않으면, 구현체에서 kind별 카탈로그 전체를 사용.
    - method_override가 있으면 해당 루브릭의 기본 방식 대신 강제.
    """
    rubrics: Optional[List[RubricID]] = None
    method_override: Optional[Dict[RubricID, EvalMethod]] = None
    evaluated_by: EvaluatorType = EvaluatorType.LLM
    run_id: Optional[str] = None


@dataclass(frozen=True)
class EvaluateInput:
    """평가 입력 데이터"""
    candidates: List[CandidateOutput]
    rubric_id: Optional[RubricID] = None
    evaluator_type: EvaluatorType = EvaluatorType.LLM
    model_name: str = "gpt-4o-mini"
    run_id: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    options: Optional[EvaluateOptions] = None


@dataclass(frozen=True)
class EvaluateOutput:
    """평가 결과"""
    records: List[EvaluationRecord]
    success_count: int = 0
    failed_count: int = 0
    total_count: int = 0

    def __post_init__(self):
        # 카운트가 설정되지 않은 경우 자동 계산
        if self.success_count == 0 and self.failed_count == 0 and self.total_count == 0:
            total = len(self.records)
            success = sum(1 for r in self.records if r.score > 0)
            failed = total - success
            object.__setattr__(self, 'total_count', total)
            object.__setattr__(self, 'success_count', success)
            object.__setattr__(self, 'failed_count', failed)

class EvaluateUseCase:
    """
    도메인 유스케이스 '채점'의 추상 경계.
    구현체는 data 계층에서 레포/판정(LLM) 서비스를 주입받아 동작한다.
    """
    # 🔧 변경: 생성자에서 EvaluationRepository를 '필수' 주입
    def __init__(self, repo: EvaluationRepository) -> None:
        self.repo = repo

    def execute(self, inp: EvaluateInput) -> EvaluateOutput:
        """
        - 입력: 이미 상위 스크립트에서 레포를 통해 모아둔 후보들(유형별 리스트)
        - 동작: 루브릭/방식에 맞춰 채점, EvaluationRepository에 저장
        - 출력: 요약 통계
        """
        output = self.repo.evaluate(inp)
        return output

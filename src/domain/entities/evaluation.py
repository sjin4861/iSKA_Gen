from __future__ import annotations
from typing import Optional, List, Literal
from datetime import datetime, timezone
from pydantic import Field, model_validator
from src.domain.entities.base import DomainModel
from src.domain.entities.contents import ContentType
from src.domain.entities.rubrics import RubricID
from src.domain.entities.enums import EvalMethod, EvaluatorType

# ---- 점수 타입(Discriminated Union) ----
class BinaryScore(DomainModel):
    method: Literal[EvalMethod.BINARY] = EvalMethod.BINARY
    value: bool

class LikertScore(DomainModel):
    method: Literal[EvalMethod.LIKERT] = EvalMethod.LIKERT
    value: int = Field(..., ge=1, le=5)

class PreferenceScore(DomainModel):
    method: Literal[EvalMethod.PREFERENCE] = EvalMethod.PREFERENCE
    value: float = Field(..., ge=-5.0, le=5.0)

class JudgeMeta(DomainModel):
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    prompt_id: Optional[str] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="판정 시각(UTC, timezone-aware)"
    )

# ---- 평가 대상(자료 혹은 자료+지시문) ----
class EvaluationTarget(DomainModel):
    """
    평가 대상 텍스트 표준화:
    - content_type: passage | audio_script | image_caption
    - content: 지문/대본/캡션 본문
    - stems: (선택) 지시문/문항 — 'content+stems' 평가에 사용
    """
    content_type: ContentType = Field(..., description="자료 종류")
    content: str = Field(..., description="원본 텍스트(지문/대본/캡션)")
    stems: Optional[List[str]] = Field(None, description="지시문(선택)")

class EvaluationRecord(DomainModel):
    """단일 루브릭에 대한 한 번의 평가 결과"""
    target: EvaluationTarget
    rubric_id: RubricID
    # 점수는 Score 모델들의 Union이며 'method'로 분기
    score: BinaryScore | LikertScore | PreferenceScore = Field(discriminator="method")
    evaluated_by: EvaluatorType
    judge_meta: Optional[JudgeMeta] = None  # ✅ 요청대로 created_at은 judge_meta 안에 둠
    notes: Optional[str] = None
    run_id: Optional[str] = Field(None, description="실험 실행 식별자(재현성)")

    @model_validator(mode="after")
    def _require_stems_for_scope(self):
        """
        R1, R6은 content+stems 기준이므로 stems가 비어 있으면 에러.
        (정교한 허용-방법 검증은 서비스 레이어에서 카탈로그와 교차검증 권장)
        """
        if self.rubric_id in {RubricID.R1_GUIDELINE_COMPLETENESS, RubricID.R6_L2_APPROPRIATENESS}:
            if not self.target.stems or len(self.target.stems) == 0:
                raise ValueError("R1/R6 평가는 content+stems 기준입니다. stems가 필요합니다.")
        return self
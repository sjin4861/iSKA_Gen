from __future__ import annotations
from typing import Optional, List, Literal
from enum import Enum
from pydantic import Field
from .base import DomainModel
from .content_types import ArtifactKind
from .rubrics import RubricID, EvalMethod

class RaterKind(str, Enum):
    HUMAN_EXPERT = "human_expert"
    GPT = "gpt"
    REWARD_MODEL = "reward_model"

# ---- 점수 타입 ----
class BinaryScore(DomainModel):
    method: Literal[EvalMethod.BINARY] = EvalMethod.BINARY
    value: bool

class LikertScore(DomainModel):
    method: Literal[EvalMethod.LIKERT] = EvalMethod.LIKERT
    value: int = Field(..., ge=1, le=5)

class PreferenceSide(str, Enum):
    CHOSEN = "chosen"
    REJECTED = "rejected"

class PreferenceScore(DomainModel):
    method: Literal[EvalMethod.PREFERENCE] = EvalMethod.PREFERENCE
    preferred: PreferenceSide

Score = BinaryScore | LikertScore | PreferenceScore

# ---- 평가 대상(자료 혹은 자료+지시문) ----
class EvaluationTarget(DomainModel):
    """
    평가 대상 텍스트를 최소 표준 형태로 통일:
    - artifact_kind: passage | audio_script | image_caption
    - content: 지문/대본/캡션 본문
    - instruction: (선택) 지시문/문항(stem) — '자료+지시문' 평가에 사용
    """
    artifact_kind: ArtifactKind
    content: str = Field(..., description="원본 텍스트(지문/대본/캡션)")
    instruction: Optional[str] = Field(None, description="지시문(선택)")

class EvaluationRecord(DomainModel):
    """단일 루브릭에 대한 한 번의 평가 결과"""
    target: EvaluationTarget
    rubric_id: RubricID
    score: Score
    rater: RaterKind
    rater_model_name: Optional[str] = Field(None, description="GPT/RM 모델명(해당 시)")
    notes: Optional[str] = None
    run_id: Optional[str] = Field(None, description="실험 실행 식별자(재현성)")

class RankingMetrics(DomainModel):
    """최종 평가 지표"""
    recall_at_25: Optional[float] = Field(None, ge=0.0, le=1.0)
    rbo: Optional[float] = Field(None, ge=0.0, le=1.0)

class SelectionResult(DomainModel):
    """RM(혹은 GPT/전문가)이 선택한 상위 N개와 지표"""
    artifact_kind: ArtifactKind
    k: int = 25
    selected_ids: List[str] = Field(..., description="선정 대상의 식별자 목록(외부 ID)")
    metrics: Optional[RankingMetrics] = None

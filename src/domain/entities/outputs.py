from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import Field

from src.domain.entities.sources import SourceItem
from .base import DomainModel
from .enums import ContentType

class CandidateOutput(DomainModel):
    """
    채점 대상으로 쓰일 '후보 한 개'의 최소 단위(도메인 VO).
    구현 저장소(JSONL/CSV/DB 등)와 무관하게 공통 필드만 규정.
    """
    source_id: str = Field(..., description="동일 소스(자료) 묶음 식별자")
    benchmark_id: int = Field(..., description="벤치마크 ID")
    model_name: str = Field(..., description="생성 모델 식별")
    candidate_id: str = Field(..., description="후보 식별자(예: source_id:model:sample)")
    content_type: ContentType = Field(..., description="passage | audio_script | image_caption")
    content: str = Field(..., description="후보 본문(모델 생성 결과)")
    stems: Optional[List[str]] = Field(None, description="지시문(선택) — content+stems 평가용")
    generated_at: Optional[datetime] = Field(None, description="생성 시각(있으면 필터링에 사용)")
    meta: Optional[Dict[str, Any]] = None
    # ★ 추가: 원자료(구조화)
    source_item: Optional[Union[SourceItem, Dict[str, Any]]] = Field(
        None, description="생성에 사용된 원 소스(구조화). 없으면 평가시에 복원."
    )

    def get(self, field: str) -> Any:
        return getattr(self, field, None)
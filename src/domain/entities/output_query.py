from __future__ import annotations
from typing import Optional, List
from datetime import datetime
from pydantic import Field
from src.domain.entities.base import DomainModel

class OutputQuery(DomainModel):
    """
    원천 후보 조회 조건(날짜/모델/벤치마크/선택 소스).
    구현체(data 계층)에서 이 조건을 해석해 데이터를 가져온다.
    """
    benchmark_id: Optional[int] = Field(None)
    model_name: Optional[str] = Field(None)
    date_from: Optional[datetime] = Field(None)
    date_to: Optional[datetime] = Field(None)
    source_ids: Optional[List[str]] = Field(None, description="특정 소스만 가져오고 싶을 때")
    limit: Optional[int] = Field(None)

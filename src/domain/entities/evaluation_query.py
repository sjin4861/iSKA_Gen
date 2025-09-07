from __future__ import annotations
from typing import Optional, List
from datetime import datetime, date, timezone
from pydantic import Field, ConfigDict, field_validator, model_validator

from .base import DomainModel
from .enums import ContentType, EvaluatorType
from .rubrics import RubricID


class EvaluationQuery(DomainModel):
    """
    Evaluation CRUD에 사용할 도메인-무관 쿼리 객체.
    - 저장 매체/경로와 무관하게 필터 조건만 표현한다.
    - data 계층 구현(예: FS, DB)이 이 쿼리를 해석해 실제 검색을 수행한다.
    """
    model_config = ConfigDict(extra="forbid")

    rubric_ids: Optional[List[RubricID]] = None
    content_types: Optional[List[ContentType]] = None
    evaluated_by: Optional[EvaluatorType] = None
    run_ids: Optional[List[str]] = None
    model_names: Optional[List[str]] = None          # judge_meta.model_name 필터
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: Optional[int] = Field(None, gt=0, description="최대 반환 개수(없으면 무제한)")

    # ---------- Validators ----------

    @field_validator("run_ids", "model_names", "rubric_ids", "content_types", mode="after")
    @classmethod
    def _dedup_preserve_order(cls, v):
        """리스트 중복 제거(순서 보존). None은 통과."""
        if v is None:
            return None
        seen = set()
        out = []
        for item in v:
            key = getattr(item, "value", item)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    @field_validator("date_from", "date_to", mode="before")
    @classmethod
    def _coerce_datetime(cls, v):
        """str/date → tz-aware datetime(UTC)로 정규화."""
        if v is None:
            return None
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        if isinstance(v, date):
            return datetime(v.year, v.month, v.day, tzinfo=timezone.utc)
        if isinstance(v, str):
            # ISO8601 또는 'YYYY-MM-DD' 지원
            try:
                if "T" in v:
                    dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                else:
                    dt = datetime.strptime(v, "%Y-%m-%d")
            except Exception as e:
                raise ValueError(f"Invalid datetime string: {v}") from e
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        raise TypeError(f"Unsupported type for datetime: {type(v)}")

    @model_validator(mode="after")
    def _check_date_range(self):
        """date_from <= date_to 보장."""
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from은 date_to보다 늦을 수 없습니다.")
        return self

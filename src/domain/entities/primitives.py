from __future__ import annotations
from typing import Optional
from pydantic import Field, field_validator
from typing_extensions import Protocol
from .base import DomainModel

class CultureText(DomainModel):
    """단일 언어 지문 블록"""
    topic: str = Field(..., description="지문 소재")   # 필요시: max_length=200 등
    context: str = Field(..., description="지문 맥락") # 필요시: max_length=5000 등

    @field_validator("topic", "context")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("빈 문자열은 허용되지 않습니다.")
        return v

class KoreanForeignPair(DomainModel):
    """복합(비교) 지문: 한국어(필수) + 외국어(선택)"""
    korean: CultureText = Field(..., description="한국어 자료")
    foreign: Optional[CultureText] = Field(None, description="외국어 자료(없을 수 있음)")

class HasCultureText(Protocol):
    def to_culture_text(self) -> "CultureText": ...

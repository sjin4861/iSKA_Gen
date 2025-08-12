from __future__ import annotations
from typing import Optional, Literal, Union
from pydantic import BaseModel, Field, ConfigDict
from .primitives import CultureText, HasCultureText, KoreanForeignPair

class SourceBase(BaseModel):
    """원본 입력 공통 베이스"""
    source_kind: str = Field(..., description="소스 형태 식별자")
    model_config = ConfigDict(extra="forbid")

# -------- passage용 입력 소스 --------
class PassageSingleSource(SourceBase):
    """단일 소재 지문 입력: topic+context 한 벌"""
    source_kind: Literal["passage_single"] = "passage_single"
    topic: str
    context: str

    def as_culture_text(self) -> CultureText:
        return CultureText(topic=self.topic, context=self.context)

class PassageMultiSource(SourceBase):
    """복합 소재(비교) 지문 입력: 한국어 필수, 외국어 선택"""
    source_kind: Literal["passage_multi"] = "passage_multi"
    korean_topic: str
    korean_context: str
    foreign_topic: Optional[str] = None
    foreign_context: Optional[str] = None

    def as_pair(self) -> KoreanForeignPair:
        return KoreanForeignPair(
            korean=CultureText(topic=self.korean_topic, context=self.korean_context),
            foreign=(CultureText(topic=self.foreign_topic, context=self.foreign_context)
                     if self.foreign_topic and self.foreign_context else None)
        )

# -------- audio_script용 입력 소스 --------
class AudioScriptSource(SourceBase):
    """오디오 대본 입력: 보통 하나의 주제/설명 본문에서 대본을 생성"""
    source_kind: Literal["audio_script"] = "audio_script"
    topic: str
    context: str

    def to_culture_text(self) -> CultureText:
        return CultureText(topic=self.topic, context=self.context)

# -------- image_caption용 입력 소스 --------
class ImageCaptionSource(SourceBase):
    """이미지 캡션/상황: 최소 topic만 제공되는 경우가 있음"""
    source_kind: Literal["image_caption"] = "image_caption"
    topic: str
    # 필요 시 참고 설명을 넣고 싶다면:
    hint: Optional[str] = Field(None, description="이미지 맥락/힌트(선택)")

# 모든 source_item 유니온
SourceItem = Union[
    PassageSingleSource,
    PassageMultiSource,
    AudioScriptSource,
    ImageCaptionSource,
]

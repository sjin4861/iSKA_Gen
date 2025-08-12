from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import Field, ConfigDict
from src.domain.entities.base import DomainModel

from .content_types import ArtifactKind  # passage | audio_script | image_caption

class StemPrompt(DomainModel):
    """단일 출제 프롬프트 묶음"""
    problem_type: str = Field(..., description="문제 유형(자유 텍스트)")
    eval_goal: str = Field(..., description="평가 목표(자유 텍스트)")
    stem: str = Field(..., description="출제 문장(지시문)")

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

class StemBundle(DomainModel):
    """
    하나의 원본(지문/오디오/이미지설명)에서 파생된 stem 묶음.
    레거시 JSON의 source_passage / source_audio_script / source_image_caption 중 하나를 source_text로 통합.
    """
    source_kind: Literal[
        ArtifactKind.passage,
        ArtifactKind.audio_script,
        ArtifactKind.image_caption
    ] = Field(..., description="원본 산출물 종류")
    source_text: str = Field(..., description="원본 텍스트(지문/대본/이미지 설명)")
    prompts: List[StemPrompt] = Field(..., description="출제 프롬프트 목록(1..N)")
    benchmark_id: Optional[int] = Field(None, description="옵션: 추적을 위한 벤치마크 ID")
    benchmark_version: Optional[str] = Field(None, description="옵션: 벤치마크 버전 태그")
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

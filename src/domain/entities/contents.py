# src/domain/entities/contents.py
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Literal, Union
from src.domain.entities.base import DomainModel
from pydantic import Field, ConfigDict
from src.domain.entities.sources import SourceItem
from src.domain.entities.enums import ContentType


# -------- 산출물 결과 모델들 --------

class PassageResult(DomainModel):
    """지문 결과 (단일/복합 모두 이 타입으로 수용)"""
    kind: Literal[ContentType.passage] = ContentType.passage
    source_item: SourceItem = Field(..., description="입력 소스")
    passage: str = Field(..., description="생성된 지문(텍스트)")

class AudioScriptTurn(DomainModel):
    """대본을 구조화하고 싶을 때(선택)"""
    speaker: str = Field(..., description="화자 라벨")
    utterance: str = Field(..., description="발화 텍스트")
    # 필요 시 timestamp, sfx 등 확장 가능

class AudioScriptResult(DomainModel):
    """오디오 대본 결과"""
    kind: Literal[ContentType.audio_script] = ContentType.audio_script
    source_item: SourceItem = Field(..., description="입력 소스")
    audio_script: str = Field(..., description="대본 원문(플랫 텍스트)")
    # 선택: 구조화 버전 병행 저장
    turns: Optional[list[AudioScriptTurn]] = Field(None, description="대본의 구조화 버전(선택)")

class ImageCaptionResult(DomainModel):
    """이미지 캡션 결과"""
    kind: Literal[ContentType.image_caption] = ContentType.image_caption
    source_item: SourceItem = Field(..., description="입력 소스")
    image_caption: str = Field(..., description="이미지 설명(캡션)")
    situation: Optional[str] = Field(None, description="상황/문제 서술(선택)")

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
        ContentType.passage,
        ContentType.audio_script,
        ContentType.image_caption
    ] = Field(..., description="원본 산출물 종류")
    source_text: str = Field(..., description="원본 텍스트(지문/대본/이미지 설명)")
    prompts: List[StemPrompt] = Field(..., description="출제 프롬프트 목록(1..N)")
    benchmark_id: Optional[int] = Field(None, description="옵션: 추적을 위한 벤치마크 ID")
    benchmark_version: Optional[str] = Field(None, description="옵션: 벤치마크 버전 태그")
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)



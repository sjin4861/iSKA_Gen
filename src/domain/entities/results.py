from __future__ import annotations
from typing import Optional, Literal, Union
from pydantic import Field, ConfigDict
from .content_types import ArtifactKind
from .sources import SourceItem
from .base import DomainModel


# -------- 산출물 결과 모델들 --------

class PassageResult(DomainModel):
    """지문 결과 (단일/복합 모두 이 타입으로 수용)"""
    kind: Literal[ArtifactKind.passage] = ArtifactKind.passage
    source_item: SourceItem = Field(..., description="입력 소스")
    passage: str = Field(..., description="생성된 지문(텍스트)")

class AudioScriptTurn(DomainModel):
    """대본을 구조화하고 싶을 때(선택)"""
    speaker: str = Field(..., description="화자 라벨")
    utterance: str = Field(..., description="발화 텍스트")
    # 필요 시 timestamp, sfx 등 확장 가능

class AudioScriptResult(DomainModel):
    """오디오 대본 결과"""
    kind: Literal[ArtifactKind.audio_script] = ArtifactKind.audio_script
    source_item: SourceItem = Field(..., description="입력 소스")
    audio_script: str = Field(..., description="대본 원문(플랫 텍스트)")
    # 선택: 구조화 버전 병행 저장
    turns: Optional[list[AudioScriptTurn]] = Field(None, description="대본의 구조화 버전(선택)")

class ImageCaptionResult(DomainModel):
    """이미지 캡션 결과"""
    kind: Literal[ArtifactKind.image_caption] = ArtifactKind.image_caption
    source_item: SourceItem = Field(..., description="입력 소스")
    image_caption: str = Field(..., description="이미지 설명(캡션)")
    situation: Optional[str] = Field(None, description="상황/문제 서술(선택)")

# -------- 결과 유니온 --------
GeneratedArtifact = Union[PassageResult, AudioScriptResult, ImageCaptionResult]

# -------- 예시 스키마 --------
class ExampleSchemas(DomainModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "kind": "passage",
                    "source_item": {
                        "source_kind": "passage_multi",
                        "korean_topic": "회식 문화",
                        "korean_context": "회식은 한국 직장 문화의 중요한 부분...",
                        "foreign_topic": "Happy Hour Culture",
                        "foreign_context": "Happy hour is a social tradition..."
                    },
                    "passage": "한국 직장에서는 회식이 중요한 사회적 활동으로..."
                },
                {
                    "kind": "audio_script",
                    "source_item": {
                        "source_kind": "audio_script",
                        "topic": "신문 구독 vs 뉴스 앱 이용",
                        "context": "세상 소식을 접하는 방법..."
                    },
                    "audio_script": "남자: ...\n여자: ..."
                },
                {
                    "kind": "image_caption",
                    "source_item": {
                        "source_kind": "image_caption",
                        "topic": "쓰레기 무단 투기 및 분리배출 문제",
                        "hint": "아파트 단지 공용 쓰레기장 상황"
                    },
                    "image_caption": "서울의 한 아파트 단지 공용 쓰레기장에...",
                    "situation": "분리배출 미준수로 환경오염과 주민 불편 발생..."
                }
            ]
        }
    )

from __future__ import annotations
from typing import Optional, Dict, Set, List
from pydantic import Field, ConfigDict, PrivateAttr, model_validator
from .base import DomainModel
from src.domain.entities.enums import ContentType, Scope, EvalMethod, RubricID

class RubricOverride(DomainModel):
    name: Optional[str] = None
    summary: Optional[str] = None
    scope: Optional[Scope] = None
    methods: Optional[Set[EvalMethod]] = None

class RubricDefinition(DomainModel):
    """루브릭 기본 정의(코어) + 유형별 override"""
    model_config = ConfigDict(frozen=True)

    id: RubricID
    name: str
    summary: str
    applies_to: Set[ContentType] = Field(
        default_factory=lambda: {ContentType.passage, ContentType.audio_script, ContentType.image_caption},
        description="이 루브릭이 적용 가능한 산출물 종류",
    )
    scope: Scope = Scope.CONTENT_ONLY
    methods: Set[EvalMethod] = Field(..., description="허용 평가 방식")

    # ★ Variant를 대체: 유형별 덮어쓰기 모음
    overrides: Dict[ContentType, RubricOverride] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _non_empty_targets(self):
        if not self.applies_to and not self.overrides:
            raise ValueError("applies_to 또는 overrides 중 최소 하나는 필요합니다.")
        if not self.methods:
            raise ValueError("methods는 최소 1개 이상이어야 합니다.")
        return self

    def for_kind(self, kind: ContentType) -> Optional["RubricDefinition"]:
        """특정 kind에 맞춰 override 반영된 사본 반환(항상 applies_to={kind})"""
        if (kind not in self.applies_to) and (kind not in self.overrides):
            return None
        ov = self.overrides.get(kind)
        if not ov:
            return self.model_copy(update={"applies_to": {kind}})
        merged = {
            "name": ov.name or self.name,
            "summary": ov.summary or self.summary,
            "scope": ov.scope or self.scope,
            "methods": ov.methods or self.methods,
            "applies_to": {kind},
        }
        return self.model_copy(update=merged)

class RubricCatalog(DomainModel):
    """코어+오버라이드 카탈로그"""
    rubrics: List[RubricDefinition]
    _cache: Dict[ContentType, List[RubricDefinition]] = PrivateAttr(default_factory=dict)

    def effective_for(self, kind: ContentType) -> List[RubricDefinition]:
        if kind in self._cache:
            return self._cache[kind]
        out = [eff for r in self.rubrics if (eff := r.for_kind(kind))]
        self._cache[kind] = out
        return out

# ====== 카탈로그 인스턴스 ======
def build_default_rubric_catalog() -> RubricCatalog:
    core = [
        RubricDefinition(
            id=RubricID.R1_GUIDELINE_COMPLETENESS,
            name="평가 지침 완전성",
            summary="주어진 평가 지침(평가 목표 세트)을 완전히 충족하는가?",
            methods={EvalMethod.BINARY, EvalMethod.PREFERENCE},
            scope=Scope.CONTENT_PLUS_STEM,  # ★ 자료+지시문 단위로 평가
        ),
        RubricDefinition(
            id=RubricID.R2_TOPIC_CLARITY,
            name="핵심 주제 명확성",
            summary="하나의 통일된 주제를 명확히 전달하는가?",
            methods={EvalMethod.BINARY, EvalMethod.PREFERENCE},
            overrides={
                # 보고 말하기(이미지 캡션) 특화
                ContentType.image_caption: RubricOverride(
                    name="핵심 주제 시각화",
                    summary="핵심 주제가 장면/사물/행동의 시각 단서로 또렷하게 표현되었는가?",
                ),
                # 듣고 말하기(오디오 스크립트) 특화  👇 새로 추가
                ContentType.audio_script: RubricOverride(
                    name="대화 주제 일관성",
                    summary="화자들이 하나의 통일된 주제를 중심으로 대화를 나누는가? "
                            "발화 간 응집성이 유지되고, 불필요한 주제 전환이 최소화되는가?",
                ),
            },
        ),
        RubricDefinition(
            id=RubricID.R3_SOURCE_GROUNDEDNESS,
            name="참고 자료 기반성",
            summary="제공된 참고 자료(배경/컨텍스트)에만 근거하는가?",
            methods={EvalMethod.BINARY, EvalMethod.PREFERENCE},
            overrides={
                ContentType.audio_script: RubricOverride(
                    name="배경 정보 기반성",
                    summary="대화 내용이 주어진 배경 정보와 일치하는가?",
                ),
                ContentType.image_caption: RubricOverride(
                    name="이미지 설명 기반성",
                    summary="제공된 이미지 설명에 근거해서 문제 상황을 설명하는가?",
                ),
            },
        ),
        RubricDefinition(
            id=RubricID.R4_LOGICAL_STRUCTURE,
            name="논리적 흐름 및 구조",
            summary="도입-본론-결론 구조가 명확하고 자연스러운가?",
            methods={EvalMethod.BINARY, EvalMethod.PREFERENCE},
            overrides={
                ContentType.audio_script: RubricOverride(
                    name="대화의 흐름 및 구조",
                    summary="화자 간 상호작용이 '문제 제기→논거 제시→반박' 흐름을 따르는가?",
                ),
                ContentType.image_caption: RubricOverride(
                    name="시각적 재현 가능성",
                    summary="이미지 설명이 사진 생성이 가능할 만큼 구체적으로 기술되었는가?",
                ),
            },
        ),
        RubricDefinition(
            id=RubricID.R5_KOREAN_QUALITY,
            name="한국어 품질",
            summary="문법·맞춤법 오류나 번역투 없이 자연스러운가?",
            methods={EvalMethod.BINARY, EvalMethod.PREFERENCE},
            overrides={
                ContentType.image_caption: RubricOverride(
                    name="문제 상황 한국어 품질",
                    summary=(
                        "이미지(또는 이미지 설명)로부터 도출된 문제 상황의 서술이 "
                        "문법·맞춤법 오류 없이 자연스럽고 불필요한 번역투나 모호한 지시어 없이 명료하게 작성되었는가?"
                    ),
                ),
            },
        ),
        RubricDefinition(
            id=RubricID.R6_L2_APPROPRIATENESS,
            name="L2 학습자 응답 적합성",
            summary="(content + stems) 기준: 어휘 수준, 문장 복잡도/길이, 지시문의 명료성이 L2 학습자가 무리 없이 응답하기에 적절한가?",
            methods={EvalMethod.LIKERT, EvalMethod.PREFERENCE},  # 권장: 세밀도 확보 + 선호 랭킹 가능
            scope=Scope.CONTENT_PLUS_STEM,                       # ★ content + stems 단위로 평가
            overrides={
                # 듣고 말하기
                ContentType.audio_script: RubricOverride(
                    name="구어체·응답 적합성",
                    summary=(
                        "(content + stems) 기준: 구어체 자연성, 발화 길이/회차, 질문·지시의 명료성이 "
                        "L2 학습자의 응답을 용이하게 하는가?"
                    ),
                ),
                # 보고 말하기
                ContentType.image_caption: RubricOverride(
                    name="시각 단서·응답 적합성",
                    summary=(
                        "(content + stems) 기준: 이미지(또는 이미지 설명)의 시각 단서가 지시문과 자연스럽게 연결되어 "
                        "학습자가 무엇을 말해야 하는지 명확히 파악할 수 있으며, "
                        "요구 어휘/문장 복잡도가 과도하지 않아 묘사·경험·제안 응답을 무리 없이 수행할 수 있는가?"
                    ),
                ),
            },
        )
    ]
    return RubricCatalog(rubrics=core)
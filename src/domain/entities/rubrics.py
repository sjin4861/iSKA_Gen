from __future__ import annotations
from typing import Optional, List, Dict, Literal
from enum import Enum
from pydantic import Field
from .base import DomainModel
from .content_types import ArtifactKind

class RubricID(str, Enum):
    # Core (Reading / 기본형)
    R1_GUIDELINE_COMPLETENESS = "R1_GUIDELINE_COMPLETENESS"   # 평가 지침 완전성
    R2_TOPIC_CLARITY = "R2_TOPIC_CLARITY"                     # 핵심 주제 명확성
    R3_SOURCE_GROUNDEDNESS = "R3_SOURCE_GROUNDEDNESS"         # 참고 자료 기반성
    R4_LOGICAL_STRUCTURE = "R4_LOGICAL_STRUCTURE"             # 논리적 흐름 및 구조
    R5_KOREAN_QUALITY = "R5_KOREAN_QUALITY"                   # 한국어 품질
    R6_L2_APPROPRIATENESS = "R6_L2_APPROPRIATENESS"           # L2 학습자 적합성

class EvalMethod(str, Enum):
    BINARY = "binary"     # O/X
    LIKERT = "likert"     # 1~5
    PREFERENCE = "preference"  # A vs B

class Scope(str, Enum):
    CONTENT_ONLY = "content_only"
    CONTENT_PLUS_STEM = "content_plus_stem"

class BinaryAggregation(str, Enum):
    ALL_MUST_PASS = "all_must_pass"   # 하나라도 실패하면 Fail
    ANY = "any"

class RubricDefinition(DomainModel):
    """루브릭 기본 정의(코어)"""
    id: RubricID
    name: str
    summary: str
    applies_to: List[ArtifactKind] = Field(
        default_factory=lambda: [ArtifactKind.passage, ArtifactKind.audio_script, ArtifactKind.image_caption],
        description="이 루브릭이 적용 가능한 산출물 종류"
    )
    scope: Scope = Scope.CONTENT_ONLY
    binary_aggregation: Optional[BinaryAggregation] = None  # binary일 때만 의미

    methods: List[EvalMethod] = Field(..., description="허용 평가 방식")

class RubricVariant(DomainModel):
    """
    유형별 특화 루브릭: 특정 ArtifactKind에서 코어 루브릭을 수정/대체
    - base_id: 어떤 코어 루브릭을 대체하는지
    - override_*: 이름/요약 대체
    """
    base_id: RubricID
    for_kind: ArtifactKind
    override_name: Optional[str] = None
    override_summary: Optional[str] = None

class RubricCatalog(DomainModel):
    """코어+변형 루브릭 카탈로그"""
    core: List[RubricDefinition]
    variants: List[RubricVariant]

    def effective_for(self, kind: ArtifactKind) -> List[RubricDefinition]:
        """
        특정 산출물 종류에 대해 실제 적용되는(변형 반영된) 루브릭 리스트를 돌려준다.
        순서는 core 정의 순서를 그대로 유지한다.
        """
        var_map: Dict[RubricID, RubricVariant] = {
            v.base_id: v for v in self.variants if v.for_kind == kind
        }
        out: List[RubricDefinition] = []
        for r in self.core:
            if kind not in r.applies_to:
                continue
            if r.id in var_map:
                v = var_map[r.id]
                out.append(RubricDefinition(
                    id=r.id,
                    name=v.override_name or r.name,
                    summary=v.override_summary or r.summary,
                    applies_to=[kind],   # 이 변형은 해당 kind에 한정
                    methods=r.methods,
                ))
            else:
                out.append(r)
        return out

# ====== 카탈로그 인스턴스 ======
def build_default_rubric_catalog() -> RubricCatalog:
    core = [
        RubricDefinition(
            id=RubricID.R1_GUIDELINE_COMPLETENESS,
            name="평가 지침 완전성",
            summary="주어진 평가 지침(평가 목표 세트)을 완전히 충족하는가?",
            applies_to=[ArtifactKind.passage, ArtifactKind.audio_script, ArtifactKind.image_caption],
            methods=[EvalMethod.BINARY, EvalMethod.PREFERENCE],
            scope=Scope.CONTENT_PLUS_STEM,                 # ★ 자료+지시문 단위로 평가
            binary_aggregation=BinaryAggregation.ALL_MUST_PASS,    # ★ 하나라도 미충족이면 Fail
        ),
        RubricDefinition(
            id=RubricID.R2_TOPIC_CLARITY,
            name="핵심 주제 명확성",
            summary="하나의 통일된 주제를 명확히 전달하는가?",
            applies_to=[ArtifactKind.passage, ArtifactKind.audio_script, ArtifactKind.image_caption],
            methods=[EvalMethod.LIKERT, EvalMethod.PREFERENCE],
        ),
        RubricDefinition(
            id=RubricID.R3_SOURCE_GROUNDEDNESS,
            name="참고 자료 기반성",
            summary="제공된 참고 자료(배경/컨텍스트)에만 근거하는가?",
            applies_to=[ArtifactKind.passage, ArtifactKind.audio_script, ArtifactKind.image_caption],
            methods=[EvalMethod.BINARY, EvalMethod.PREFERENCE],
        ),
        RubricDefinition(
            id=RubricID.R4_LOGICAL_STRUCTURE,
            name="논리적 흐름 및 구조",
            summary="도입-본론-결론 구조가 명확하고 자연스러운가?",
            applies_to=[ArtifactKind.passage, ArtifactKind.audio_script, ArtifactKind.image_caption],
            methods=[EvalMethod.LIKERT, EvalMethod.PREFERENCE],
        ),
        RubricDefinition(
            id=RubricID.R5_KOREAN_QUALITY,
            name="한국어 품질",
            summary="문법·맞춤법 오류나 번역투 없이 자연스러운가?",
            applies_to=[ArtifactKind.passage, ArtifactKind.audio_script, ArtifactKind.image_caption],
            methods=[EvalMethod.LIKERT, EvalMethod.PREFERENCE],
        ),
        RubricDefinition(
            id=RubricID.R6_L2_APPROPRIATENESS,
            name="L2 학습자 적합성",
            summary="어휘 수준과 문장 복잡도가 학습자에게 적절한가?",
            applies_to=[ArtifactKind.passage, ArtifactKind.audio_script, ArtifactKind.image_caption],
            methods=[EvalMethod.LIKERT, EvalMethod.PREFERENCE],
        ),
    ]

    variants = [
        # 듣고 말하기(= audio_script) 특화
        RubricVariant(
            base_id=RubricID.R3_SOURCE_GROUNDEDNESS,
            for_kind=ArtifactKind.audio_script,
            override_name="배경 정보 기반성",
            override_summary="대화 내용이 주어진 배경 정보와 일치하는가?",
        ),
        RubricVariant(
            base_id=RubricID.R4_LOGICAL_STRUCTURE,
            for_kind=ArtifactKind.audio_script,
            override_name="대화의 흐름 및 구조",
            override_summary="화자 간 상호작용이 '문제 제기→논거 제시→반박' 흐름을 따르는가?",
        ),
        RubricVariant(
            base_id=RubricID.R6_L2_APPROPRIATENESS,
            for_kind=ArtifactKind.audio_script,
            override_name="구어체 적합성",
            override_summary="딱딱한 문어체가 아닌 자연스러운 구어체로 작성되었는가?",
        ),
        # 보고 말하기(= image_caption) 특화
        RubricVariant(
            base_id=RubricID.R1_GUIDELINE_COMPLETENESS,
            for_kind=ArtifactKind.image_caption,
            override_name="평가 목표 연계성",
            override_summary="묘사·경험·제안 응답을 이끌 단서를 분명히 제시하는가?",
        ),
        RubricVariant(
            base_id=RubricID.R2_TOPIC_CLARITY,
            for_kind=ArtifactKind.image_caption,
            override_name="핵심 주제 시각화",
            override_summary="핵심 주제가 장면/사물/행동의 시각 단서로 또렷하게 표현되었는가?",
        ),
        RubricVariant(
            base_id=RubricID.R4_LOGICAL_STRUCTURE,  # ← R3 → R4로 변경
            for_kind=ArtifactKind.image_caption,
            override_name="시각적 재현 가능성",
            override_summary="사진 생성이 가능할 만큼 구체적으로(주체/행동/배치/배경/광원/시점/색·수량 등) 기술되었는가?",
        ),
    ]
    return RubricCatalog(core=core, variants=variants)

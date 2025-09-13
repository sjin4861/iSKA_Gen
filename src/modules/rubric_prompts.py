# src/modules/rubric_prompts.py
"""
루브릭 메타데이터 & 유틸.

- RUBRIC_DESCRIPTIONS: 각 루브릭의 짧은 설명
- ALL_RUBRICS: 사용 가능한 전체 루브릭 ID 목록 (평가 시 --rubrics all 에서 사용)
- normalize_rubric_id: 축약/별칭을 표준 ID로 매핑
- resolve_rubrics: CLI 입력 리스트를 표준 ID 리스트로 확정 (all 지원)
"""

from __future__ import annotations
from typing import Iterable, List

# 표준 루브릭 ID → 설명
RUBRIC_DESCRIPTIONS = {
    # R1
    "completeness_for_guidelines": (
        "R1. 평가 지침 완전성 (Binary). 세 문항이 각 평가 목표를 모두 충실히 반영하는지 판정."
    ),
    # R2
    "core_theme_clarity": (
        "R2. 핵심 주제 명확성 (Binary). 상위 주제가 일관되게 전개되고 문단/소주제가 명시적으로 기여하는지."
    ),
    # R3
    "reference_groundedness": (
        "R3. 참고 자료 기반성 (Binary). 자료와 상충/무근거 수치·인용·고유명사 사실 제시 여부 점검."
    ),
    # R4
    "logical_flow_and_structure": (
        "R4. 논리적 흐름 및 구조 (Binary). 도입–본론–결론 구조, 문장/문단 연결의 논리성·자연스러움."
    ),
    # R5
    "korean_quality": (
        "R5. 한국어 품질 (Binary). 문법·맞춤법·띄어쓰기·호응·자연스러운 표현 여부."
    ),
    # R6
    "l2_learner_suitability": (
        "R6. L2 학습자 응답 적합성 (Likert 1~5). 어휘/구문 난이도와 명료성 관점에서의 응답 가능성."
    ),
}

# 전 범위 평가용 목록
ALL_RUBRICS: List[str] = list(RUBRIC_DESCRIPTIONS.keys())

# 축약/별칭 → 표준 ID 매핑
_RUBRIC_ALIASES = {
    # R1
    "r1": "completeness_for_guidelines",
    "completeness": "completeness_for_guidelines",
    "guideline_completeness": "completeness_for_guidelines",
    # R2
    "r2": "core_theme_clarity",
    "core_theme": "core_theme_clarity",
    "theme_clarity": "core_theme_clarity",
    "clarity": "core_theme_clarity",
    # R3
    "r3": "reference_groundedness",
    "groundedness": "reference_groundedness",
    "reference": "reference_groundedness",
    # R4
    "r4": "logical_flow_and_structure",
    "logic": "logical_flow_and_structure",
    "logical_flow": "logical_flow_and_structure",
    "structure": "logical_flow_and_structure",
    # R5
    "r5": "korean_quality",
    "ko_quality": "korean_quality",
    "korean": "korean_quality",
    # R6
    "r6": "l2_learner_suitability",
    "l2": "l2_learner_suitability",
    "suitability": "l2_learner_suitability",
}

def normalize_rubric_id(rubric: str) -> str:
    """입력 문자열을 표준 루브릭 ID로 정규화. (별칭/대소문자 허용)"""
    key = (rubric or "").strip().lower()
    if key in RUBRIC_DESCRIPTIONS:
        return key
    if key in _RUBRIC_ALIASES:
        return _RUBRIC_ALIASES[key]
    # 그대로 반환(상위에서 검증)
    return key

def resolve_rubrics(rubrics: Iterable[str]) -> List[str]:
    """
    CLI 입력을 표준 루브릭 ID 목록으로 변환.
    - ["all"] 이면 ALL_RUBRICS 전체 반환
    - 별칭 허용, 존재하지 않는 ID는 예외
    """
    r = list(rubrics or [])
    if len(r) == 1 and (r[0] or "").strip().lower() == "all":
        return ALL_RUBRICS[:]
    out: List[str] = []
    for x in r:
        norm = normalize_rubric_id(x)
        if norm not in RUBRIC_DESCRIPTIONS:
            raise ValueError(f"알 수 없는 루브릭: {x} (정규화 결과: {norm})")
        out.append(norm)
    return out

__all__ = [
    "RUBRIC_DESCRIPTIONS",
    "ALL_RUBRICS",
    "normalize_rubric_id",
    "resolve_rubrics",
]

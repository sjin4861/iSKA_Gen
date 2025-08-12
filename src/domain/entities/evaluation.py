from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

RUBRIC_KEYS = [
    "completeness_for_guidelines",
    "clarity_of_core_theme",
    "reference_groundedness",
    "logical_flow",
    "korean_quality",
    "l2_learner_suitability",
]

@dataclass(frozen=True)
class RubricScores:
    completeness_for_guidelines_score: Optional[float] = None
    clarity_of_core_theme_score: Optional[float] = None
    reference_groundedness_score: Optional[float] = None
    logical_flow_score: Optional[float] = None
    korean_quality_score: Optional[float] = None
    l2_learner_suitability_score: Optional[float] = None
    # 선택: 코멘트/근거
    notes: Optional[Dict[str, str]] = None

@dataclass(frozen=True)
class EvalTarget:
    text: str                       # 평가 대상 텍스트 (자료 or 자료+지시문)
    meta: Dict[str, Any]            # model_name, template_key, benchmark_id, idx 등

@dataclass(frozen=True)
class EvalResult:
    meta: Dict[str, Any]
    scores: RubricScores

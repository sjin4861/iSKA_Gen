# src/domain/entities/enums.py
from __future__ import annotations
from enum import Enum

class ContentType(str, Enum):
    passage = "passage"
    audio_script = "audio_script"
    image_caption = "image_caption"
    stem = "stem"

class RubricID(str, Enum):
    # 기존 루브릭
    R1_GUIDELINE_COMPLETENESS = "R1_GUIDELINE_COMPLETENESS"
    R2_TOPIC_CLARITY          = "R2_TOPIC_CLARITY"
    R3_SOURCE_GROUNDEDNESS    = "R3_SOURCE_GROUNDEDNESS"
    R4_LOGICAL_STRUCTURE      = "R4_LOGICAL_STRUCTURE"
    R5_KOREAN_QUALITY         = "R5_KOREAN_QUALITY"
    R6_L2_APPROPRIATENESS     = "R6_L2_APPROPRIATENESS"
    
    # Stem 평가용 루브릭 (기존 채점 시스템과 호환)
    completeness_for_guidelines = "completeness_for_guidelines"
    clarity_of_core_theme = "clarity_of_core_theme"
    reference_groundedness = "reference_groundedness"
    logical_flow = "logical_flow"
    korean_quality = "korean_quality"
    l2_learner_suitability = "l2_learner_suitability"
    all_in_one = "all_in_one"

class EvalMethod(str, Enum):
    BINARY = "binary"
    LIKERT = "likert"
    PREFERENCE = "preference"

class Scope(str, Enum):
    CONTENT_ONLY = "content_only"
    CONTENT_PLUS_STEM = "content_plus_stem"

class EvaluatorType(str, Enum):
    HUMAN_EXPERT = "human_expert"
    LLM = "llm"
    REWARD_MODEL = "reward_model"


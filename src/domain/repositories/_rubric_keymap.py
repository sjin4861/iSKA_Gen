# src/data/repositories/_rubric_keymap.py
from __future__ import annotations
from src.domain.entities.rubrics import RubricID

RUBRIC_TO_JSONKEY = {
    RubricID.R1_GUIDELINE_COMPLETENESS: "completeness_for_guidelines",
    RubricID.R2_TOPIC_CLARITY:          "clarity_of_core_theme",
    RubricID.R3_SOURCE_GROUNDEDNESS:    "reference_groundedness",
    RubricID.R4_LOGICAL_STRUCTURE:      "logical_flow",
    RubricID.R5_KOREAN_QUALITY:         "korean_quality",
    RubricID.R6_L2_APPROPRIATENESS:     "l2_learner_suitability",
}

# src/domain/repositories/rubric_evaluation_repository.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from src.domain.entities.content_types import ArtifactKind
from src.domain.entities.rubrics import RubricID

class RubricEvaluationRepository(ABC):
    @abstractmethod
    def evaluate_and_save(
        self,
        *,
        date_str: str,
        target_mode: str,  # "content" | "content+instruction" (오늘은 "content")
        artifact_kind: ArtifactKind,  # passage
        bench_ids: List[int],
        benchmark_version: str,
        rubric_ids: List[RubricID],  # 오늘은 3개만 사용
        source_model_filter: Optional[List[str]] = None,
        template_filter: Optional[List[str]] = None,
        limit_per_benchmark: Optional[int] = None,
        evaluator_client_type: str = "vllm",
        evaluator_model_name: str = "gpt-oss-20b",
        evaluator_client_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ...

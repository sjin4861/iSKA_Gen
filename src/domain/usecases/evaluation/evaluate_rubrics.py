from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Literal
from src.domain.repositories.rubric_evaluation_repository import RubricEvaluationRepository
from src.domain.entities.content_types import ArtifactKind
from src.domain.entities.rubrics import RubricID

@dataclass(frozen=True)
class EvaluateRubricsInput:
    date_str: str
    target_mode: Literal["content", "content+instruction"]
    artifact_kind: ArtifactKind
    bench_ids: List[int]
    benchmark_version: str
    rubric_ids: List[RubricID]
    source_model_filter: Optional[List[str]] = None
    template_filter: Optional[List[str]] = None
    limit_per_benchmark: Optional[int] = None
    evaluator_client_type: str = "vllm"
    evaluator_model_name: str = "gpt-oss-20b"
    evaluator_client_kwargs: Optional[Dict[str, Any]] = None  # {"base_url": ...}

class EvaluateRubricsUseCase:
    def __init__(self, repo: RubricEvaluationRepository):
        self.repo = repo

    def execute(self, i: EvaluateRubricsInput) -> Dict[str, Any]:
        return self.repo.evaluate_and_save(
            date_str=i.date_str,
            target_mode=i.target_mode,
            artifact_kind=i.artifact_kind,
            bench_ids=i.bench_ids,
            benchmark_version=i.benchmark_version,
            rubric_ids=i.rubric_ids,
            source_model_filter=i.source_model_filter,
            template_filter=i.template_filter,
            limit_per_benchmark=i.limit_per_benchmark,
            evaluator_client_type=i.evaluator_client_type,
            evaluator_model_name=i.evaluator_model_name,
            evaluator_client_kwargs=i.evaluator_client_kwargs,
        )

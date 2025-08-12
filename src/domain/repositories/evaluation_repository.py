from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any, List, Optional
from src.domain.entities.evaluation import EvalTarget, EvalResult

class EvaluationRepository(ABC):
    @abstractmethod
    def iter_targets(
        self,
        date_str: str,
        mode: str,                         # "materials" | "stem"
        *,
        only_bench_ids: Optional[List[int]] = None,
        only_models: Optional[List[str]] = None,
    ) -> Iterable[EvalTarget]:
        """평가 가능한 대상(자료/자료+지시문)을 순회 제공."""

    @abstractmethod
    def save_results(
        self,
        *,
        date_str: str,
        model_name: str,                   # 평가 수행 모델명 (예: gpt-oss-20b)
        benchmark_id: int,
        benchmark_version: str,
        results: List[Dict[str, Any]],     # JSON serializable rows
        task_dir: str = "eval_rubric",
    ) -> str:
        """벤치마크 단위로 결과 파일 저장, 저장 경로 반환."""

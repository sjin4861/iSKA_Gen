# src/domain/usecases/benchmark/get_guideline_by_id.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any
from src.domain.repositories.benchmark_repository import BenchmarkRepository

@dataclass(frozen=True)
class GetGuidelineByIdInput:
    set_id: int

@dataclass(frozen=True)
class GetGuidelineByIdOutput:
    data: Dict[str, Any]  # {"problem_types": [...], "eval_goals": [...]}

class GetGuidelineByIdUseCase:
    """UC-04: 가이드라인/평가목표만 추출"""
    def __init__(self, repo: BenchmarkRepository):
        self.repo = repo

    def execute(self, inp: GetGuidelineByIdInput) -> GetGuidelineByIdOutput:
        # 신규 인터페이스에 helper가 있다면 사용, 없으면 도메인에서 직접 추출
        if hasattr(self.repo, "get_guideline_by_id"):
            data = self.repo.get_guideline_by_id(inp.set_id)  # type: ignore[attr-defined]
        else:
            s = self.repo.get_set_by_id(inp.set_id)
            data = {}
            if getattr(s, "problem_types", None):
                data["problem_types"] = s.problem_types
            if getattr(s, "eval_goals", None):
                data["eval_goals"] = s.eval_goals
        return GetGuidelineByIdOutput(data=data)

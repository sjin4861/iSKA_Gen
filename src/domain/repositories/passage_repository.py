# src/domain/repositories/passage_repository.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class PassageRepository(ABC):
    """지문 생성/수정/저장까지 단일 창구"""
    @abstractmethod
    def generate_and_fill_missing(
        self, *, model_name: str, template_key: str,
        benchmark_id: int, benchmark_version: str,
        problem_types: List[str], eval_goals: List[str],
        sources: List[Dict[str, Any]], date_str: Optional[str],
        min_length: int, max_length: int, max_retries: int
    ) -> dict:  # {filled: [...], failed: [...], total: int}
        ...

    @abstractmethod
    def generate_one(
        self, *, source: Dict[str, Any], problem_types: List[str], eval_goals: List[str],
        model_name: str, template_key: str, min_length: int, max_length: int, max_retries: int
    ) -> Optional[str]:
        ...

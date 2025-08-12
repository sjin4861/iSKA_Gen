from __future__ import annotations
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

class StemRepository(ABC):
    """Stem 생성/수정/저장까지 단일 창구"""
    
    @abstractmethod
    def generate_and_fill_missing(
        self, *, 
        model_name: str, 
        template_key: str,
        benchmark_id: int, 
        benchmark_version: str,
        problem_types: List[str], 
        eval_goals: List[str],
        passages: List[Dict[str, Any]], 
        date_str: Optional[str],
        max_retries: int,
        passage_model_name: Optional[str] = None
    ) -> dict:  # {filled: [...], failed: [...], total: int}
        """기존 stem 데이터에서 누락된 부분을 찾아 생성하여 채움"""
        ...

    @abstractmethod
    def generate_one(
        self, *, 
        passage: str, 
        problem_type: str, 
        eval_goal: str,
        model_name: str, 
        template_key: str, 
        max_retries: int
    ) -> Optional[str]:
        """단일 stem 생성"""
        ...

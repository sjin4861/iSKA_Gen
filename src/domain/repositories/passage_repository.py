# src/domain/repositories/passage_repository.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from src.domain.entities.output_query import OutputQuery
from typing import Iterable
from src.domain.entities.outputs import CandidateOutput

class PassageRepository(ABC):
    """지문 생성/수정/저장까지 단일 창구"""
    
    # TODO: [ARCHITECTURE] 다음 메서드들을 domain/usecases/passage/로 분리 필요:
    # - generate_and_fill_missing -> GenerateAndFillMissingPassagesUseCase
    # - generate_one -> GenerateOnePassageUseCase
    # - find -> FindPassagesUseCase (이미 존재하지만 확인 필요)
    # Repository는 순수한 데이터 접근 인터페이스만 제공해야 함
    
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
    @abstractmethod
    def find(self, query: OutputQuery) -> Iterable[CandidateOutput]:
        raise NotImplementedError
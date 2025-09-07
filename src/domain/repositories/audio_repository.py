from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, List, Dict, Any, Optional
from src.domain.entities.outputs import CandidateOutput
from src.domain.entities.output_query import OutputQuery

class AudioRepository(ABC):
    """audio_script 후보를 제공하는 추상 포트"""
    
    # TODO: [ARCHITECTURE] 다음 메서드들을 domain/usecases/audio/로 분리 필요:
    # - generate_and_fill_missing -> GenerateAndFillMissingAudioScriptsUseCase
    # - generate_one -> GenerateOneAudioScriptUseCase
    # - find -> FindAudioScriptsUseCase
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

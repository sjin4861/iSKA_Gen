from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class AnalysisRepository(ABC):
    """data_store의 결과물(생성/평가)을 조회하기 위한 읽기 전용 리포지토리"""

    @abstractmethod
    def list_passage_records_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        """
        raw_outputs/<date>/passage/**.json 을 모두 로드해
        [{..., "generated_passage": str|None, "model_name":..., "task_name":..., "benchmark_id": int, "file_path":...}, ...] 형태로 반환
        """
        ...

    @abstractmethod
    def list_evaluation_records_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        """
        evaluations/<date>/misc/**/eval_rubric/*.json 을 모두 로드해
        [{"model_name":..., "benchmark_id": int, "file_path":..., "<rubric>_score": float, ...}, ...] 형태로 반환
        """
        ...

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class AnalysisRepository(ABC):
    """data_store의 결과물(생성/평가)을 조회하기 위한 읽기 전용 리포지토리"""

    # TODO: [ARCHITECTURE] 다음 메서드들을 domain/usecases/analysis/로 분리 필요:
    # - list_passage_records_by_date -> LoadPassageRecordsByDateUseCase (이미 존재함)
    # - list_evaluation_records_by_date -> LoadEvaluationRecordsByDateUseCase (이미 존재함)  
    # Repository는 순수한 데이터 접근 인터페이스만 제공해야 함
    # 기존 UseCase들과 Repository 메서드들 간의 일관성 확인 필요

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

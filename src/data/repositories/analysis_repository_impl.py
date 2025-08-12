from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path

from src.domain.repositories.analysis_repository import AnalysisRepository
from src.data.datasources.fs.raw_outputs_reader import RawOutputsFSReader
from src.data.datasources.fs.evaluations_reader import EvaluationsFSReader

class AnalysisRepositoryImpl(AnalysisRepository):
    def __init__(self, data_store_root: Path = Path("data_store")):
        self._raw = RawOutputsFSReader(data_store_root)
        self._eval = EvaluationsFSReader(data_store_root)

    def list_passage_records_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        return self._raw.list_passage_records_by_date(date_str)

    def list_evaluation_records_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        return self._eval.list_evaluation_records_by_date(date_str)

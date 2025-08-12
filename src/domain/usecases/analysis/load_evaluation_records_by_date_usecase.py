from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from domain.repositories.analysis_repository import AnalysisRepository

@dataclass(frozen=True)
class LoadEvaluationRecordsByDateInput:
    date_str: str

class LoadEvaluationRecordsByDateUseCase:
    def __init__(self, repo: AnalysisRepository):
        self.repo = repo

    def execute(self, i: LoadEvaluationRecordsByDateInput) -> List[Dict[str, Any]]:
        return self.repo.list_evaluation_records_by_date(i.date_str)

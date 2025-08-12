from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from domain.repositories.analysis_repository import AnalysisRepository

@dataclass(frozen=True)
class LoadPassageRecordsByDateInput:
    date_str: str

class LoadPassageRecordsByDateUseCase:
    def __init__(self, repo: AnalysisRepository):
        self.repo = repo

    def execute(self, i: LoadPassageRecordsByDateInput) -> List[Dict[str, Any]]:
        return self.repo.list_passage_records_by_date(i.date_str)

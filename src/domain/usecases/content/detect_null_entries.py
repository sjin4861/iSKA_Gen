# src/domain/usecases/content/detect_null_entries.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from ...repositories.content_repository import ContentRepository

@dataclass
class DetectNullEntriesInput:
    path: str

@dataclass
class DetectNullEntriesOutput:
    null_indices: List[int]
    total_count: int

class DetectNullEntriesUseCase:
    def __init__(self, repo: ContentRepository):
        self.repo = repo

    def execute(self, inp: DetectNullEntriesInput) -> DetectNullEntriesOutput:
        rows = self.repo.load_outputs(inp.path) or []
        return DetectNullEntriesOutput(
            null_indices=self.repo.find_null_indices(rows),
            total_count=len(rows)
        )

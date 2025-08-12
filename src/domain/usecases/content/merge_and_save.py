# src/domain/usecases/content/merge_and_save.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from ...repositories.content_repository import ContentRepository

@dataclass
class MergeAndSaveInput:
    path: str
    regenerated: Dict[int, Dict[str, Any]]

class MergeAndSaveUseCase:
    def __init__(self, repo: ContentRepository):
        self.repo = repo

    def execute(self, inp: MergeAndSaveInput) -> None:
        base = self.repo.load_outputs(inp.path) or []
        merged = self.repo.merge_by_indices(base, inp.regenerated)
        self.repo.save_outputs(inp.path, merged)

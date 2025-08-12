# src/domain/usecases/passage/generate_single.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from ...repositories.passage_repository import PassageRepository

@dataclass
class GenerateSingleInput:
    korean_topic: str
    korean_context: str
    problem_types: List[str]
    eval_goals: List[str]
    template_key: str
    foreign_topic: str = ""
    foreign_context: str = ""

@dataclass
class GenerateSingleOutput:
    row: Dict[str, Any]

class GenerateSingleUseCase:
    def __init__(self, repo: PassageRepository):
        self.repo = repo

    def execute(self, inp: GenerateSingleInput) -> GenerateSingleOutput:
        row = self.repo.generate_single(
            korean_topic=inp.korean_topic,
            korean_context=inp.korean_context,
            foreign_topic=inp.foreign_topic,
            foreign_context=inp.foreign_context,
            problem_types=inp.problem_types,
            eval_goals=inp.eval_goals,
            template_key=inp.template_key,
        )
        return GenerateSingleOutput(row=row)

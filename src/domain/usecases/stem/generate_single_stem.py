# src/domain/usecases/stem/generate_single_stem.py
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass

from src.domain.repositories.stem_repository import StemRepository


@dataclass
class GenerateSingleStemInput:
    """단일 stem 생성 입력"""
    passage: str
    problem_type: str
    eval_goal: str
    model_name: str
    template_key: str
    max_retries: int = 3


@dataclass 
class GenerateSingleStemOutput:
    """단일 stem 생성 출력"""
    stem: Optional[str]
    success: bool


class GenerateSingleStemUseCase:
    """단일 stem 생성 유스케이스"""
    
    def __init__(self, stem_repository: StemRepository):
        self.stem_repository = stem_repository
    
    def execute(self, input_data: GenerateSingleStemInput) -> GenerateSingleStemOutput:
        """단일 stem 생성 실행"""
        stem = self.stem_repository.generate_one(
            passage=input_data.passage,
            problem_type=input_data.problem_type,
            eval_goal=input_data.eval_goal,
            model_name=input_data.model_name,
            template_key=input_data.template_key,
            max_retries=input_data.max_retries,
        )
        
        return GenerateSingleStemOutput(
            stem=stem,
            success=stem is not None
        )

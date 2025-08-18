# src/data/repositories/raw_output_repository_impl.py
from __future__ import annotations
from typing import Iterable

from src.domain.repositories.passage_repository import PassageRepository
from src.domain.entities.outputs import CandidateOutput
from src.domain.entities.output_query import OutputQuery
from src.domain.entities.enums import ContentType
from src.data.datasources.fs.raw_output_fs import RawOutputFSDataSource


class RawOutputRepositoryImpl:
    """
    Raw Output 데이터를 관리하는 리포지토리 구현체
    """
    
    def __init__(self, root_dir: str = "data_store"):
        self.ds = RawOutputFSDataSource(root_dir)
    
    def find(self, content_type: ContentType, query: OutputQuery) -> Iterable[CandidateOutput]:
        """
        지정된 콘텐츠 타입과 쿼리 조건에 맞는 후보들을 찾아 반환
        """
        return self.ds.find_candidates(content_type, query)

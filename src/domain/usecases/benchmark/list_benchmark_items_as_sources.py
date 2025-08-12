from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

from src.domain.repositories.benchmark_repository import BenchmarkRepository
from src.domain.entities.benchmark import BenchmarkItemFlat
from src.domain.entities.content_types import ArtifactKind

@dataclass(frozen=True)
class ListBenchmarkItemsAsSourcesInput:
    benchmark_id: int
    artifact_kind: ArtifactKind  # 현재 passage만 지원

@dataclass(frozen=True)
class ListBenchmarkItemsAsSourcesOutput:
    # Passage 생성에 바로 넣어 쓸 수 있는 dict들
    sources_as_dicts: List[Dict[str, Any]]

class ListBenchmarkItemsAsSourcesUseCase:
    def __init__(self, repo: BenchmarkRepository) -> None:
        self.repo = repo

    def execute(self, i: ListBenchmarkItemsAsSourcesInput) -> ListBenchmarkItemsAsSourcesOutput:
        assert i.artifact_kind == ArtifactKind.passage, "현재 passage만 지원"
        items: List[BenchmarkItemFlat] = list(self.repo.iter_items(i.benchmark_id))

        # foreign 정보가 있으면 비교형(korean/foreign), 없으면 단일형(topic/context)
        out: List[Dict[str, Any]] = []
        for it in items:
            if it.foreign_topic is not None or it.foreign_context is not None:
                out.append({
                    "korean_topic": it.topic,
                    "korean_context": it.context,
                    "foreign_topic": it.foreign_topic,
                    "foreign_context": it.foreign_context,
                })
            else:
                out.append({
                    "topic": it.topic,
                    "context": it.context,
                })
        return ListBenchmarkItemsAsSourcesOutput(sources_as_dicts=out)

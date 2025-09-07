# src/domain/repositories/benchmark_repository.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, Literal, Optional

from src.domain.entities.benchmark import (
    BenchmarkSet,
    BenchmarkCollection,
    BenchmarkItemFlat,
)

# ---- Value Objects / DTOs ----------------------------------------------------

Kind = Literal["single", "compare", "any"]  # 단일형/비교형/전체

@dataclass(frozen=True, slots=True)
class ItemFilter:
    """
    평탄화된 아이템 스트림을 위한 간단 필터.
    - kind: 'single' | 'compare' | 'any'
      * 비교형은 (korean_topic|korean_context) 존재하는 항목으로 간주
      * 단일형은 위 조건이 없는 항목
    - offset/limit: 페이징 (메모리 절약 목적)
    """
    kind: Kind = "any"
    offset: int = 0
    limit: Optional[int] = None


# ---- Repository Interface ----------------------------------------------------

class BenchmarkRepository(ABC):
    """
    벤치마크 컬렉션/세트/아이템(평탄화 스트림) 접근을 추상화한다.

    설계 원칙
    - 항상 도메인 모델(BenchmarkCollection/Set/ItemFlat)로 노출
    - iter_items 는 **지연(generator)** 제공으로 대용량 친화
    - '파일/버전 선택'은 상위 Usecase/Service 레이어 책임
    - 캐시는 구현체가 선택적으로 제공하고, 무효화 훅을 노출
    """

    # --- 컬렉션 단위 ---

    @abstractmethod
    def load_collection(self) -> BenchmarkCollection:
        """
        벤치마크 전체 컬렉션을 로드/검증하여 반환한다.
        구현체는 pydantic(v2) 등으로 스키마 검증을 수행해야 한다.
        실패 시 ValidationError/FileNotFoundError 등을 일관되게 전달한다.
        """
        raise NotImplementedError

    # --- 세트 단위 ---

    @abstractmethod
    def get_set_by_id(self, set_id: int) -> BenchmarkSet:
        """
        지정 ID의 벤치마크 세트를 반환한다.
        존재하지 않으면 ValueError를 발생시킨다.
        """
        raise NotImplementedError

    # --- 아이템(평탄화) 스트림 ---

    @abstractmethod
    def iter_items(
        self,
        set_id: int,
        *,
        flt: Optional[ItemFilter] = None,
    ) -> Iterable[BenchmarkItemFlat]:
        """
        지정 세트의 아이템을 '평탄화'하여 **스트리밍**으로 반환한다.

        평탄화 규칙(UC-08):
        - 비교형: korean_topic/context 를 topic/context 로 매핑하고,
                 foreign_topic/context 는 보조 필드로 채움
        - 단일형: 원래 topic/context 를 사용, foreign_* 는 None
        - None 은 빈 문자열("") 또는 None 으로 안전 처리

        필터(UC-10):
        - kind: 'single' | 'compare' | 'any'
        - offset/limit: 페이징

        구현체는 generator 를 사용해 메모리 사용을 최소화해야 한다.
        """
        raise NotImplementedError

    # --- 편의 메서드(선택 구현, 기본 제공 가능) ---

    def count_items(self, set_id: int, *, flt: Optional[ItemFilter] = None) -> int:
        """
        스트림을 소비하지 않는 범위에서 가능한 구현이면 override 권장.
        기본 구현은 스트림을 모두 순회하여 카운트한다.
        """
        cnt = 0
        for _ in self.iter_items(set_id, flt=flt):
            cnt += 1
        return cnt

    def get_guideline_by_id(self, set_id: int) -> dict:
        """
        상위 유스케이스(기존 benchmark_loader.get_guideline_by_id)를
        도메인 모델에서 바로 꺼낼 수 있도록 보조 메서드로 제공.
        존재하지 않는 키는 포함하지 않는다.
        """
        s = self.get_set_by_id(set_id)
        out: dict = {}
        if getattr(s, "problem_types", None):
            out["problem_types"] = s.problem_types
        if getattr(s, "eval_goals", None):
            out["eval_goals"] = s.eval_goals
        return out

    # --- 캐시/진단 훅(UC-11/12) ---

    def invalidate_cache(self) -> None:
        """구현체 내부 캐시가 있다면 무효화한다. (선택)"""
        pass

    def last_loaded_at(self) -> Optional[datetime]:
        """컬렉션이 마지막으로 로드된 시각(있다면) 반환. (선택)"""
        return None

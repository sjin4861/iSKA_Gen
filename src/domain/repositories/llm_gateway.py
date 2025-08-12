# src/domain/repositories/llm_gateway.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterable

class LLMGateway(ABC):
    """LLM 호출을 추상화한 도메인 레벨 게이트웨이(리포지토리 인터페이스)."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **params: Any) -> str:
        """단일 생성 호출."""
        raise NotImplementedError

    @abstractmethod
    def generate_batch(self, batch_messages: Iterable[List[Dict[str, str]]], **params: Any) -> List[str]:
        """배치 생성 호출(미구현 백엔드는 순차 호출로 대체 가능)."""
        raise NotImplementedError

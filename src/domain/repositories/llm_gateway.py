# src/domain/repositories/llm_gateway.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Iterable

class LLMGateway(ABC):
    """LLM 호출을 추상화한 도메인 레벨 게이트웨이(리포지토리 인터페이스)."""

    # TODO: [ARCHITECTURE] LLMGateway는 이미 적절히 추상화되어 있음
    # 하지만 복잡한 LLM 호출 로직이 있다면 domain/usecases/llm/로 분리 고려:
    # - generate -> GenerateTextUseCase
    # - generate_batch -> GenerateBatchTextUseCase
    # 현재는 단순한 인터페이스이므로 그대로 유지 가능

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **params: Any) -> str:
        """단일 생성 호출."""
        raise NotImplementedError

    @abstractmethod
    def generate_batch(self, batch_messages: Iterable[List[Dict[str, str]]], **params: Any) -> List[str]:
        """배치 생성 호출(미구현 백엔드는 순차 호출로 대체 가능)."""
        raise NotImplementedError

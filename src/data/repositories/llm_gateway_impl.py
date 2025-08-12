from __future__ import annotations
from typing import List, Dict, Any, Iterable, Optional
import time
from modules.client_factory import ModelClientFactory  # ← 기존 팩토리 사용
from modules.model_client import BaseModelClient
from src.domain.repositories.llm_gateway import LLMGateway

class LLMGatewayImpl(LLMGateway):
    """
    ModelClientFactory를 감싸는 Gateway.
    - 리트라이/백오프
    - 공통 파라미터 머지
    - JSON 모드, 스트리밍(옵션) 등 일관 처리
    """
    def __init__(
        self,
        *,
        client_type: str,         # "openai" | "local" | "vllm"
        model_name: str,
        default_params: Optional[Dict[str, Any]] = None,
        max_retries: int = 2,
        retry_backoff_sec: float = 1.5,
        **client_kwargs: Any,     # base_url, api_key 등
    ):
        self.client: BaseModelClient = ModelClientFactory.create_model_client(
            client_type=client_type, model_name=model_name, **client_kwargs
        )
        self.default_params = default_params or {}
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec

    # ---- private helpers ----
    def _merge_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(self.default_params)
        merged.update(params or {})
        return merged

    # ---- public API ----
    def generate(self, messages: List[Dict[str, str]], **params) -> str:
        merged = self._merge_params(params)
        for attempt in range(self.max_retries + 1):
            try:
                out = self.client.call(messages, **merged)  # model_client 통일 인터페이스
                return out or ""
            except Exception as e:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff_sec * (attempt + 1))
        return ""

    def generate_batch(self, batch_messages: Iterable[List[Dict[str, str]]], **params) -> List[str]:
        """
        OpenAIModelClient.call_batch가 있으면 사용, 없으면 순차 호출로 폴백
        """
        merged = self._merge_params(params)
        # 최적 경로: call_batch 지원 여부
        call_batch = getattr(self.client, "call_batch", None)
        if callable(call_batch):
            return call_batch(list(batch_messages), **merged)

        # 폴백: 순차 호출 + 간단 리트라이
        outputs: List[str] = []
        for msgs in batch_messages:
            outputs.append(self.generate(msgs, **merged))
        return outputs

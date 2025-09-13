# src/modules/model_client.py
"""LangChain LLM 초기화 유틸

모델별 포트:
- A.X-* 모델: localhost:8000/v1
- EXAONE-* 모델: localhost:8001/v1

환경 변수 기반 구성도 지원:
    VLLM_URL / VLLM_BASE_URL : base URL 강제 지정
    VLLM_API_KEY             : vLLM 서버 key (필요 없으면 ANY)
    VLLM_MODEL               : 기본 모델명
    OPENAI_API_KEY           : OpenAI 공식 API 키

노출 함수:
    get_vllm_chat, get_openai_chat, auto_get_chat, get_chat_from_env
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import os
import time
import urllib.parse
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel



# 환경 변수 키
ENV_VLLM_URL = "VLLM_URL"
ENV_VLLM_BASE_URL = "VLLM_BASE_URL"
ENV_VLLM_API_KEY = "VLLM_API_KEY"
ENV_VLLM_MODEL = "VLLM_MODEL"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"


def _default_vllm_url_for_model(model_name: str) -> str:
    """모델명 패턴으로 기본 base_url 선택."""
    lower = model_name.lower()
    if lower.startswith("a.x") or "a.x-" in lower:
        return "http://localhost:8000/v1"
    if lower.startswith("exaone") or "exaone-" in lower:
        return "http://localhost:8001/v1"
    # fallback
    return "http://localhost:8000/v1"


def get_vllm_chat(
    model_name: str,
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseChatModel:
    """vLLM(OpenAI 호환) ChatOpenAI 인스턴스."""
    if ChatOpenAI is None:  # pragma: no cover
        raise ImportError("langchain_openai 미설치: pip install langchain-openai")

    base_url = (
        base_url
        or os.getenv(ENV_VLLM_URL)
        or os.getenv(ENV_VLLM_BASE_URL)
        or _default_vllm_url_for_model(model_name)
    )
    api_key = api_key or os.getenv(ENV_VLLM_API_KEY, "EMPTY")

    # /v1 정규화
    parsed = urllib.parse.urlparse(base_url)
    if parsed.path.rstrip("/") != "/v1":
        if parsed.path in ("", "/"):
            base_url = base_url.rstrip("/") + "/v1"

    params: Dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "base_url": base_url,
        "api_key": api_key,
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if extra_kwargs:
        params.update(extra_kwargs)

    last_err = None
    for attempt in range(3):
        try:
            return ChatOpenAI(**params)
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(
        f"vLLM ChatOpenAI 초기화 실패: model={model_name}, base_url={base_url}, error={last_err}"
    )


def get_openai_chat(
    model_name: str = "gpt-4o-mini",
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    api_key: Optional[str] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseChatModel:
    """공식 OpenAI API ChatOpenAI 인스턴스."""
    if ChatOpenAI is None:  # pragma: no cover
        raise ImportError("langchain_openai 미설치: pip install langchain-openai")
    api_key = api_key or os.getenv(ENV_OPENAI_API_KEY)
    if not api_key:
        raise ValueError("OPENAI_API_KEY 미설정")

    params: Dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "api_key": api_key,
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if extra_kwargs:
        params.update(extra_kwargs)
    return ChatOpenAI(**params)


def auto_get_chat(model_name: str, **kwargs) -> BaseChatModel:
    """모델명 패턴으로 openai vs vllm 자동 선택."""
    lower = model_name.lower()
    if lower.startswith("gpt-") or "o1-" in lower:
        return get_openai_chat(model_name, **kwargs)
    return get_vllm_chat(model_name, **kwargs)


def get_chat_from_env(
    *,
    model_name: str | None = None,
    prefer_openai: bool | None = None,
    **kwargs,
) -> BaseChatModel:
    """환경 변수 기반 Chat 모델 생성."""
    resolved = model_name or os.getenv(ENV_VLLM_MODEL)
    if not resolved:
        raise ValueError("model_name 미지정 & VLLM_MODEL 환경 변수 없음")
    if prefer_openai is True:
        return get_openai_chat(resolved, **kwargs)
    return auto_get_chat(resolved, **kwargs)


__all__ = [
    "get_vllm_chat",
    "get_openai_chat",
    "auto_get_chat",
    "get_chat_from_env",
    "ENV_VLLM_URL",
    "ENV_VLLM_BASE_URL",
    "ENV_VLLM_API_KEY",
    "ENV_VLLM_MODEL",
    "ENV_OPENAI_API_KEY",
    "BaseChatModel", 
]

# src/modules/client_factory.py
"""
모델 클라이언트 팩토리 - 클라이언트 유형에 따라 적절한 클라이언트를 생성합니다.
"""

from pathlib import Path
import sys
import os

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))

from modules.model_client import (
    BaseModelClient, OpenAIModelClient, LocalModelClient, VLLMOpenAIClient
)
from typing import Optional, Dict, Any, List

class ModelClientFactory:
    _REGISTRY = {
        "openai": OpenAIModelClient,
        "local": LocalModelClient,
        "vllm": VLLMOpenAIClient,
    }
    """모델 클라이언트를 생성하고 관리하는 팩토리 클래스"""
    
    @staticmethod
    def get_available_client_types() -> Dict[str, str]:
        return {
            "openai": "OpenAI API 클라이언트",
            "local": "로컬 모델 클라이언트",
            "vllm": "vLLM(OpenAI 호환) 클라이언트",
        }

    @staticmethod
    def get_default_model_for_client(client_type: str) -> str:
        defaults = {
            "openai": "gpt-4o-mini",
            "local": "Qwen3-8B",
            "vllm": "Llama-3.1-8B-Instruct",
        }
        return defaults.get(client_type.lower(), "")

    @staticmethod
    def create_model_client(client_type: str, model_name: str, **kwargs) -> BaseModelClient:
        client_type = client_type.lower()
        try:
            cls = ModelClientFactory._REGISTRY[client_type]
        except KeyError:
            raise ValueError(f"지원하지 않는 클라이언트 유형: {client_type}. 사용 가능: {list(ModelClientFactory._REGISTRY)}")
        # 파라미터 정리
        if client_type == "openai":
            kwargs.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        if client_type == "vllm":
            kwargs.setdefault("api_key", os.getenv("VLLM_API_KEY", "EMPTY"))
            kwargs.setdefault("base_url", os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
        return cls(model_name=model_name, **kwargs)

    @staticmethod
    def validate_client_config(client_type: str, model_name: str, **kwargs) -> Dict[str, Any]:
        result = {"valid": True, "errors": [], "warnings": []}
        ct = client_type.lower()
        if ct not in ModelClientFactory._REGISTRY:
            return {"valid": False, "errors": [f"지원하지 않는 클라이언트 유형: {client_type}"], "warnings": []}
        if not model_name or not model_name.strip():
            return {"valid": False, "errors": ["모델 이름이 필요합니다"], "warnings": []}

        if ct == "openai":
            if not (kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")):
                result["valid"] = False
                result["errors"].append("OpenAI API 키가 필요합니다")
        elif ct == "local":
            if not os.getenv("LOCAL_MODELS_PATH"):
                result["warnings"].append("LOCAL_MODELS_PATH 미설정. 기본값 ~/models 사용")
        elif ct == "vllm":
            base_url = kwargs.get("base_url") or os.getenv("VLLM_BASE_URL")
            if not base_url:
                result["warnings"].append("vLLM base_url 미설정 (기본 http://localhost:8000/v1 사용)")
        return result
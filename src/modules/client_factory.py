# src/modules/client_factory.py
"""
모델 클라이언트 팩토리 - 클라이언트 유형에 따라 적절한 클라이언트를 생성합니다.
"""

from pathlib import Path
import sys
import os

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))

from modules.model_client import BaseModelClient, OpenAIModelClient, LocalModelClient
from src.utils.settings_loader import get_settings
from typing import Optional, Dict, Any, List

class ModelClientFactory:
    """모델 클라이언트를 생성하고 관리하는 팩토리 클래스"""
    
    @staticmethod
    def create_model_client(client_type: str, model_name: str, **kwargs) -> BaseModelClient:
        """
        클라이언트 유형에 따라 적절한 모델 클라이언트를 생성합니다.
        
        Args:
            client_type: 클라이언트 유형 ("openai", "local")
            model_name: 모델 이름
            **kwargs: 추가 설정
        
        Returns:
            BaseModelClient: 생성된 클라이언트
        
        Raises:
            ValueError: 지원하지 않는 클라이언트 유형
        """
        client_type = client_type.lower()
        
        if client_type == "openai":
            api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
            return OpenAIModelClient(
                model_name=model_name,
                api_key=api_key,
                **{k: v for k, v in kwargs.items() if k != 'api_key'}
            )
        
        elif client_type == "local":
            return LocalModelClient(
                model_name=model_name,
                **kwargs
            )
        
        else:
            raise ValueError(f"지원하지 않는 클라이언트 유형: {client_type}. 사용 가능한 유형: openai, local")

    @staticmethod
    def get_available_client_types() -> Dict[str, str]:
        """사용 가능한 클라이언트 유형들을 반환합니다."""
        return {
            "openai": "OpenAI API 클라이언트",
            "local": "로컬 모델 클라이언트"
        }

    @staticmethod
    def get_default_model_for_client(client_type: str) -> str:
        """클라이언트 유형별 기본 모델을 반환합니다."""
        defaults = {
            "openai": "gpt-4o-mini",
            "local": "Qwen3-8B"
        }
        return defaults.get(client_type.lower(), "")

    @staticmethod
    def create_default_client() -> BaseModelClient:
        """설정 파일 기반으로 기본 클라이언트를 생성합니다."""
        cfg = get_settings()
        llm_cfg = cfg.get('llm', {})
        
        # 기본 클라이언트 유형 결정
        default_client_type = llm_cfg.get('default_client', 'local')
        default_model = llm_cfg.get('model') or ModelClientFactory.get_default_model_for_client(default_client_type)
        
        return ModelClientFactory.create_model_client(
            client_type=default_client_type,
            model_name=default_model
        )

    @staticmethod
    def validate_client_config(client_type: str, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        클라이언트 설정을 검증하고 문제가 있으면 오류 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: {"valid": bool, "errors": List[str], "warnings": List[str]}
        """
        result = {"valid": True, "errors": [], "warnings": []}
        
        client_type = client_type.lower()
        
        # 클라이언트 유형 검증
        if client_type not in ModelClientFactory.get_available_client_types():
            result["valid"] = False
            result["errors"].append(f"지원하지 않는 클라이언트 유형: {client_type}")
            return result
        
        # 모델 이름 검증
        if not model_name or not model_name.strip():
            result["valid"] = False
            result["errors"].append("모델 이름이 필요합니다")
            return result
        
        # OpenAI 특별 검증
        if client_type == "openai":
            api_key = kwargs.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                result["valid"] = False
                result["errors"].append("OpenAI API 키가 필요합니다")
            
            # OpenAI 모델명 검증
            valid_openai_models = [
                "gpt-4o", "gpt-4o-mini"
            ]
            if model_name not in valid_openai_models:
                result["warnings"].append(f"일반적이지 않은 OpenAI 모델명: {model_name}")
        
        # Local 특별 검증
        elif client_type == "local":
            models_dir = os.getenv('LOCAL_MODELS_PATH')
            if not models_dir:
                result["warnings"].append("LOCAL_MODELS_PATH 환경변수가 설정되지 않음. 기본값 사용: ~/models")
            
            # GPU 설정 검증
            gpus = kwargs.get('gpus')
            if gpus is not None:
                import torch
                if not torch.cuda.is_available():
                    result["warnings"].append("CUDA가 사용 불가능하지만 GPU 설정이 지정됨")
                elif isinstance(gpus, list):
                    available_gpus = torch.cuda.device_count()
                    invalid_gpus = [gpu for gpu in gpus if gpu >= available_gpus or gpu < 0]
                    if invalid_gpus:
                        result["errors"].append(f"유효하지 않은 GPU 인덱스: {invalid_gpus}. 사용 가능한 GPU: 0~{available_gpus-1}")
                else:
                    result["errors"].append("gpus 파라미터는 정수 리스트여야 합니다 (예: [0, 1])")
        
        return result

    # 편의 함수들
    @staticmethod
    def create_openai_client(model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, **kwargs) -> OpenAIModelClient:
        """OpenAI 클라이언트 생성 편의 함수"""
        return ModelClientFactory.create_model_client("openai", model_name, api_key=api_key, **kwargs)

    @staticmethod
    def create_local_client(model_name: str = "Qwen3-8B", gpus: Optional[List[int]] = None, **kwargs) -> LocalModelClient:
        """
        로컬 클라이언트 생성 편의 함수
        
        Args:
            model_name: 모델 이름
            gpus: 사용할 GPU 리스트 (예: [2, 3])
            **kwargs: 추가 설정
        """
        if gpus is not None:
            kwargs['gpus'] = gpus
        return ModelClientFactory.create_model_client("local", model_name, **kwargs)

    # GPU 유틸리티 함수들
    @staticmethod
    def get_available_gpus() -> List[int]:
        """사용 가능한 GPU 인덱스 리스트를 반환합니다."""
        try:
            import torch
            if torch.cuda.is_available():
                return list(range(torch.cuda.device_count()))
            return []
        except ImportError:
            return []

    @staticmethod
    def get_gpu_memory_info(gpu_id: int) -> Dict[str, float]:
        """특정 GPU의 메모리 정보를 반환합니다 (GB 단위)."""
        try:
            import torch
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                cached = torch.cuda.memory_reserved(gpu_id) / 1024**3
                total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                return {
                    "allocated": allocated,
                    "cached": cached,
                    "total": total,
                    "free": total - allocated
                }
            return {}
        except ImportError:
            return {}

    @staticmethod
    def suggest_optimal_gpus(required_memory_gb: float = 8.0) -> List[int]:
        """
        필요한 메모리 양을 기준으로 최적의 GPU들을 추천합니다.
        
        Args:
            required_memory_gb: 필요한 메모리 양 (GB)
        
        Returns:
            추천 GPU 인덱스 리스트
        """
        available_gpus = ModelClientFactory.get_available_gpus()
        if not available_gpus:
            return []
        
        # 각 GPU의 사용 가능한 메모리 계산
        gpu_scores = []
        for gpu_id in available_gpus:
            memory_info = ModelClientFactory.get_gpu_memory_info(gpu_id)
            if memory_info:
                free_memory = memory_info["free"]
                if free_memory >= required_memory_gb:
                    # 여유 메모리가 많을수록 높은 점수
                    score = free_memory
                    gpu_scores.append((gpu_id, score))
        
        # 메모리 여유 공간 기준으로 정렬
        gpu_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 GPU들 반환 (최대 2개)
        return [gpu_id for gpu_id, _ in gpu_scores[:2]]

    @staticmethod
    def create_local_client_auto_gpu(model_name: str, **kwargs) -> LocalModelClient:
        """
        GPU를 자동으로 선택하여 로컬 클라이언트를 생성합니다.
        
        Args:
            model_name: 모델 이름
            **kwargs: 추가 설정
        """
        # 모델 크기에 따른 메모리 요구량 추정
        memory_requirements = {
            "3B": 6.0,
            "8B": 12.0,
            "32B": 48.0,
            "30B": 45.0
        }
        
        required_memory = 8.0  # 기본값
        for size, memory in memory_requirements.items():
            if size in model_name:
                required_memory = memory
                break
        
        # 최적 GPU 추천
        suggested_gpus = ModelClientFactory.suggest_optimal_gpus(required_memory)
        
        if suggested_gpus:
            print(f"🎯 모델 {model_name}에 대해 GPU {suggested_gpus} 추천 (예상 메모리: {required_memory}GB)")
            kwargs['gpus'] = suggested_gpus
        else:
            print("⚠️ 적절한 GPU를 찾을 수 없어 auto 모드로 설정합니다.")
        
        return ModelClientFactory.create_model_client("local", model_name, **kwargs)


# 하위 호환성을 위한 함수들 (기존 코드에서 사용하던 함수명 유지)
def create_model_client(client_type: str, model_name: str, **kwargs) -> BaseModelClient:
    """하위 호환성을 위한 래퍼 함수"""
    return ModelClientFactory.create_model_client(client_type, model_name, **kwargs)

def get_available_client_types() -> Dict[str, str]:
    """하위 호환성을 위한 래퍼 함수"""
    return ModelClientFactory.get_available_client_types()

def get_default_model_for_client(client_type: str) -> str:
    """하위 호환성을 위한 래퍼 함수"""
    return ModelClientFactory.get_default_model_for_client(client_type)

def create_default_client() -> BaseModelClient:
    """하위 호환성을 위한 래퍼 함수"""
    return ModelClientFactory.create_default_client()

def validate_client_config(client_type: str, model_name: str, **kwargs) -> Dict[str, Any]:
    """하위 호환성을 위한 래퍼 함수"""
    return ModelClientFactory.validate_client_config(client_type, model_name, **kwargs)

def create_openai_client(model_name: str = "gpt-4o-mini", api_key: Optional[str] = None, **kwargs) -> OpenAIModelClient:
    """하위 호환성을 위한 래퍼 함수"""
    return ModelClientFactory.create_openai_client(model_name, api_key, **kwargs)

def create_local_client(model_name: str = "Qwen3-8B", gpus: Optional[List[int]] = None, **kwargs) -> LocalModelClient:
    """하위 호환성을 위한 래퍼 함수"""
    return ModelClientFactory.create_local_client(model_name, gpus, **kwargs)

def get_available_gpus() -> List[int]:
    """하위 호환성을 위한 래퍼 함수"""
    return ModelClientFactory.get_available_gpus()

def get_gpu_memory_info(gpu_id: int) -> Dict[str, float]:
    """하위 호환성을 위한 래퍼 함수"""
    return ModelClientFactory.get_gpu_memory_info(gpu_id)

def suggest_optimal_gpus(required_memory_gb: float = 8.0) -> List[int]:
    """하위 호환성을 위한 래퍼 함수"""
    return ModelClientFactory.suggest_optimal_gpus(required_memory_gb)

def create_local_client_auto_gpu(model_name: str, **kwargs) -> LocalModelClient:
    """하위 호환성을 위한 래퍼 함수"""
    return ModelClientFactory.create_local_client_auto_gpu(model_name, **kwargs)

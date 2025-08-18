from __future__ import annotations
from typing import Optional
from src.modules.model_client import BaseModelClient

class TextGenerationDataSource:
    """텍스트 생성을 위한 데이터소스 - 모델 클라이언트를 래핑"""
    
    def __init__(self, model_client: BaseModelClient):
        self.model_client = model_client
    
    def generate(self, prompt: str) -> str:
        """
        주어진 프롬프트로 텍스트를 생성합니다.
        
        Args:
            prompt: 생성을 위한 프롬프트
            min_length: 최소 길이 (현재는 사용하지 않음)
            max_length: 최대 길이
            
        Returns:
            생성된 텍스트
        """
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.model_client.call(messages)
            return response.strip() if response else ""
        except Exception as e:
            print(f"텍스트 생성 중 오류 발생: {e}")
            return ""

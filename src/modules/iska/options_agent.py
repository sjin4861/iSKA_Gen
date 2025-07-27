import json
from typing import Optional, List, Dict
import sys
from pathlib import Path

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))

from src.utils.prompt_loader import get_prompt
from src.modules.model_client import BaseModelClient

class OptionsAgent:
    """
    주어진 지문과 문항(stem)에 따라 4지선다형 선택지를 생성하는 에이전트.
    """
    def __init__(self, llm_client: BaseModelClient):
        """
        클래스를 초기화하고, LLM 클라이언트를 설정합니다.

        Args:
            llm_client (BaseModelClient): API 호출을 담당하는 클라이언트.
        """
        self.llm_client = llm_client
        print("✅ 4지선다 선택지 생성 에이전트가 초기화되었습니다.")

    def generate_options(self, passage: str, stem: str, template_key: str = 'options_agent.fewshot') -> Optional[Dict]:
        """
        주어진 지문과 문항에 맞는 4지선다형 선택지를 생성합니다.

        Args:
            passage (str): RAG를 통해 생성된 비교 설명 지문.
            stem (str): 생성된 핵심 질문 (stem).
            template_key (str): 사용할 프롬프트 템플릿 키 (기본값: 'options_agent.fewshot')

        Returns:
            Optional[Dict]: {"options": [...], "answer_idx": ...} 형식의 딕셔너리.
        """
        print(f"\n✨ 4지선다 선택지 생성을 시작합니다...")
        
        # YAML 파일에서 프롬프트를 로드합니다.
        prompt = get_prompt(
            template_key, 
            agent='iska',
            passage=passage,
            stem=stem
        )
        
        try:
            messages = [{"role": "user", "content": prompt}]
            # LLM을 호출하여 JSON 형식의 선택지를 생성합니다.
            response_str = self.llm_client.call(
                messages=messages,
                temperature=0.5  # 일관된 형식을 위해 온도를 낮춤
            )
            
            # 응답 정리 및 JSON 파싱
            options_data = self._parse_options_response(response_str)
            
            if options_data:
                print("✅ 선택지 생성 성공!")
                return options_data
            else:
                print(f"❌ 선택지 파싱에 실패했습니다: {response_str}")
                return None

        except Exception as e:
            print(f"❌ 선택지 생성 중 오류 발생: {e}")
            return None

    def _parse_options_response(self, response_str: str) -> Optional[Dict]:
        """
        LLM 응답에서 선택지 정보를 파싱하는 함수
        """
        if not response_str or not response_str.strip():
            return None
            
        response_str = response_str.strip()
        
        try:
            # 1차 시도: 전체 응답을 JSON으로 파싱
            options_data = json.loads(response_str)
            if self._validate_options_data(options_data):
                return options_data
        except json.JSONDecodeError:
            pass
        
        try:
            # 2차 시도: JSON 패턴 찾기
            import re
            # {"options": [...], "answer_idx": ...} 패턴 찾기
            json_pattern = r'\{[^}]*"options"[^}]*"answer_idx"[^}]*\}'
            json_match = re.search(json_pattern, response_str, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                options_data = json.loads(json_str)
                if self._validate_options_data(options_data):
                    return options_data
        except (json.JSONDecodeError, AttributeError):
            pass
        
        try:
            # 3차 시도: 수동으로 options와 answer_idx 추출
            options = []
            answer_idx = None
            
            # options 배열 추출
            options_pattern = r'"options"\s*:\s*\[(.*?)\]'
            options_match = re.search(options_pattern, response_str, re.DOTALL)
            if options_match:
                options_str = options_match.group(1)
                # 각 선택지 추출
                option_pattern = r'"([^"]+)"'
                options = re.findall(option_pattern, options_str)
            
            # answer_idx 추출
            idx_pattern = r'"answer_idx"\s*:\s*(\d+)'
            idx_match = re.search(idx_pattern, response_str)
            if idx_match:
                answer_idx = int(idx_match.group(1))
            
            if options and len(options) >= 4 and answer_idx is not None:
                return {
                    "options": options[:4],  # 처음 4개만 사용
                    "answer_idx": answer_idx
                }
        except Exception:
            pass
        
        return None
    
    def _validate_options_data(self, data: Dict) -> bool:
        """
        선택지 데이터의 유효성을 검증하는 함수
        """
        if not isinstance(data, dict):
            return False
        
        if 'options' not in data or 'answer_idx' not in data:
            return False
        
        options = data['options']
        answer_idx = data['answer_idx']
        
        if not isinstance(options, list) or len(options) < 4:
            return False
        
        if not isinstance(answer_idx, int) or answer_idx < 0 or answer_idx >= len(options):
            return False
        
        return True

# --- 실행 예시 ---
if __name__ == "__main__":
    from src.modules.model_client import LocalModelClient
    
    # 로컬 모델을 사용하는 클라이언트 초기화
    llm_client = LocalModelClient(model_name="dummy_model")
    
    # 에이전트 인스턴스 생성
    options_agent = OptionsAgent(llm_client=llm_client)
    
    # 예시 지문 및 문항
    sample_passage = "사람들은 휴가철이 되면 일상을 떠나 어디론가 가고 싶어 한다. ... 그래서 요즘 혼자만의 휴가를 즐기는 사람이 늘고 있다. ... 혼자서라도 충분히 쉬는 것이 더 낫다고 여기는 것 같다."
    sample_stem = "글의 제목으로 가장 알맞은 것을 고르십시오."
    
    # 4지선다 선택지 생성
    generated_options = options_agent.generate_options(sample_passage, sample_stem)
    
    if generated_options:
        print("\n" + "="*50)
        print("      최종 생성된 4지선다 선택지")
        print("="*50)
        for i, option in enumerate(generated_options['options']):
            is_answer = " (정답)" if i == generated_options['answer_idx'] else ""
            print(f"  {i+1}. {option}{is_answer}")
        print("="*50)
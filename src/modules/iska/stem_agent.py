from typing import Optional
import sys
from pathlib import Path

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))

from src.utils.prompt_loader import get_prompt
from src.modules.model_client import BaseModelClient

class StemAgent:
    """
    주어진 지문과 과제 유형에 따라 최종 문항(stem)을 생성하는 에이전트.
    """
    def __init__(self, llm_client: BaseModelClient):
        """
        클래스를 초기화하고, LLM 클라이언트를 설정합니다.

        Args:
            llm_client (BaseModelClient): API 호출을 담당하는 클라이언트.
        """
        self.llm_client = llm_client
        print("✅ 문항(Stem) 생성 에이전트가 초기화되었습니다.")

    def generate_stem(self, passage: str, problem_type: str, eval_goal: str, template: str = 'stem_agent.few_shot') -> Optional[str]:
        """
        주어진 지문과 과제 유형에 맞는 문항(stem)을 생성합니다.

        Args:
            passage (str): RAG를 통해 생성된 비교 설명 지문.
            problem_type (str): 생성할 문항의 유형 (예: '자문화와 비교하기').
            eval_goal (str): 평가 목표 (예: '문화적 차이 이해 및 표현 능력 평가').
            template (str): 사용할 프롬프트 템플릿의 키.

        Returns:
            Optional[str]: 생성된 최종 문항(stem) 문자열.
        """
        print(f"\n✨ '{problem_type}' 유형의 문항 생성을 시작합니다...")
        
        # YAML 파일에서 해당 과제 유형에 맞는 프롬프트를 로드합니다.
        prompt = get_prompt(
            template, 
            agent='iska',
            passage=passage,
            problem_type=problem_type,
            eval_goal=eval_goal
        )
        
        try:
            messages = [{"role": "user", "content": prompt}]
            # LLM을 호출하여 문항을 생성합니다.
            stem = self.llm_client.call(messages=messages)
            print("✅ 최종 문항 생성 성공!")
            return stem.strip() if stem else None
        except Exception as e:
            print(f"❌ 최종 문항 생성 중 오류 발생: {e}")
            return None

# --- 실행 예시 ---
if __name__ == "__main__":
    from src.modules.model_client import LocalModelClient
    
    # 로컬 모델을 사용하는 클라이언트 초기화
    llm_client = LocalModelClient(model_name="dummy_model")
    
    # 에이전트 인스턴스 생성
    stem_agent = StemAgent(llm_client=llm_client)
    
    # RAG로 생성된 예시 지문
    sample_passage = "한국의 설날은 음력 1월 1일로, 가족들이 모여 차례를 지내고 떡국을 먹는 명절입니다. 반면, 서구권의 새해 첫날(New Year's Day)은 양력 1월 1일로, 주로 파티를 열거나 불꽃놀이를 보며 새해를 맞이하는 축제 분위기가 강합니다."
    
    # '문화 비교하기' 유형의 문항 생성
    problem_type = "자문화와 비교하기"
    eval_goal = "문화적 차이 이해 및 표현 능력 평가"
    final_stem = stem_agent.generate_stem(sample_passage, problem_type, eval_goal)
    
    print("\n" + "="*50)
    print("      최종 생성된 문항(Stem)")
    print("="*50)
    print(final_stem)
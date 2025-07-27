from typing import Optional, List
import sys
from pathlib import Path

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))

from src.utils.prompt_loader import get_prompt
from src.modules.model_client import BaseModelClient

class PassageAgent:
    """
    요약된 컨텍스트와 평가 목표를 바탕으로 최종 비교 지문을 생성하는 에이전트.
    """
    def __init__(self, llm_client: BaseModelClient):
        """
        클래스를 초기화하고, LLM 클라이언트를 설정합니다.
        """
        self.llm_client = llm_client
        print("✅ 지문 생성 에이전트가 초기화되었습니다.")

    def generate_passage(
        self,
        korean_topic: str,
        korean_context: str,
        foreign_topic: str,
        foreign_context: str,
        problem_types: List[str],
        eval_goals: List[str],
        template_key: str = 'passage_agent.create_passage'
    ) -> Optional[str]:
        """
        주어진 컨텍스트와 평가 목표에 맞는 최종 지문을 생성합니다.

        Args:
            korean_topic (str): 한국 문화 주제.
            korean_context (str): 한국 문화에 대한 요약 컨텍스트.
            foreign_topic (str): 외국 문화 주제.
            foreign_context (str): 외국 문화에 대한 요약 컨텍스트.
            problem_types (List[str]): 문제 유형 리스트 (3개).
            eval_goals (List[str]): 평가 목표 리스트 (3개).
            template_key (str): 사용할 프롬프트 템플릿 키 (기본값: 'passage_agent.create_passage')

        Returns:
            Optional[str]: 생성된 최종 지문.
        """
        if len(problem_types) < 3 or len(eval_goals) < 3:
            raise ValueError("문제 유형과 평가 목표는 각각 3개가 필요합니다.")

        print(f"\n✨ '{korean_topic}' vs '{foreign_topic}' 지문 생성을 시작합니다...")
        
        # 프롬프트에 전달할 인자 구성
        prompt_kwargs = {
            "korean_topic": korean_topic,
            "korean_context": korean_context,
            "foreign_topic": foreign_topic,
            "foreign_context": foreign_context,
            "eval_goal1": eval_goals[0],
            "eval_goal2": eval_goals[1],
            "eval_goal3": eval_goals[2],
            "problem_type1": problem_types[0],
            "problem_type2": problem_types[1],
            "problem_type3": problem_types[2],
        }
        
        # 프롬프트 로드 및 포맷팅
        prompt = get_prompt(
            template_key,
            agent='iska',
            **prompt_kwargs
        )
        
        try:
            # LLM 호출
            final_passage = self.llm_client.call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            if final_passage:
                # 제목 제거 후처리
                cleaned_passage = self._remove_title_from_passage(final_passage.strip())
                print("✅ 최종 지문 생성 성공!")
                return cleaned_passage
            else:
                return None
                
        except Exception as e:
            print(f"❌ 최종 지문 생성 중 오류 발생: {e}")
            return None

    def _remove_title_from_passage(self, passage: str) -> str:
        """
        지문에서 제목 부분을 제거하는 후처리 함수
        """
        import re
        
        # [지문]: 부분 제거
        if "[지문]:" in passage:
            passage = passage.split("[지문]:")[1].strip()
        
        # 제목 패턴들 제거
        title_patterns = [
            r'^\*\*제목:.*?\*\*\s*\n+',  # **제목: ...** 형태
            r'^제목:.*?\n+',              # 제목: ... 형태
            r'^\*\*.*?\*\*\s*\n+',       # **임의 제목** 형태
            r'^Title:.*?\n+',             # Title: ... 형태
            r'^#.*?\n+',                  # # 제목 형태
        ]
        
        for pattern in title_patterns:
            passage = re.sub(pattern, '', passage, flags=re.MULTILINE)
        
        # 앞뒤 공백 및 개행 정리
        passage = passage.strip()
        
        # 연속된 개행을 하나로 정리
        passage = re.sub(r'\n+', ' ', passage)
        
        return passage

# --- 실행 예시 ---
if __name__ == "__main__":
    from src.modules.model_client import LocalModelClient
    
    # 로컬 모델을 사용하는 클라이언트 초기화
    llm_client = LocalModelClient(model_name="dummy_model")
    
    # 에이전트 인스턴스 생성
    passage_agent = PassageAgent(llm_client=llm_client)
    
    # 이전 단계(ContextAgent)에서 생성되었다고 가정한 데이터
    k_topic = "단오"
    k_context = "단오는 음력 5월 5일로, 한국의 주요 명절 중 하나이며 씨름, 그네뛰기 등 공동체 놀이를 즐깁니다."
    f_topic = "Halloween"
    f_context = "Halloween is a holiday celebrated each year on October 31. The tradition originated with the ancient Celtic festival of Samhain."
    problem_types = ["제목과 이유 설명하기", "문화 비교하기", "의견 제시하기"]
    eval_goals = [
        "두 명절의 공통점과 차이점을 중심으로 글의 제목을 정하고 이유 설명하기",
        "각 명절에 즐기는 전통 놀이나 활동을 비교하여 설명하기", 
        "두 명절이 현대 사회에서 갖는 의미에 대한 자신의 생각 말하기"
    ]
    
    # 최종 지문 생성
    final_passage = passage_agent.generate_passage(
        korean_topic=k_topic,
        korean_context=k_context,
        foreign_topic=f_topic,
        foreign_context=f_context,
        problem_types=problem_types,
        eval_goals=eval_goals
    )
    
    print("\n" + "="*50)
    print("      최종 생성된 비교 설명 지문")
    print("="*50)
    print(final_passage)
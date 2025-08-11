from typing import Optional, List
import sys
from pathlib import Path
import re
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
        foreign_topic: str = None,
        foreign_context: str = None,
        problem_types: List[str] = None,
        eval_goals: List[str] = None,
        template_key: str = 'passage_agent.create_passage',
        topic: str = None,
        context: str = None
    ) -> Optional[str]:
        """
        주어진 컨텍스트와 평가 목표에 맞는 최종 지문을 생성합니다.
        대화형/domestic 템플릿의 경우 topic, context 파라미터를 사용합니다.

        Args:
            korean_topic (str): 한국 문화 주제.
            korean_context (str): 한국 문화에 대한 요약 컨텍스트.
            foreign_topic (str, optional): 외국 문화 주제.
            foreign_context (str, optional): 외국 문화에 대한 요약 컨텍스트.
            problem_types (List[str], optional): 문제 유형 리스트 (3개).
            eval_goals (List[str], optional): 평가 목표 리스트 (3개).
            template_key (str): 사용할 프롬프트 템플릿 키 (기본값: 'passage_agent.create_passage')
            topic (str, optional): 단일 주제 (대화형/domestic 템플릿용)
            context (str, optional): 단일 컨텍스트 (대화형/domestic 템플릿용)

        Returns:
            Optional[str]: 생성된 최종 지문.
        """
        # 템플릿 유형 확인 - 대화형이나 domestic 템플릿인지 판단
        is_domestic_template = ('domestic' in template_key or 
                               'dialogue' in template_key or
                               'violate_' in template_key and '_domestic' in template_key)
        
        if is_domestic_template:
            # 단일 주제 템플릿의 경우
            if not topic or not context:
                # topic/context가 제공되지 않았으면 korean_topic/korean_context를 사용
                topic = topic or korean_topic
                context = context or korean_context
            
            print(f"\n✨ '{topic}' 단일 주제 지문 생성을 시작합니다... (템플릿: {template_key})")
            
            # 단일 주제 템플릿용 프롬프트 인자 구성
            prompt_kwargs = {
                "topic": topic,
                "context": context,
            }
            
            # eval_goals와 problem_types가 제공된 경우 추가
            if eval_goals and len(eval_goals) >= 3:
                prompt_kwargs.update({
                    "eval_goal1": eval_goals[0],
                    "eval_goal2": eval_goals[1],
                    "eval_goal3": eval_goals[2],
                })
            
            if problem_types and len(problem_types) >= 3:
                prompt_kwargs.update({
                    "problem_type1": problem_types[0],
                    "problem_type2": problem_types[1],
                    "problem_type3": problem_types[2],
                })
        else:
            # 기존 비교 템플릿의 경우
            if len(problem_types) < 3 or len(eval_goals) < 3:
                raise ValueError("문제 유형과 평가 목표는 각각 3개가 필요합니다.")

            print(f"\n✨ '{korean_topic}' vs '{foreign_topic}' 지문 생성을 시작합니다...")
            
            # 비교 템플릿용 프롬프트 인자 구성
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
        
        # LLM 호출
        result = self._call_llm_with_prompt(prompt, 0.7)
        
        if result:
            # 제목 제거 후처리 (비교 지문의 경우에만)
            if not is_domestic_template:
                cleaned_passage = self._remove_title_from_passage(result)
                print("✅ 최종 지문 생성 성공!")
                return cleaned_passage
            else:
                print("✅ 최종 지문 생성 성공!")
                return result
        else:
            print("❌ 최종 지문 생성 실패")
            return None

    def _remove_title_from_passage(self, passage: str) -> str:
        """
        지문에서 제목 부분을 제거하는 후처리 함수
        """
        
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

    def _call_llm_with_prompt(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """
        LLM을 호출하여 결과를 반환하는 공통 메서드
        
        Args:
            prompt (str): LLM에 전달할 프롬프트
            temperature (float): LLM 호출 시 사용할 temperature 값
            
        Returns:
            Optional[str]: LLM 응답 결과
        """
        try:
            result = self.llm_client.call(
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
        
            result = re.sub(r'\(.*?\)', '', result)
            result = result.replace('**', '')
            # 여기에 \n\n이런 거 다 제거해줘.
            result = result.replace('\n\n', ' ')

            return result.strip() if result else None
        except Exception as e:
            print(f"❌ LLM 호출 중 오류 발생: {e}")
            return None

    def generate_image_caption_and_situation(
        self, 
        topic: str,
        template_key: str = 'passage_agent.create_image_caption_and_situation',
        temperature: float = 0.7
    ) -> Optional[str]:
        """
        보고 말하기 유형(ID 5)을 위한 이미지 캡션과 문제 상황 설명을 생성합니다.

        Args:
            topic (str): 주제 (예: "{쓰레기 분리배출}" 또는 "쓰레기 분리배출")
            template_key (str): 사용할 프롬프트 템플릿 키 (기본값: 'passage_agent.create_image_caption_and_situation')
            temperature (float): LLM 호출 시 사용할 temperature 값 (기본값: 0.7)

        Returns:
            Optional[str]: 생성된 이미지 설명과 문제 상황.
        """
        # 중괄호 제거
        clean_topic = topic.strip('{}')
        
        print(f"\n🖼️ '{clean_topic}' 주제의 이미지 캡션과 상황 설명 생성을 시작합니다... (템플릿: {template_key})")
        
        # 프롬프트 인자 구성
        prompt_kwargs = {
            "topic": clean_topic
        }
        
        # 프롬프트 로드 및 포맷팅
        prompt = get_prompt(
            template_key,
            agent='iska',
            **prompt_kwargs
        )
        
        # LLM 호출
        result = self._call_llm_with_prompt(prompt, temperature)
        
        if result:
            print("✅ 이미지 캡션과 상황 설명 생성 성공!")
            return result
        else:
            print("❌ 이미지 캡션과 상황 설명 생성 실패")
            return None

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
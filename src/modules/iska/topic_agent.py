import json
from typing import Optional, Dict, List
import sys
from pathlib import Path

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))

from src.utils.prompt_loader import get_prompt
from src.modules.model_client import BaseModelClient


class TopicAgent:
    """
    LLM을 사용하여 비교 연구를 위한 <한국 문화, 외국 문화> 토픽 쌍을 생성하는 에이전트.
    """
    def __init__(self, llm_client: BaseModelClient):
        """
        클래스를 초기화하고, LLM 클라이언트를 설정합니다.

        Args:
            llm_client (BaseModelClient): API 호출을 담당하는 클라이언트.
        """
        self.llm_client = llm_client
        print("✅ 토픽 생성 에이전트가 초기화되었습니다.")

    def generate_topic_pair(self, category: str, problem_types: List[str], eval_goals: List[str], template_key: str = 'topic_agent.generate_pair') -> Optional[Dict[str, str]]:
        """
        주어진 상위 카테고리에 맞는 문화 토픽 쌍을 생성합니다.

        Args:
            category (str): 토픽 생성을 위한 상위 카테고리 (예: '명절', '음식').
            problem_types (List[str]): 문제 유형 리스트 (3개)
            eval_goals (List[str]): 평가 목표 리스트 (3개)
            template_key (str): 사용할 프롬프트 템플릿 키 (기본값: 'topic_agent.generate_pair')

        Returns:
            Optional[Dict[str, str]]: {'korean_topic': '...', 'foreign_topic': '...'} 형식의 딕셔너리.
        """
        print(f"\n✨ '{category}' 카테고리에 대한 토픽 쌍 생성을 시작합니다...")
        
        # 평가 지침을 개별 파라미터로 전달
        prompt_kwargs = {
            'category': category,
            'problem_type1': problem_types[0] if len(problem_types) > 0 else '',
            'problem_type2': problem_types[1] if len(problem_types) > 1 else '',
            'problem_type3': problem_types[2] if len(problem_types) > 2 else '',
            'eval_goal1': eval_goals[0] if len(eval_goals) > 0 else '',
            'eval_goal2': eval_goals[1] if len(eval_goals) > 1 else '',
            'eval_goal3': eval_goals[2] if len(eval_goals) > 2 else '',
        }
        
        prompt = get_prompt(template_key, agent='iska', **prompt_kwargs)
        
        try:
            # LLM 호출
            response_str = self.llm_client.call(
                messages=[{"role": "user", "content": prompt}],
            )
            
            # 응답에서 불필요한 텍스트 제거 및 정리
            response_str = response_str.strip()
            
            # JSON 리스트 형태 찾기 ([...] 패턴)
            import re
            json_pattern = r'\[([^\]]+)\]'
            json_match = re.search(json_pattern, response_str)
            
            if json_match:
                json_content = json_match.group(0)
                try:
                    # JSON 파싱 시도
                    topic_list = json.loads(json_content)
                    if isinstance(topic_list, list) and len(topic_list) >= 2:
                        # 토픽에서 불필요한 공백과 따옴표 제거
                        korean_topic = str(topic_list[0]).strip().strip('"').strip("'")
                        foreign_topic = str(topic_list[1]).strip().strip('"').strip("'")
                        
                        topic_pair = {
                            'korean_topic': korean_topic,
                            'foreign_topic': foreign_topic
                        }
                        print(f"✅ 토픽 쌍 생성 성공: {topic_pair}")
                        return topic_pair
                except (json.JSONDecodeError, IndexError):
                    pass
            
            # JSON 파싱 실패 시 수동으로 파싱
            # 쉼표로 구분된 두 토픽 찾기
            if ',' in response_str:
                # 대괄호 제거 후 쉼표로 분할
                cleaned_text = response_str.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
                parts = [part.strip() for part in cleaned_text.split(',')]
                
                if len(parts) >= 2:
                    # 첫 번째와 두 번째 부분만 사용
                    korean_topic = parts[0].strip()
                    foreign_topic = parts[1].strip()
                    
                    # 불필요한 설명 제거 (콜론 뒤의 내용 제거)
                    if ':' in korean_topic:
                        korean_topic = korean_topic.split(':')[0].strip()
                    if ':' in foreign_topic:
                        foreign_topic = foreign_topic.split(':')[0].strip()
                    
                    topic_pair = {
                        'korean_topic': korean_topic,
                        'foreign_topic': foreign_topic
                    }
                    print(f"✅ 토픽 쌍 생성 성공 (수동 파싱): {topic_pair}")
                    return topic_pair
            
            print(f"❌ 토픽 쌍을 추출할 수 없습니다: {response_str}")
            return None

        except Exception as e:
            print(f"❌ LLM 호출 중 오류 발생: {e}")
            return None

# --- 실행 예시 ---
if __name__ == "__main__":
    from src.modules.model_client import LocalModelClient
    
    # 로컬 모델을 사용하는 클라이언트 초기화
    llm_client = LocalModelClient(model_name="dummy_model")
    
    # 에이전트 인스턴스 생성
    topic_agent = TopicAgent(llm_client=llm_client)
    
    # '전통 놀이' 카테고리에서 토픽 쌍 생성
    problem_types = ["문화 비교하기", "의견 제시하기", "원인 분석하기"]
    eval_goals = ["문화적 차이 이해", "개인적 견해 표현", "현상 원인 분석"]
    topic_pair = topic_agent.generate_topic_pair("전통 놀이", problem_types, eval_goals)
    
    if topic_pair:
        print("\n" + "="*50)
        print("      최종 생성된 토픽 쌍")
        print("="*50)
        print(f"  - 한국 문화: {topic_pair['korean_topic']}")
        print(f"  - 외국 문화: {topic_pair['foreign_topic']}")
        print("="*50)
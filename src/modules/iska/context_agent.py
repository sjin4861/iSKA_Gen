import wikipediaapi
import json
from typing import Optional, List, Dict
import sys
from pathlib import Path

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))

from src.utils.prompt_loader import get_prompt
from src.modules.model_client import BaseModelClient

class ContextAgent:
    """
    위키피디아 검색을 통해 정보를 수집하고, LLM을 이용해 컨텍스트를 생성하는 에이전트.
    """
    def __init__(self, user_agent: str, llm_client: BaseModelClient):
        self.user_agent = user_agent
        self.llm_client = llm_client
        self.wiki_ko = wikipediaapi.Wikipedia(user_agent, 'ko')
        self.wiki_en = wikipediaapi.Wikipedia(user_agent, 'en')
        print("✅ 컨텍스트 생성 에이전트가 초기화되었습니다.")

    def _get_wiki_documents(self, topic: str, lang: str, top_k: int = 5) -> str:
        """주어진 토픽으로 위키피디아를 검색하여, 상위 K개 문서 내용을 합쳐서 반환합니다."""
        print(f"  - 위키피디아({lang})에서 '{topic}' 관련 문서를 {top_k}개 검색합니다...")
        wiki = self.wiki_ko if lang == 'ko' else self.wiki_en
        main_page = wiki.page(topic)
        if not main_page.exists():
            print(f"  - ❌ '{topic}'({lang}) 메인 페이지가 없습니다.")
            return ""
        
        candidate_pages = list(main_page.links.values())[:top_k-1]
        all_pages = [main_page] + candidate_pages
        
        documents = "\n\n---\n\n".join([
            f"문서 제목: {p.title}\n내용: {p.summary[:500]}..." 
            for p in all_pages if p.summary
        ])
        print(f"  - ✅ {len(all_pages)}개 문서에서 컨텍스트를 추출했습니다.")
        return documents

    def _synthesize_context(
        self, topic: str, template_key: str, documents: List[Dict[str, str]], 
        problem_types: List[str], eval_goals: List[str]
    ) -> Optional[str]:
        """LLM을 호출하여 컨텍스트를 요약/합성합니다."""
        if not documents.strip():
            print("  - ❌ 참고할 문서가 없어 요약을 건너뜁니다.")
            return None
            
        print("  - 수집된 정보를 바탕으로 LLM에게 최종 컨텍스트 생성을 요청합니다...")
        
        # 평가 지침을 개별 파라미터로 전달
        prompt_key = template_key
        
        # 개별 매개변수로 변환
        prompt_kwargs = {
            'retrieved_documents': documents
        }
        
        # problem_types와 eval_goals를 개별 매개변수로 전환
        for i, problem_type in enumerate(problem_types[:3], 1):
            prompt_kwargs[f'problem_type{i}'] = problem_type
        for i, eval_goal in enumerate(eval_goals[:3], 1):
            prompt_kwargs[f'eval_goal{i}'] = eval_goal
        
        # home_context vs foreign_context 구분하여 토픽 매개변수 설정
        if 'home_context' in prompt_key:
            prompt_kwargs['home_topic'] = topic
        else:
            prompt_kwargs['foreign_topic'] = topic
        
        prompt = get_prompt(prompt_key, agent='iska', **prompt_kwargs)
        
        try:
            context = self.llm_client.call(
                messages=[{"role": "user", "content": prompt}], 
            )
            return context.strip() if context else None
        except Exception as e:
            print(f"  - ❌ LLM 요약 중 오류 발생: {e}")
            return None

    def generate_home_context(self, korean_topic: str, problem_types: List[str], eval_goals: List[str], template_key: str = 'context_agent.home_context') -> Optional[str]:
        """주어진 한국 문화 토픽에 대한 home_context를 생성합니다."""
        print(f"\n✨ '{korean_topic}'에 대한 Home Context 생성을 시작합니다...")
        documents = self._get_wiki_documents(korean_topic, 'ko', top_k=5)
        home_context = self._synthesize_context(
            korean_topic, template_key, documents, problem_types, eval_goals
        )
        if home_context:
            print("✅ Home Context 생성 성공!")
        return home_context

    def generate_foreign_context(self, foreign_topic: str, problem_types: List[str], eval_goals: List[str], template_key: str = 'context_agent.foreign_context') -> Optional[str]:
        """주어진 외국 문화 토픽에 대한 foreign_context를 생성합니다."""
        print(f"\n✨ '{foreign_topic}'에 대한 Foreign Context 생성을 시작합니다...")
        documents = self._get_wiki_documents(foreign_topic, 'en', top_k=5)
        foreign_context = self._synthesize_context(
            foreign_topic, template_key, documents, problem_types, eval_goals
        )
        if foreign_context:
            print("✅ Foreign Context 생성 성공!")
        return foreign_context

# --- 실행 예시 ---
if __name__ == "__main__":
    from src.modules.model_client import LocalModelClient
    
    USER_AGENT = "iSKA_Project/1.0 (sjun24530@gmail.com)"
    llm_client = LocalModelClient(model_name="dummy_model") # 실제 LLM 클라이언트로 교체
    
    agent = ContextAgent(user_agent=USER_AGENT, llm_client=llm_client)
    
    # 문제 유형과 평가 목표
    problem_types = ["문화 비교하기", "의견 제시하기", "배경 설명하기"]
    eval_goals = [
        "두 문화의 공통점과 차이점 비교하기",
        "문화적 배경과 의미 설명하기",
        "개인적 견해 표현하기"
    ]
    
    # home_context와 foreign_context를 각각 생성
    home_context = agent.generate_home_context("단오", problem_types, eval_goals)
    foreign_context = agent.generate_foreign_context("Halloween", problem_types, eval_goals)
    
    print("\n" + "="*50)
    print("      최종 생성된 Context 결과")
    print("="*50)
    print(f"[Home Context]\n{home_context}")
    print("\n" + "-"*50)
    print(f"[Foreign Context]\n{foreign_context}")
    print("="*50)
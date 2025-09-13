# src/modules/content_chain.py
"""콘텐츠 생성 체인 (LangChain Runnable)

- get_prompt 재사용
- domestic/dialogue 여부 분기
- 비-domestic일 때 제목 제거 후처리
- 입력 dict에 llm(BaseChatModel 호환)과 템플릿 변수들을 넣어 .invoke 로 호출
"""
from __future__ import annotations
from typing import Dict, Any, List
import re

from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.prompt_loader import get_prompt

_TITLE_PATTERNS = [
    r'^\*\*제목:.*?\*\*\s*\n+',
    r'^제목:.*?\n+',
    r'^\*\*.*?\*\*\s*\n+',
    r'^Title:.*?\n+',
    r'^#.*?\n+',
]

def _is_domestic(template_key: str) -> bool:
    return (
        'domestic' in template_key
        or 'dialogue' in template_key
        or ('violate_' in template_key and '_domestic' in template_key)
    )

def _remove_title(text: str) -> str:
    if '[지문]:' in text:
        text = text.split('[지문]:', 1)[1].strip()
    for pat in _TITLE_PATTERNS:
        text = re.sub(pat, '', text, flags=re.MULTILINE)
    text = re.sub(r'\n+', ' ', text).strip()
    text = text.replace('**', '')
    text = re.sub(r'\(.*?\)', '', text)
    return text.strip()

def _build_prompt_args(template_key: str, vars: Dict[str, Any]) -> Dict[str, Any]:
    domestic = _is_domestic(template_key)
    if domestic:
        topic = vars.get('topic') or vars.get('korean_topic')
        context = vars.get('context') or vars.get('korean_context')
        prompt_kwargs: Dict[str, Any] = {"topic": topic, "context": context}

        eval_goals: List[str] = vars.get('eval_goals') or []
        problem_types: List[str] = vars.get('problem_types') or []

        if len(eval_goals) >= 3:
            prompt_kwargs.update({
                'eval_goal1': eval_goals[0],
                'eval_goal2': eval_goals[1],
                'eval_goal3': eval_goals[2],
            })
        if len(problem_types) >= 3:
            prompt_kwargs.update({
                'problem_type1': problem_types[0],
                'problem_type2': problem_types[1],
                'problem_type3': problem_types[2],
            })
    else:
        problem_types = vars.get('problem_types') or []
        eval_goals = vars.get('eval_goals') or []
        if len(problem_types) < 3 or len(eval_goals) < 3:
            raise ValueError('problem_types / eval_goals 최소 3개 필요')
        prompt_kwargs = {
            'korean_topic': vars['korean_topic'],
            'korean_context': vars['korean_context'],
            'foreign_topic': vars.get('foreign_topic'),
            'foreign_context': vars.get('foreign_context'),
            'eval_goal1': eval_goals[0],
            'eval_goal2': eval_goals[1],
            'eval_goal3': eval_goals[2],
            'problem_type1': problem_types[0],
            'problem_type2': problem_types[1],
            'problem_type3': problem_types[2],
        }
    return prompt_kwargs

def build_content_chain(template_key: str):
    """입력(dict) 예시
    - 공통: llm(BaseChatModel), template 변수들
    - domestic/dialogue:
        topic, context, (선택) eval_goals: List[str], problem_types: List[str]
    - 비교형:
        korean_topic, korean_context, foreign_topic, foreign_context,
        eval_goals: List[str], problem_types: List[str]
    """
    domestic = _is_domestic(template_key)

    def build_prompt(vars: Dict[str, Any]) -> Dict[str, Any]:
        llm: BaseChatModel = vars['llm']
        prompt_kwargs = _build_prompt_args(template_key, vars)
        prompt = get_prompt(template_key, agent='iska', **prompt_kwargs)
        return {'prompt': prompt, 'llm': llm}

    def to_messages(d: Dict[str, Any]):
        return [{"role": "user", "content": d['prompt']}]

    def run_llm(d: Dict[str, Any]) -> str:
        llm: BaseChatModel = d['llm']
        msgs = to_messages(d)
        return llm.invoke(msgs).content

    def post(text: str) -> str:
        if not domestic:
            return _remove_title(text)
        return text

    chain = (
        RunnableLambda(build_prompt)
        | RunnableLambda(run_llm)
        | RunnableLambda(post)
        | RunnableLambda(lambda x: x.strip())
    )
    return chain

__all__ = ["build_content_chain"]

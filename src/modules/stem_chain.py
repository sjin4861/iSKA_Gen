# src/modules/stem_chain.py
"""LangChain 기반 Stem 생성 체인"""

from __future__ import annotations
from typing import Dict, Any

from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.prompt_loader import get_prompt


def build_stem_chain(template_key: str = "stem_agent.few_shot"):
    """문항(stem) 생성 Runnable.

    입력 dict 필드:
      - passage: str                 # 지문
      - problem_type: str            # 문항 유형
      - eval_goal: str               # 평가 목표
      - llm: BaseChatModel           # ChatOpenAI 호환 LangChain LLM
      - (선택) k: int                # 생성 개수 힌트(템플릿에서 사용할 때만 전달됨)

    출력:
      - 모델 원문 출력 문자열 (전처리: strip 만 적용)
    """

    def build_prompt(vars: Dict[str, Any]) -> Dict[str, Any]:
        llm: BaseChatModel = vars["llm"]
        passage_text = vars["passage"]

        prompt_kwargs: Dict[str, Any] = {
            "agent": "iska",
            "content": passage_text,            # 템플릿은 {content}를 기대
            "problem_type": vars["problem_type"],
            "eval_goal": vars["eval_goal"],
        }
        # 템플릿에서 k를 사용할 경우에만 전달 (없으면 넣지 않음)
        if "k" in vars and vars["k"] is not None:
            prompt_kwargs["k"] = int(vars["k"])

        prompt = get_prompt(template_key, **prompt_kwargs)
        return {"prompt": prompt, "llm": llm}

    def to_messages(d: Dict[str, Any]):
        return [{"role": "user", "content": d["prompt"]}]

    def run_llm(d: Dict[str, Any]) -> str:
        llm: BaseChatModel = d["llm"]
        messages = to_messages(d)
        return llm.invoke(messages).content

    chain = (
        RunnableLambda(build_prompt)
        | RunnableLambda(run_llm)
        | RunnableLambda(lambda x: x.strip())
    )
    return chain


__all__ = ["build_stem_chain"]

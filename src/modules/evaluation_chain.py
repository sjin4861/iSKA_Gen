# src/modules/evaluation_chain.py
"""루브릭 기반 평가 체인 (LangChain Runnable)

- rubric_evaluation.yaml 내 프롬프트를 사용
- rubric_id를 지정하면 해당 평가 프롬프트 체인을 반환
- 입력: { "llm": BaseChatModel, "passage": str, ... (rubric 프롬프트에 필요한 변수) }
- 출력: 모델의 원문 응답 문자열 (strip 처리)
"""

from __future__ import annotations
from typing import Dict, Any

from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel

from src.utils.prompt_loader import get_prompt


def build_evaluation_chain(rubric_id: str):
    """주어진 rubric_id에 맞는 평가 체인을 생성한다.

    Args:
        rubric_id: rubric_evaluation.yaml 내 key
            예: "completeness_for_guidelines", "core_theme_clarity",
                "reference_groundedness", "logical_flow_and_structure",
                "korean_quality", "l2_learner_suitability"

    Returns:
        Runnable 체인
    """

    def build_prompt(vars: Dict[str, Any]) -> Dict[str, Any]:
        llm: BaseChatModel = vars["llm"]
        # rubric 프롬프트 키 = "rubric_evaluation.{rubric_id}"
        prompt = get_prompt(f"rubric_evaluation.{rubric_id}", agent="iska", **vars)
        return {"prompt": prompt, "llm": llm}

    def to_messages(d: Dict[str, Any]):
        return [{"role": "user", "content": d["prompt"]}]

    def run_llm(d: Dict[str, Any]) -> str:
        llm: BaseChatModel = d["llm"]
        msgs = to_messages(d)
        return llm.invoke(msgs).content

    chain = (
        RunnableLambda(build_prompt)
        | RunnableLambda(run_llm)
        | RunnableLambda(lambda x: x.strip())
    )
    return chain


__all__ = ["build_evaluation_chain"]

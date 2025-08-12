# src/data/repositories/passage_repository_impl.py
from __future__ import annotations
from typing import List, Dict, Any, Optional  # ✅ Any 임포트 추가!

from src.domain.repositories.llm_gateway import LLMGateway
from src.domain.repositories.passage_repository import PassageRepository
from src.data.datasources.fs.content_store import ContentFSStore
from src.data.repositories.llm_gateway_impl import LLMGatewayImpl  # infra 구현

from src.utils.prompt_loader import get_prompt
import re

def _strip_and_squash(text: str) -> str:
    if not text:
        return ""
    text = text.replace("**", "")
    text = text.replace("[지문]:", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

class PassageRepositoryImpl(PassageRepository):
    """
    FS(ContentFSStore) + LLMGateway 조합.
    - 도메인에선 LLM 백엔드를 몰라도 됨 (OpenAI/Local/vLLM 교체 용이).
    """

    def __init__(
        self,
        fs: Optional[ContentFSStore] = None,
        llm: Optional[LLMGateway] = None,
        *,
        # llm 미주입 시 내부에서 기본 게이트웨이 생성에 쓰일 옵션
        client_type: str = "local",
        model_name: str = "EXAONE-3.5-7.8B-Instruct",
        gpus: Optional[List[int]] = None,
        default_llm_params: Optional[Dict[str, Any]] = None,
        **client_kwargs: Any,
    ):
        self.fs = fs or ContentFSStore()

        # ✅ 모호성 가드: llm이 주입되었는데 또 빌더 옵션도 들어오면 경고/예외
        if llm is not None and (client_type or model_name or gpus or default_llm_params or client_kwargs):
            # 여기서는 조용히 무시해도 되지만, 디버깅 편의를 위해 명시적으로 막는 쪽 권장
            # raise ValueError("llm 인스턴스가 주입된 경우 client_type/model_name 등 빌더 옵션을 함께 전달하지 마세요.")
            pass

        self.llm: LLMGateway = llm or LLMGatewayImpl(
            client_type=client_type,
            model_name=model_name,
            default_params=default_llm_params or {"temperature": 0.7},
            gpus=gpus or [0],
            **client_kwargs,
        )

    # ---------- 내부: 프롬프트 구성 ----------
    def _build_prompt(
        self,
        *,
        template_key: str,
        source: Dict[str, Any],
        problem_types: List[str],
        eval_goals: List[str],
    ) -> str:
        is_domestic_like = any(k in template_key for k in ("domestic", "dialogue"))
        if is_domestic_like:
            topic = source.get("topic") or source.get("korean_topic") or ""
            context = source.get("context") or source.get("korean_context") or ""
            prompt = get_prompt(
                template_key, agent="iska",
                topic=topic, context=context,
                problem_type1=problem_types[0] if len(problem_types) > 0 else "",
                problem_type2=problem_types[1] if len(problem_types) > 1 else "",
                problem_type3=problem_types[2] if len(problem_types) > 2 else "",
                eval_goal1=eval_goals[0] if len(eval_goals) > 0 else "",
                eval_goal2=eval_goals[1] if len(eval_goals) > 1 else "",
                eval_goal3=eval_goals[2] if len(eval_goals) > 2 else "",
            )
        else:
            prompt = get_prompt(
                template_key, agent="iska",
                korean_topic=source.get("korean_topic", ""),
                korean_context=source.get("korean_context", ""),
                foreign_topic=source.get("foreign_topic", ""),
                foreign_context=source.get("foreign_context", ""),
                problem_type1=problem_types[0] if len(problem_types) > 0 else "",
                problem_type2=problem_types[1] if len(problem_types) > 1 else "",
                problem_type3=problem_types[2] if len(problem_types) > 2 else "",
                eval_goal1=eval_goals[0] if len(eval_goals) > 0 else "",
                eval_goal2=eval_goals[1] if len(eval_goals) > 1 else "",
                eval_goal3=eval_goals[2] if len(eval_goals) > 2 else "",
            )
        return prompt

    def _call_llm_once(
        self,
        *,
        template_key: str,
        source: Dict[str, Any],
        problem_types: List[str],
        eval_goals: List[str],
        gen_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        prompt = self._build_prompt(
            template_key=template_key, source=source,
            problem_types=problem_types, eval_goals=eval_goals,
        )
        messages = [{"role": "user", "content": prompt}]
        raw = self.llm.generate(messages, **(gen_params or {}))
        return _strip_and_squash(raw)

    # ---------- 공개 API ----------
    def generate_one(
        self,
        *,
        source: Dict[str, Any],
        problem_types: List[str],
        eval_goals: List[str],
        model_name: str,          # 인터페이스 호환용(지금은 내부 사용 X)
        template_key: str,
        min_length: int,
        max_length: int,
        max_retries: int,
    ) -> Optional[str]:
        tries = 0
        while tries < max_retries:
            tries += 1
            out = self._call_llm_once(
                template_key=template_key,
                source=source,
                problem_types=problem_types,
                eval_goals=eval_goals,
                gen_params=None,
            )
            if not out:
                continue
            n = len(out)
            if n < min_length or n > max_length:
                continue
            return out
        return None

    def generate_and_fill_missing(
        self,
        *,
        model_name: str,
        template_key: str,
        benchmark_id: int,
        benchmark_version: str,
        problem_types: List[str],
        eval_goals: List[str],
        sources: List[Dict[str, Any]],
        date_str: Optional[str],
        min_length: int,
        max_length: int,
        max_retries: int,
    ) -> dict:
        rows = self.fs.load_passage_list(
            model_name, benchmark_id, benchmark_version, template_key, date_str
        ) or []
        null_idxs = self.fs.find_null_indices(rows)

        patch: Dict[int, Dict[str, Any]] = {}
        filled: List[int] = []
        failed: List[int] = []

        for idx in null_idxs:
            src = sources[idx]
            text = self.generate_one(
                source=src,
                problem_types=problem_types,
                eval_goals=eval_goals,
                model_name=model_name,
                template_key=template_key,
                min_length=min_length,
                max_length=max_length,
                max_retries=max_retries,
            )
            if text:
                if "korean_topic" in src:
                    si = {
                        "topic": src.get("korean_topic", ""),
                        "context": src.get("korean_context", ""),
                        "foreign_topic": src.get("foreign_topic"),
                        "foreign_context": src.get("foreign_context"),
                    }
                else:
                    si = {"topic": src.get("topic", ""), "context": src.get("context", "")}
                patch[idx] = {"source_item": si, "generated_passage": text}
                filled.append(idx)
            else:
                failed.append(idx)

        if patch:
            self.fs.patch_by_indices(
                model_name, benchmark_id, benchmark_version, template_key, patch, date_str
            )

        final = self.fs.load_passage_list(
            model_name, benchmark_id, benchmark_version, template_key, date_str
        ) or []
        return {"filled": filled, "failed": failed, "total": len(final)}

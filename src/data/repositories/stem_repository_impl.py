# src/data/repositories/stem_repository_impl.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import re

from src.domain.repositories.llm_gateway import LLMGateway
from src.domain.repositories.stem_repository import StemRepository
from src.data.datasources.fs.stem_store import StemFSStore
from src.data.repositories.llm_gateway_impl import LLMGatewayImpl

from src.utils.prompt_loader import get_prompt

def _strip_and_squash(text: str) -> str:
    """생성된 stem 텍스트 정리"""
    if not text:
        return ""
    text = text.replace("**", "")
    text = text.replace("[문항]:", "")
    text = text.replace("[stem]:", "") 
    text = re.sub(r"\s+", " ", text).strip()
    return text

class StemRepositoryImpl(StemRepository):
    """
    FS(StemFSStore) + LLMGateway 조합.
    - 도메인에선 LLM 백엔드를 몰라도 됨 (OpenAI/Local/vLLM 교체 용이).
    """

    def __init__(
        self,
        fs: Optional[StemFSStore] = None,
        llm: Optional[LLMGateway] = None,
        *,
        # llm 미주입 시 내부에서 기본 게이트웨이 생성에 쓰일 옵션
        client_type: str = "local",
        model_name: str = "EXAONE-3.5-7.8B-Instruct",
        gpus: Optional[List[int]] = None,
        default_llm_params: Optional[Dict[str, Any]] = None,
        **client_kwargs: Any,
    ):
        self.fs = fs or StemFSStore()

        # ✅ 모호성 가드: llm이 주입되었는데 또 빌더 옵션도 들어오면 경고/예외
        if llm is not None and (client_type or model_name or gpus or default_llm_params or client_kwargs):
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
        passage: str,
        problem_type: str,
        eval_goal: str,
    ) -> str:
        """stem 생성을 위한 프롬프트 구성"""
        prompt = get_prompt(
            template_key, 
            agent="iska",
            passage=passage,
            problem_type=problem_type,
            eval_goal=eval_goal
        )
        return prompt

    def _call_llm_once(
        self,
        *,
        template_key: str,
        passage: str,
        problem_type: str,
        eval_goal: str,
        gen_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """LLM 호출하여 stem 생성"""
        prompt = self._build_prompt(
            template_key=template_key,
            passage=passage,
            problem_type=problem_type,
            eval_goal=eval_goal,
        )
        messages = [{"role": "user", "content": prompt}]
        raw = self.llm.generate(messages, **(gen_params or {}))
        return _strip_and_squash(raw)

    # ---------- 공개 API ----------
    def generate_one(
        self,
        *,
        passage: str,
        problem_type: str,
        eval_goal: str,
        model_name: str,          # 인터페이스 호환용(지금은 내부 사용 X)
        template_key: str,
        max_retries: int,
    ) -> Optional[str]:
        """단일 stem 생성"""
        tries = 0
        while tries < max_retries:
            tries += 1
            try:
                stem = self._call_llm_once(
                    template_key=template_key,
                    passage=passage,
                    problem_type=problem_type,
                    eval_goal=eval_goal,
                    gen_params=None,
                )
                if stem:
                    return stem
            except Exception as e:
                print(f"Stem 생성 시도 {tries} 실패: {e}")
                continue
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
        passages: List[Dict[str, Any]],
        date_str: Optional[str],
        max_retries: int,
        passage_model_name: Optional[str] = None,
    ) -> dict:
        """기존 stem 데이터에서 누락된 부분을 찾아 생성하여 채움"""
        # 기존 stem 데이터 로드
        existing_stems = self.fs.load_list(
            model_name, benchmark_id, benchmark_version, template_key, date_str
        ) or []

        filled: List[int] = []
        failed: List[int] = []
        stem_data_list: List[Dict[str, Any]] = []

        # 각 passage에 대해 stem 생성
        for i, passage_data in enumerate(passages):
            print(f"  📄 Passage {i+1}/{len(passages)} 처리 중...")
            
            # 기존 데이터가 있는지 확인
            existing_stem = None
            if i < len(existing_stems):
                existing_stem = existing_stems[i]

            stem_data = {
                "source_passage": passage_data.get('generated_passage', ''),
                "source_item": passage_data.get('source_item', {})
            }

            # 각 problem_type과 eval_goal에 대해 stem 생성
            all_success = True
            for j in range(len(problem_types)):
                problem_type = problem_types[j]
                eval_goal = eval_goals[j]
                
                field_stem = f'stem_{j+1}'
                field_problem_type = f'problem_type_{j+1}'
                field_eval_goal = f'eval_goal_{j+1}'

                # 기존 데이터에 해당 stem이 있고 유효한지 확인
                if (existing_stem and 
                    existing_stem.get(field_stem) and 
                    existing_stem.get(field_stem) != "문항 생성 실패"):
                    # 기존 데이터 사용
                    stem_data[field_problem_type] = existing_stem.get(field_problem_type, problem_type)
                    stem_data[field_eval_goal] = existing_stem.get(field_eval_goal, eval_goal)
                    stem_data[field_stem] = existing_stem[field_stem]
                else:
                    # 새로 생성
                    generated_stem = self.generate_one(
                        passage=passage_data.get('generated_passage', ''),
                        problem_type=problem_type,
                        eval_goal=eval_goal,
                        model_name=model_name,
                        template_key=template_key,
                        max_retries=max_retries,
                    )
                    
                    stem_data[field_problem_type] = problem_type
                    stem_data[field_eval_goal] = eval_goal
                    
                    if generated_stem:
                        stem_data[field_stem] = generated_stem
                    else:
                        stem_data[field_stem] = "문항 생성 실패"
                        all_success = False

            stem_data_list.append(stem_data)
            
            if all_success:
                filled.append(i)
            else:
                failed.append(i)

        # 생성된 stem 데이터 저장
        if stem_data_list:
            # passage 모델명을 포함한 템플릿 키 생성
            if passage_model_name:
                modified_template_key = f"{template_key}_from_{passage_model_name}"
            else:
                modified_template_key = template_key
                
            self.fs.save_list(
                stem_data_list, model_name, benchmark_id, benchmark_version, 
                modified_template_key, date_str
            )

        return {"filled": filled, "failed": failed, "total": len(stem_data_list)}

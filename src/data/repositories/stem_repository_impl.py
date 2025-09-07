# src/data/repositories/stem_repository_impl.py
from __future__ import annotations
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

from src.domain.entities import sources
from src.domain.entities.enums import ContentType
from src.domain.entities.outputs import CandidateOutput
from src.domain.repositories.stem_repository import StemRepository
from src.data.datasources.fs.stem_store import StemStoreFSDataSource

from src.data.datasources.fs.raw_output_fs import RawOutputFSDataSource
from src.data.datasources.fs.data_store_fs import DataStoreFSDataSource
from src.data.datasources.fs.templates_fs import TemplatesFSDataSource
from src.data.datasources.fs.text_generation import TextGenerationDataSource
from src.utils.prompt_loader import get_prompt

def _strip_and_squash(text: str) -> str:
    """생성된 stem 텍스트 정리"""
    if not text:
        return ""
    # 불필요 포맷 제거
    text = text.replace("**", "")
    text = text.replace("[문항]:", "")
    text = text.replace("[stem]:", "")
    text = text.replace("[출력]:", "")
    text = text.replace("출력:", "")
    # 공백 정규화
    text = re.sub(r"\s+", " ", text).strip()
    return text

class StemRepositoryImpl(StemRepository):
    """
    - PassageRepositoryImpl과 동일한 DI/인터페이스 스타일
      * templates_ds.get(...) 으로 프롬프트 구성
      * textgen_ds.generate(...) 로 생성
      * raw_outputs/stem/... 구조(StemStoreFSDataSource)에 JSON 리스트 저장
    """

    def __init__(
        self,
        raw_output_ds: RawOutputFSDataSource,
        data_store_ds: DataStoreFSDataSource,
        templates_ds: TemplatesFSDataSource,
        textgen_ds: TextGenerationDataSource,
    ):
        self.raw_output = raw_output_ds
        self.store = data_store_ds
        self.templates = templates_ds
        self.textgen = textgen_ds

    # --- 단일 생성 ---
    def generate_one(
        self,
        *,
        content: str,
        problem_type: str,
        eval_goal: str,
        model_name: str,       # 인터페이스 정합성 유지(내부 사용 안 함)
        template_key: str,
        max_retries: int,
    ) -> Optional[str]:
        """
        템플릿 키 예:
          - stem_agent.few_shot_new (권장)
          - stem_agent.basic / stem_agent.few_shot / stem_agent.unified_fewshot
        """
        # 프롬프트 구성
        prompt = self.templates.get(
            template_key,
            content=content,
            problem_type=problem_type,
            eval_goal=eval_goal,
        )

        # 생성 시도
        for i in range(max_retries):
            try:
                out = self.textgen.generate(prompt)
                out = _strip_and_squash(out)
                if out:
                    return out
            except Exception as e:
                print(f"  ⚠️ stem generate_one 실패(시도 {i+1}/{max_retries}): {e}")
                continue
        return None

    # --- 일괄 생성/보강 ---
    def generate_and_fill_missing(
        self,
        *,
        model_name: str,
        template_key: str,
        benchmark_id: int,
        benchmark_version: str,
        problem_types: List[str],
        eval_goals: List[str],
        contents: List[Dict[str, Any]],
        date_str: Optional[str],
        max_retries: int,
        content_model_name: Optional[str] = None,
    ) -> dict:
        """
        contents 원소 예:
          {
            "generated_content": "...",      # 필수
            "source_item": {...},            # 선택(메타)
            "source_id": "bench_1_item_0"    # 선택(있으면 보존)
          }

        저장 경로는 기존 파이프라인과 동일:
          data_store/raw_outputs/{date}/stem/{model}/benchmark_{id}_v{ver}_{template_key}.json
          * content_model_name가 주어지면 template_key 뒤에 `_from_{content_model_name}`를 덧붙여 동일 키로 로드/세이브
        """
        # 로드/세이브 키를 통일(중요!)
        filled, failed = [], []
        for i, row in enumerate(contents):
            
            print(f"  📄 Passage {i+1}/{len(contents)} 처리 중...")

            src_content = row.get("content")
            src_item = row.get("source_item")
            src_id = row.get("source_id")

            stems = []
            all_pass = True
            for j in range(3):
                pt = problem_types[j]
                eg = eval_goals[j]

                # 새로 생성
                gen = self.generate_one(
                    content=src_content,
                    problem_type=pt,
                    eval_goal=eg,
                    model_name=model_name,
                    template_key=template_key,  # 프롬프트 키는 원본 사용
                    max_retries=max_retries,
                )
                if gen:
                    stems.append(gen)
                else:
                    failed.append(src_id)
                    stems.append("문항 생성 실패")
                    all_pass = False
                    break
            if all_pass:
                c = CandidateOutput(source_id=src_id,
                                benchmark_id=benchmark_id,
                                model_name=content_model_name,
                                candidate_id=f"bench_{benchmark_id}_item_{i}",
                                content_type=ContentType.stem,
                                content=src_content,
                                stems=stems,
                                generated_at=datetime.utcnow(),
                                meta={ "benchmark_version": benchmark_version, "template_key": template_key},
                                source_item=src_item
                            )
                self.store.append_candidate(c, date_str=date_str)
                filled.append(c.candidate_id)

        return {"filled": filled, "failed": failed, "total": len(filled) + len(failed)}

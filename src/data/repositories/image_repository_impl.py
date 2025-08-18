from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterable
from datetime import datetime
from src.domain.repositories.image_repository import ImageRepository
from src.domain.entities.output_query import OutputQuery
from src.domain.entities.outputs import CandidateOutput
from src.domain.entities.enums import ContentType

from src.data.datasources.fs.raw_output_fs import RawOutputFSDataSource
from src.data.datasources.fs.data_store_fs import DataStoreFSDataSource
from src.data.datasources.fs.templates_fs import TemplatesFSDataSource
from src.data.datasources.fs.text_generation import TextGenerationDataSource

class ImageRepositoryImpl(ImageRepository):
    """
    - find(): Raw_Output에서 image_caption 후보를 로드
    - generate_one(): 템플릿 + 생성기(text-gen DS)로 한 건 생성
    - generate_and_fill_missing(): 존재하지 않는 소스에 대해 생성 후 DataStore에 저장
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

    # --- 조회 ---
    def find(self, query: OutputQuery) -> Iterable[CandidateOutput]:
        return self.raw_output.find_candidates(ContentType.image_caption, query)

    # --- 생성 ---
    def generate_one(
        self, *, source: Dict[str, Any], problem_types: List[str], eval_goals: List[str],
        model_name: str, template_key: str, min_length: int, max_length: int, max_retries: int
    ) -> Optional[str]:
        """
        source 예시: {"source_id": "...", "text": "..."} 또는 {"source_id": "...", "content": "..."}
        """
        prompt = self.templates.get(
            template_key,
            problem_type1=problem_types[0],
            eval_goal1=eval_goals[0],
            problem_type2=problem_types[1],
            eval_goal2=eval_goals[1],
            problem_type3=problem_types[2],
            eval_goal3=eval_goals[2],
            topic=source.get("topic")
        )
        # prompt = None
        # if template_key == "passage_agent.create_image_caption_and_situation":
        #     prompt = self.templates.get(
        #         template_key,
        #         problem_type1=problem_types[0],
        #         eval_goal1=eval_goals[0],
        #         problem_type2=problem_types[1],
        #         eval_goal2=eval_goals[1],
        #         problem_type3=problem_types[2],
        #         eval_goal3=eval_goals[2],
        #         topic=source.get("topic")
        #     )
        # else: 
        #     return None  # 지원하지 않는 템플릿

        for i in range(max_retries):
            gen = self.textgen.generate(prompt)
            if len(gen) < min_length:
                print(f"  ❌ 생성 실패 (길이 부족): {len(gen)} < {min_length} (시도 {i+1}/{max_retries})")
                continue
            elif len(gen) > max_length:
                print(f"  ❌ 생성 실패 (길이 초과): {len(gen)} > {max_length} (시도 {i+1}/{max_retries})")
                continue
            else: 
                return gen.strip()
        return None

    def generate_and_fill_missing(
        self, *, model_name: str, template_key: str,
        benchmark_id: int, benchmark_version: str,
        problem_types: List[str], eval_goals: List[str],
        sources: List[Dict[str, Any]], date_str: Optional[str],
        min_length: int, max_length: int, max_retries: int
    ) -> dict:
        filled, failed = [], []
        day = date_str
        for s in sources:
            source_id = s.get("source_id") or s.get("id") or s.get("name")
            if not source_id:
                failed.append({"source": s, "reason": "missing source_id"})
                continue
            # 이미 저장돼 있으면 skip
            if self.store.exists(source_id=source_id, benchmark_id=benchmark_id, model_name=model_name, kind=ContentType.passage):
                continue
        
            text = self.generate_one(
                source=s, problem_types=problem_types, eval_goals=eval_goals,
                model_name=model_name, template_key=template_key,
                min_length=min_length, max_length=max_length, max_retries=max_retries
            )
            if not text:
                failed.append({"source_id": source_id, "reason": "generation_failed"})
                continue

            c = CandidateOutput(
                source_id=source_id,
                benchmark_id=benchmark_id,
                model_name=model_name,
                candidate_id=f"{source_id}:{model_name}:{benchmark_version}",
                content_type=ContentType.image_caption,
                content=text,
                stems=None,            # stems가 필요하면 sources에서 가져와 채우기
                generated_at=datetime.utcnow(),
                meta={"benchmark_version": benchmark_version, "template_key" : template_key}
            )
            self.store.append_candidate(c, date_str=day)
            filled.append(c.candidate_id)

        return {"filled": filled, "failed": failed, "total": len(sources)}

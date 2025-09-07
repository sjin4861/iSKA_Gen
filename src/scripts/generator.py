#!/usr/bin/env python3
"""
Passage Generation Script
벤치마크 데이터를 기반으로 passage, audio_script, image_caption을 생성하는 스크립트
"""
import gc
from pathlib import Path
import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))  # 프로젝트 루트 경로 추가

# Repository implementations
from src.data.repositories.passage_repository_impl import PassageRepositoryImpl
from src.data.repositories.audio_repository_impl import AudioRepositoryImpl
from src.data.repositories.image_repository_impl import ImageRepositoryImpl
from src.data.repositories.benchmark_repository_impl import BenchmarkRepositoryImpl

# Data sources
from src.data.datasources.fs.raw_output_fs import RawOutputFSDataSource
from src.data.datasources.fs.data_store_fs import DataStoreFSDataSource
from src.data.datasources.fs.templates_fs import TemplatesFSDataSource
from src.data.datasources.fs.text_generation import TextGenerationDataSource

# Model client
from src.modules.model_client import LocalModelClient, create_model_client

# Use cases
from src.domain.usecases.benchmark.load_collection import LoadCollectionUseCase

def main():
    # Configuration
    # A.X-4.0-Light                     llama3-bllossom-3b
# EXAONE-3.5-7.8B-Instruct          Midm-2.0-Base-Instruct
# llama3.1_korean_v1.1_sft_by_aidx
    MODEL_LIST = ["A.X-4.0-Light", "EXAONE-3.5-7.8B-Instruct", "Midm-2.0-Base-Instruct", "llama3.1_korean_v1.1_sft_by_aidx"]
    # MODEL_LIST = ["Midm-2.0-Base-Instruct", "llama3.1_korean_v1.1_sft_by_aidx"]

    DATE = "2025-08-19"
    TEMPLATES = {
        1: "passage_agent.create_passage_rubric_aware",
        2: "passage_agent.create_domestic_passage", 
        3: "audio_agent.create_dialogue_passage",
        4: "audio_agent.create_dialogue_passage",
        5: "image_agent.create_image_caption_and_situation",
    }
    
    # 벤치마크 로드
    benchmarks_root = Path("data_store/benchmarks/v1")
    benchmark_filename = "iSKA-Gen_Benchmark_v1.1.0_20250808_test.json"
    benchmark_repo = BenchmarkRepositoryImpl(benchmarks_root, benchmark_filename)
    load_collection_uc = LoadCollectionUseCase(benchmark_repo)
    collection_output = load_collection_uc.execute()
    benchmarks = collection_output.collection.benchmarks
    print(f"✅ 벤치마크 로드 완료: {len(benchmarks)}개 세트")
    raw_output_ds = RawOutputFSDataSource(Path("data_store/raw_outputs"))
    data_store_ds = DataStoreFSDataSource(Path("data_store"))
    templates_ds = TemplatesFSDataSource(agent="iska", user_path=Path("src/config/prompts"))
    # Data sources 초기화
    for MODEL in MODEL_LIST:
        print(f"\n🚀 모델 시작: {MODEL}")
        # 모델 클라이언트 초기화
        # 메모리 정리
        model_client = LocalModelClient(model_name=MODEL,  gpus=[1])
        # model_client = create_model_client(client_type="local", model_name=MODEL, gpus=[3])
        textgen_ds = TextGenerationDataSource(model_client)
        passage_repo = PassageRepositoryImpl(raw_output_ds, data_store_ds, templates_ds, textgen_ds)
        audio_repo = AudioRepositoryImpl(raw_output_ds, data_store_ds, templates_ds, textgen_ds)
        image_repo = ImageRepositoryImpl(raw_output_ds, data_store_ds, templates_ds, textgen_ds)
        try:           
            for bench_id, template_key in TEMPLATES.items():
                if bench_id > len(benchmarks):
                    print(f"⚠️ 벤치마크 ID {bench_id}가 범위를 벗어남 (총 {len(benchmarks)}개)")
                    continue

                bench = benchmarks[bench_id - 1]
                print(f"\n📝 벤치마크 {bench_id} 처리 중... (베이스 템플릿: {template_key})")

                # 공통 유틸: 이미지 주제에서 중괄호 제거
                def _strip_braces(s: str | None) -> str:
                    s = (s or "").strip()
                    return s[1:-1] if len(s) >= 2 and s.startswith("{") and s.endswith("}") else s

                try:
                    # ─────────────────────────────────────────────
                    # 1, 2번 → Passage 전용
                    #   - 1: 복합 비교형(ko/foreign)
                    #   - 2: 단일형(topic/context)
                    # ─────────────────────────────────────────────
                    if bench_id in (1, 2):
                        if bench_id == 1:
                            # 비교형: korean_* / foreign_* 사용
                            sources = []
                            for idx, item in enumerate(bench.items):
                                sources.append({
                                    "source_id": f"bench_{bench_id}_item_{idx}",
                                    "korean_topic": getattr(item, "korean_topic", None) or getattr(item, "topic", "") or "",
                                    "korean_context": getattr(item, "korean_context", None) or getattr(item, "context", "") or "",
                                    "foreign_topic": getattr(item, "foreign_topic", None),
                                    "foreign_context": getattr(item, "foreign_context", None),
                                })
                        else:
                            # 단일형: topic/context 사용
                            sources = []
                            for idx, item in enumerate(bench.items):
                                sources.append({
                                    "source_id": f"bench_{bench_id}_item_{idx}",
                                    "topic": getattr(item, "topic", None) or getattr(item, "korean_topic", "") or "",
                                    "context": getattr(item, "context", None) or getattr(item, "korean_context", "") or "",
                                })

                        min_length = 200
                        max_length = 800
                        print(f"  📖 Passage 생성 중... (템플릿: {template_key})")
                        passage_result = passage_repo.generate_and_fill_missing(
                            model_name=MODEL,
                            template_key=template_key,                          # passage용 템플릿 그대로
                            benchmark_id=bench_id,
                            benchmark_version="v1.1.0",
                            problem_types=bench.problem_types,
                            eval_goals=bench.eval_goals,
                            sources=sources,
                            date_str=DATE,
                            min_length=min_length,
                            max_length=max_length,
                            max_retries=10,
                        )
                        print(f"    ✅ Passage: {len(passage_result['filled'])} 생성, {len(passage_result['failed'])} 실패")

                    # ─────────────────────────────────────────────
                    # 3, 4번 → Audio 전용
                    #   - items: topic/context
                    # ─────────────────────────────────────────────
                    elif bench_id in (3, 4):
                        sources = []
                        for idx, item in enumerate(bench.items):
                            sources.append({
                                "source_id": f"bench_{bench_id}_item_{idx}",
                                "topic": getattr(item, "topic", None) or getattr(item, "korean_topic", "") or "",
                                "context": getattr(item, "context", None) or getattr(item, "korean_context", "") or "",
                            })

                        min_length = 300
                        max_length = 1000
                        print(f"  🎵 Audio script 생성 중... (템플릿: {template_key})")
                        audio_result = audio_repo.generate_and_fill_missing(
                            model_name=MODEL,
                            template_key=template_key,
                            benchmark_id=bench_id,
                            benchmark_version="v1.1.0",
                            problem_types=bench.problem_types,
                            eval_goals=bench.eval_goals,
                            sources=sources,
                            date_str=DATE,
                            min_length=min_length,
                            max_length=max_length,
                            max_retries=10,
                        )
                        print(f"    ✅ Audio: {len(audio_result['filled'])} 생성, {len(audio_result['failed'])} 실패")

                    # ─────────────────────────────────────────────
                    # 5번 → Image 전용
                    #   - items: topic만 존재 (중괄호 포함 가능) → braces 제거
                    # ─────────────────────────────────────────────
                    elif bench_id == 5:
                        sources = []
                        for idx, item in enumerate(bench.items):
                            raw_topic = getattr(item, "topic", None) or getattr(item, "korean_topic", "") or ""
                            sources.append({
                                "source_id": f"bench_{bench_id}_item_{idx}",
                                "topic": _strip_braces(raw_topic),
                            })

                        min_length = 200
                        max_length = 800
                        print(f"  🖼️ Image caption 생성 중... (템플릿: {template_key})")
                        image_result = image_repo.generate_and_fill_missing(
                            model_name=MODEL,
                            template_key=template_key,
                            benchmark_id=bench_id,
                            benchmark_version="v1.1.0",
                            problem_types=bench.problem_types,
                            eval_goals=bench.eval_goals,
                            sources=sources,
                            date_str=DATE,
                            min_length=min_length,
                            max_length=max_length,
                            max_retries=10,
                        )
                        print(f"    ✅ Image: {len(image_result['filled'])} 생성, {len(image_result['failed'])} 실패")

                    else:
                        print(f"⚠️ 미지원 벤치마크 ID: {bench_id} (스킵)")
                        continue

                except Exception as e:
                    print(f"    ❌ 벤치마크 {bench_id} 처리 중 오류: {e}")
                    continue
            
        except Exception as e:
            print(f"❌ 모델 {MODEL} 초기화 실패: {e}")
            continue

        finally:
            model_client.close()
    
    print(f"\n🎉 모든 생성 작업 완료!")

if __name__ == "__main__":
    main()

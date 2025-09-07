# src/scripts/stem_generator.py
"""
passage_repository처럼 클린 아키텍처로 stem 생성하는 예시 스크립트
"""
from __future__ import annotations
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # 프로젝트 루트 경로 추가

from src.domain.entities.enums import ContentType
from src.domain.entities.output_query import OutputQuery
from src.domain.usecases.benchmark.load_collection import LoadCollectionUseCase
from src.data.repositories.stem_repository_impl import StemRepositoryImpl
from src.data.datasources.fs.raw_output_fs import RawOutputFSDataSource
from src.data.datasources.fs.data_store_fs import DataStoreFSDataSource
from src.data.datasources.fs.templates_fs import TemplatesFSDataSource
from src.data.datasources.fs.text_generation import TextGenerationDataSource
from src.modules.model_client import LocalModelClient
from src.data.repositories.benchmark_repository_impl import BenchmarkRepositoryImpl

# ===== 설정 =====
# MODEL = "EXAONE-3.5-7.8B-Instruct"
MODEL_LIST = [
    "A.X-4.0-Light",
    "EXAONE-3.5-7.8B-Instruct",
    "Midm-2.0-Base-Instruct",
    "llama3.1_korean_v1.1_sft_by_aidx",
]

# ✅ few_shot_new 템플릿 사용
TEMPLATE_KEY = "stem_agent.few_shot_new"

# ✅ 실제 저장본 날짜 고정
DATE = "2025-08-23"

# 벤치마크 2(단일 지문) 우선
BENCH_ID_LIST = [1, 2]


def main():
    print("🚀 Stem 생성기 (Clean Architecture) 시작")

    benchmarks_root = Path("data_store/benchmarks/v1")
    benchmark_filename = "iSKA-Gen_Benchmark_v1.1.0_20250808_test.json"
    benchmark_repo = BenchmarkRepositoryImpl(benchmarks_root, benchmark_filename)
    load_collection_uc = LoadCollectionUseCase(benchmark_repo)
    collection_output = load_collection_uc.execute()
    benchmarks = collection_output.collection.benchmarks    
    print(f"✅ 벤치마크 로드 완료: {len(benchmarks)}개 세트")


    # Repository 초기화
    raw_output_ds = RawOutputFSDataSource(Path("data_store/raw_outputs"))
    data_store_ds = DataStoreFSDataSource(Path("data_store"))
    templates_ds = TemplatesFSDataSource(agent="iska", user_path=Path("src/config/prompts"))

    for model_name in MODEL_LIST:
        model_client = LocalModelClient(model_name=model_name, gpus=[2])
        print(f"✅ 모델 클라이언트 초기화: {model_name}")
        textgen_ds = TextGenerationDataSource(model_client)
        stem_repo = StemRepositoryImpl(raw_output_ds, data_store_ds, templates_ds, textgen_ds)

        for bench_id in BENCH_ID_LIST:
            print(f"\n📝 벤치마크 ID {bench_id}에 대한 stem 생성 중...")
            benchmark = benchmarks[bench_id - 1]  # 벤치마크 ID는 1부터 시작하므로 -1
            problem_types = benchmark.problem_types
            eval_goals = benchmark.eval_goals

            # ✅ 2025-08-19, bench_id=2, 해당 템플릿만 조회
            q = OutputQuery(
                date_from=datetime.strptime(DATE, "%Y-%m-%d"),
                date_to=datetime.strptime(DATE, "%Y-%m-%d"),
                model_name=model_name,
                benchmark_id=bench_id,
                limit=None,
            )

            candidates = list(raw_output_ds.find_candidates(ContentType.passage, q))
            # ✅ 생성 실행 (few_shot_new 사용, 저장 날짜 DATE, passage_model_name 지정)
            result = stem_repo.generate_and_fill_missing(
                model_name=model_name,
                template_key=TEMPLATE_KEY,          # "stem_agent.few_shot_new"
                benchmark_id=bench_id,
                benchmark_version="v1.1.0",
                problem_types=problem_types,
                eval_goals=eval_goals,
                contents=candidates,
                date_str=DATE,
                max_retries=3,
                content_model_name=model_name,   # 저장 키에 *_from_{model_name} suffix
            )

            print(f"✅ 생성 성공: {len(result.get('filled', []))}개 | ❌ 실패: {len(result.get('failed', []))}개 | 📊 전체: {result.get('total', 0)}개")
        model_client.close()
if __name__ == "__main__":
    # 실사용: 메인 실행
    main()
    # 필요할 때만 단일 테스트
    # test_single_stem()

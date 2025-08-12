#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Null 지문만 재생성하여 기존 raw_outputs 파일에 패치합니다.
- Clean Architecture: UseCase 1개 + Repository 1개 수동 주입
- 대상: passage 계열 (id=1 비교형, id=2 단일형 기본)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

from src.data.repositories.llm_gateway_impl import LLMGatewayImpl

# ---- project path bootstrap ----
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.extend([str(PROJECT_ROOT), str(SRC_ROOT)])

# ---- domain usecases & repos ----
from domain.usecases.benchmark.get_benchmark_set_by_id import (
    GetBenchmarkSetByIdUseCase, GetBenchmarkSetByIdInput
)
from domain.usecases.benchmark.list_benchmark_items_as_sources import (
    ListBenchmarkItemsAsSourcesUseCase, ListBenchmarkItemsAsSourcesInput
)
from domain.usecases.content.fill_null_passages import (
    FillNullPassagesUseCase, FillNullPassagesInput
)
from data.repositories.benchmark_repository_impl import BenchmarkRepositoryImpl
from data.repositories.passage_repository_impl import PassageRepositoryImpl

# ---- artifact kind selector (passage만) ----
from domain.entities.content_types import ArtifactKind

# -------------------------------
# settings & helpers
# -------------------------------
DEFAULT_BENCHMARK_FILE = "iSKA-Gen_Benchmark_v1.1.0_20250808.json"
DEFAULT_TEMPLATE_BY_ID = {
    1: "passage_agent.create_passage_rubric_aware",
    2: "passage_agent.create_domestic_passage",
    3: "passage_agent.create_dialogue_passage",
    4: "passage_agent.create_dialogue_passage",
    5: "passage_agent.create_image_caption_and_situation",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill null passages in raw_outputs by regenerating only missing items.")
    p.add_argument("--model", required=True, help="생성에 사용할 모델명 (예: EXAONE-3.5-7.8B-Instruct)")
    p.add_argument("--gpus", default="0", help="GPU 인덱스 CSV (예: 1 또는 1,2)")
    p.add_argument("--bench-ids", default="1,2", help="대상 벤치마크 ID CSV (예: 1,2)")
    p.add_argument("--benchmark-file", default=DEFAULT_BENCHMARK_FILE, help="벤치마크 JSON 파일명 또는 절대경로")
    p.add_argument("--benchmark-version", default="v1.1.0", help="벤치마크 버전 표기 (파일명에 사용)")
    p.add_argument("--date", default="2025-08-08", help="raw_outputs 날짜 디렉토리 (예: 2025-08-08)")
    p.add_argument("--min-length", type=int, default=300)
    p.add_argument("--max-length", type=int, default=800)
    p.add_argument("--max-retries", type=int, default=10)
    return p.parse_args()

def split_benchmark_root_and_name(arg: str) -> tuple[Path, str]:
    """
    절대경로면 (부모, 파일명)으로 쪼개고,
    상대경로면 data_store/benchmarks/v1 아래에 있다고 가정.
    """
    p = Path(arg)
    if p.is_absolute():
        return p.parent, p.name
    return Path("data_store/benchmarks/v1"), arg

# -------------------------------
# main
# -------------------------------
def main() -> None:
    args = parse_args()
    model_name: str = args.model
    gpus: List[int] = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    bench_ids: List[int] = [int(x) for x in args.bench_ids.split(",") if x.strip() != ""]
    benchmarks_root, benchmark_filename = split_benchmark_root_and_name(args.benchmark_file)
    benchmark_file_path = benchmarks_root / benchmark_filename

    # --- repositories wiring (manual DI) ---
    benchmark_repo = BenchmarkRepositoryImpl(
        benchmarks_root=benchmarks_root,
        benchmark_filename=benchmark_filename,
    )
    llm = LLMGatewayImpl(
        client_type="local",              # "openai" | "vllm" 로 쉽게 교체
        model_name=args.model,
        default_params={"temperature": 0.7},
        # openai일 경우: api_key=..., vllm일 경우: base_url=...
        gpus=[int(x) for x in args.gpus.split(",") if x],   # LocalModelClient에 그대로 전달됨
    )
    passage_repo = PassageRepositoryImpl(llm=llm)
    # --- usecases wiring ---
    get_set_uc = GetBenchmarkSetByIdUseCase(benchmark_repo)
    list_src_uc = ListBenchmarkItemsAsSourcesUseCase(benchmark_repo)
    fill_uc = FillNullPassagesUseCase(passage_repo)

    print(f"\n🚀 Fill null passages")
    print(f"   • Model: {model_name}")
    print(f"   • GPUs: {gpus}")
    print(f"   • Date: {args.date}")
    print(f"   • Benchmark file: {benchmark_file_path}\n")

    for bench_id in bench_ids:
        template_key = DEFAULT_TEMPLATE_BY_ID.get(bench_id)
        if not template_key or "image_caption" in template_key:
            print(f"⚠️  benchmark_id={bench_id} 는 passage 대상이 아니거나(이미지) 현재 스크립트 범위 외라 건너뜁니다.")
            continue

        # 1) 벤치마크 세트 로드
        set_out = get_set_uc.execute(GetBenchmarkSetByIdInput(benchmark_id=bench_id))
        problem_types = set_out.benchmark_set.problem_types
        eval_goals = set_out.benchmark_set.eval_goals

        # 2) 소스 아이템을 passage용 Source dict 리스트로 변환
        src_out = list_src_uc.execute(
            ListBenchmarkItemsAsSourcesInput(
                benchmark_id=bench_id,
                artifact_kind=ArtifactKind.passage
            )
        )
        sources: List[Dict[str, Any]] = src_out.sources_as_dicts

        # 3) null 항목만 재생성 + 패치 저장
        result = fill_uc.execute(
            FillNullPassagesInput(
                model_name=model_name,
                template_key=template_key,
                benchmark_id=bench_id,
                benchmark_version=args.benchmark_version,
                problem_types=problem_types,
                eval_goals=eval_goals,
                sources=sources,
                date_str=args.date,
                min_length=args.min_length,
                max_length=args.max_length if bench_id not in (3, 4) else max(args.max_length, 800),
                max_retries=args.max_retries,
            )
        )

        print(f"✅ bench_id={bench_id} | filled={len(result['filled'])} | failed={len(result['failed'])} | total={result['total']}")
        if result["failed"]:
            print(f"   └─ failed indices: {sorted(result['failed'])}")

    print("\n✨ Done.")

if __name__ == "__main__":
    main()

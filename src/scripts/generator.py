# scripts/fill_null_passages.py
from pathlib import Path
import json

from src.data.repositories.content_repository_impl import ContentRepositoryImpl
from src.data.services.passage_generation_service_impl import PassageGenerationServiceImpl
from src.domain.usecases.content.generate_missing_passages import (
    GenerateMissingPassagesUseCase, GenerateMissingPassagesInput
)

# 벤치마크 로드(방식 자유) — 예: data_store/benchmarks/v1/… 파일
bench_path = Path("data_store/benchmarks/v1/iSKA-Gen_Benchmark_v1.1.0_20250808.json")
benchmarks = json.loads(bench_path.read_text(encoding="utf-8"))

MODEL = "EXAONE-3.5-7.8B-Instruct"
DATE  = "2025-08-08"
TEMPLATES = {
    1: "passage_agent.create_passage_rubric_aware",
    2: "passage_agent.create_domestic_passage",
}

repo = ContentRepositoryImpl()
generator = PassageGenerationServiceImpl(model_name=MODEL, gpus=[1])  # 필요한 GPU 인덱스

for bench_id, template_key in TEMPLATES.items():
    bench = benchmarks[bench_id - 1]
    uc = GenerateMissingPassagesUseCase(repo, generator)

    result = uc.execute(GenerateMissingPassagesInput(
        model_name=MODEL,
        template_key=template_key,
        benchmark_id=bench_id,
        benchmark_version="v1.1.0",
        problem_types=bench["problem_types"],
        eval_goals=bench["eval_goals"],
        sources=bench["items"],
        date_str=DATE,
        min_length=300,
        max_length=800 if bench_id in (3,4) else 500,
        max_retries=10,
    ))
    print(f"[bench {bench_id}] filled={result.filled_indices} failed={result.failed_indices} total={result.total_after}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List

THIS = Path(__file__).resolve()
PROJECT = THIS.parents[3]
SRC = PROJECT / "src"
sys.path.extend([str(PROJECT), str(SRC)])

from src.domain.entities.content_types import ArtifactKind
from src.domain.entities.rubrics import RubricID
from src.domain.usecases.evaluation.evaluate_rubrics import (
    EvaluateRubricsInput, EvaluateRubricsUseCase
)
from src.data.repositories.rubric_evaluation_repository_impl import RubricEvaluationRepositoryImpl

def _parse_ids(csv: str) -> List[int]:
    return [int(x) for x in csv.split(",") if x.strip()]

def _parse_rubrics(csv: str) -> List[RubricID]:
    name_map = {r.name.lower(): r for r in RubricID}
    out=[]
    for x in csv.split(","):
        t=x.strip().lower()
        if not t: continue
        # 이름 또는 value 모두 허용
        r = name_map.get(t) or RubricID(t)
        out.append(r)
    return out

def main():
    ap = argparse.ArgumentParser("Evaluate passages with gpt-oss-20b (vLLM)")
    ap.add_argument("--date", required=True, help="예: 2025-08-08")
    ap.add_argument("--bench-ids", default="1,2", help="예: 1,2")
    ap.add_argument("--benchmark-version", default="v1.1.0")
    ap.add_argument("--target-mode", default="content", choices=["content","content+instruction"])
    ap.add_argument("--artifact-kind", default="passage")
    ap.add_argument("--rubrics", default="completeness_for_guidelines,clarity_of_core_theme,reference_groundedness,logical_flow,korean_quality,l2_learner_suitability")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--src-models", default=None, help="평가 대상 생성모델 필터 CSV")
    ap.add_argument("--templates", default=None, help="템플릿 키 필터 CSV")
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")

    args = ap.parse_args()
    bench_ids = _parse_ids(args.bench_ids)
    rubrics = _parse_rubrics(args.rubrics)
    src_models = args.src_models.split(",") if args.src_models else None
    templates = args.templates.split(",") if args.templates else None

    repo = RubricEvaluationRepositoryImpl()
    uc = EvaluateRubricsUseCase(repo)
    out = uc.execute(EvaluateRubricsInput(
        date_str=args.date,
        target_mode=args.target_mode,
        artifact_kind=ArtifactKind[args.artifact_kind],
        bench_ids=bench_ids,
        benchmark_version=args.benchmark_version,
        rubric_ids=rubrics,
        source_model_filter=src_models,
        template_filter=templates,
        limit_per_benchmark=args.limit,
        evaluator_client_type="vllm",
        evaluator_model_name="gpt-oss-20b",
        evaluator_client_kwargs={"base_url": args.vllm_url},
    ))
    print("\n✅ Summary")
    for bid, s in out["benchmarks"].items():
        print(f"  - bench_id={bid}: scored={s['scored']}, saved={s['saved_file']}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px

# path bootstrap
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.extend([str(ROOT), str(ROOT / "src")])

from data.repositories.analysis_repository_impl import AnalysisRepositoryImpl
from domain.usecases.analysis.load_evaluation_records_by_date_usecase import (
    LoadEvaluationRecordsByDateUseCase, LoadEvaluationRecordsByDateInput
)

RUBRICS = [
    "completeness_for_guidelines",
    "clarity_of_core_theme",
    "reference_groundedness",
    "logical_flow",
    "korean_quality",
    "l2_learner_suitability",
]
RUBRIC_COLS = [f"{r}_score" for r in RUBRICS]
RUBRIC_NAMES = {
    "completeness_for_guidelines": "평가 지침 완전성",
    "clarity_of_core_theme": "핵심 주제 명확성",
    "reference_groundedness": "참고자료 기반성",
    "logical_flow": "논리적 흐름",
    "korean_quality": "한국어 품질",
    "l2_learner_suitability": "L2 학습자 적합성",
}

def to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    if "benchmark_id" in df.columns:
        df["benchmark_id"] = pd.Categorical(
            df["benchmark_id"],
            categories=sorted([x for x in df["benchmark_id"].unique() if isinstance(x, int)]),
            ordered=True,
        )
    return df

def analyze_and_visualize(df: pd.DataFrame, out_dir: Path) -> None:
    df_scores = df.dropna(subset=RUBRIC_COLS, how="all")
    if df_scores.empty:
        print("⚠️ 유효한 점수 없음")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 전체 평균
    avg_overall = df_scores[RUBRIC_COLS].mean().reset_index()
    avg_overall.columns = ["rubric", "average_score"]
    avg_overall["rubric_display"] = avg_overall["rubric"].str.replace("_score","",regex=False).map(RUBRIC_NAMES)
    print("\n📊 전체 루브릭 평균:\n", avg_overall[["rubric_display","average_score"]])

    fig = px.bar(avg_overall, x="rubric_display", y="average_score",
                 title="전체 루브릭별 평균 점수", template="plotly_white")
    fig.write_image(out_dir / "overall_rubric_scores.png")

    # 모델별
    if "model_name" in df_scores.columns and df_scores["model_name"].nunique() > 0:
        avg_by_model = df_scores.groupby("model_name")[RUBRIC_COLS].mean().reset_index()
        melted = avg_by_model.melt(id_vars="model_name", var_name="rubric", value_name="average_score")
        melted["rubric_display"] = melted["rubric"].str.replace("_score","",regex=False).map(RUBRIC_NAMES)
        fig = px.bar(melted, x="rubric_display", y="average_score", color="model_name",
                     barmode="group", title="모델별 루브릭 평균 점수", template="plotly_white")
        fig.write_image(out_dir / "model_rubric_scores.png")

    # 벤치마크별
    if "benchmark_id" in df_scores.columns and df_scores["benchmark_id"].nunique() > 0:
        avg_by_bench = df_scores.groupby("benchmark_id")[RUBRIC_COLS].mean().reset_index()
        melted = avg_by_bench.melt(id_vars="benchmark_id", var_name="rubric", value_name="average_score")
        melted["rubric_display"] = melted["rubric"].str.replace("_score","",regex=False).map(RUBRIC_NAMES)
        fig = px.bar(melted, x="rubric_display", y="average_score", color="benchmark_id",
                     barmode="group", title="벤치마크 ID별 루브릭 평균 점수", template="plotly_white")
        fig.write_image(out_dir / "benchmark_rubric_scores.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", type=str, help="예: 2025-08-08 (data_store/evaluations/<date>)")
    ap.add_argument("--out", default="analysis_results/evaluations")
    args = ap.parse_args()

    repo = AnalysisRepositoryImpl(Path("data_store"))
    uc = LoadEvaluationRecordsByDateUseCase(repo)
    records = uc.execute(LoadEvaluationRecordsByDateInput(date_str=args.date))

    if not records:
        print("❌ 해당 날짜의 평가 데이터가 없습니다.")
        return

    print(f"✅ 로드: {len(records)} rows")
    df = to_dataframe(records)
    out_dir = Path(args.out) / args.date
    analyze_and_visualize(df, out_dir)
    print("\n🎉 완료. 결과:", out_dir)

if __name__ == "__main__":
    main()

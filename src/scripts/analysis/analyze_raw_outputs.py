#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px

# path bootstrap
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.extend([str(ROOT), str(ROOT / "src")])

from data.repositories.analysis_repository_impl import AnalysisRepositoryImpl
from domain.usecases.analysis.load_passage_records_by_date_usecase import (
    LoadPassageRecordsByDateUseCase, LoadPassageRecordsByDateInput
)

def to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    # benchmark_id 정렬용 카테고리
    if "benchmark_id" in df.columns:
        df["benchmark_id"] = pd.Categorical(
            df["benchmark_id"],
            categories=sorted([x for x in df["benchmark_id"].unique() if isinstance(x, int)]),
            ordered=True,
        )
    return df

def analyze_passage_lengths(df: pd.DataFrame, out_dir: Path) -> None:
    if "generated_passage" not in df.columns:
        print("⚠️ 'generated_passage' 컬럼이 없어 길이 분석 스킵")
        return
    df["passage_length"] = df["generated_passage"].apply(lambda x: len(str(x)) if x else 0)

    print("\n📊 지문 길이 전체 통계:")
    print(df["passage_length"].describe())

    out_dir.mkdir(parents=True, exist_ok=True)
    fig = px.histogram(df, x="passage_length", nbins=50, title="전체 지문 길이 분포", template="plotly_white")
    fig.write_image(out_dir / "overall_passage_length_distribution.png")

    if "model_name" in df.columns and df["model_name"].nunique() > 1:
        fig = px.box(df, x="model_name", y="passage_length", title="모델별 지문 길이 분포", template="plotly_white")
        fig.write_image(out_dir / "passage_length_by_model.png")

    if "task_name" in df.columns and df["task_name"].nunique() > 1:
        fig = px.box(df, x="task_name", y="passage_length", title="태스크별 지문 길이 분포", template="plotly_white")
        fig.write_image(out_dir / "passage_length_by_task.png")

    if all(c in df.columns for c in ["benchmark_id","task_name","model_name"]):
        if df["benchmark_id"].nunique() > 0 and df["task_name"].nunique() > 0 and df["model_name"].nunique() > 0:
            fig = px.box(df, x="benchmark_id", y="passage_length", color="model_name",
                         facet_col="task_name", title="벤치마크/태스크/모델별 지문 길이 분포",
                         template="plotly_white")
            fig.update_xaxes(matches=None)
            fig.update_yaxes(range=[0, 1000])
            fig.write_image(out_dir / "passage_length_by_benchmark_task_model.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", type=str, help="예: 2025-08-08 (data_store/raw_outputs/<date>)")
    ap.add_argument("--out", default="analysis_results", help="결과 저장 루트")
    args = ap.parse_args()

    repo = AnalysisRepositoryImpl(Path("data_store"))
    uc = LoadPassageRecordsByDateUseCase(repo)
    records = uc.execute(LoadPassageRecordsByDateInput(date_str=args.date))

    if not records:
        print("❌ 해당 날짜의 passage 데이터가 없습니다.")
        return

    print(f"✅ 로드: {len(records)} rows")
    df = to_dataframe(records)
    out_dir = Path(args.out) / args.date
    analyze_passage_lengths(df, out_dir)
    print("\n🎉 완료. 결과:", out_dir)

if __name__ == "__main__":
    main()

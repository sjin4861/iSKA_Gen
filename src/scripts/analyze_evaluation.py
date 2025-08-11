import argparse
import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import re

def load_evaluation_data(date_str: str) -> pd.DataFrame:
    """
    주어진 날짜의 평가 JSON 파일을 로드하여 DataFrame으로 반환합니다.
    """
    base_path = Path(f"src/data/evaluations/{date_str}/misc/")
    if not base_path.exists():
        print(f"❌ 오류: 지정된 날짜의 평가 데이터 디렉토리를 찾을 수 없습니다: {base_path}")
        return pd.DataFrame()

    all_data = []
    # Find all JSON files in eval_rubric subdirectories
    json_files = list(base_path.glob('**/*/eval_rubric/*.json'))

    if not json_files:
        print(f"❌ 오류: {base_path} 경로에서 평가 JSON 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract model_name and benchmark_id from the file path
            # Example path: .../misc/A.X-4.0-Light_evaluation/eval_rubric/benchmark_1_v1.0.0_eval_rubric.json
            parts = file_path.parts
            model_dir_name = [p for p in parts if p.endswith('_evaluation')][0]
            model_name = model_dir_name.replace('_evaluation', '')

            benchmark_id = "unknown_benchmark"
            match = re.search(r'benchmark_(\d+)', file_path.name)
            if match:
                benchmark_id = f"benchmark_{match.group(1)}"

            for item in data:
                if 'evaluation' in item:
                    eval_data = item['evaluation']
                    row = {
                        'model_name': model_name,
                        'benchmark_id': benchmark_id,
                        'file_path': str(file_path)
                    }
                    for key, value in eval_data.items():
                        if key.endswith('_score'):
                            row[key] = value
                    all_data.append(row)

        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️ 파일 로드 또는 파싱 오류: {file_path} - {e}")
            continue
    
    if not all_data:
        print(f"❌ 로드된 데이터가 없습니다. JSON 파일 형식을 확인해주세요.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    
    # Convert benchmark_id to categorical for consistent plotting order
    if 'benchmark_id' in df.columns:
        df['benchmark_id_numeric'] = df['benchmark_id'].apply(lambda x: int(x.split('_')[1]) if isinstance(x, str) and x.startswith('benchmark_') else -1)
        df['benchmark_id'] = pd.Categorical(df['benchmark_id'], categories=df.sort_values('benchmark_id_numeric')['benchmark_id'].unique(), ordered=True)
        df = df.drop(columns=['benchmark_id_numeric'])

    return df

def analyze_and_visualize_evaluations(df: pd.DataFrame, output_dir: Path):
    """
    평가 데이터를 분석하고 시각화합니다.
    """
    rubrics = [
        "completeness_for_guidelines",
        "clarity_of_core_theme", 
        "reference_groundedness",
        "logical_flow",
        "korean_quality",
        "l2_learner_suitability"
    ]
    
    rubric_names = {
        "completeness_for_guidelines": "평가 지침 완전성",
        "clarity_of_core_theme": "핵심 주제 명확성", 
        "reference_groundedness": "참고자료 기반성",
        "logical_flow": "논리적 흐름",
        "korean_quality": "한국어 품질",
        "l2_learner_suitability": "L2 학습자 적합성"
    }

    score_columns = [f"{r}_score" for r in rubrics]
    df_scores = df.dropna(subset=score_columns, how='all')

    if df_scores.empty:
        print("⚠️ 유효한 루브릭 점수 데이터가 없어 분석을 건너뜁니다.")
        return

    print("\n📊 평가 결과 분석:")

    # 1. Overall Average Scores per Rubric
    avg_overall_rubric = df_scores[score_columns].mean().reset_index()
    avg_overall_rubric.columns = ['rubric', 'average_score']
    avg_overall_rubric['rubric_display'] = avg_overall_rubric['rubric'].apply(lambda x: rubric_names.get(x.replace('_score', ''), x))
    print("\n전체 루브릭별 평균 점수:")
    print(avg_overall_rubric[['rubric_display', 'average_score']])

    fig = px.bar(avg_overall_rubric, x='rubric_display', y='average_score',
                 title="전체 루브릭별 평균 점수",
                 labels={'rubric_display': '루브릭', 'average_score': '평균 점수'},
                 template="plotly_white")
    fig.write_image(output_dir / "overall_rubric_scores.png")
    print(f"✅ 'overall_rubric_scores.png' 저장 완료.")

    # 2. Average Scores by Model and Rubric
    if 'model_name' in df_scores.columns and df_scores['model_name'].nunique() > 0:
        avg_model_rubric = df_scores.groupby('model_name')[score_columns].mean().reset_index()
        df_melted_model = avg_model_rubric.melt(id_vars='model_name', var_name='rubric', value_name='average_score')
        df_melted_model['rubric_display'] = df_melted_model['rubric'].apply(lambda x: rubric_names.get(x.replace('_score', ''), x))
        
        print("\n모델별 루브릭 평균 점수:")
        print(avg_model_rubric)

        fig = px.bar(df_melted_model, x="rubric_display", y="average_score", color="model_name",
                     barmode="group", title="모델별 루브릭 평균 점수 비교",
                     labels={'rubric_display': '루브릭', 'average_score': '평균 점수', 'model_name': '모델'},
                     template="plotly_white")
        fig.write_image(output_dir / "model_rubric_scores.png")
        print(f"✅ 'model_rubric_scores.png' 저장 완료.")

    # 3. Average Scores by Benchmark ID and Rubric
    if 'benchmark_id' in df_scores.columns and df_scores['benchmark_id'].nunique() > 0:
        avg_benchmark_rubric = df_scores.groupby('benchmark_id')[score_columns].mean().reset_index()
        df_melted_benchmark = avg_benchmark_rubric.melt(id_vars='benchmark_id', var_name='rubric', value_name='average_score')
        df_melted_benchmark['rubric_display'] = df_melted_benchmark['rubric'].apply(lambda x: rubric_names.get(x.replace('_score', ''), x))

        print("\n벤치마크 ID별 루브릭 평균 점수:")
        print(avg_benchmark_rubric)

        fig = px.bar(df_melted_benchmark, x="rubric_display", y="average_score", color="benchmark_id",
                     barmode="group", title="벤치마크 ID별 루브릭 평균 점수 비교",
                     labels={'rubric_display': '루브릭', 'average_score': '평균 점수', 'benchmark_id': '벤치마크 ID'},
                     template="plotly_white")
        fig.write_image(output_dir / "benchmark_rubric_scores.png")
        print(f"✅ 'benchmark_rubric_scores.png' 저장 완료.")

    # 4. Average Scores by Model, Benchmark ID, and Rubric (Faceted)
    if all(col in df_scores.columns for col in ['model_name', 'benchmark_id']) and \
       df_scores['model_name'].nunique() > 0 and df_scores['benchmark_id'].nunique() > 0:
        
        avg_all_grouped = df_scores.groupby(['model_name', 'benchmark_id'])[score_columns].mean().reset_index()
        df_melted_all_grouped = avg_all_grouped.melt(id_vars=['model_name', 'benchmark_id'], 
                                                     var_name='rubric', value_name='average_score')
        df_melted_all_grouped['rubric_display'] = df_melted_all_grouped['rubric'].apply(lambda x: rubric_names.get(x.replace('_score', ''), x))

        fig = px.bar(df_melted_all_grouped, x="benchmark_id", y="average_score", color="model_name",
                     facet_col="rubric_display",
                     title="모델 및 벤치마크 ID별 루브릭 평균 점수 비교",
                     labels={'benchmark_id': '벤치마크 ID', 'average_score': '평균 점수', 'model_name': '모델', 'rubric_display': '루브릭'},
                     barmode="group", template="plotly_white", height=600)
        fig.update_xaxes(matches=None)
        fig.write_image(output_dir / "model_benchmark_rubric_scores.png")
        print(f"✅ 'model_benchmark_rubric_scores.png' 저장 완료.")

def main():
    parser = argparse.ArgumentParser(description="지정된 날짜의 평가 데이터를 분석하고 시각화합니다.")
    parser.add_argument("date", type=str, help="분석할 평가 데이터의 날짜 (예: 2025-08-05)")
    args = parser.parse_args()

    date_to_analyze = args.date
    output_base_dir = Path("analysis_results/evaluations")
    output_dir = output_base_dir / date_to_analyze
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"✨ {date_to_analyze} 날짜의 평가 데이터 분석을 시작합니다.")
    print(f"결과는 {output_dir} 디렉토리에 저장됩니다.")

    df = load_evaluation_data(date_to_analyze)

    if df.empty:
        print("분석할 데이터가 없습니다. 스크립트를 종료합니다.")
        return

    print(f"✅ 총 {len(df)}개의 평가 항목을 로드했습니다.")

    analyze_and_visualize_evaluations(df, output_dir)

    print("\n🎉 분석 및 시각화가 완료되었습니다.")
    print(f"생성된 이미지 파일은 {output_dir} 디렉토리에서 확인해주세요.")

if __name__ == "__main__":
    main()

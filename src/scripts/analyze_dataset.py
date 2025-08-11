import argparse
import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import re

def load_data_for_date(date_str: str) -> pd.DataFrame:
    """
    주어진 날짜에 해당하는 모든 JSON 파일을 로드하여 DataFrame으로 반환합니다.
    """
    base_path = Path(f"src/data/raw_outputs/{date_str}/passage/")
    if not base_path.exists():
        print(f"❌ 오류: 지정된 날짜의 데이터 디렉토리를 찾을 수 없습니다: {base_path}")
        return pd.DataFrame()

    all_data = []
    json_files = list(base_path.glob('**/*.json'))

    if not json_files:
        print(f"❌ 오류: {base_path} 경로에서 JSON 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 파일 경로에서 model_name, task_name, benchmark_id 추출
            relative_path = file_path.relative_to(base_path)
            parts = relative_path.parts
            
            model_name = parts[0] if len(parts) > 0 else "unknown_model"
            task_name = parts[1] if len(parts) > 1 else "unknown_task"

            benchmark_id = "unknown_benchmark"
            match = re.search(r'benchmark_(\d+)', file_path.name)
            if match:
                benchmark_id = f"benchmark_{match.group(1)}"

            # 각 항목에 메타데이터 추가
            for item in data:
                item['model_name'] = model_name
                item['task_name'] = task_name
                item['benchmark_id'] = benchmark_id # Add benchmark_id
                item['file_path'] = str(file_path)
                all_data.append(item)

        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️ 파일 로드 또는 파싱 오류: {file_path} - {e}")
            continue
    
    if not all_data:
        print(f"❌ 로드된 데이터가 없습니다. JSON 파일 형식을 확인해주세요.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    # Sort benchmark_id for consistent plotting order
    if 'benchmark_id' in df.columns:
        # Extract the numeric part of benchmark_id for sorting
        df['benchmark_id_numeric'] = df['benchmark_id'].apply(lambda x: int(x.split('_')[1]) if isinstance(x, str) and x.startswith('benchmark_') else -1)
        df['benchmark_id'] = pd.Categorical(df['benchmark_id'], categories=df.sort_values('benchmark_id_numeric')['benchmark_id'].unique(), ordered=True)
        df = df.drop(columns=['benchmark_id_numeric'])
    return df

def analyze_passage_lengths(df: pd.DataFrame, output_dir: Path):
    """
    지문 길이를 분석하고 시각화합니다.
    """
    if 'generated_passage' not in df.columns:
        print("⚠️ 'generated_passage' 컬럼이 없어 지문 길이 분석을 건너뜁니다.")
        return

    df['passage_length'] = df['generated_passage'].apply(lambda x: len(str(x)) if x else 0)
    
    print("\n📊 지문 길이 분석 결과:")
    print(df['passage_length'].describe())

    if 'model_name' in df.columns and df['model_name'].nunique() > 0:
        print("\n모델별 지문 길이 통계 요약:")
        model_length_summary = df.groupby('model_name')['passage_length'].describe()
        print(model_length_summary)

    if 'benchmark_id' in df.columns and df['benchmark_id'].nunique() > 0:
        print("\n벤치마크 ID별 지문 길이 통계 요약:")
        benchmark_length_summary = df.groupby('benchmark_id')['passage_length'].describe()
        print(benchmark_length_summary)

    # 전체 지문 길이 분포
    fig = px.histogram(df, x="passage_length", nbins=50, 
                       title="전체 지문 길이 분포",
                       labels={'passage_length': '지문 길이 (글자 수)'},
                       template="plotly_white")
    fig.write_image(output_dir / "overall_passage_length_distribution.png")
    print(f"✅ 'overall_passage_length_distribution.png' 저장 완료.")

    # 모델별 지문 길이 분포
    if 'model_name' in df.columns and df['model_name'].nunique() > 1:
        fig = px.box(df, x="model_name", y="passage_length", 
                     title="모델별 지문 길이 분포",
                     labels={'model_name': '모델', 'passage_length': '지문 길이 (글자 수)'},
                     template="plotly_white")
        fig.write_image(output_dir / "passage_length_by_model.png")
        print(f"✅ 'passage_length_by_model.png' 저장 완료.")

    # 태스크별 지문 길이 분포
    if 'task_name' in df.columns and df['task_name'].nunique() > 1:
        fig = px.box(df, x="task_name", y="passage_length", 
                     title="태스크별 지문 길이 분포",
                     labels={'task_name': '태스크', 'passage_length': '지문 길이 (글자 수)'},
                     template="plotly_white")
        fig.write_image(output_dir / "passage_length_by_task.png")
        print(f"✅ 'passage_length_by_task.png' 저장 완료.")

    # 벤치마크 ID별 세부 유형 (태스크) 및 모델별 지문 길이 분포
    if all(col in df.columns for col in ['benchmark_id', 'task_name', 'model_name']) and \
       df['benchmark_id'].nunique() > 0 and df['task_name'].nunique() > 0 and df['model_name'].nunique() > 0:
        
        fig = px.box(df, x="benchmark_id", y="passage_length", color="model_name", facet_col="task_name",
                     title="벤치마크 ID별 태스크 및 모델 지문 길이 분포",
                     labels={'benchmark_id': '벤치마크 ID', 'passage_length': '지문 길이 (글자 수)', 'model_name': '모델', 'task_name': '태스크'},
                     template="plotly_white")
        fig.update_xaxes(matches=None) # Allow independent x-axes for facets
        fig.update_yaxes(range=[0, 1000]) # Set y-axis range to clean up outliers
        fig.write_image(output_dir / "passage_length_by_benchmark_task_model.png")
        print(f"✅ 'passage_length_by_benchmark_task_model.png' 저장 완료.")

def analyze_rubric_scores(df: pd.DataFrame, output_dir: Path):
    """
    루브릭 점수를 분석하고 시각화합니다.
    """
    if 'evaluation' not in df.columns:
        print("⚠️ 'evaluation' 컬럼이 없어 루브릭 점수 분석을 건너뜁니다.")
        return

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

    # evaluation 딕셔너리에서 점수 추출
    for rubric in rubrics:
        score_key = f"{rubric}_score"
        df[score_key] = df['evaluation'].apply(lambda x: x.get(score_key) if isinstance(x, dict) else None)
    
    # 점수가 있는 행만 필터링
    score_columns = [f"{r}_score" for r in rubrics]
    df_scores = df.dropna(subset=score_columns, how='all')

    if df_scores.empty:
        print("⚠️ 유효한 루브릭 점수 데이터가 없어 루브릭 점수 분석을 건너뜁니다.")
        return

    print("\n📊 루브릭 점수 분석 결과:")
    
    # 모델별 평균 점수
    if 'model_name' in df_scores.columns and df_scores['model_name'].nunique() > 0:
        avg_scores_by_model = df_scores.groupby('model_name')[score_columns].mean()
        print("\n모델별 평균 루브릭 점수:")
        print(avg_scores_by_model)

        # 시각화: 모델별 루브릭 점수 막대 그래프
        for score_col in score_columns:
            rubric_display_name = rubric_names.get(score_col.replace('_score', ''), score_col)
            fig = px.bar(avg_scores_by_model, y=score_col, 
                         title=f"모델별 {rubric_display_name} 평균 점수",
                         labels={'model_name': '모델', score_col: '평균 점수'},
                         template="plotly_white")
            fig.write_image(output_dir / f"avg_{score_col}_by_model.png")
            print(f"✅ 'avg_{score_col}_by_model.png' 저장 완료.")
        
        # 모든 루브릭 점수를 한 그래프에 (모델별)
        df_melted_model = avg_scores_by_model.reset_index().melt(id_vars='model_name', var_name='rubric', value_name='average_score')
        df_melted_model['rubric_display'] = df_melted_model['rubric'].apply(lambda x: rubric_names.get(x.replace('_score', ''), x))
        fig = px.bar(df_melted_model, x="model_name", y="average_score", color="rubric_display",
                     barmode="group", title="모델별 루브릭 평균 점수 비교",
                     labels={'model_name': '모델', 'average_score': '평균 점수', 'rubric_display': '루브릭'},
                     template="plotly_white")
        fig.write_image(output_dir / "all_rubrics_by_model.png")
        print(f"✅ 'all_rubrics_by_model.png' 저장 완료.")

    # 태스크별 평균 점수
    if 'task_name' in df_scores.columns and df_scores['task_name'].nunique() > 0:
        avg_scores_by_task = df_scores.groupby('task_name')[score_columns].mean()
        print("\n태스크별 평균 루브릭 점수:")
        print(avg_scores_by_task)

        # 시각화: 태스크별 루브릭 점수 막대 그래프
        for score_col in score_columns:
            rubric_display_name = rubric_names.get(score_col.replace('_score', ''), score_col)
            fig = px.bar(avg_scores_by_task, y=score_col, 
                         title=f"태스크별 {rubric_display_name} 평균 점수",
                         labels={'task_name': '태스크', score_col: '평균 점수'},
                         template="plotly_white")
            fig.write_image(output_dir / f"avg_{score_col}_by_task.png")
            print(f"✅ 'avg_{score_col}_by_task.png' 저장 완료.")
        
        # 모든 루브릭 점수를 한 그래프에 (태스크별)
        df_melted_task = avg_scores_by_task.reset_index().melt(id_vars='task_name', var_name='rubric', value_name='average_score')
        df_melted_task['rubric_display'] = df_melted_task['rubric'].apply(lambda x: rubric_names.get(x.replace('_score', ''), x))
        fig = px.bar(df_melted_task, x="task_name", y="average_score", color="rubric_display",
                     barmode="group", title="태스크별 루브릭 평균 점수 비교",
                     labels={'task_name': '태스크', 'average_score': '평균 점수', 'rubric_display': '루브릭'},
                     template="plotly_white")
        fig.write_image(output_dir / "all_rubrics_by_task.png")
        print(f"✅ 'all_rubrics_by_task.png' 저장 완료.")

    # 벤치마크 ID별 평균 점수
    if 'benchmark_id' in df_scores.columns and df_scores['benchmark_id'].nunique() > 0:
        avg_scores_by_benchmark = df_scores.groupby('benchmark_id')[score_columns].mean()
        print("\n벤치마크 ID별 평균 루브릭 점수:")
        print(avg_scores_by_benchmark)

        # 시각화: 벤치마크 ID별 루브릭 점수 막대 그래프
        for score_col in score_columns:
            rubric_display_name = rubric_names.get(score_col.replace('_score', ''), score_col)
            fig = px.bar(avg_scores_by_benchmark, y=score_col, 
                         title=f"벤치마크 ID별 {rubric_display_name} 평균 점수",
                         labels={'benchmark_id': '벤치마크 ID', score_col: '평균 점수'},
                         template="plotly_white")
            fig.write_image(output_dir / f"avg_{score_col}_by_benchmark.png")
            print(f"✅ 'avg_{score_col}_by_benchmark.png' 저장 완료.")
        
        # 모든 루브릭 점수를 한 그래프에 (벤치마크 ID별)
        df_melted_benchmark = avg_scores_by_benchmark.reset_index().melt(id_vars='benchmark_id', var_name='rubric', value_name='average_score')
        df_melted_benchmark['rubric_display'] = df_melted_benchmark['rubric'].apply(lambda x: rubric_names.get(x.replace('_score', ''), x))
        fig = px.bar(df_melted_benchmark, x="benchmark_id", y="average_score", color="rubric_display",
                     barmode="group", title="벤치마크 ID별 루브릭 평균 점수 비교",
                     labels={'benchmark_id': '벤치마크 ID', 'average_score': '평균 점수', 'rubric_display': '루브릭'},
                     template="plotly_white")
        fig.write_image(output_dir / "all_rubrics_by_benchmark.png")
        print(f"✅ 'all_rubrics_by_benchmark.png' 저장 완료.")

    # 벤치마크 ID별 세부 유형 (태스크) 및 모델별 루브릭 점수 분포
    if all(col in df_scores.columns for col in ['benchmark_id', 'task_name', 'model_name']) and \
       df_scores['benchmark_id'].nunique() > 0 and df_scores['task_name'].nunique() > 0 and df_scores['model_name'].nunique() > 0:
        
        # Group by benchmark_id, task_name, model_name and calculate mean for all score columns
        avg_scores_by_benchmark_task_model = df_scores.groupby(['benchmark_id', 'task_name', 'model_name'])[score_columns].mean().reset_index()
        
        # Melt the DataFrame for plotting all rubrics in one chart
        df_melted_all = avg_scores_by_benchmark_task_model.melt(id_vars=['benchmark_id', 'task_name', 'model_name'], 
                                                                 var_name='rubric', value_name='average_score')
        df_melted_all['rubric_display'] = df_melted_all['rubric'].apply(lambda x: rubric_names.get(x.replace('_score', ''), x))

        fig = px.bar(df_melted_all, x="benchmark_id", y="average_score", color="model_name", 
                     facet_col="task_name", facet_row="rubric_display",
                     title="벤치마크 ID별 태스크 및 모델 루브릭 평균 점수 비교",
                     labels={'benchmark_id': '벤치마크 ID', 'average_score': '평균 점수', 'model_name': '모델', 
                             'task_name': '태스크', 'rubric_display': '루브릭'},
                     barmode="group", template="plotly_white", height=800) # Adjust height for better readability
        fig.update_xaxes(matches=None) # Allow independent x-axes for facets
        fig.write_image(output_dir / "rubric_scores_by_benchmark_task_model.png")
        print(f"✅ 'rubric_scores_by_benchmark_task_model.png' 저장 완료.")


def main():
    parser = argparse.ArgumentParser(description="지정된 날짜의 raw_outputs 데이터를 분석하고 시각화합니다.")
    parser.add_argument("date", type=str, help="분석할 데이터의 날짜 (예: 2025-08-08)")
    args = parser.parse_args()

    date_to_analyze = args.date
    output_base_dir = Path("analysis_results")
    output_dir = output_base_dir / date_to_analyze
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"✨ {date_to_analyze} 날짜의 데이터 분석을 시작합니다.")
    print(f"결과는 {output_dir} 디렉토리에 저장됩니다.")

    df = load_data_for_date(date_to_analyze)

    if df.empty:
        print("분석할 데이터가 없습니다. 스크립트를 종료합니다.")
        return

    print(f"✅ 총 {len(df)}개의 데이터 항목을 로드했습니다.")
    analyze_passage_lengths(df, output_dir)
    analyze_rubric_scores(df, output_dir)

    print("\n🎉 분석 및 시각화가 완료되었습니다.")
    print(f"생성된 이미지 파일은 {output_dir} 디렉토리에서 확인해주세요.")

if __name__ == "__main__":
    main()
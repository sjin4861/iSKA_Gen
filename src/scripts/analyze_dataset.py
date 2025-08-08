import json
import numpy as np
import pandas as pd
from pathlib import Path
import re  # 정규표현식을 사용한 문장 분리

def split_sentences(text: str):
    """
    정규표현식을 사용하여 한국어 문장을 분리합니다.
    """
    # 한국어 문장 끝 기호: 마침표(.), 물음표(?), 느낌표(!)
    # 따옴표나 괄호 뒤에 오는 경우도 고려
    sentence_pattern = r'[.!?]+[\s]*'
    sentences = re.split(sentence_pattern, text)
    # 빈 문자열 제거 및 공백 정리
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def preprocess_passage(text: str) -> str:
    """
    지문 텍스트 전처리: 연속된 개행 문자 제거 및 공백 정리
    """
    if not text:
        return text
    
    # 연속된 개행 문자(\n\n, \n\n\n 등)를 단일 공백으로 변환
    cleaned_text = re.sub(r'\n+', ' ', text)
    
    # 연속된 공백을 단일 공백으로 변환
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # 앞뒤 공백 제거
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def find_long_passages(file_path: str, length_threshold: int = 500):
    """
    지정된 길이를 초과하는 지문들을 찾아서 출력합니다.
    """
    input_path = Path(file_path)
    if not input_path.exists():
        print(f"❌ 오류: 파일을 찾을 수 없습니다 - {input_path}")
        return

    # JSON 파일 로드
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return
    
    if not data:
        print("분석할 데이터가 없습니다.")
        return

    print(f"\n🔍 길이가 {length_threshold}자를 초과하는 지문 검색 결과")
    print("=" * 80)
    
    long_passages = []
    
    for idx, item in enumerate(data):
        if 'generated_passage' in item:
            original_passage = item['generated_passage']
            preprocessed_passage = preprocess_passage(original_passage)
            
            original_length = len(original_passage)
            preprocessed_length = len(preprocessed_passage)
            
            if original_length > length_threshold:
                # 소스 정보 추출
                korean_topic = "N/A"
                foreign_topic = "N/A"
                if 'source_item' in item:
                    source = item['source_item']
                    korean_topic = source.get('korean_topic', 'N/A')
                    foreign_topic = source.get('foreign_topic', 'N/A')
                
                long_passages.append({
                    'index': idx,
                    'korean_topic': korean_topic,
                    'foreign_topic': foreign_topic,
                    'original_length': original_length,
                    'preprocessed_length': preprocessed_length,
                    'original_passage': original_passage,
                    'preprocessed_passage': preprocessed_passage
                })
    
    if not long_passages:
        print(f"✅ {length_threshold}자를 초과하는 지문이 없습니다.")
        return
    
    print(f"📊 총 {len(long_passages)}개의 긴 지문을 발견했습니다.\n")
    
    for i, passage_info in enumerate(long_passages, 1):
        print(f"🔸 {i}번째 긴 지문 (인덱스: {passage_info['index']})")
        print(f"   📚 주제: {passage_info['korean_topic']} vs {passage_info['foreign_topic']}")
        print(f"   📏 원본 길이: {passage_info['original_length']}자")
        print(f"   📏 전처리 후 길이: {passage_info['preprocessed_length']}자")
        print(f"   📉 길이 감소: {passage_info['original_length'] - passage_info['preprocessed_length']}자")
        
        print(f"\n   📝 원본 지문:")
        print(f"   {passage_info['original_passage'][:200]}...")
        if len(passage_info['original_passage']) > 200:
            print(f"   ... (총 {passage_info['original_length']}자)")
        
        print(f"\n   ✨ 전처리 후 지문:")
        print(f"   {passage_info['preprocessed_passage'][:200]}...")
        if len(passage_info['preprocessed_passage']) > 200:
            print(f"   ... (총 {passage_info['preprocessed_length']}자)")
        
        print("\n" + "-" * 80 + "\n")
    
    # 통계 요약
    original_lengths = [p['original_length'] for p in long_passages]
    preprocessed_lengths = [p['preprocessed_length'] for p in long_passages]
    length_reductions = [p['original_length'] - p['preprocessed_length'] for p in long_passages]
    
    print(f"📈 긴 지문 통계 요약:")
    print(f"   📏 원본 길이 - 평균: {np.mean(original_lengths):.1f}자, 최대: {np.max(original_lengths)}자, 최소: {np.min(original_lengths)}자")
    print(f"   ✨ 전처리 후 길이 - 평균: {np.mean(preprocessed_lengths):.1f}자, 최대: {np.max(preprocessed_lengths)}자, 최소: {np.min(preprocessed_lengths)}자")
    print(f"   📉 평균 길이 감소: {np.mean(length_reductions):.1f}자")
    
    return long_passages

def analyze_passage_statistics(file_path: str):
    """
    JSON 형식의 지문 데이터셋 통계를 종합적으로 분석하여 표로 출력합니다.
    """
    input_path = Path(file_path)
    if not input_path.exists():
        print(f"❌ 오류: 파일을 찾을 수 없습니다 - {input_path}")
        return

    # JSON 파일 로드
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return
    
    if not data:
        print("분석할 데이터가 없습니다.")
        return

    # 지문 데이터 추출
    passages = []
    korean_topics = []
    foreign_topics = []
    
    for item in data:
        if 'generated_passage' in item:
            passages.append(item['generated_passage'])
        
        # source_item에서 토픽 정보 추출
        if 'source_item' in item:
            source = item['source_item']
            if 'korean_topic' in source:
                korean_topics.append(source['korean_topic'])
            if 'foreign_topic' in source:
                foreign_topics.append(source['foreign_topic'])
    
    if not passages:
        print("분석할 지문이 없습니다.")
        return

    # 1. 기본 통계 계산
    char_counts = [len(p) for p in passages]
    word_counts = [len(p.split()) for p in passages]
    
    # 2. 가독성 통계 계산 (문장 분리)
    sentence_counts = []
    avg_sentence_lengths = []
    for p in passages:
        sentences = split_sentences(p)
        sentence_counts.append(len(sentences))
        if sentences:  # 문장이 있는 경우에만 평균 계산
            avg_sentence_lengths.append(np.mean([len(s.split()) for s in sentences]))
        else:
            avg_sentence_lengths.append(0)

    # 3. 어휘 다양성 통계 계산
    all_words = [word for p in passages for word in p.split()]
    total_tokens = len(all_words)
    unique_types = len(set(all_words))
    ttr = unique_types / total_tokens if total_tokens > 0 else 0

    # 4. 결과 정리 (핵심 5개 지표만)
    summary = {
        "지표 (Metric)": [
            "총 지문 수 (Num Passages)",
            "글자 수 (Characters)",
            "단어 수 (Words)",
            "문장 수 (Sentences)",
            "평균 문장 길이 (Avg Sent Length)"
        ],
        "평균 (Mean)": [
            f"{len(passages)}개",
            f"{np.mean(char_counts):.2f}",
            f"{np.mean(word_counts):.2f}",
            f"{np.mean(sentence_counts):.2f}",
            f"{np.mean(avg_sentence_lengths):.2f} 단어"
        ],
        "표준편차 (Std)": [
            "-",
            f"{np.std(char_counts):.2f}",
            f"{np.std(word_counts):.2f}",
            f"{np.std(sentence_counts):.2f}",
            f"{np.std(avg_sentence_lengths):.2f}"
        ],
        "최소 (Min)": [
            "-",
            np.min(char_counts),
            np.min(word_counts),
            np.min(sentence_counts),
            f"{np.min(avg_sentence_lengths):.2f}"
        ],
        "최대 (Max)": [
            "-",
            np.max(char_counts),
            np.max(word_counts),
            np.max(sentence_counts),
            f"{np.max(avg_sentence_lengths):.2f}"
        ]
    }
    
    df = pd.DataFrame(summary)
    print(f"\n📊 '{input_path.name}' 파일 통계 분석 결과")
    print(df.to_string(index=False))
    
    # LaTeX 형식으로 출력
    print(f"\n📋 LaTeX 형식:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{|l|c|c|c|c|}")
    print("\\hline")
    print("지표 (Metric) & 평균 (Mean) & 표준편차 (Std) & 최소 (Min) & 최대 (Max) \\\\")
    print("\\hline")
    
    for i in range(len(summary["지표 (Metric)"])):
        metric = summary["지표 (Metric)"][i]
        mean = summary["평균 (Mean)"][i]
        std = summary["표준편차 (Std)"][i]
        min_val = summary["최소 (Min)"][i]
        max_val = summary["최대 (Max)"][i]
        print(f"{metric} & {mean} & {std} & {min_val} & {max_val} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{지문 데이터셋 통계 분석 결과}")
    print("\\label{tab:passage_statistics}")
    print("\\end{table}")

def analyze_rubric_scores_by_model(evaluation_dir: str):
    """
    각 모델별로 6개 루브릭의 평균 점수를 계산하여 LaTeX 표 형식으로 출력합니다.
    
    Args:
        evaluation_dir: 평가 결과가 저장된 디렉토리 경로 (src/data/evaluations/2025-08-05/misc)
    """
    eval_path = Path(evaluation_dir)
    if not eval_path.exists():
        print(f"❌ 평가 디렉토리를 찾을 수 없습니다: {evaluation_dir}")
        return
    
    # 루브릭 정의
    rubrics = [
        "completeness_for_guidelines",
        "clarity_of_core_theme", 
        "reference_groundedness",
        "logical_flow",
        "korean_quality",
        "l2_learner_suitability"
    ]
    
    # 루브릭 한국어 이름
    rubric_names = {
        "completeness_for_guidelines": "평가 지침 완전성",
        "clarity_of_core_theme": "핵심 주제 명확성", 
        "reference_groundedness": "참고자료 기반성",
        "logical_flow": "논리적 흐름",
        "korean_quality": "한국어 품질",
        "l2_learner_suitability": "L2 학습자 적합성"
    }
    
    model_scores = {}
    
    # 각 모델별 평가 파일 찾기
    for eval_file in eval_path.rglob("*.json"):
        if "eval_rubric" in eval_file.name:
            # 모델명 추출
            parts = eval_file.parts
            model_name = "unknown"
            for part in parts:
                if "_evaluation" in part:
                    model_name = part.replace("_evaluation", "")
                    break
            
            if model_name == "unknown":
                continue
                
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if model_name not in model_scores:
                    model_scores[model_name] = {rubric: [] for rubric in rubrics}
                
                # 각 항목에서 점수 추출
                for item in data:
                    if 'evaluation' not in item:
                        continue
                    
                    evaluation = item['evaluation']
                    for rubric in rubrics:
                        score_key = f"{rubric}_score"
                        if score_key in evaluation:
                            model_scores[model_name][rubric].append(evaluation[score_key])
                            
            except Exception as e:
                print(f"⚠️ 파일 로드 실패: {eval_file.name} - {e}")
    
    if not model_scores:
        print("❌ 평가 데이터를 찾을 수 없습니다.")
        return
    
    # 평균 계산
    model_averages = {}
    for model_name, scores in model_scores.items():
        model_averages[model_name] = {}
        for rubric in rubrics:
            if scores[rubric]:
                model_averages[model_name][rubric] = np.mean(scores[rubric])
            else:
                model_averages[model_name][rubric] = 0.0
    
    # 모델명 정리 (언더스코어를 하이픈으로 변경)
    clean_model_names = {}
    for model_name in model_averages.keys():
        clean_name = model_name.replace("_", "-")
        clean_model_names[model_name] = clean_name
    
    print(f"\n📊 모델별 루브릭 점수 분석 결과")
    print("=" * 100)
    
    # 데이터프레임으로 정리
    df_data = {}
    df_data['루브릭'] = [rubric_names[rubric] for rubric in rubrics]
    
    for model_name in sorted(model_averages.keys()):
        clean_name = clean_model_names[model_name]
        df_data[clean_name] = [f"{model_averages[model_name][rubric]:.2f}" for rubric in rubrics]
    
    df = pd.DataFrame(df_data)
    print(df.to_string(index=False))
    
    # LaTeX 표 생성
    print(f"\n📋 LaTeX 표 형식:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\small")  # 표 크기 조정
    
    # 모델 수에 따라 컬럼 수 조정
    model_names_sorted = sorted([clean_model_names[name] for name in model_averages.keys()])
    num_models = len(model_names_sorted)
    
    # 테이블 헤더 생성
    col_spec = "|l|" + "c|" * num_models
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\hline")
    
    # 헤더 행 생성
    header = "루브릭 (Rubric)"
    for model_name in model_names_sorted:
        header += f" & {model_name}"
    header += " \\\\"
    print(header)
    print("\\hline")
    
    # 데이터 행 생성
    for i, rubric in enumerate(rubrics):
        row = rubric_names[rubric]
        for model_name in model_averages.keys():
            clean_name = clean_model_names[model_name]
            if clean_name in model_names_sorted:
                score = model_averages[model_name][rubric]
                row += f" & {score:.2f}"
        row += " \\\\"
        print(row)
    
    print("\\hline")
    
    # 전체 평균 계산 및 추가
    overall_averages = {}
    for model_name in model_averages.keys():
        scores = [model_averages[model_name][rubric] for rubric in rubrics]
        overall_averages[model_name] = np.mean(scores)
    
    # 전체 평균 행
    avg_row = "\\textbf{전체 평균}"
    for model_name in model_averages.keys():
        clean_name = clean_model_names[model_name]
        if clean_name in model_names_sorted:
            avg_score = overall_averages[model_name]
            avg_row += f" & \\textbf{{{avg_score:.2f}}}"
    avg_row += " \\\\"
    print(avg_row)
    print("\\hline")
    
    print("\\end{tabular}")
    print("\\caption{모델별 루브릭 평가 점수 비교}")
    print("\\label{tab:model_rubric_scores}")
    print("\\end{table}")
    
    # 통계 요약
    print(f"\n📈 통계 요약:")
    for model_name in sorted(model_averages.keys()):
        clean_name = clean_model_names[model_name]
        total_items = sum(len(model_scores[model_name][rubric]) for rubric in rubrics)
        avg_score = overall_averages[model_name]
        print(f"   {clean_name}: 평균 {avg_score:.2f}점 (총 {total_items}개 평가)")
    
    return model_averages

def analyze_passage_quality(file_path: str):
    """
    지문의 품질 관련 세부 분석을 수행합니다. (사용 안함)
    """
    pass

# --- 실행 ---
if __name__ == "__main__":
    DATASET_PATH = "src/data/raw_outputs/2025-08-05/passage/A.X-4.0-Light/create_passage_rubric_aware/benchmark_2_v1.0.0_passage_agent.create_passage_rubric_aware.json"
    DATASET_PATH_2 = "src/data/raw_outputs/2025-07-28/passage/Gemini-2.5-Pro/create_passage_rubric_aware/benchmark_1_v1.0.0_passage_agent.create_passage_rubric_aware.json"
    EVALUATION_DIR = "src/data/evaluations/2025-08-05/misc"  # 평가 결과 디렉토리
    
    # 1. 기본 통계 분석
    # analyze_passage_statistics(DATASET_PATH_2)
    
    # 2. 길이 500자 초과 지문 검색 및 전처리
    # print("\n" + "=" * 80)
    # find_long_passages(DATASET_PATH_2, length_threshold=500)
    
    # 3. 모델별 루브릭 점수 분석
    analyze_rubric_scores_by_model(EVALUATION_DIR)


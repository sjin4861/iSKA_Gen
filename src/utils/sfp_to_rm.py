#!/usr/bin/env python
# coding: utf-8

"""
Score-based Pairwise Data Generator for Reward Model Training

이 스크립트는 지문 채점 결과를 바탕으로 Reward Model 훈련용 pairwise 데이터를 생성합니다.

단계별 과정:
1. 기준별로 데이터 분류 및 동일한 source item에 대해서 점수별로 정렬
2. 점수 차이가 나는 지문끼리 쌍을 생성 (5>4, 5>3, 4>3 등)
3. 점수 차이만큼 margin을 설정
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
from collections import defaultdict, Counter
import itertools

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'src'))

# 루브릭 정의
RUBRICS = [
    "completeness_for_guidelines",
    "clarity_of_core_theme", 
    "reference_groundedness",
    "logical_flow",
    "korean_quality",
    "l2_learner_suitability"
]

def load_evaluation_data(evaluation_dir: str) -> Dict[str, List[Dict]]:
    """
    평가 결과 데이터를 로드합니다.
    
    Args:
        evaluation_dir: 평가 결과가 저장된 디렉토리 경로
        
    Returns:
        Dict[str, List[Dict]]: 모델별 평가 결과 딕셔너리
    """
    evaluation_data = {}
    eval_path = Path(evaluation_dir)
    
    if not eval_path.exists():
        print(f"❌ 평가 디렉토리를 찾을 수 없습니다: {evaluation_dir}")
        return evaluation_data
    
    print(f"🔍 디버깅: 검색 시작 디렉토리 = {eval_path}")
    
    # 평가 파일 검색 (eval_rubric 패턴으로 검색)
    for eval_file in eval_path.rglob("*.json"):
        print(f"🔍 디버깅: 찾은 파일 = {eval_file}")
        if "eval_rubric" in eval_file.name:
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 파일 경로에서 모델명 추출
                parts = eval_file.parts
                model_name = "unknown"
                for part in parts:
                    if "_evaluation" in part:
                        model_name = part.replace("_evaluation", "")
                        break
                
                # 파일명에서 모델명과 벤치마크 정보 추출
                filename = eval_file.name
                # benchmark_1_v1.0.0_eval_rubric.json 형식
                if "eval_rubric" in filename:
                    key = f"{model_name}_{filename}"
                    evaluation_data[key] = data
                    print(f"✅ 로드됨: {key} ({len(data)}개 항목)")
                    print(f"🔍 디버깅: 전체 경로 = {eval_file}")
                    
            except Exception as e:
                print(f"⚠️ 파일 로드 실패: {eval_file.name} - {e}")
    
    print(f"🔍 디버깅: 총 로드된 파일 수 = {len(evaluation_data)}")
    print(f"🔍 디버깅: 로드된 키 목록 = {list(evaluation_data.keys())}")
    
    return evaluation_data

def extract_source_key(source_item: Dict) -> str:
    """
    source_item에서 고유 키를 생성합니다.
    """
    korean_topic = source_item.get('korean_topic', 'N/A')
    foreign_topic = source_item.get('foreign_topic', 'N/A')
    return f"{korean_topic}||{foreign_topic}"

def group_by_source_and_score(evaluation_data: Dict[str, List[Dict]], rubric: str) -> Dict[str, Dict[int, List[Dict]]]:
    """
    1단계: 동일한 source item에 대해 점수별로 데이터를 그룹화합니다.
    
    Args:
        evaluation_data: 평가 결과 데이터
        rubric: 평가 기준 (예: "completeness_for_guidelines")
        
    Returns:
        Dict[source_key, Dict[score, List[passage_data]]]
    """
    grouped_data = defaultdict(lambda: defaultdict(list))
    
    print(f"🔍 디버깅: group_by_source_and_score 시작")
    print(f"🔍 디버깅: 처리할 파일 수 = {len(evaluation_data)}")
    print(f"🔍 디버깅: 찾을 루브릭 = {rubric}")
    
    for file_path, data in evaluation_data.items():
        print(f"🔍 디버깅: 처리 중인 파일 = {file_path}")
        print(f"🔍 디버깅: 해당 파일의 데이터 수 = {len(data)}")
        
        valid_items = 0
        for item in data:
            if 'evaluation' not in item or 'source_item' not in item:
                continue
                
            evaluation = item['evaluation']
            score_key = f"{rubric}_score"
            
            if score_key not in evaluation:
                continue
                
            score = evaluation[score_key]
            source_key = extract_source_key(item['source_item'])
            
            print(f"🔍 디버깅: source_key = {source_key[:50]}...")
            print(f"🔍 디버깅: score = {score}")
            
            # 메타데이터 추가
            passage_data = {
                'source_item': item['source_item'],
                'generated_passage': item['generated_passage'],
                'evaluation': evaluation,
                'score': score,
                'file_path': file_path,
                'rubric': rubric
            }
            
            grouped_data[source_key][score].append(passage_data)
            valid_items += 1
        
        print(f"🔍 디버깅: 해당 파일에서 유효한 아이템 수 = {valid_items}")
    
    print(f"🔍 디버깅: 그룹화된 source_key 수 = {len(grouped_data)}")
    
    # 각 source_key별 점수 분포 출력
    for source_key, score_groups in list(grouped_data.items())[:3]:  # 첫 3개만 출력
        print(f"🔍 디버깅: source_key = {source_key[:50]}...")
        print(f"🔍 디버깅:   점수별 분포 = {dict(score_groups.keys())}")
        for score, passages in score_groups.items():
            print(f"🔍 디버깅:     점수 {score}: {len(passages)}개 지문")

    return grouped_data

def create_pairs(grouped_data: Dict[str, Dict[int, List[Dict]]], min_score_diff: int = 1) -> List[Dict]:
    """
    2단계: 점수 차이가 나는 지문끼리 쌍을 생성합니다.
    
    Args:
        grouped_data: 그룹화된 데이터
        min_score_diff: 최소 점수 차이 (기본값: 1)
        
    Returns:
        List[Dict]: 생성된 쌍 데이터 리스트
    """
    pairs = []
    
    print(f"🔍 디버깅: create_pairs 시작")
    print(f"🔍 디버깅: 처리할 source_key 수 = {len(grouped_data)}")
    
    for source_key, score_groups in grouped_data.items():
        scores = sorted(score_groups.keys())
        print(f"🔍 디버깅: source_key = {source_key[:50]}...")
        print(f"🔍 디버깅:   사용 가능한 점수들 = {scores}")
        
        # 모든 점수 조합에 대해 쌍 생성
        for high_score, low_score in itertools.combinations(scores, 2):
            score_diff = high_score - low_score
            print(f"🔍 디버깅:   점수 조합 시도: {high_score} vs {low_score} (차이: {score_diff})")
            
            if score_diff < min_score_diff:
                print(f"🔍 디버깅:     점수 차이가 최소값({min_score_diff})보다 작음 - 건너뜀")
                continue
                
            high_passages = score_groups[high_score]
            low_passages = score_groups[low_score]
            
            print(f"🔍 디버깅:     높은 점수({high_score}) 지문 수: {len(high_passages)}")
            print(f"🔍 디버깅:     낮은 점수({low_score}) 지문 수: {len(low_passages)}")
            
            # 높은 점수와 낮은 점수 지문들 간의 모든 조합
            pair_count = 0
            for high_passage in high_passages:
                for low_passage in low_passages:
                    # 3단계: margin 계산
                    margin = high_score - low_score
                    
                    pair = {
                        'source_key': source_key,
                        'high_score': high_score,
                        'low_score': low_score,
                        'margin': margin,
                        'chosen_passage': high_passage,
                        'rejected_passage': low_passage,
                        'rubric': high_passage['rubric']
                    }
                    
                    pairs.append(pair)
                    pair_count += 1
            
            print(f"🔍 디버깅:     생성된 쌍 수: {pair_count}")
    
    print(f"🔍 디버깅: 총 생성된 쌍 수 = {len(pairs)}")
    return pairs

def format_for_reward_model(pairs: List[Dict], rubric: str) -> List[Dict]:
    """
    Reward Model 훈련용 형식으로 변환합니다.
    IMP_...json 파일 형식을 참고하여 생성합니다.
    """
    formatted_data = []
    
    for pair in pairs:
        # 공통 조건 생성 (source_item에서 추출)
        source_item = pair['chosen_passage']['source_item']
        
        # 평가 기준 설명 생성
        rubric_descriptions = {
            "completeness_for_guidelines": "지문이 주어진 3개의 `공통 조건` 각각에 대한 질문을 모두 만들 수 있을 만큼, 충분하고 균형 잡힌 정보를 포함하고 있는가?",
            "clarity_of_core_theme": "지문이 한국 문화와 외국 문화의 핵심 주제를 명확하고 이해하기 쉽게 제시하고 있는가?",
            "reference_groundedness": "지문이 제공된 참고 자료의 내용을 정확하게 반영하고 근거로 활용하고 있는가?",
            "logical_flow": "지문의 내용이 논리적 순서와 연결성을 갖추어 일관성 있게 구성되어 있는가?",
            "korean_quality": "지문이 문법적으로 올바르고 자연스러운 한국어로 작성되어 있는가?",
            "l2_learner_suitability": "지문이 L2 학습자(한국어를 외국어로 학습하는 사람)에게 적합한 수준과 표현으로 작성되어 있는가?"
        }
        
        rubric_names = {
            "completeness_for_guidelines": "평가 지침 완전성",
            "clarity_of_core_theme": "핵심 주제 명확성", 
            "reference_groundedness": "참고자료 기반성",
            "logical_flow": "논리적 흐름",
            "korean_quality": "한국어 품질",
            "l2_learner_suitability": "L2 학습자 적합성"
        }
        
        # 문제 유형과 평가 목표는 source_item이 없으므로 기본값 설정
        problem_types = "문화 비교하기, 내용 분석하기, 의견 제시하기"
        eval_goals = "문화적 차이점과 공통점을 파악하여 비교 분석하는 능력을 평가한다., 주어진 내용의 핵심을 파악하고 분석하는 능력을 평가한다., 문화적 현상에 대한 자신의 의견을 논리적으로 제시하는 능력을 평가한다."
        
        # 대화 형식 생성
        conversation = {
            "conversations": [
                {
                    "from": "human",
                    "value": f"두 개의 한국어 독해 지문 A와 B가 주어집니다.\n두 지문은 모두 아래 [공통 조건]을 바탕으로 생성되었습니다.\n\n[공통 조건]:\n- 문제 유형: {problem_types}\n- 평가 목표: {eval_goals}\n\n다음은 생성된 지문의 품질을 평가하는 기준입니다:\n\n**평가 기준:**\n- {rubric_descriptions[rubric]}\n\n위 기준에 따라, 두 지문 중 [{rubric_names[rubric]}] 측면에서 더 우수한 것을 선택하세요.\n"
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": pair['chosen_passage']['generated_passage']
            },
            "rejected": {
                "from": "gpt", 
                "value": pair['rejected_passage']['generated_passage']
            },
            "metadata": {
                "source_key": pair['source_key'],
                "rubric": rubric,
                "high_score": pair['high_score'],
                "low_score": pair['low_score'],
                "margin": pair['margin'],
                "korean_topic": source_item.get('korean_topic', 'N/A'),
                "foreign_topic": source_item.get('foreign_topic', 'N/A'),
                "korean_context": source_item.get('korean_context', 'N/A'),
                "foreign_context": source_item.get('foreign_context', 'N/A'),
                "guidelines": {
                    "problem_types": problem_types,
                    "eval_goals": eval_goals
                }
            }
        }
        
        formatted_data.append(conversation)
    
    return formatted_data

def save_pairwise_data(data: List[Dict], rubric: str, output_dir: str, date_str: str = None):
    """
    생성된 pairwise 데이터를 저장합니다.
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"IMP_{rubric}_{date_str}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 저장됨: {filepath} ({len(data)}개 쌍)")
    return filepath

def analyze_pair_statistics(pairs: List[Dict], rubric: str):
    """
    생성된 쌍 데이터의 통계를 분석합니다.
    """
    print(f"\n📊 {rubric} 루브릭 쌍 생성 통계:")
    print(f"   총 쌍 개수: {len(pairs)}")
    
    # 점수 차이별 분포
    margin_counts = Counter([pair['margin'] for pair in pairs])
    print(f"   점수 차이별 분포:")
    for margin in sorted(margin_counts.keys()):
        print(f"     차이 {margin}점: {margin_counts[margin]}개")
    
    # 점수별 분포
    high_score_counts = Counter([pair['high_score'] for pair in pairs])
    low_score_counts = Counter([pair['low_score'] for pair in pairs])
    print(f"   높은 점수 분포: {dict(sorted(high_score_counts.items()))}")
    print(f"   낮은 점수 분포: {dict(sorted(low_score_counts.items()))}")
    
    # 소스별 분포
    source_counts = Counter([pair['source_key'] for pair in pairs])
    print(f"   소스 아이템별 쌍 개수 (상위 5개):")
    for source_key, count in source_counts.most_common(5):
        topics = source_key.split('||')
        korean_topic = topics[0][:20] + "..." if len(topics[0]) > 20 else topics[0]
        foreign_topic = topics[1][:20] + "..." if len(topics[1]) > 20 else topics[1]
        print(f"     {korean_topic} vs {foreign_topic}: {count}개")

def generate_pairwise_data_for_rubric(evaluation_dir: str, rubric: str, output_dir: str, 
                                     min_score_diff: int = 1, date_str: str = None) -> str:
    """
    특정 루브릭에 대한 pairwise 데이터를 생성합니다.
    
    Args:
        evaluation_dir: 평가 결과 디렉토리
        rubric: 평가 기준
        output_dir: 출력 디렉토리
        min_score_diff: 최소 점수 차이
        date_str: 날짜 문자열
        
    Returns:
        str: 저장된 파일 경로
    """
    print(f"\n🔄 {rubric} 루브릭 pairwise 데이터 생성 시작...")
    
    # 1단계: 평가 데이터 로드
    print("1️⃣ 평가 데이터 로드 중...")
    evaluation_data = load_evaluation_data(evaluation_dir)
    
    if not evaluation_data:
        print(f"❌ {rubric}: 평가 데이터를 찾을 수 없습니다.")
        return None
    
    # 2단계: 소스별/점수별 그룹화
    print("2️⃣ 데이터 그룹화 중...")
    grouped_data = group_by_source_and_score(evaluation_data, rubric)
    
    if not grouped_data:
        print(f"❌ {rubric}: 그룹화할 데이터가 없습니다.")
        return None
    
    # 3단계: 쌍 생성
    print("3️⃣ 쌍 생성 중...")
    pairs = create_pairs(grouped_data, min_score_diff)
    
    if not pairs:
        print(f"❌ {rubric}: 생성할 쌍이 없습니다.")
        return None
    
    # 4단계: 형식 변환
    print("4️⃣ 형식 변환 중...")
    formatted_data = format_for_reward_model(pairs, rubric)
    
    # 5단계: 통계 분석
    analyze_pair_statistics(pairs, rubric)
    
    # 6단계: 저장
    print("5️⃣ 데이터 저장 중...")
    filepath = save_pairwise_data(formatted_data, rubric, output_dir, date_str)
    
    return filepath

def generate_all_pairwise_data(evaluation_dir: str, output_dir: str, 
                              min_score_diff: int = 1, date_str: str = None):
    """
    모든 루브릭에 대해 pairwise 데이터를 생성합니다.
    """
    print("🚀 모든 루브릭에 대한 pairwise 데이터 생성 시작...")
    print(f"   평가 데이터 디렉토리: {evaluation_dir}")
    print(f"   출력 디렉토리: {output_dir}")
    print(f"   최소 점수 차이: {min_score_diff}")
    print(f"   날짜: {date_str or 'auto'}")
    
    generated_files = []
    
    for rubric in RUBRICS:
        try:
            filepath = generate_pairwise_data_for_rubric(
                evaluation_dir, rubric, output_dir, min_score_diff, date_str
            )
            if filepath:
                generated_files.append(filepath)
        except Exception as e:
            print(f"❌ {rubric} 처리 중 오류 발생: {e}")
    
    print(f"\n🎉 완료! {len(generated_files)}개 파일 생성:")
    for filepath in generated_files:
        print(f"   ✅ {filepath}")

# 실행 예시
if __name__ == "__main__":
    # 설정
    # 실제 평가 결과 구조: /home/sjin4861/25-1/HCLT/iSKA_Gen/src/data/evaluations/2025-08-05/misc/{MODEL_NAME}_evaluation/eval_rubric/benchmark_{ID}_v1.0.0_eval_rubric.json
    EVALUATION_DIR = "src/data/evaluations/2025-08-05/misc"  # 모든 모델 디렉토리를 포함하는 상위 경로
    OUTPUT_DIR = "src/data/pairwise_data/train/v4/generated"  # 생성된 pairwise 데이터 저장 디렉토리
    MIN_SCORE_DIFF = 1  # 최소 점수 차이
    DATE_STR = "2025-08-08"  # 날짜 문자열
    
    # 단일 루브릭 테스트
    print("🧪 단일 루브릭 테스트:")
    test_rubric = "completeness_for_guidelines"
    generate_pairwise_data_for_rubric(
        EVALUATION_DIR, test_rubric, OUTPUT_DIR, MIN_SCORE_DIFF, DATE_STR
    )
    
    # 전체 루브릭 생성 (주석 해제하여 사용)
    # print("\n" + "="*80)
    # generate_all_pairwise_data(EVALUATION_DIR, OUTPUT_DIR, MIN_SCORE_DIFF, DATE_STR)

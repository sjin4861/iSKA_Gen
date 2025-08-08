#!/usr/bin/env python
# coding: utf-8

# # RM 훈련 데이터셋 생성
# 
# 이 노트북은 Reward Model 훈련을 위한 세 가지 유형의 선호도 쌍 데이터셋을 생성합니다:
# 
# 1. **SPF (Supervised Preference Dataset by Filtering)**: GPT-4o 루브릭 평가 기반
# 2. **IMP (Inter-Model Performance Preference Dataset)**: 모델 성능 차이 기반
# 3. **ICP (Intra-Model Contrastive Preference Dataset)**: 대조적 지문 생성 기반
# 
# 각 데이터셋은 6개 루브릭(평가 지침 완전성, 핵심 주제 명확성, 참고 자료 기반성, 논리적 흐름 및 구조, 한국어 품질, L2 학습자 적합성)에 대해 생성됩니다.
# - completeness_for_guidelines
# - core_theme_clarity
# - reference_groundedness
# - logical_flow_and_structure
# - korean_quality
# - l2_learner_suitability

# In[1]:


import sys
from pathlib import Path
import json
import random
from datetime import datetime

# 프로젝트 경로 설정
sys.path.append(str(Path.cwd().parent))
sys.path.append(str(Path.cwd().parent / 'modules'))
sys.path.append(str(Path.cwd().parent / 'utils'))
print("✅ 경로 설정 완료")


from utils.rm_dataset_generator import (
    create_spf_dataset,
    create_imp_dataset,
    create_icp_dataset,
)
from utils.output_loader import load_passages
from utils.benchmark_loader import get_guideline_by_id


# In[2]:


# 실제 사용 모델 리스트
MODEL_LIST = [
    "Midm-2.0-Base-Instruct",
    "EXAONE-3.5-7.8B-Instruct",
    "A.X-4.0-Light",
    "llama3.1_korean_v1.1_sft_by_aidx",
    "llama3-bllossom-3b"
]
# 기본 설정
OPENAI_MODEL = 'gpt-4o'
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
BENCH_ID_LIST = [1, 2, 3, 4, 5]

# 날짜 문자열 변수
DATE_STR = "2025-07-28"


# In[3]:


print(f'🎯 설정 완료:')
print(f'   OpenAI 모델: {OPENAI_MODEL}')
print(f'   랜덤 시드: {RANDOM_SEED}')


# In[6]:


IMP_HIGH_LIST = ["A.X-4.0-Light"]
IMP_LOW_LIST = ["llama3.1_korean_v1.1_sft_by_aidx"]
for high_model, low_model in zip(IMP_HIGH_LIST, IMP_LOW_LIST):
    print(f'🚀 IMP 데이터셋 생성 시작... High: {high_model}, Low: {low_model}')

    raw_imp_high_passages = []
    raw_imp_low_passages = []
    for bench_id in BENCH_ID_LIST:
        guideline = get_guideline_by_id(file_or_version="v1.0.0_20250725", benchmark_id=bench_id)
        print(f'   - 벤치마크 ID: {bench_id}, {guideline}')
        imp_high_passages = load_passages(
            model_name=high_model,
            benchmark_id=bench_id,
            benchmark_version='v1.0.0',
            template_key="passage_agent.create_passage_rubric_aware",
            date_str=DATE_STR
        )
        imp_low_passages = load_passages(
            model_name=low_model,
            benchmark_id=bench_id,
            benchmark_version='v1.0.0',
            template_key="passage_agent.create_passage_rubric_aware",
            date_str=DATE_STR
        )
        problem_types = guideline.get('problem_types', [])
        eval_goals = guideline.get('eval_goals', [])

        for p in imp_high_passages:
            p['source_item']['problem_types'] = problem_types
            p['source_item']['eval_goals'] = eval_goals
        for p in imp_low_passages:
            p['source_item']['problem_types'] = problem_types
            p['source_item']['eval_goals'] = eval_goals
        raw_imp_high_passages += imp_high_passages
        raw_imp_low_passages += imp_low_passages


    # print(raw_imp_high_passages)
    # 두 리스트의 길이가 다를 경우 최소 길이만큼만 매칭
    min_len = min(len(raw_imp_high_passages), len(raw_imp_low_passages))
    imp_high_passages = raw_imp_high_passages[:min_len]
    imp_low_passages = raw_imp_low_passages[:min_len]
    imp_dataset, imp_files = create_imp_dataset(
        high_perf=raw_imp_high_passages,
        low_perf=raw_imp_low_passages,
    )
    print('\n✅ IMP 데이터셋 생성 완료! ')
    print('📁 저장된 파일들: ')
    for rubric, file_path in imp_files.items():
        print(f'   {rubric}: {Path(file_path).name}')


# In[4]:


BASE_MODEL = "Midm-2.0-Base-Instruct"
LOW_TEMPLATE_KEY_LIST = [
    # "passage_agent.violate_flow_severely",
    # "passage_agent.violate_korean_quality_severely",
    # "passage_agent.violate_l2_suitability_severely",
    # "passage_agent.violate_completeness_severely",
    "passage_agent.violate_clarity_severely",
    # "passage_agent.violate_groundedness_severely",
]
RUBRIC_DICT = {
    "passage_agent.violate_flow_severely": "logical_flow",
    "passage_agent.violate_korean_quality_severely": "korean_quality",
    "passage_agent.violate_l2_suitability_severely": "l2_learner_suitability",
    "passage_agent.violate_completeness_severely": "completeness_for_guidelines",
    "passage_agent.violate_clarity_severely": "clarity_of_core_theme",
    "passage_agent.violate_groundedness_severely": "reference_groundedness",

}
print(f'🚀 ICP 데이터셋 생성 시작... ') 
#  ['completeness_for_guidelines', 'clarity_of_core_theme', 'reference_groundedness', 'logical_flow', 'korean_quality', 'l2_learner_suitability']


for low_template_key in LOW_TEMPLATE_KEY_LIST:
    raw_icp_high_passages = []
    raw_icp_low_passages = []
    for bench_id in BENCH_ID_LIST:
        guideline = get_guideline_by_id(file_or_version="v1.0.0_20250725", benchmark_id=bench_id)
        print(f'   - 벤치마크 ID: {bench_id}, {guideline}')
        icp_high_passages = load_passages(
            model_name=BASE_MODEL,
            benchmark_id=bench_id,
            benchmark_version='v1.0.0',
            template_key="passage_agent.create_passage_rubric_aware",
            date_str=DATE_STR
        )
        icp_low_passages = load_passages(
            model_name=BASE_MODEL,
            benchmark_id=bench_id,
            benchmark_version='v1.0.0',
            template_key= low_template_key,
            date_str=DATE_STR
        )
        problem_types = guideline.get('problem_types', [])
        eval_goals = guideline.get('eval_goals', [])

        for p in icp_high_passages:
            p['source_item']['problem_types'] = problem_types
            p['source_item']['eval_goals'] = eval_goals
        for p in icp_low_passages:
            p['source_item']['problem_types'] = problem_types
            p['source_item']['eval_goals'] = eval_goals
        raw_icp_high_passages += icp_high_passages
        raw_icp_low_passages += icp_low_passages

    # print(raw_icp_high_passages)
    # 두 리스트의 길이가 다를 경우 최소 길이만큼만 매칭
    min_len = min(len(raw_icp_high_passages), len(raw_icp_low_passages))
    icp_high_passages = raw_icp_high_passages[:min_len]
    icp_low_passages = raw_icp_low_passages[:min_len]
    icp_dataset, icp_files = create_icp_dataset(
        high_perf=raw_icp_high_passages,
        low_perf=raw_icp_low_passages,
        rubric=RUBRIC_DICT[low_template_key],
    )
    print('\n✅ ICP 데이터셋 생성 완료! ')
    print('📁 저장된 파일들: ')
    for rubric, file_path in icp_files.items():
        print(f'   {rubric}: {Path(file_path).name}')


# In[ ]:





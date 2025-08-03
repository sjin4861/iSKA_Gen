#!/usr/bin/env python3
import os
import json
import yaml
import re
from pathlib import Path

"""
IMP → RM 데이터셋 변환 유틸리티
IMP 데이터셋을 RM 훈련용 pairwise 포맷으로 변환합니다.
"""

# Example usage:
# imp_data = ... # List of IMP dicts
# rubric_prompts = load_yaml_prompts('src/config/prompts/iska/preference_eval.yaml')
# rm_dataset = imp_to_rm_format(imp_data, rubric_prompts)
"""
IMP (Inter-Model Performance Preference Dataset) 생성 전용 모듈

이 모듈은 고성능/저성능 모델의 지문을 비교하여 루브릭별 선호도 쌍을 생성합니다.

IMP 데이터셋 스키마 예시:
{
  "pair_id": "imp_completeness_for_guidelines_001_0001",
  "rubric": "completeness_for_guidelines",
  "source_item": {
    "korean_topic": "1인 가구 증가 현상 분석 및 사회적 시사점",
    "korean_context": "오늘 저는 한국 사회의 가장 큰 변화 중 하나인 '1인 가구의 증가' 현상에 대해 발표하겠습니다. 먼저 지난 10년간 1인 가구의 증가 추이를 연령대별 그래프로 보여드리고, 그 원인을 사회경제적 측면에서 분석하겠습니다. 특히 청년층과 노년층에서 1인 가구가 급증하고 있으며, 이는 개인의 가치관 변화와 함께 주거 및 고용 불안의 심화를 의미합니다. 결론적으로, 우리 사회는 이제 1인 가구를 위한 맞춤형 주거, 복지, 그리고 사회적 관계망 지원 정책을 시급히 마련해야 합니다.",
    "foreign_topic": "Analysis of the Rise in Single-Person Households and its Social Implications",
    "foreign_context": "Today, I will discuss one of Korea's most significant social changes: the increase in single-person households. I will first show the trend over the past 10 years with a graph broken down by age, and then analyze the socioeconomic causes. The sharp increase, particularly among the young and elderly, reflects not only changing values but also growing housing and employment instability. In conclusion, our society must urgently develop policies for housing, welfare, and social networks tailored to this demographic.",
    "problem_types": ["제목을 붙인 근거 설명하기", "자문화와 비교하기", "원인과 전망 예측하기"],
    "eval_goals": [
        "글의 전체적인 주제와 핵심 내용을 정확히 파악하는 능력을 평가한다.",
        "지문에 제시된 특정 문화 현상을 자신의 문화적 배경과 관련지어 공통점과 차이점을 구체적으로 비교 설명하는 능력을 평가한다.",
        "글에 제시된 사회/문화적 현상의 원인을 추론하고, 이를 근거로 미래에 나타날 변화나 결과를 논리적으로 설명하는 능력을 평가한다."
    ],
    ... # 기타 메타데이터
  },
  "chosen": "고성능 모델이 생성한 지문 텍스트",
  "rejected": "저성능 모델이 생성한 지문 텍스트",
  "dataset_type": "IMP",
  "created_at": "2025-07-29T12:34:56.789012"
}
"""

"""
만들어야 하는 데이터셋 예시
    {
        "conversations": [
            {
                "from": "human",
                "value": ""
            }
        ],
        "chosen": {
            "from": "gpt",
            "value": ""
        },
        "rejected": {
            "from": "gpt",
            "value": ""
        }
    },
"""

def load_yaml_prompts(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def fill_prompt_template(template, source_item):
    def repl(match):
        key = match.group(1)
        if key.startswith('problem_type'):
            idx = int(key.replace('problem_type','')) - 1
            return source_item.get('problem_types', [''])[idx] if 'problem_types' in source_item and idx < len(source_item['problem_types']) else ''
        if key.startswith('eval_goal'):
            idx = int(key.replace('eval_goal','')) - 1
            return source_item.get('eval_goals', [''])[idx] if 'eval_goals' in source_item and idx < len(source_item['eval_goals']) else ''
        return source_item.get(key, '')
    return re.sub(r'\{([a-zA-Z0-9_]+)\}', repl, template)

def imp_to_rm_format(imp_data, rubric_prompts):
    rm_dataset = []
    for entry in imp_data:
        rubric = entry.get('rubric')
        source_item = entry.get('source_item', {})
        chosen = entry.get('chosen', '')
        rejected = entry.get('rejected', '')
        prompt_template = rubric_prompts.get('preference_evaluation', {}).get(rubric, '')
        if not prompt_template:
            continue
        prompt_filled = fill_prompt_template(prompt_template, source_item)
        rm_entry = {
            "conversations": [
                {
                    "from": "human",
                    "value": prompt_filled
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": chosen
            },
            "rejected": {
                "from": "gpt",
                "value": rejected
            }
        }
        rm_dataset.append(rm_entry)
    return rm_dataset

def find_imp_files():
    """
    imp 데이터셋 폴더에서 모든 imp 파일들을 찾아서 반환합니다.
    반환값: {rubric: filepath} 형태의 딕셔너리
    """
    imp_dir = Path('src/data/rm_training/imp/2025-08-02/midm-aidx')
    imp_files = {}
    
    # 유효한 루브릭 목록
    valid_rubrics = {
        'logical_flow', 'korean_quality', 'l2_learner_suitability',
        'clarity_of_core_theme', 'completeness_for_guidelines',
        'reference_groundedness'
    }
    
    # imp 파일들을 찾아서 rubric별로 정리
    for file in imp_dir.glob('IMP_*_*.json'):
        basename = file.stem
        parts = basename.split('_')
        if len(parts) >= 3 and parts[0] == 'IMP':
            # 타임스탬프 부분을 찾아서 그 이전까지를 루브릭으로 사용
            timestamp_idx = -1
            for i, part in enumerate(parts):
                if re.match(r'\d{8}', part):  # YYYYMMDD 형식 찾기
                    timestamp_idx = i
                    break
            
            if timestamp_idx > 1:  # imp_ 다음부터 타임스탬프 이전까지가 루브릭
                rubric = '_'.join(parts[1:timestamp_idx])
            else:
                continue
            
            # 유효한 루브릭인 경우에만 처리
            if rubric in valid_rubrics and rubric not in imp_files:
                imp_files[rubric] = str(file)
            else:
                if rubric not in valid_rubrics:
                    print(f'⚠️ Warning: Skipping invalid rubric "{rubric}" in file {file.name}')
    
    return imp_files

def process_single_rubric(imp_file, rubric, prompt_yaml, out_base_dir):
    """
    단일 루브릭의 imp 데이터셋을 처리하여 RM 데이터셋으로 변환합니다.
    """
    print(f"📝 Processing rubric: {rubric}")
    
    # imp 파일 읽기
    with open(imp_file, encoding='utf-8') as f:
        imp_data = json.load(f)
    
    # YAML 프롬프트 템플릿 로드
    rubric_prompts = load_yaml_prompts(prompt_yaml)
    
    # imp 데이터를 리스트로 변환
    if isinstance(imp_data, dict):
        imp_entries = []
        for value in imp_data.values():
            if isinstance(value, dict):
                imp_entries.append(value)
            elif isinstance(value, list):
                imp_entries.extend(value)
    else:
        imp_entries = imp_data if isinstance(imp_data, list) else [imp_data]

    # imp 엔트리들을 RM 포맷으로 변환
    rm_dataset = []
    for entry in imp_entries:
        if isinstance(entry, dict):
            entry['rubric'] = rubric
            rm_dataset.extend(imp_to_rm_format([entry], rubric_prompts))

    if rm_dataset:
        # 루브릭 디렉토리 생성 및 데이터 저장
        rubric_dir = Path(out_base_dir) / rubric
        rubric_dir.mkdir(parents=True, exist_ok=True)
        out_path = rubric_dir / 'rm_pairwise.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(rm_dataset, f, ensure_ascii=False, indent=2)
        print(f'✅ Saved: {out_path} ({len(rm_dataset)} pairs)')
    else:
        print(f'⚠️ Warning: No valid entries found for rubric {rubric}')


if __name__ == "__main__":
    prompt_yaml = 'src/config/prompts/iska/preference_eval.yaml'
    out_base_dir = 'src/data/pairwise_data/v3/imp/midm-aidx'
    imp_file = find_imp_files()
    if not imp_file:
        print('❌ imp 데이터셋 파일을 찾을 수 없습니다.')
    else:
        print(f'📂 Found {len(imp_file)} imp files for rubrics: {", ".join(imp_file.keys())}')
        print(f'📂 Using prompt YAML: {prompt_yaml}')
        print(f'📂 Output base directory: {out_base_dir}')
        
        # 각 루브릭별로 처리
        for rubric, imp_file in imp_file.items():
            process_single_rubric(imp_file, rubric, prompt_yaml, out_base_dir)

#!/usr/bin/env python3
import os
import json
import yaml
import re
from pathlib import Path

"""
ICP → RM 데이터셋 변환 유틸리티
ICP (Inter-Client Performance) 데이터셋을 RM 훈련용 pairwise 포맷으로 변환합니다.
"""

"""
ICP 데이터셋 스키마 예시:
{
    "conversations": [
        {
            "from": "human",
            "value": "문제"
        }
    ],
    "chosen": {
        "from": "client1",
        "value": "좋은 답변"
    },
    "rejected": {
        "from": "client2",
        "value": "나쁜 답변"
    },
    "rubric": "logical_flow_and_structure",
    "created_at": "2025-08-01T11:46:34.102422"
}
"""

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

def load_yaml_prompts(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def icp_to_rm_format(icp_data, rubric_prompts):
    """
    ICP 데이터셋을 RM 훈련용 포맷으로 변환합니다.
    """
    rm_dataset = []
    for entry in icp_data:
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

def find_icp_files():
    """
    ICP 데이터셋 폴더에서 모든 ICP 파일들을 찾아서 반환합니다.
    반환값: {rubric: filepath} 형태의 딕셔너리
    """
    icp_dir = Path('src/data/rm_training/icp/2025-08-02/aidx')
    icp_files = {}
    
    # 유효한 루브릭 목록
    valid_rubrics = {
        'logical_flow', 'korean_quality', 'l2_learner_suitability',
        'clarity_of_core_theme', 'completeness_for_guidelines',
        'reference_groundedness'
    }
    
    # ICP 파일들을 찾아서 rubric별로 정리
    for file in icp_dir.glob('ICP_*_*.json'):
        basename = file.stem
        parts = basename.split('_')
        if len(parts) >= 3 and parts[0] == 'ICP':
            # 타임스탬프 부분을 찾아서 그 이전까지를 루브릭으로 사용
            timestamp_idx = -1
            for i, part in enumerate(parts):
                if re.match(r'\d{8}', part):  # YYYYMMDD 형식 찾기
                    timestamp_idx = i
                    break
            
            if timestamp_idx > 1:  # ICP_ 다음부터 타임스탬프 이전까지가 루브릭
                rubric = '_'.join(parts[1:timestamp_idx])
            else:
                continue
            
            # 유효한 루브릭인 경우에만 처리
            if rubric in valid_rubrics and rubric not in icp_files:
                icp_files[rubric] = str(file)
            else:
                if rubric not in valid_rubrics:
                    print(f'⚠️ Warning: Skipping invalid rubric "{rubric}" in file {file.name}')
    
    return icp_files

def process_single_rubric(icp_file, rubric, prompt_yaml, out_base_dir):
    """
    단일 루브릭의 ICP 데이터셋을 처리하여 RM 데이터셋으로 변환합니다.
    """
    print(f"📝 Processing rubric: {rubric}")
    
    # ICP 파일 읽기
    with open(icp_file, encoding='utf-8') as f:
        icp_data = json.load(f)
    
    # YAML 프롬프트 템플릿 로드
    rubric_prompts = load_yaml_prompts(prompt_yaml)
    
    # ICP 데이터를 리스트로 변환
    if isinstance(icp_data, dict):
        icp_entries = []
        for value in icp_data.values():
            if isinstance(value, dict):
                icp_entries.append(value)
            elif isinstance(value, list):
                icp_entries.extend(value)
    else:
        icp_entries = icp_data if isinstance(icp_data, list) else [icp_data]

    # ICP 엔트리들을 RM 포맷으로 변환
    rm_dataset = []
    for entry in icp_entries:
        if isinstance(entry, dict):
            entry['rubric'] = rubric
            rm_dataset.extend(icp_to_rm_format([entry], rubric_prompts))

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
    out_base_dir = 'src/data/pairwise_data/v3/icp/aidx'
    
    # 모든 ICP 파일 찾기
    icp_files = find_icp_files()
    
    if not icp_files:
        print('❌ ICP 데이터셋 파일을 찾을 수 없습니다.')
    else:
        print(f'📂 Found {len(icp_files)} ICP files for rubrics: {", ".join(icp_files.keys())}')
        print(f'📂 Using prompt YAML: {prompt_yaml}')
        print(f'📂 Output base directory: {out_base_dir}')
        
        # 각 루브릭별로 처리
        for rubric, icp_file in icp_files.items():
            process_single_rubric(icp_file, rubric, prompt_yaml, out_base_dir)

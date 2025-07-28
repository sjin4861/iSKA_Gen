import sys
from pathlib import Path
import json
import random
from datetime import datetime
from utils.rm_dataset_generator import (
    RMDatasetGenerator,
    create_spf_dataset,
    create_imp_dataset,
    create_icp_dataset,
)

# 프로젝트 경로 설정
sys.path.append(str(Path.cwd().parent))
sys.path.append(str(Path.cwd().parent / 'modules'))
sys.path.append(str(Path.cwd().parent / 'utils'))
print("✅ 경로 설정 완료")

# 기본 설정
OPENAI_MODEL = 'gpt-4o'
TARGET_PAIRS_PER_RUBRIC = 50  # 테스트용으로 적게 설정
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

print(f'🎯 설정 완료:')
print(f'   OpenAI 모델: {OPENAI_MODEL}')
print(f'   루브릭당 목표 쌍 수: {TARGET_PAIRS_PER_RUBRIC}')
print(f'   랜덤 시드: {RANDOM_SEED}')

# 샘플 기본 지문 데이터 생성 (실제로는 기존 생성된 지문을 사용)
sample_base_passages = [
    {
        'id': f'passage_{i:03d}',
        'text': f'이것은 샘플 지문 {i}입니다. 한국어로 작성된 교육용 텍스트로, 다양한 주제를 다루고 있습니다. 문법적으로 정확하고 논리적으로 구성되어 있으며, 교육 목적에 적합한 내용을 담고 있습니다. 이 지문은 학습자들이 한국어 읽기 능력을 향상시킬 수 있도록 도와줍니다.',
        'topic': f'주제_{i%10}',
        'source': 'sample_generation',
        'created_at': datetime.now().isoformat()
    }
    for i in range(1, 101)
]
sample_file_path = Path.cwd().parent / 'data' / 'base_passages' / 'sample_passages_100.json'
sample_file_path.parent.mkdir(exist_ok=True)
with open(sample_file_path, 'w', encoding='utf-8') as f:
    json.dump(sample_base_passages, f, ensure_ascii=False, indent=2)
print(f'📝 샘플 기본 지문 생성 완료: {len(sample_base_passages)}개')
print(f'💾 저장 위치: {sample_file_path}')

# SPF 데이터셋 생성
print('🚀 SPF 데이터셋 생성 시작...')
spf_dataset, spf_files = create_spf_dataset(
    passages_file='sample_passages_100.json',
    target_pairs_per_rubric=TARGET_PAIRS_PER_RUBRIC,
    openai_model=OPENAI_MODEL,
)
print('\n✅ SPF 데이터셋 생성 완료!')
print('📁 저장된 파일들:')
for rubric, file_path in spf_files.items():
    print(f'   {rubric}: {Path(file_path).name}')

# IMP 데이터셋 생성
high_perf_passages = [
    {
        'id': f'high_perf_{i:03d}',
        'text': f'고성능 모델이 생성한 우수한 품질의 지문 {i}입니다. 문법적으로 정확하고, 주제에 충실하며, 논리적으로 잘 구성되어 있습니다. 표현이 자연스럽고 완전한 정보를 제공합니다.',
        'model': 'high_performance_7B',
        'rubric_achievement_rate': 0.98,
        'created_at': datetime.now().isoformat()
    }
    for i in range(1, 51)
]
low_perf_passages = [
    {
        'id': f'low_perf_{i:03d}',
        'text': f'저성능 모델이 생성한 지문 {i}입니다. 문법에 일부 오류가 있고, 주제에서 벗어나는 경우가 있으며, 논리적 구성이 부족할 수 있습니다.',
        'model': 'low_performance_3B',
        'rubric_achievement_rate': 0.40,
        'created_at': datetime.now().isoformat()
    }
    for i in range(1, 51)
]

high_perf_path = Path.cwd().parent / 'data' / 'base_passages' / 'high_performance_passages.json'
low_perf_path = Path.cwd().parent / 'data' / 'base_passages' / 'low_performance_passages.json'

with open(high_perf_path, 'w', encoding='utf-8') as f:
    json.dump(high_perf_passages, f, ensure_ascii=False, indent=2)
with open(low_perf_path, 'w', encoding='utf-8') as f:
    json.dump(low_perf_passages, f, ensure_ascii=False, indent=2)

print(f'📝 고성능 모델 지문: {len(high_perf_passages)}개')
print(f'📝 저성능 모델 지문: {len(low_perf_passages)}개')
print(f'💾 저장 완료')
print('🚀 IMP 데이터셋 생성 시작...')

imp_dataset, imp_files = create_imp_dataset(
    high_perf_file='high_performance_passages.json',
    low_perf_file='low_performance_passages.json',
    target_pairs_per_rubric=TARGET_PAIRS_PER_RUBRIC,
    openai_model=OPENAI_MODEL,
)
print('\n✅ IMP 데이터셋 생성 완료!')
print('📁 저장된 파일들:')
for rubric, file_path in imp_files.items():
    print(f'   {rubric}: {Path(file_path).name}')

# ICP 데이터셋 생성
print('🚀 ICP 데이터셋 생성 시작...')
icp_dataset, icp_files = create_icp_dataset(
    base_passages_file='sample_passages_100.json',
    target_pairs_per_rubric=TARGET_PAIRS_PER_RUBRIC,
    openai_model=OPENAI_MODEL,
)
print('\n✅ ICP 데이터셋 생성 완료!')
print('📁 저장된 파일들:')
for rubric, file_path in icp_files.items():
    print(f'   {rubric}: {Path(file_path).name}')

# 데이터셋 통계 출력
generator = RMDatasetGenerator(OPENAI_MODEL)
print('📊 생성된 데이터셋 통계 요약:')
print('\n1. SPF 데이터셋:')
spf_stats = generator.get_dataset_stats(spf_dataset)
for rubric, count in spf_stats['pairs_per_rubric'].items():
    print(f'   {rubric}: {count}개 쌍')
print(f'   총합: {spf_stats['total_pairs']}개 쌍')
print('\n2. IMP 데이터셋:')
imp_stats = generator.get_dataset_stats(imp_dataset)
for rubric, count in imp_stats['pairs_per_rubric'].items():
    print(f'   {rubric}: {count}개 쌍')
print(f'   총합: {imp_stats['total_pairs']}개 쌍')
print('\n3. ICP 데이터셋:')
icp_stats = generator.get_dataset_stats(icp_dataset)
for rubric, count in icp_stats['pairs_per_rubric'].items():
    print(f'   {rubric}: {count}개 쌍')
print(f'   총합: {icp_stats['total_pairs']}개 쌍')
total_pairs = spf_stats['total_pairs'] + imp_stats['total_pairs'] + icp_stats['total_pairs']
print(f'\n🎯 전체 생성된 선호도 쌍: {total_pairs}개')

# 샘플 데이터 확인
rubric_keys = [
    'completeness_for_guidelines',
    'core_theme_clarity',
    'reference_groundedness',
    'logical_flow_and_structure',
    'korean_quality',
    'l2_learner_suitability'
]
print('📝 샘플 데이터 확인:\n')
# SPF 샘플
for rubric in rubric_keys:
    if rubric in spf_dataset and spf_dataset[rubric]:
        spf_sample = spf_dataset[rubric][0]
        print(f'1. SPF 샘플 ({rubric}):')
        print(f'   쌍 ID: {spf_sample['pair_id']}')
        print(f'   Chosen: {spf_sample['chosen'][:100]}...')
        print(f'   Rejected: {spf_sample['rejected'][:100]}...')
        print()
# IMP 샘플
for rubric in rubric_keys:
    if rubric in imp_dataset and imp_dataset[rubric]:
        imp_sample = imp_dataset[rubric][0]
        print(f'2. IMP 샘플 ({rubric}):')
        print(f'   쌍 ID: {imp_sample['pair_id']}')
        print(f'   Chosen: {imp_sample['chosen'][:100]}...')
        print(f'   Rejected: {imp_sample['rejected'][:100]}...')
        print()
# ICP 샘플
for rubric in rubric_keys:
    if rubric in icp_dataset and icp_dataset[rubric]:
        icp_sample = icp_dataset[rubric][0]
        print(f'3. ICP 샘플 ({rubric}):')
        print(f'   쌍 ID: {icp_sample['pair_id']}')
        print(f'   Chosen: {icp_sample['chosen'][:100]}...')
        print(f'   Rejected: {icp_sample['rejected'][:100]}...')
        print()

print("\n생성된 데이터셋은 다음과 같이 활용할 수 있습니다:")
print("1. Reward Model 훈련: 각 루브릭별로 생성된 선호도 쌍을 사용하여 RM 훈련")
print("2. 데이터셋 품질 검증: 생성된 쌍의 품질을 수동으로 검증")
print("3. 하이퍼파라미터 조정: 더 많은 데이터나 다른 설정으로 재생성")
print("4. 모델 평가: 훈련된 RM의 성능 평가")
print("\n데이터셋 파일들은 src/data/rm_training/ 디렉토리에 저장됩니다.")

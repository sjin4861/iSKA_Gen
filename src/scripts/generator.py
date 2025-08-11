#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from pathlib import Path
import torch
import gc
import json
import os

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # iSKA_Gen 디렉토리
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'modules'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'utils'))

# import pdb
# pdb.set_trace()  # 디버깅을 위한 중단점 설정

from utils.make_passage import generate_passage
from utils.make_stem import generate_stem

MODEL_LIST = [
    "EXAONE-3.5-7.8B-Instruct",
    # "Midm-2.0-Base-Instruct",
    # "A.X-4.0-Light",
    # "llama3.1_korean_v1.1_sft_by_aidx",
    # "llama3-bllossom-3b"
]
USER_AGENT = "iSKA (sjun24530@gmail.com)"
BENCH_ID_LIST = [2]#, 3, 4, 5]  # 전체 벤치마크 ID 리스트
BENCH_FILE = "iSKA-Gen_Benchmark_v1.1.0_20250808.json"  # 파일명만 지정
DATE_STR = "2025-08-10"

BENCHMARK_TEMPLATES = {
    1: "passage_agent.create_passage_rubric_aware",  # 읽기 (비교형)
    2: "passage_agent.create_domestic_passage",      # 읽기 (단일 주제형)
    3: "passage_agent.create_dialogue_passage",      # 듣기 (대화형)
    4: "passage_agent.create_dialogue_passage",      # 듣기 (대화형)
    5: "passage_agent.create_image_caption_and_situation"  # 보기 (이미지 캡션)
}
# LOW_TEMPLATE_KEYS = [
#     "passage_agent.violate_completeness_severely",
#     "passage_agent.violate_clarity_severely",
#     "passage_agent.violate_groundedness_severely",
#     "passage_agent.violate_flow_severely",
#     "passage_agent.violate_korean_quality_severely",
#     "passage_agent.violate_l2_suitability_severely",
# ]


# ===== 기존 배치 생성 (주석 처리) =====
# for model_name in MODEL_LIST:
#     print(f"\n🤖 모델: {model_name}")
#     print("="*50)
    
#     # 모든 벤치마크 ID에 대해 generate_passage 함수로 통일
#     for benchmark_id in BENCH_ID_LIST:
#         template_key = BENCHMARK_TEMPLATES[benchmark_id]
        
#         print(f"\n📋 벤치마크 ID {benchmark_id} 생성 시작")
#         print(f"   📝 템플릿: {template_key}")
        
#         try:
#             # 모든 ID를 generate_passage 함수로 통일 처리
#             print(f"   📚 벤치마크 ID {benchmark_id} 전체 배치 생성...")
            
#             # 대화문(ID 3, 4)은 더 긴 길이 허용
#             max_len = 800 if benchmark_id in [3, 4] else 500
            
#             generate_passage(
#                 benchmark_file=BENCH_FILE,
#                 model_name=model_name,
#                 template_key=template_key,
#                 benchmark_version="v1.1.0",
#                 gpus=[1],
#                 BENCH_ID_LIST=[benchmark_id],
#                 date_str=DATE_STR,
#                 max_length=max_len
#             )
            
#             print(f"   ✅ 벤치마크 ID {benchmark_id} 생성 완료!")
            
#         except Exception as e:
#             print(f"   ❌ 벤치마크 ID {benchmark_id} 생성 실패: {e}")
        
#         # 메모리 정리
#         torch.cuda.empty_cache()
#         gc.collect()
#         print(f"   🧹 메모리 정리 완료")
    
#     print(f"\n✨ 모델 {model_name} 전체 배치 생성 완료!")


# ===== NULL 항목 재생성 기능 =====
def find_null_passages_in_file(file_path):
    """JSON 파일에서 generated_passage가 null인 항목들의 인덱스를 찾음"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        null_indices = []
        for i, item in enumerate(data):
            if item.get('generated_passage') is None:
                null_indices.append(i)
        
        return null_indices, len(data)
    except Exception as e:
        print(f"파일 읽기 오류 {file_path}: {e}")
        return [], 0

def regenerate_null_passages():
    """모든 모델의 생성된 파일에서 null인 항목들을 재생성"""
    base_path = PROJECT_ROOT / "src" / "data" / "raw_outputs" / DATE_STR / "passage"
    
    for model_name in MODEL_LIST:
        print(f"\n🔄 {model_name} 모델의 null 항목 재생성 시작")
        print("="*60)
        
        model_path = base_path / model_name
        if not model_path.exists():
            print(f"❌ 모델 경로가 존재하지 않음: {model_path}")
            continue
        
        for benchmark_id in BENCH_ID_LIST:
            template_key = BENCHMARK_TEMPLATES[benchmark_id]
            template_path = model_path / template_key
            
            if not template_path.exists():
                print(f"❌ 템플릿 경로가 존재하지 않음: {template_path}")
                continue
            
            # JSON 파일 찾기
            json_files = list(template_path.glob(f"benchmark_{benchmark_id}_v1.1.0_*.json"))
            
            for json_file in json_files:
                print(f"\n📁 파일 확인: {json_file.name}")
                
                # null 항목 찾기
                null_indices, total_count = find_null_passages_in_file(json_file)
                
                if not null_indices:
                    print(f"   ✅ null 항목 없음 (총 {total_count}개 항목)")
                    continue
                
                print(f"   🔍 발견된 null 항목: {len(null_indices)}개 / 총 {total_count}개")
                print(f"   📋 null 인덱스: {null_indices}")
                
                # null 항목들만 재생성
                try:
                    print(f"   🔄 null 항목 재생성 시작...")
                    
                    # 대화문(ID 3, 4)은 더 긴 길이 허용
                    max_len = 800 if benchmark_id in [3, 4] else 500
                    
                    generate_passage(
                        benchmark_file=BENCH_FILE,
                        model_name=model_name,
                        template_key=template_key,
                        benchmark_version="v1.1.0",
                        gpus=[1],
                        BENCH_ID_LIST=[benchmark_id],
                        date_str=DATE_STR,
                        max_length=max_len,
                        only_indices=null_indices
                    )
                    
                    print(f"   ✅ 재생성 완료!")
                    
                except Exception as e:
                    print(f"   ❌ 재생성 실패: {e}")
                
                # 메모리 정리
                torch.cuda.empty_cache()
                gc.collect()
        
        print(f"\n✨ {model_name} 모델의 null 항목 재생성 완료!")

def regenerate_specific_null_passages(target_file_path, null_indices_to_regenerate=None):
    """특정 파일의 특정 인덱스만 재생성"""
    if not os.path.exists(target_file_path):
        print(f"❌ 파일이 존재하지 않음: {target_file_path}")
        return
    
    # 파일 경로에서 정보 추출
    file_path = Path(target_file_path)
    file_name = file_path.name
    
    # 파일명에서 벤치마크 ID와 템플릿 키 추출
    if "benchmark_2_" in file_name and "create_domestic_passage" in file_name:
        benchmark_id = 2
        template_key = "passage_agent.create_domestic_passage"
        model_name = file_path.parent.parent.name
    else:
        print(f"❌ 지원되지 않는 파일 형식: {file_name}")
        return
    
    print(f"\n🎯 특정 파일 재생성: {file_name}")
    print(f"   📂 모델: {model_name}")
    print(f"   📋 벤치마크 ID: {benchmark_id}")
    print(f"   📝 템플릿: {template_key}")
    
    # null 항목 찾기
    null_indices, total_count = find_null_passages_in_file(target_file_path)
    
    if null_indices_to_regenerate is None:
        null_indices_to_regenerate = null_indices
    
    if not null_indices_to_regenerate:
        print(f"   ✅ 재생성할 null 항목 없음")
        return
    
    print(f"   🔍 재생성할 null 항목: {len(null_indices_to_regenerate)}개")
    print(f"   📋 인덱스: {null_indices_to_regenerate}")
    
    try:
        max_len = 800 if benchmark_id in [3, 4] else 500
        
        generate_passage(
            benchmark_file=BENCH_FILE,
            model_name=model_name,
            template_key=template_key,
            benchmark_version="v1.1.0",
            gpus=[1],
            BENCH_ID_LIST=[benchmark_id],
            date_str=DATE_STR,
            max_length=max_len,
            only_indices=null_indices_to_regenerate
        )
        
        print(f"   ✅ 재생성 완료!")
        
    except Exception as e:
        print(f"   ❌ 재생성 실패: {e}")
    
    # 메모리 정리
    torch.cuda.empty_cache()
    gc.collect()


def regenerate_indices_for_benchmark(model_name: str, benchmark_id: int, indices: list[int]):
    """모델/벤치마크 기준으로 지정 인덱스만 부분 재생성 (파일 경로 지정 불필요).

    기존 결과 파일이 있으면 해당 인덱스만 교체하여 덮어씁니다.
    """
    if benchmark_id not in BENCHMARK_TEMPLATES:
        print(f"❌ 지원되지 않는 벤치마크 ID: {benchmark_id}")
        return

    template_key = BENCHMARK_TEMPLATES[benchmark_id]
    max_len = 800 if benchmark_id in [3, 4] else 500

    print(f"\n🎯 부분 재생성 실행")
    print(f"   🤖 모델: {model_name}")
    print(f"   🆔 벤치마크 ID: {benchmark_id}")
    print(f"   📝 템플릿: {template_key}")
    print(f"   📋 인덱스: {sorted(indices)}")

    try:
        generate_passage(
            benchmark_file=BENCH_FILE,
            model_name=model_name,
            template_key=template_key,
            benchmark_version="v1.1.0",
            gpus=[1],
            BENCH_ID_LIST=[benchmark_id],
            date_str=DATE_STR,
            max_length=max_len,
            only_indices=indices,
        )
        print("   ✅ 부분 재생성 완료")
    except Exception as e:
        print(f"   ❌ 부분 재생성 실패: {e}")


# 실행: null 항목 재생성
print("\n🚀 NULL 항목 재생성 시작")
print("="*60)

# 방법 1: 모든 파일의 null 항목 자동 재생성 (필요 시만 사용)
# regenerate_null_passages()

# 방법 2: 특정 파일만 재생성 (주석 해제하여 사용)
# target_file = "/home/sjin4861/25-1/HCLT/iSKA_Gen/src/data/raw_outputs/2025-08-08/passage/EXAONE-3.5-7.8B-Instruct/passage_agent.create_domestic_passage/benchmark_2_v1.1.0_passage_agent.create_domestic_passage.json"
# regenerate_specific_null_passages(target_file)

# 방법 3: 지정 인덱스만 재생성 (예: 사용자가 제공한 인덱스)
example_indices = [0, 4, 8, 10, 14, 28, 29, 33, 44, 48]
regenerate_indices_for_benchmark("EXAONE-3.5-7.8B-Instruct", 2, example_indices)


# Stem Generation
# print("\n--- Stem Generation ---")
# STEM_TEMPLATE_KEY = "stem_agent.few_shot" # From stem_agent.yaml

# for stem_model_name in MODEL_LIST: # Model used to generate stems
#     print(f"\n📝 Stem 생성 모델: {stem_model_name}")
#     print("="*50)
#     try:
#         for i in range(2, 6):  # 벤치마크 ID 2부터 5까지
#             generate_stem(
#                 benchmark_file=BENCH_FILE,
#                 passage_model_name=stem_model_name,
#                 model_name=stem_model_name,
#                 template_key=STEM_TEMPLATE_KEY,
#                 passage_template_key=BENCHMARK_TEMPLATES[i], # Assuming all passages use the same template for a given benchmark ID
#                 benchmark_version="v1.1.0",
#                 gpus=[1],
#                 BENCH_ID_LIST=[i],
#                 date_str=DATE_STR
#             )
#             print(f"  ✅ {stem_model_name}의 passage로 stem 생성 완료!")
#     except Exception as e:
#         print(f"  ❌ {stem_model_name}의 passage로 stem 생성 실패: {e}")
    
#     # 메모리 정리
#     torch.cuda.empty_cache()
#     gc.collect()
#     print(f"  🧹 메모리 정리 완료")

# print("\n--- Stem Generation Complete ---")



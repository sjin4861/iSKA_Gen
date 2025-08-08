#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from pathlib import Path
import torch
import gc

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # iSKA_Gen 디렉토리
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'modules'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'utils'))

# import pdb
# pdb.set_trace()  # 디버깅을 위한 중단점 설정

from utils.make_passage import generate_passage, generate_single_passage, generate_single_passage_from_benchmark
from utils.make_stem import generate_stem

MODEL_LIST = [
    "EXAONE-3.5-7.8B-Instruct",
    "Midm-2.0-Base-Instruct",
    "A.X-4.0-Light",
    "llama3.1_korean_v1.1_sft_by_aidx",
    # "llama3-bllossom-3b"
]
USER_AGENT = "iSKA (sjun24530@gmail.com)"
BENCH_ID_LIST = [1, 2, 3, 4, 5]
BENCH_FILE = "v1/iSKA-Gen_Benchmark_v1.0.0_20250725_Initial.json"
DATE_STR = "2025-08-05"  # 날짜 문자열 추가
LOW_TEMPLATE_KEYS = [
    "passage_agent.violate_completeness_severely",
    "passage_agent.violate_clarity_severely",
    "passage_agent.violate_groundedness_severely",
    "passage_agent.violate_flow_severely",
    "passage_agent.violate_korean_quality_severely",
    "passage_agent.violate_l2_suitability_severely",
]
# for model_name in MODEL_LIST:
#     generate_passage(benchmark_file=BENCH_FILE, model_name=model_name, template_key="passage_agent.create_passage_rubric_aware", gpus=[1], BENCH_ID_LIST=BENCH_ID_LIST, date_str=DATE_STR)
#     # 메모리 정리
#     torch.cuda.empty_cache()
#     gc.collect()
#     for low_template_key in LOW_TEMPLATE_KEYS:
#         generate_passage(benchmark_file=BENCH_FILE, model_name=model_name, template_key=low_template_key, gpus=[0], BENCH_ID_LIST=BENCH_ID_LIST, date_str=DATE_STR)
#         # 메모리 정리
#         torch.cuda.empty_cache()
#         gc.collect()


# ===== 단일 지문 생성 테스트 =====
print("\n" + "="*60)
print("🔧 단일 지문 생성 함수 테스트")
print("="*60)

# 1. 직접 정보 입력으로 단일 지문 생성 테스트
# print("\n📝 1. 직접 정보 입력으로 단일 지문 생성")
# test_data = {
#     "korean_topic": "분기별 실적 보고 및 목표 달성 전략 발표",
#     "korean_context": "지금부터 2024년 2분기 영업 실적에 대한 발표를 시작하겠습니다. 오늘 발표에서는 먼저 2분기 매출 실적 데이터를 전 분기와 비교 분석하고, 이어서 실적 부진의 원인을 진단한 뒤, 마지막으로 3분기 매출 목표 달성을 위한 구체적인 전략을 제안해 드리고자 합니다. 보시는 그래프와 같이, 2분기 전체 매출은 전 분기 대비 15% 하락했으며, 이는 경쟁사의 신제품 출시와 마케팅 활동 부족 때문으로 분석됩니다. 결론적으로, 3분기에는 공격적인 프로모션과 SNS 마케팅 예산 증액을 통해 실적을 회복해야 하며, 이에 대한 구체적인 실행 계획을 중심으로 질의응답을 진행하겠습니다.",
#     "foreign_topic": "Quarterly Performance Review and Strategy Presentation",
#     "foreign_context": "Today, I will present the sales performance for Q2 2024. I will begin by analyzing the sales data compared to Q1, then identify the key reasons for our underperformance, and finally propose a strategy for Q3. As the chart indicates, our revenue saw a 15% decline, primarily due to competitor actions and insufficient marketing. Therefore, my conclusion is that we must launch an aggressive promotional campaign and increase our social media budget; I am now ready to answer your questions on this action plan.",
#     "problem_types": ["발표의 핵심 목적 파악하기", "세부 내용 및 근거 설명하기", "전체 내용 요약 및 재구성하기"],
#     "eval_goals": [
#         "주어진 발표문(지문)의 전체적인 주제와 핵심 목적이 무엇인지 정확히 파악하는 능력을 평가한다.",
#         "발표의 주장을 뒷받침하기 위해 사용된 데이터와 근거를 파악하고, 그 내용을 상세히 설명하는 능력을 평가한다.",
#         "발표 전체 내용을 자신의 말로 간결하게 요약하고, 핵심 메시지를 재구성하여 전달하는 능력을 평가한다."
#     ],
# }



# test_result = generate_single_passage(
#     korean_topic=test_data["korean_topic"],
#     korean_context=test_data["korean_context"],
#     foreign_topic=test_data["foreign_topic"],
#     foreign_context=test_data["foreign_context"],
#     problem_types=test_data["problem_types"],
#     eval_goals=test_data["eval_goals"],
#     model_name="A.X-4.0-Light",
#     template_key="passage_agent.create_passage_rubric_aware",
#     gpus=[0],
#     max_retries=5
# )

# if test_result["success"]:
#     print(f"✅ 지문 생성 성공!")
#     print(f"   📏 길이: {test_result['passage_length']}자")
#     print(f"   🔄 재시도 횟수: {test_result['retries_used']}회")
#     print(f"   📝 지문:")
#     print(f"   {test_result['generated_passage']}")
# else:
#     print(f"❌ 지문 생성 실패: {test_result['error']}")

# # 메모리 정리
# torch.cuda.empty_cache()
# gc.collect()

# ===== 기존 배치 생성 =====

for model_name in MODEL_LIST:
    generate_stem(benchmark_file=BENCH_FILE, passage_model_name=model_name, model_name=model_name, template_key="stem_agent.few_shot", passage_template_key="passage_agent.create_passage_rubric_aware", gpus=[0], BENCH_ID_LIST=[1], date_str=DATE_STR)


# %%


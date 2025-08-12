#!/usr/bin/env python
# coding: utf-8

import json
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

from modules.iska.passage_agent import PassageAgent
from modules.model_client import LocalModelClient
from utils.make_passage import clean_passage_text

def regenerate_null_passages(input_file_path: str, output_file_path: str, model_name: str, template_key: str, benchmark_id: int, gpus: list = [0]):
    """
    JSON 파일에서 null인 passage들을 찾아서 재생성하는 함수
    
    Args:
        input_file_path (str): 입력 JSON 파일 경로
        output_file_path (str): 출력 JSON 파일 경로  
        model_name (str): 사용할 모델 이름
        template_key (str): 사용할 템플릿 키
        benchmark_id (int): 벤치마크 ID (1-5)
        gpus (list): 사용할 GPU 번호 리스트
    """
    
    # 벤치마크 ID에 따른 problem_types와 eval_goals 설정
    benchmark_configs = {
        1: {
            "problem_types": ["제목을 붙인 근거 설명하기", "자문화와 비교하기", "원인과 전망 예측하기"],
            "eval_goals": [
                "글의 전체적인 주제와 핵심 내용을 정확히 파악하는 능력을 평가한다.",
                "지문에 제시된 특정 문화 현상을 자신의 문화적 배경과 관련지어 공통점과 차이점을 구체적으로 비교 설명하는 능력을 평가한다.",
                "글에 제시된 사회/문화적 현상의 원인을 추론하고, 이를 근거로 미래에 나타날 변화나 결과를 논리적으로 설명하는 능력을 평가한다."
            ]
        },
        2: {
            "problem_types": ["찬성/반대 입장 논거 파악하기", "논리적 근거 제시하기", "예상 반론에 재반박하기"],
            "eval_goals": [
                "제시된 지문에서 특정 입장(찬성 또는 반대)의 핵심 논거를 정확히 파악하고, 그 근거를 자신의 말로 요약하여 설명하는 능력을 평가한다.",
                "자신의 주장을 뒷받침하기 위해 타당한 이유와 구체적인 사례를 들어 논리적으로 설명하는 능력을 평가한다.",
                "자신의 주장과 반대되는 견해를 예상하고, 그에 대한 논리적인 재반박을 통해 자신의 주장을 강화하는 능력을 평가한다."
            ]
        },
        3: {
            "problem_types": ["문제 상황 요약하기", "문제 해결 방안 제안하기", "기대 효과 및 부작용 설명하기"],
            "eval_goals": [
                "제시된 갈등 상황의 핵심 원인과 현재 상태를 정확히 분석하고, 이를 바탕으로 문제점을 간결하게 요약하는 능력을 평가한다.",
                "주어진 문제를 해결하기 위한 독창적이면서 실현 가능한 방안을 제시하는 능력을 평가한다.",
                "자신이 제안한 해결 방안이 가져올 긍정적인 기대 효과와 발생 가능한 부작용을 균형 있게 설명하는 능력을 평가한다."
            ]
        },
        4: {
            "problem_types": ["두 대안의 핵심 차이점 파악하기", "주어진 기준에 따라 장단점 분석하기", "최종 선택 및 결정 이유 정당화하기"],
            "eval_goals": [
                "제시된 두 가지 선택지의 가장 본질적인 차이점이 무엇인지 정확히 파악하는 능력을 평가한다.",
                "가격, 시간, 편의성 등 주어진 특정 기준에 따라 각 옵션의 장점과 단점을 체계적으로 분석하는 능력을 평가한다.",
                "모든 정보를 종합하여 최종적으로 하나의 옵션을 선택하고, 자신의 선택을 논리적으로 정당화하는 능력을 평가한다."
            ]
        },
        5: {
            "problem_types": ["발표의 핵심 목적 파악하기", "세부 내용 및 근거 설명하기", "전체 내용 요약 및 재구성하기"],
            "eval_goals": [
                "주어진 발표문(지문)의 전체적인 주제와 핵심 목적이 무엇인지 정확히 파악하는 능력을 평가한다.",
                "발표의 주장을 뒷받침하기 위해 사용된 데이터와 근거를 파악하고, 그 내용을 상세히 설명하는 능력을 평가한다.",
                "발표 전체 내용을 자신의 말로 간결하게 요약하고, 핵심 메시지를 재구성하여 전달하는 능력을 평가한다."
            ]
        }
    }
    
    if benchmark_id not in benchmark_configs:
        raise ValueError(f"지원하지 않는 benchmark_id: {benchmark_id}. 1-5 사이의 값을 입력하세요.")
    
    config = benchmark_configs[benchmark_id]
    problem_types = config["problem_types"]
    eval_goals = config["eval_goals"]
    
    print(f"📋 벤치마크 ID {benchmark_id} 설정:")
    print(f"   Problem Types: {problem_types}")
    print(f"   Eval Goals: {eval_goals[:1]}..." if len(eval_goals) > 1 else f"   Eval Goals: {eval_goals}")
    
    # 입력 파일 로드
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 입력 파일 로드 완료: {len(data)}개 항목")
    
    # null인 항목들 찾기
    null_indices = []
    for i, item in enumerate(data):
        if item.get('generated_passage') is None:
            null_indices.append(i)
    
    print(f"🔍 null인 항목 발견: {len(null_indices)}개")
    
    if len(null_indices) == 0:
        print("✅ 모든 passage가 이미 생성되어 있습니다.")
        return
    
    # 모델 클라이언트 초기화
    print(f"🔄 모델 로딩 중: {model_name}")
    llm_client = LocalModelClient(model_name=model_name, gpus=gpus)
    passage_agent = PassageAgent(llm_client=llm_client)
    print("✅ 모델 로딩 완료!")
    
    # null인 항목들에 대해 passage 재생성
    for idx, data_idx in enumerate(null_indices):
        item = data[data_idx]
        source_item = item['source_item']
        
        print(f"\n🔄 {idx+1}/{len(null_indices)} 재생성 중 (인덱스: {data_idx})")
        print(f"   주제: {source_item['korean_topic']}")
        
        # 최대 5번까지 재시도
        max_retries = 5
        retry_count = 0
        generated_passage = None
        
        while generated_passage is None and retry_count < max_retries:
            temp_passage = passage_agent.generate_passage(
                korean_topic=source_item['korean_topic'],
                korean_context=source_item['korean_context'],
                foreign_topic=source_item['foreign_topic'],
                foreign_context=source_item['foreign_context'],
                problem_types=problem_types,
                eval_goals=eval_goals,
                template_key=template_key
            )
            
            if temp_passage is None:
                retry_count += 1
                print(f"   ❌ 생성 실패, 재시도 중... ({retry_count}/{max_retries})")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                # 후처리: 괄호와 그 안의 내용 제거
                temp_passage = clean_passage_text(temp_passage)
                
                # 길이 검증 (300-500자)
                passage_length = len(temp_passage)
                if passage_length < 300:
                    retry_count += 1
                    print(f"   ⚠️  너무 짧음 ({passage_length}자), 재시도 중... ({retry_count}/{max_retries})")
                    torch.cuda.empty_cache()
                    gc.collect()
                elif passage_length > 500:
                    # 500자 초과시 자르기
                    last_period_index = temp_passage.rfind('.')
                    if last_period_index != -1 and last_period_index < 500:
                        temp_passage = temp_passage[:last_period_index + 1]
                        passage_length = len(temp_passage)
                        print(f"   ✂️  500자 초과로 마침표 기준 자름 ({passage_length}자)")
                    else:
                        temp_passage = temp_passage[:500]
                        passage_length = 500
                        print(f"   ✂️  500자 초과로 강제 자름 ({passage_length}자)")
                    
                    generated_passage = temp_passage
                    break
                else:
                    # 길이 조건 만족
                    generated_passage = temp_passage
                    print(f"   ✅ 생성 성공! ({passage_length}자)")
                    break
        
        if generated_passage is None:
            print(f"   ❌ {max_retries}번 시도 후에도 생성 실패")
            generated_passage = "생성 실패"
        
        # 결과 저장
        data[data_idx]['generated_passage'] = generated_passage
        
        # 메모리 정리
        torch.cuda.empty_cache()
        gc.collect()
    
    # 결과를 파일에 저장
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 재생성 완료! 결과 저장: {output_file_path}")
    print(f"📊 총 {len(null_indices)}개 항목 중 재생성 완료")

if __name__ == "__main__":
    # 설정
    INPUT_FILE = "/home/sjin4861/25-1/HCLT/iSKA_Gen/src/data/raw_outputs/2025-08-05/passage/A.X-4.0-Light/passage_agent.create_passage_rubric_aware/benchmark_5_v1.0.0_passage_agent.create_passage_rubric_aware.json"
    OUTPUT_FILE = "/home/sjin4861/25-1/HCLT/iSKA_Gen/src/data/raw_outputs/2025-08-05/passage/A.X-4.0-Light/passage_agent.create_passage_rubric_aware/benchmark_5_v1.0.0_passage_agent.create_passage_rubric_aware_regenerated.json"
    MODEL_NAME = "A.X-4.0-Light"
    TEMPLATE_KEY = "passage_agent.create_passage_rubric_aware"
    BENCHMARK_ID = 5  # 벤치마크 ID 추가
    GPUS = [3]  # 사용할 GPU 번호
    
    print("🚀 null인 passage 재생성 시작!")
    print(f"입력 파일: {INPUT_FILE}")
    print(f"출력 파일: {OUTPUT_FILE}")
    print(f"모델: {MODEL_NAME}")
    print(f"템플릿: {TEMPLATE_KEY}")
    print(f"벤치마크 ID: {BENCHMARK_ID}")
    print(f"GPU: {GPUS}")
    
    regenerate_null_passages(
        input_file_path=INPUT_FILE,
        output_file_path=OUTPUT_FILE,
        model_name=MODEL_NAME,
        template_key=TEMPLATE_KEY,
        benchmark_id=BENCHMARK_ID,
        gpus=GPUS
    )

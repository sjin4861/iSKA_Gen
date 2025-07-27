import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from utils.output_loader import load_passages
from utils.benchmark_loader import load_benchmarks, get_benchmark_by_id
from utils.output_saver import save_model_output, DEFAULT_PAIRWISE_DATA_DIR


def create_passage_pairs(
    model_name: str,
    benchmark_file: str,
    benchmark_version: str = "v1.0.0",
    BENCH_ID_LIST: List[int] = [1, 2, 3, 4, 5]
) -> List[Dict[str, Any]]:
    """
    특정 모델의 두 가지 템플릿으로 생성된 passage들을 비교용 pair 데이터로 만듭니다.
    
    Args:
        model_name (str): 모델 이름
        benchmark_file (str): 벤치마크 파일 이름
        benchmark_version (str): 벤치마크 버전
        BENCH_ID_LIST (List[int]): 처리할 벤치마크 ID 리스트
        
    Returns:
        List[Dict[str, Any]]: Reward Model 학습용 pair 데이터
    """
    
    pair_data = []
    
    for benchmark_id in BENCH_ID_LIST:
        print(f"\n📝 벤치마크 ID {benchmark_id}에 대한 passage pair 생성 중...")
        
        # 1. 벤치마크 정보 로드
        benchmark = get_benchmark_by_id(benchmark_file, benchmark_id)
        if not benchmark:
            print(f"❌ 벤치마크 ID {benchmark_id}를 찾을 수 없습니다.")
            continue
            
        # 2. 두 가지 템플릿으로 생성된 passage 로드
        chosen_passages = load_passages(
            model_name=model_name,
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
            template_key="passage_agent.create_passage"
        )
        
        rejected_passages = load_passages(
            model_name=model_name,
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
            template_key="passage_agent.create_passage_with_korean_errors"
        )
        
        if not chosen_passages:
            print(f"❌ 'passage_agent.create_passage' 템플릿의 passage를 찾을 수 없습니다.")
            continue
            
        if not rejected_passages:
            print(f"❌ 'passage_agent.create_passage_with_korean_errors' 템플릿의 passage를 찾을 수 없습니다.")
            continue
            
        print(f"✅ Chosen passages: {len(chosen_passages)}개")
        print(f"✅ Rejected passages: {len(rejected_passages)}개")
        
        # 3. 벤치마크 정보에서 프롬프트 구성 요소 추출
        problem_types = benchmark.get("problem_types", [])
        eval_goals = benchmark.get("eval_goals", [])
        
        # 4. Pair 데이터 생성
        min_length = min(len(chosen_passages), len(rejected_passages))
        
        for i in range(min_length):
            chosen_passage_data = chosen_passages[i]
            rejected_passage_data = rejected_passages[i]
            
            # source_item이 동일한지 확인
            chosen_source = chosen_passage_data.get("source_item", {})
            rejected_source = rejected_passage_data.get("source_item", {})
            
            # 동일한 source_item인 경우에만 pair 생성
            if chosen_source == rejected_source:
                # 프롬프트 구성
                prompt = create_prompt_from_source(chosen_source, problem_types, eval_goals)
                
                pair_item = {
                    "data_id": f"benchmark_{benchmark_id}.item_{i}",
                    "reply_id": len(pair_data) + 1,
                    "reply_type": ["chosen", "rejected"],
                    "model_name": model_name,
                    "prompt": prompt,
                    "chosen": chosen_passage_data["generated_passage"],
                    "rejected": rejected_passage_data["generated_passage"],
                    "metadata": {
                        "benchmark_id": benchmark_id,
                        "benchmark_version": benchmark_version,
                        "source_item": chosen_source,
                        "problem_types": problem_types,
                        "eval_goals": eval_goals,
                        "chosen_template": "passage_agent.create_passage",
                        "rejected_template": "passage_agent.create_passage_with_korean_errors",
                        "created_at": datetime.now().isoformat()
                    }
                }
                
                pair_data.append(pair_item)
                
            else:
                print(f"⚠️ source_item이 다릅니다. item {i} 건너뜀")
                
        print(f"✅ 벤치마크 ID {benchmark_id}: {min_length}개 pair 생성 완료")
    
    return pair_data


def create_prompt_from_source(
    source_item: Dict[str, Any],
    problem_types: List[str],
    eval_goals: List[str]
) -> str:
    """
    source_item과 벤치마크 정보를 바탕으로 프롬프트를 구성합니다.
    
    Args:
        source_item (Dict[str, Any]): 원본 아이템 정보
        problem_types (List[str]): 문제 유형 리스트
        eval_goals (List[str]): 평가 목표 리스트
        
    Returns:
        str: 구성된 프롬프트
    """
    korean_topic = source_item.get("korean_topic", "")
    korean_context = source_item.get("korean_context", "")
    foreign_topic = source_item.get("foreign_topic", "")
    foreign_context = source_item.get("foreign_context", "")
    
    prompt = f"""다음 조건에 맞는 한국어 독해 지문을 생성해주세요.

**주제 정보:**
- 한국어 주제: {korean_topic}
- 외국어 주제: {foreign_topic}

**배경 맥락:**
- 한국어 맥락: {korean_context}
- 외국어 맥락: {foreign_context}

**문제 유형:** {', '.join(problem_types)}

**평가 목표:** {', '.join(eval_goals)}

위 조건들을 모두 반영하여 적절한 한국어 독해 지문을 작성해주세요. 지문은 주어진 주제와 맥락을 포함하고, 지정된 문제 유형과 평가 목표에 적합해야 합니다."""

    return prompt


def save_passage_pairs(
    model_name: str,
    benchmark_file: str,
    benchmark_version: str = "v1.0.0",
    BENCH_ID_LIST: List[int] = [1, 2, 3, 4, 5]
) -> Path:
    """
    passage pair 데이터를 생성하고 저장합니다.
    
    Args:
        model_name (str): 모델 이름
        benchmark_file (str): 벤치마크 파일 이름
        benchmark_version (str): 벤치마크 버전
        BENCH_ID_LIST (List[int]): 처리할 벤치마크 ID 리스트
        
    Returns:
        Path: 저장된 파일 경로
    """
    print(f"🔄 모델 '{model_name}'의 passage pair 데이터 생성 시작...")
    
    # pair 데이터 생성
    pair_data = create_passage_pairs(
        model_name=model_name,
        benchmark_file=benchmark_file,
        benchmark_version=benchmark_version,
        BENCH_ID_LIST=BENCH_ID_LIST
    )
    
    if not pair_data:
        print("❌ 생성된 pair 데이터가 없습니다.")
        return None
    
    # 모든 벤치마크 ID를 하나의 파일에 저장
    saved_file = save_model_output(
        model_name=f"{model_name}_reward_pairs",
        benchmark_id=0,  # 여러 벤치마크를 포함하므로 0으로 설정
        benchmark_version=benchmark_version,
        template_key="passage_reward_pairs",
        data=pair_data,
        base_dir=DEFAULT_PAIRWISE_DATA_DIR
    )
    
    print(f"✅ 총 {len(pair_data)}개의 passage pair 생성 완료")
    print(f"💾 저장 위치: {saved_file}")
    
    return saved_file


def validate_pair_data(pair_data: List[Dict[str, Any]]) -> None:
    """
    생성된 pair 데이터의 유효성을 검증합니다.
    
    Args:
        pair_data (List[Dict[str, Any]]): 검증할 pair 데이터
    """
    print(f"\n🔍 Pair 데이터 검증 중...")
    
    if not pair_data:
        print("❌ 검증할 데이터가 없습니다.")
        return
    
    required_fields = ["data_id", "reply_id", "reply_type", "model_name", "prompt", "chosen", "rejected"]
    
    for i, item in enumerate(pair_data):
        # 필수 필드 확인
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            print(f"❌ Item {i}: 누락된 필드 {missing_fields}")
            continue
            
        # 데이터 타입 확인
        if not isinstance(item["prompt"], str) or not item["prompt"].strip():
            print(f"❌ Item {i}: prompt가 비어있거나 올바르지 않습니다.")
            
        if not isinstance(item["chosen"], str) or not item["chosen"].strip():
            print(f"❌ Item {i}: chosen passage가 비어있거나 올바르지 않습니다.")
            
        if not isinstance(item["rejected"], str) or not item["rejected"].strip():
            print(f"❌ Item {i}: rejected passage가 비어있거나 올바르지 않습니다.")
    
    print(f"✅ {len(pair_data)}개 항목 검증 완료")


# --- 실행 예시 ---
if __name__ == "__main__":
    # 사용 예시
    MODEL_NAME = "A.X-4.0-Light"  # 실제 모델명으로 변경
    BENCHMARK_FILE = "v1/iSKA-Gen_Benchmark_v1.0.0_20250725_Initial.json"
    BENCH_ID_LIST = [1, 2, 3, 4, 5]
    
    print("🚀 Passage Pair 데이터 생성 시작...")
    
    try:
        # Pair 데이터 생성 및 저장
        saved_file = save_passage_pairs(
            model_name=MODEL_NAME,
            benchmark_file=BENCHMARK_FILE,
            benchmark_version="v1.0.0",
            BENCH_ID_LIST=BENCH_ID_LIST
        )
        
        if saved_file:
            print(f"\n🎉 Passage pair 데이터 생성 완료!")
            print(f"📁 파일 위치: {saved_file}")
            
            # 생성된 데이터 검증
            with open(saved_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            validate_pair_data(saved_data)
            
            # 첫 번째 예시 출력
            if saved_data:
                print(f"\n📋 첫 번째 pair 예시:")
                print(json.dumps(saved_data[0], ensure_ascii=False, indent=2)[:500] + "...")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path.cwd().parent.parent))
sys.path.append(str(Path.cwd().parent / 'modules'))

from modules.iska.passage_eval import PassageEvaluator
from modules.model_client import OpenAIModelClient, LocalModelClient
from utils.benchmark_loader import get_guideline_by_id
from utils.output_loader import load_passages, debug_available_files
from utils.output_saver import save_model_output, DEFAULT_EVALUATION_DIR


def evaluate_passages(
    benchmark_file: str,
    passage_model_name: str,
    evaluator_model: str = "gpt-4o-mini",
    benchmark_version: str = "v1.0.0",
    template_key: str = "passage_eval.binary_rubric",
    passage_template_key: Optional[str] = None,
    BENCH_ID_LIST: List[int] = [1, 2, 3, 4, 5],
    date_str: Optional[str] = None
) -> Dict[int, List[Dict[str, Any]]]:
    """
    특정 모델이 생성한 passage들을 벤치마크에 따라 평가합니다.
    
    Args:
        benchmark_file (str): 벤치마크 파일 이름
        passage_model_name (str): 평가할 모델 이름 (passage 생성한 모델)
        evaluator_model (str): 평가에 사용할 모델 (기본: gpt-4o-mini)
        benchmark_version (str): 벤치마크 버전
        template_key (str): 평가 템플릿 키
        passage_template_key (Optional[str]): passage 생성에 사용된 템플릿 키 (None이면 자동 검색)
        BENCH_ID_LIST (List[int]): 평가할 벤치마크 ID 리스트
        date_str (Optional[str]): 특정 날짜의 데이터를 로드할 때 사용 (None이면 최신 데이터)
        
    Returns:
        Dict[int, List[Dict[str, Any]]]: 벤치마크 ID별 평가 결과
    """
    
    # 1. 평가자 설정
    if evaluator_model.startswith("gpt"):
        evaluator_client = OpenAIModelClient(model_name=evaluator_model)
    else:
        evaluator_client = LocalModelClient(model_name=evaluator_model)
    
    evaluator = PassageEvaluator(llm_client=evaluator_client)
    
    # 2. 벤치마크 로드

    all_results = {}
    
    for benchmark_id in BENCH_ID_LIST:
        print(f"\n📊 벤치마크 ID {benchmark_id} 평가 중...")
        
        # 3. 벤치마크 정보 가져오기
        guideline = get_guideline_by_id(benchmark_file, benchmark_id)
        if not guideline:
            print(f"❌ 벤치마크 ID {benchmark_id}를 찾을 수 없습니다.")
            continue
        problem_types = guideline["problem_types"]
        eval_goals = guideline["eval_goals"]

        # 4. 해당 모델이 생성한 passage 데이터 로드
        passages = load_passages(
            model_name=passage_model_name,
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
            template_key=passage_template_key,  # 지정된 템플릿 키 사용 (None이면 자동 검색)
            date_str=date_str
        )
        
        if not passages:
            print(f"❌ 모델 '{passage_model_name}'의 벤치마크 ID {benchmark_id} passage 데이터를 찾을 수 없습니다.")
            print("🔍 디버깅 정보:")
            debug_available_files(passage_model_name)
            continue
        
        print(f"✅ {len(passages)}개의 passage 로드 완료")
        
        # 5. 각 passage 평가
        results = []
        for i, passage_data in enumerate(passages):
            print(f"  📝 Passage {i+1}/{len(passages)} 평가 중...")
            
            source_item = passage_data["source_item"]
            korean_topic = source_item['korean_topic']
            foreign_topic = source_item['foreign_topic']
            korean_context = source_item['korean_context']
            foreign_context = source_item['foreign_context']
            passage = passage_data['generated_passage']

            # 평가 실행
            if "binary" in template_key:
                result = evaluator.evaluate_binary_rubric(
                    problem_types=problem_types,
                    eval_goals=eval_goals,
                    korean_topic=korean_topic,
                    foreign_topic=foreign_topic,
                    korean_context=korean_context,
                    foreign_context=foreign_context,
                    passage=passage,
                    template_key=template_key
                )
            else:
                result = evaluator.evaluate_passage_metrics(
                    problem_types=problem_types,
                    eval_goals=eval_goals,
                    home_topic=korean_topic,
                    foreign_topic=foreign_topic,
                    home_context=korean_context,
                    foreign_context=foreign_context,
                    passage=passage,
                )

            # 원본 데이터와 평가 결과를 함께 저장
            evaluation_result = {
                "source_item": source_item,
                "generated_passage": passage,
                "evaluation": result
            }
            results.append(evaluation_result)
        
        # 6. 결과 저장 (evaluations 디렉토리에 저장)
        saved_file = save_model_output(
            model_name=f"{passage_model_name}_evaluation",
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
            template_key=f"eval_{template_key.split('.')[-1]}",  # eval_binary_rubric
            data=results,
            base_dir=DEFAULT_EVALUATION_DIR,  # evaluations 디렉토리 사용
            date_str=date_str
        )
        
        print(f"✅ 벤치마크 ID {benchmark_id} 평가 완료 및 저장: {saved_file}")
        
        # 첫 번째 결과 예시 출력
        if results:
            print(f"📋 첫 번째 평가 결과 예시:")
            print(json.dumps(results[0]["evaluation"], ensure_ascii=False, indent=2))
        
        all_results[benchmark_id] = results
    
    return all_results


def evaluate_single_benchmark(
    benchmark_file: str,
    passage_model_name: str,
    benchmark_id: int,
    evaluator_model: str = "gpt-4o-mini",
    benchmark_version: str = "v1.0.0",
    template_key: str = "passage_eval.binary_rubric",
    passage_template_key: Optional[str] = None,
    date_str: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    특정 벤치마크 ID에 대해서만 평가를 수행합니다.
    
    Args:
        benchmark_file (str): 벤치마크 파일 이름
        passage_model_name (str): 평가할 모델 이름
        benchmark_id (int): 평가할 벤치마크 ID
        evaluator_model (str): 평가에 사용할 모델
        benchmark_version (str): 벤치마크 버전
        template_key (str): 평가 템플릿 키
        passage_template_key (Optional[str]): passage 생성에 사용된 템플릿 키 (None이면 자동 검색)
        date_str (Optional[str]): 특정 날짜의 데이터를 로드할 때 사용 (None이면 최신 데이터)
        
    Returns:
        Optional[List[Dict[str, Any]]]: 평가 결과 또는 None
    """
    results = evaluate_passages(
        benchmark_file=benchmark_file,
        passage_model_name=passage_model_name,
        evaluator_model=evaluator_model,
        benchmark_version=benchmark_version,
        template_key=template_key,
        passage_template_key=passage_template_key,
        BENCH_ID_LIST=[benchmark_id],
        date_str=date_str
    )
    
    return results.get(benchmark_id)


# --- 실행 예시 ---
if __name__ == "__main__":
    # 사용 예시
    MODEL_LIST = ["EXAONE-3.5B", "Qwen-8B"]  # 평가할 모델 리스트
    BENCH_ID_LIST = [1, 2, 3]  # 평가할 벤치마크 ID 리스트
    
    print("🔍 Passage 평가 시작...")
    
    for passage_model_name in MODEL_LIST:
        print(f"\n🤖 모델 '{passage_model_name}' 평가 중...")
        
        try:
            results = evaluate_passages(
                benchmark_file="v1/iSKA-Gen_Benchmark_v1.0.0_20250725_Initial.json",
                passage_model_name=passage_model_name,
                evaluator_model="gpt-4o-mini",
                benchmark_version="v1.0.0",
                template_key="passage_eval.binary_rubric",  # 올바른 템플릿 키 사용
                BENCH_ID_LIST=BENCH_ID_LIST
            )
            
            print(f"✅ 모델 '{passage_model_name}' 평가 완료")
            
            # 간단한 통계 출력
            for benchmark_id, benchmark_results in results.items():
                print(f"  📊 벤치마크 {benchmark_id}: {len(benchmark_results)}개 평가 완료")
                
        except Exception as e:
            print(f"❌ 모델 '{passage_model_name}' 평가 중 오류 발생: {e}")
            continue
    
    print("\n🎉 모든 평가 완료!")
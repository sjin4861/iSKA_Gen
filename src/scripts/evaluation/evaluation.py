#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stem 평가 실행 스크립트

생성된 stem들에 대해 다양한 루브릭으로 평가를 수행합니다.

사용법 예시:
    # 특정 벤치마크와 특정 stem 모델로 평가
    python src/scripts/evaluation/evaluation.py --model-name gpt-4o-mini --benchmark-id 1 --stem-model EXAONE-3.5-32B-Instruct
    
    # 모든 벤치마크에 대해 평가
    python src/scripts/evaluation/evaluation.py --model-name gpt-4o-mini --all-benchmarks --stem-model EXAONE-3.5-32B-Instruct
    
    # 모든 stem 모델에 대해 평가  
    python src/scripts/evaluation/evaluation.py --model-name gpt-4o-mini --benchmark-id 3 --all-stem-models
    
    # 사용 가능한 벤치마크와 모델 목록 확인
    python src/scripts/evaluation/evaluation.py --list-available
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime


# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # iSKA_Gen 디렉토리
sys.path.append(str(PROJECT_ROOT))
from src.data.repositories.evaluation_repository_impl import EvaluationRepositoryImpl
from src.data.repositories.raw_output_repository_impl import RawOutputRepositoryImpl
from src.domain.usecases.evaluation.evaluate_stems import EvaluateStemsUseCase, EvaluateStemsInput
from src.domain.entities.rubrics import RubricID
from src.domain.entities.enums import ContentType
from src.domain.entities.output_query import OutputQuery
from src.modules.client_factory import ModelClientFactory
from src.modules.model_client import LocalModelClient

MODEL_LIST = [
    "A.X-4.0-Light",
    "Midm-2.0-Base-Instruct",
    "EXAONE-3.5-7.8B-Instruct", 
    "llama3.1_korean_v1.1_sft_by_aidx"
]

# Stem 평가용 루브릭 (기존 채점 시스템과 호환)
# completeness_for_guidelines = "completeness_for_guidelines"
# clarity_of_core_theme = "clarity_of_core_theme"
# reference_groundedness = "reference_groundedness"
# logical_flow = "logical_flow"
# korean_quality = "korean_quality"
# l2_learner_suitability = "l2_learner_suitability"


def main():
    parser = argparse.ArgumentParser(description="Stem 평가 실행기")
    parser.add_argument("--benchmark-id", type=int, help="특정 벤치마크 ID (1-5)")
    parser.add_argument("--all-benchmarks", action="store_true", help="모든 벤치마크에 대해 평가")
    parser.add_argument("--model-name", type=str, required=True, help="평가에 사용할 모델명")
    parser.add_argument("--stem-model", type=str, help="특정 stem 생성 모델명")
    parser.add_argument("--all-stem-models", action="store_true", help="모든 stem 모델에 대해 평가")
    parser.add_argument("--date", type=str, help="평가할 stem 생성 날짜 (YYYY-MM-DD)")
    parser.add_argument("--run-id", type=str, default=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                        help="평가 실행 ID")
    parser.add_argument("--rubrics", type=str, nargs="+",
                        default=["l2_learner_suitability"], 
                        # default=["completeness_for_guidelines", 
                        #         "clarity_of_core_theme", 
                        #         "reference_groundedness", 
                        #         "logical_flow",
                        #         "korean_quality", 
                                # "l2_learner_suitability"],
                        help="평가할 루브릭 목록 (구버전 호환성 지원)")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 2, 3], help="사용할 GPU 번호들")
    parser.add_argument("--limit", type=int, help="평가할 stem 개수 제한")
    parser.add_argument("--temperature", type=float, default=0.1, help="평가 모델 temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="평가 모델 max tokens")
    
    args = parser.parse_args()
    
    # 저장소 초기화
    evaluation_repo = EvaluationRepositoryImpl("data_store")
    raw_output_repo = RawOutputRepositoryImpl("data_store")
    

    # 벤치마크 ID 결정
    if args.all_benchmarks:
        benchmark_ids = [1, 2, 3, 4, 5]
        print(f"🔄 모든 벤치마크 평가: {benchmark_ids}")
    elif args.benchmark_id:
        benchmark_ids = [args.benchmark_id]
        print(f"🎯 특정 벤치마크 평가: {benchmark_ids}")
    else:
        print("❌ --benchmark-id 또는 --all-benchmarks 중 하나를 지정해야 합니다.")
        return 1

    # Stem 모델 결정  
    if args.all_stem_models:
        stem_models = MODEL_LIST
        print(f"🔄 모든 stem 모델 평가: {stem_models}")
    elif args.stem_model:
        stem_models = [args.stem_model]
        print(f"🎯 특정 stem 모델 평가: {stem_models}")
    
    print("🔍 Stem 평가 실행기")
    print("=" * 60)
    print(f"벤치마크 ID: {benchmark_ids}")
    print(f"평가 모델: {args.model_name}")
    print(f"GPU 사용: {args.gpus}")
    print(f"Stem 모델: {stem_models}")
    print(f"생성 날짜: {args.date or '전체'}")
    print(f"실행 ID: {args.run_id}")
    print(f"루브릭: {', '.join(args.rubrics)}")
    print(f"제한: {args.limit or '없음'}")
    print(f"온도: {args.temperature}")
    print("-" * 60)
    
    # 루브릭 ID 변환 (구버전 호환성 지원)
    rubric_mapping = {
        # 구버전 이름 -> 새 버전 RubricID
        "completeness_for_guidelines": RubricID.completeness_for_guidelines,
        "clarity_of_core_theme": RubricID.clarity_of_core_theme, 
        "reference_groundedness": RubricID.reference_groundedness,
        "logical_flow": RubricID.logical_flow,
        "korean_quality": RubricID.korean_quality,
        "l2_learner_suitability": RubricID.l2_learner_suitability,
        # 새 버전 이름들도 지원
        "R1_GUIDELINE_COMPLETENESS": RubricID.R1_GUIDELINE_COMPLETENESS,
        "R2_TOPIC_CLARITY": RubricID.R2_TOPIC_CLARITY,
        "R3_SOURCE_GROUNDEDNESS": RubricID.R3_SOURCE_GROUNDEDNESS,
        "R4_LOGICAL_STRUCTURE": RubricID.R4_LOGICAL_STRUCTURE,
        "R5_KOREAN_QUALITY": RubricID.R5_KOREAN_QUALITY,
        "R6_L2_APPROPRIATENESS": RubricID.R6_L2_APPROPRIATENESS,
    }
    
    try:
        rubric_ids = []
        for rubric in args.rubrics:
            if rubric in rubric_mapping:
                rubric_ids.append(rubric_mapping[rubric])
            else:
                # 직접 RubricID로 시도
                rubric_ids.append(RubricID(rubric))
    except ValueError as e:
        print(f"❌ 잘못된 루브릭 이름: {e}")
        print(f"💡 사용 가능한 루브릭: {list(rubric_mapping.keys())}")
        return 1
    
    # 전체 결과 통계
    total_evaluations = 0
    total_success = 0
    total_failed = 0
    total_count = 0
    
    # 평가 클라이언트를 한 번만 생성 (CUDA 재초기화 방지)
    print(f"\n🤖 평가 모델 클라이언트 생성 중... ({args.model_name})")
    try:
        # 모델 타입에 따라 적절한 클라이언트 생성
        if "gpt" in args.model_name.lower() or "openai" in args.model_name.lower():
            shared_client = ModelClientFactory.create_model_client(
                client_type="openai", 
                model_name=args.model_name
            )
        else:
            # 로컬 모델의 경우
            shared_client = LocalModelClient(
                model_name=args.model_name,
                temperature=args.temperature,
                max_new_tokens=args.max_tokens,
                gpus=args.gpus
            )
        print(f"✅ 평가 클라이언트 생성 완료")
    except Exception as e:
        print(f"❌ 평가 클라이언트 생성 실패: {e}")
        return 1
    
    # Repository와 UseCase를 한 번만 생성
    usecase = EvaluateStemsUseCase(evaluation_repo)
    # 각 벤치마크와 stem 모델 조합에 대해 평가 수행
    for benchmark_id in benchmark_ids:
        for stem_model in stem_models:
            print(f"\n🎯 벤치마크 {benchmark_id}, stem 모델 '{stem_model}' 평가 시작")
            print("=" * 40)
            
            # 평가할 stem 후보 조회
            query = OutputQuery(
                benchmark_id=benchmark_id,
                model_name=stem_model,
                date_from=datetime.strptime(args.date, "%Y-%m-%d") if args.date else None,
                date_to=datetime.strptime(args.date, "%Y-%m-%d") if args.date else None,
                limit=args.limit
            )
            print(f"📋 Stem 후보 조회 중...")
            stem_candidates = list(raw_output_repo.find(ContentType.stem, query))
            if not stem_candidates:
                print(f"❌ 벤치마크 {benchmark_id}, 모델 '{stem_model}'에 대한 stem을 찾을 수 없습니다.")
                continue
            
            print(f"✅ {len(stem_candidates)}개의 stem 후보를 찾았습니다.")
            
            # 현재 조합에 대한 run_id 생성
            current_run_id = f"{args.run_id}_b{benchmark_id}_{stem_model.replace('/', '_')}"
            
            evaluate_input = EvaluateStemsInput(
                stem_candidates=stem_candidates,
                evaluator_model=args.model_name,
                rubric_ids=rubric_ids,
                run_id=current_run_id,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            print(f"🚀 평가 시작... (Run ID: {current_run_id})")
            print(f"📋 루브릭별 평가 방식:")
            for rubric_id in rubric_ids:
                evaluation_type = "content + stems" if rubric_id in [
                    RubricID.completeness_for_guidelines, 
                    # RubricID.l2_learner_suitability,
                    RubricID.R1_GUIDELINE_COMPLETENESS,
                    # RubricID.R6_L2_APPROPRIATENESS
                ] else "content only"
                print(f"  - {rubric_id.value}: {evaluation_type}")
            try:
                # 공유 클라이언트를 사용하여 평가 수행 (CUDA 재초기화 방지)
                result = usecase.execute_with_shared_client(evaluate_input, shared_client)
                
                print(f"\n📊 벤치마크 {benchmark_id}, 모델 '{stem_model}' 평가 완료!")
                print(f"성공: {result.total_success}, 실패: {result.total_failed}, 총계: {result.total_count}")
                
                # 전체 통계에 추가
                total_evaluations += 1
                total_success += result.total_success
                total_failed += result.total_failed
                total_count += result.total_count
                
                print(f"루브릭별 결과:")
                for i, (rubric_id, evaluation) in enumerate(zip(rubric_ids, result.evaluations)):
                    print(f"  {i+1}. {rubric_id.value}: {evaluation.success_count}/{evaluation.total_count}")
                
            except Exception as e:
                print(f"❌ 벤치마크 {benchmark_id}, 모델 '{stem_model}' 평가 중 오류: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 전체 결과 요약
    print("\n" + "=" * 80)
    print("🎉 전체 평가 완료!")
    print("=" * 80)
    print(f"평가한 조합 수: {total_evaluations}")
    print(f"총 성공: {total_success}")
    print(f"총 실패: {total_failed}")
    print(f"총 개수: {total_count}")
    
    if total_count > 0:
        success_rate = (total_success / total_count) * 100
        print(f"전체 성공률: {success_rate:.1f}%")
    
    print(f"\n💾 평가 결과는 data_store/evaluations/ 에 저장되었습니다.")
    print(f"기본 실행 ID: {args.run_id}")
    
    # 클라이언트 정리
    try:
        if hasattr(shared_client, 'cleanup'):
            shared_client.cleanup()
        print("🧹 평가 클라이언트 정리 완료")
    except:
        pass
    
    return 0 if total_evaluations > 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

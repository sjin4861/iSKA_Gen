#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 평가 스크립트

생성된 stem들과 다른 콘텐츠들에 대해 다양한 루브릭으로 평가를 수행합니다.

주요 기능:
1. Stem 평가 (기존 기능)
2. 다른 콘텐츠 타입 평가 (확장 가능)
3. 배치 평가 (여러 모델/벤치마크 조합)
4. 결과 분석 및 리포트 생성

사용법 예시:
    # 기본 stem 평가
    python unified_evaluation.py --content-type stem --eval-model gpt-4o-mini --benchmark-id 1 --content-model EXAONE-3.5-7.8B-Instruct
    
    # 모든 벤치마크에 대해 평가
    python unified_evaluation.py --content-type stem --eval-model gpt-4o-mini --all-benchmarks --content-model EXAONE-3.5-7.8B-Instruct
    
    # 모든 콘텐츠 모델에 대해 평가  
    python unified_evaluation.py --content-type stem --eval-model gpt-4o-mini --benchmark-id 3 --all-content-models
    
    # 배치 평가 (여러 조합)
    python unified_evaluation.py --content-type stem --eval-model gpt-4o-mini --batch-mode --config batch_config.json
    
    # 사용 가능한 옵션 확인
    python unified_evaluation.py --list-available
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.repositories.evaluation_repository_impl import EvaluationRepositoryImpl
from src.data.repositories.raw_output_repository_impl import RawOutputRepositoryImpl
from src.domain.usecases.evaluation.evaluate_stems import EvaluateStemsUseCase, EvaluateStemsInput
from src.domain.entities.rubrics import RubricID
from src.domain.entities.enums import ContentType
from src.domain.entities.output_query import OutputQuery
from src.modules.client_factory import ModelClientFactory
from src.modules.model_client import LocalModelClient

# ================= 설정 및 상수 =================

DEFAULT_MODELS = [
    "A.X-4.0-Light",
    "Midm-2.0-Base-Instruct",
    "EXAONE-3.5-7.8B-Instruct", 
    "llama3.1_korean_v1.1_sft_by_aidx"
]

# 루브릭 매핑 (구버전 호환성 지원)
RUBRIC_MAPPING = {
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

DEFAULT_RUBRICS = ["l2_learner_suitability"]

# ================= 데이터 클래스 =================

@dataclass
class EvaluationConfig:
    """평가 설정"""
    content_type: ContentType
    eval_model: str
    content_models: List[str]
    benchmark_ids: List[int]
    rubrics: List[str]
    date: Optional[str] = None
    run_id: Optional[str] = None
    gpus: List[int] = None
    limit: Optional[int] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    data_store_path: str = "data_store"

@dataclass
class EvaluationResult:
    """평가 결과"""
    benchmark_id: int
    content_model: str
    rubric_results: Dict[str, Dict[str, Any]]
    total_success: int
    total_failed: int
    total_count: int
    run_id: str
    timestamp: datetime

@dataclass
class BatchResult:
    """배치 평가 결과"""
    results: List[EvaluationResult]
    total_evaluations: int
    total_success: int
    total_failed: int
    total_count: int
    success_rate: float
    config: EvaluationConfig

# ================= 유틸리티 함수 =================

def parse_comma_separated(value: str) -> List[str]:
    """쉼표로 구분된 문자열을 리스트로 변환"""
    return [item.strip() for item in value.split(",") if item.strip()]

def parse_benchmark_ids(value: str) -> List[int]:
    """벤치마크 ID 문자열을 정수 리스트로 변환"""
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"벤치마크 ID는 숫자여야 합니다: {value}")

def validate_rubrics(rubrics: List[str]) -> List[RubricID]:
    """루브릭 이름을 검증하고 RubricID로 변환"""
    rubric_ids = []
    for rubric in rubrics:
        if rubric in RUBRIC_MAPPING:
            rubric_ids.append(RUBRIC_MAPPING[rubric])
        else:
            try:
                rubric_ids.append(RubricID(rubric))
            except ValueError:
                raise ValueError(f"잘못된 루브릭 이름: {rubric}")
    return rubric_ids

def create_evaluation_client(model_name: str, temperature: float, max_tokens: int, gpus: List[int]):
    """평가용 모델 클라이언트 생성"""
    if "gpt" in model_name.lower() or "openai" in model_name.lower():
        return ModelClientFactory.create_model_client(
            client_type="openai", 
            model_name=model_name
        )
    else:
        return LocalModelClient(
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_tokens,
            gpus=gpus or [0]
        )

# ================= 메인 평가 클래스 =================

class UnifiedEvaluator:
    """통합 평가기"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.evaluation_repo = EvaluationRepositoryImpl(config.data_store_path)
        self.raw_output_repo = RawOutputRepositoryImpl(config.data_store_path)
        self.evaluation_client = None
        self.usecase = EvaluateStemsUseCase(self.evaluation_repo)
        
    def initialize_client(self):
        """평가 클라이언트 초기화"""
        print(f"\n🤖 평가 모델 클라이언트 생성 중... ({self.config.eval_model})")
        try:
            self.evaluation_client = create_evaluation_client(
                self.config.eval_model,
                self.config.temperature,
                self.config.max_tokens,
                self.config.gpus
            )
            print("✅ 평가 클라이언트 생성 완료")
        except Exception as e:
            print(f"❌ 평가 클라이언트 생성 실패: {e}")
            raise
    
    def cleanup_client(self):
        """평가 클라이언트 정리"""
        try:
            if self.evaluation_client and hasattr(self.evaluation_client, 'cleanup'):
                self.evaluation_client.cleanup()
            print("🧹 평가 클라이언트 정리 완료")
        except Exception:
            pass
    
    def find_content_candidates(self, benchmark_id: int, content_model: str) -> List:
        """콘텐츠 후보 조회"""
        query = OutputQuery(
            benchmark_id=benchmark_id,
            model_name=content_model,
            date_from=datetime.strptime(self.config.date, "%Y-%m-%d") if self.config.date else None,
            date_to=datetime.strptime(self.config.date, "%Y-%m-%d") if self.config.date else None,
            limit=self.config.limit
        )
        
        return list(self.raw_output_repo.find(self.config.content_type, query))
    
    def evaluate_single_combination(self, benchmark_id: int, content_model: str) -> Optional[EvaluationResult]:
        """단일 조합에 대한 평가 수행"""
        print(f"\n🎯 벤치마크 {benchmark_id}, 콘텐츠 모델 '{content_model}' 평가 시작")
        print("=" * 50)
        
        # 콘텐츠 후보 조회
        print(f"📋 {self.config.content_type.value} 후보 조회 중...")
        candidates = self.find_content_candidates(benchmark_id, content_model)
        
        if not candidates:
            print(f"❌ 벤치마크 {benchmark_id}, 모델 '{content_model}'에 대한 {self.config.content_type.value}을 찾을 수 없습니다.")
            return None
        
        print(f"✅ {len(candidates)}개의 {self.config.content_type.value} 후보를 찾았습니다.")
        
        # 루브릭 변환
        try:
            rubric_ids = validate_rubrics(self.config.rubrics)
        except ValueError as e:
            print(f"❌ {e}")
            return None
        
        # 실행 ID 생성
        run_id = self.config.run_id or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        current_run_id = f"{run_id}_b{benchmark_id}_{content_model.replace('/', '_')}"
        
        # 평가 입력 준비
        evaluate_input = EvaluateStemsInput(
            stem_candidates=candidates,  # TODO: 다른 콘텐츠 타입도 지원하도록 확장
            evaluator_model=self.config.eval_model,
            rubric_ids=rubric_ids,
            run_id=current_run_id,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        print(f"🚀 평가 시작... (Run ID: {current_run_id})")
        print(f"📋 루브릭별 평가 방식:")
        for rubric_id in rubric_ids:
            evaluation_type = "content + stems" if rubric_id in [
                RubricID.completeness_for_guidelines, 
                RubricID.R1_GUIDELINE_COMPLETENESS,
            ] else "content only"
            print(f"  - {rubric_id.value}: {evaluation_type}")
        
        try:
            # 평가 수행
            result = self.usecase.execute_with_shared_client(evaluate_input, self.evaluation_client)
            
            print(f"\n📊 벤치마크 {benchmark_id}, 모델 '{content_model}' 평가 완료!")
            print(f"성공: {result.total_success}, 실패: {result.total_failed}, 총계: {result.total_count}")
            
            # 루브릭별 결과 생성
            rubric_results = {}
            for i, (rubric_id, evaluation) in enumerate(zip(rubric_ids, result.evaluations)):
                rubric_results[rubric_id.value] = {
                    "success_count": evaluation.success_count,
                    "total_count": evaluation.total_count,
                    "success_rate": (evaluation.success_count / evaluation.total_count * 100) if evaluation.total_count > 0 else 0
                }
                print(f"  {i+1}. {rubric_id.value}: {evaluation.success_count}/{evaluation.total_count} ({rubric_results[rubric_id.value]['success_rate']:.1f}%)")
            
            return EvaluationResult(
                benchmark_id=benchmark_id,
                content_model=content_model,
                rubric_results=rubric_results,
                total_success=result.total_success,
                total_failed=result.total_failed,
                total_count=result.total_count,
                run_id=current_run_id,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"❌ 벤치마크 {benchmark_id}, 모델 '{content_model}' 평가 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_batch(self) -> BatchResult:
        """배치 평가 수행"""
        print("🔍 통합 평가기 시작")
        print("=" * 60)
        print(f"콘텐츠 타입: {self.config.content_type.value}")
        print(f"벤치마크 ID: {self.config.benchmark_ids}")
        print(f"평가 모델: {self.config.eval_model}")
        print(f"콘텐츠 모델: {self.config.content_models}")
        print(f"GPU 사용: {self.config.gpus}")
        print(f"생성 날짜: {self.config.date or '전체'}")
        print(f"루브릭: {', '.join(self.config.rubrics)}")
        print(f"제한: {self.config.limit or '없음'}")
        print(f"온도: {self.config.temperature}")
        print("-" * 60)
        
        self.initialize_client()
        
        results = []
        total_success = 0
        total_failed = 0
        total_count = 0
        
        try:
            for benchmark_id in self.config.benchmark_ids:
                for content_model in self.config.content_models:
                    result = self.evaluate_single_combination(benchmark_id, content_model)
                    if result:
                        results.append(result)
                        total_success += result.total_success
                        total_failed += result.total_failed
                        total_count += result.total_count
        
        finally:
            self.cleanup_client()
        
        success_rate = (total_success / total_count * 100) if total_count > 0 else 0
        
        return BatchResult(
            results=results,
            total_evaluations=len(results),
            total_success=total_success,
            total_failed=total_failed,
            total_count=total_count,
            success_rate=success_rate,
            config=self.config
        )

# ================= 결과 분석 및 리포트 =================

class ResultAnalyzer:
    """결과 분석기"""
    
    @staticmethod
    def print_summary(batch_result: BatchResult):
        """결과 요약 출력"""
        print("\n" + "=" * 80)
        print("🎉 전체 평가 완료!")
        print("=" * 80)
        print(f"평가한 조합 수: {batch_result.total_evaluations}")
        print(f"총 성공: {batch_result.total_success}")
        print(f"총 실패: {batch_result.total_failed}")
        print(f"총 개수: {batch_result.total_count}")
        print(f"전체 성공률: {batch_result.success_rate:.1f}%")
        
        # 벤치마크별 결과
        print(f"\n📊 벤치마크별 결과:")
        benchmark_stats = {}
        for result in batch_result.results:
            if result.benchmark_id not in benchmark_stats:
                benchmark_stats[result.benchmark_id] = {
                    "success": 0, "failed": 0, "count": 0, "models": []
                }
            benchmark_stats[result.benchmark_id]["success"] += result.total_success
            benchmark_stats[result.benchmark_id]["failed"] += result.total_failed
            benchmark_stats[result.benchmark_id]["count"] += result.total_count
            benchmark_stats[result.benchmark_id]["models"].append(result.content_model)
        
        for bench_id, stats in benchmark_stats.items():
            rate = (stats["success"] / stats["count"] * 100) if stats["count"] > 0 else 0
            print(f"  벤치마크 {bench_id}: {stats['success']}/{stats['count']} ({rate:.1f}%) - 모델: {len(set(stats['models']))}개")
        
        # 모델별 결과
        print(f"\n🤖 모델별 결과:")
        model_stats = {}
        for result in batch_result.results:
            if result.content_model not in model_stats:
                model_stats[result.content_model] = {
                    "success": 0, "failed": 0, "count": 0, "benchmarks": []
                }
            model_stats[result.content_model]["success"] += result.total_success
            model_stats[result.content_model]["failed"] += result.total_failed
            model_stats[result.content_model]["count"] += result.total_count
            model_stats[result.content_model]["benchmarks"].append(result.benchmark_id)
        
        for model, stats in model_stats.items():
            rate = (stats["success"] / stats["count"] * 100) if stats["count"] > 0 else 0
            print(f"  {model}: {stats['success']}/{stats['count']} ({rate:.1f}%) - 벤치마크: {len(set(stats['benchmarks']))}개")
        
        print(f"\n💾 평가 결과는 {batch_result.config.data_store_path}/evaluations/ 에 저장되었습니다.")
    
    @staticmethod
    def save_report(batch_result: BatchResult, output_path: Path):
        """결과를 JSON 리포트로 저장"""
        report = {
            "summary": {
                "total_evaluations": batch_result.total_evaluations,
                "total_success": batch_result.total_success,
                "total_failed": batch_result.total_failed,
                "total_count": batch_result.total_count,
                "success_rate": batch_result.success_rate,
                "timestamp": datetime.now().isoformat()
            },
            "config": {
                "content_type": batch_result.config.content_type.value,
                "eval_model": batch_result.config.eval_model,
                "content_models": batch_result.config.content_models,
                "benchmark_ids": batch_result.config.benchmark_ids,
                "rubrics": batch_result.config.rubrics,
                "date": batch_result.config.date
            },
            "results": []
        }
        
        for result in batch_result.results:
            report["results"].append({
                "benchmark_id": result.benchmark_id,
                "content_model": result.content_model,
                "rubric_results": result.rubric_results,
                "total_success": result.total_success,
                "total_failed": result.total_failed,
                "total_count": result.total_count,
                "run_id": result.run_id,
                "timestamp": result.timestamp.isoformat()
            })
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 상세 리포트 저장: {output_path}")

# ================= 명령줄 인터페이스 =================

def create_parser():
    """명령줄 인자 파서 생성"""
    parser = argparse.ArgumentParser(
        description="통합 평가 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 stem 평가
  python unified_evaluation.py --content-type stem --eval-model gpt-4o-mini --benchmark-id 1 --content-model EXAONE-3.5-7.8B-Instruct
  
  # 모든 벤치마크 평가
  python unified_evaluation.py --content-type stem --eval-model gpt-4o-mini --all-benchmarks --content-model EXAONE-3.5-7.8B-Instruct
  
  # 모든 콘텐츠 모델 평가
  python unified_evaluation.py --content-type stem --eval-model gpt-4o-mini --benchmark-id 3 --all-content-models
  
  # 사용자 정의 설정
  python unified_evaluation.py --content-type stem --eval-model gpt-4o-mini --content-models "A.X-4.0-Light,EXAONE-3.5-7.8B-Instruct" --benchmark-ids "1,2" --rubrics "korean_quality,l2_learner_suitability"
        """
    )
    
    # 기본 설정
    parser.add_argument("--content-type", type=str, default="stem", 
                        choices=["stem", "passage", "audio", "image"],
                        help="평가할 콘텐츠 타입")
    parser.add_argument("--eval-model", type=str, required=True,
                        help="평가에 사용할 모델명")
    
    # 콘텐츠 모델 선택
    content_group = parser.add_mutually_exclusive_group()
    content_group.add_argument("--content-model", type=str,
                               help="특정 콘텐츠 생성 모델명")
    content_group.add_argument("--content-models", type=str,
                               help="콘텐츠 모델 리스트 (쉼표로 구분)")
    content_group.add_argument("--all-content-models", action="store_true",
                               help="모든 콘텐츠 모델에 대해 평가")
    
    # 벤치마크 선택
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument("--benchmark-id", type=int,
                                 help="특정 벤치마크 ID (1-5)")
    benchmark_group.add_argument("--benchmark-ids", type=str,
                                 help="벤치마크 ID 리스트 (쉼표로 구분)")
    benchmark_group.add_argument("--all-benchmarks", action="store_true",
                                 help="모든 벤치마크에 대해 평가")
    
    # 평가 설정
    parser.add_argument("--rubrics", type=str, nargs="+",
                        default=DEFAULT_RUBRICS,
                        help="평가할 루브릭 목록")
    parser.add_argument("--date", type=str,
                        help="평가할 콘텐츠 생성 날짜 (YYYY-MM-DD)")
    parser.add_argument("--run-id", type=str,
                        help="평가 실행 ID")
    
    # 모델 설정
    parser.add_argument("--gpus", type=str, default="0",
                        help="사용할 GPU 번호들 (쉼표로 구분)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="평가 모델 temperature")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="평가 모델 max tokens")
    
    # 기타
    parser.add_argument("--limit", type=int,
                        help="평가할 콘텐츠 개수 제한")
    parser.add_argument("--data-store", type=str, default="data_store",
                        help="데이터 저장소 경로")
    parser.add_argument("--output-report", type=str,
                        help="결과 리포트 저장 경로")
    parser.add_argument("--list-available", action="store_true",
                        help="사용 가능한 옵션들 표시")
    
    return parser

def list_available_options():
    """사용 가능한 옵션들 표시"""
    print("📋 사용 가능한 옵션들")
    print("=" * 50)
    
    print(f"\n🤖 기본 콘텐츠 모델:")
    for i, model in enumerate(DEFAULT_MODELS, 1):
        print(f"  {i}. {model}")
    
    print(f"\n📊 사용 가능한 루브릭:")
    for rubric, rubric_id in RUBRIC_MAPPING.items():
        print(f"  - {rubric} ({rubric_id.value})")
    
    print(f"\n🎯 콘텐츠 타입:")
    for content_type in ["stem", "passage", "audio", "image"]:
        print(f"  - {content_type}")
    
    print(f"\n🔢 벤치마크 ID: 1-5")

def main():
    """메인 실행 함수"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.list_available:
        list_available_options()
        return 0
    
    # 설정 검증 및 생성
    try:
        # 콘텐츠 타입
        content_type = ContentType(args.content_type)
        
        # 콘텐츠 모델 결정
        if args.all_content_models:
            content_models = DEFAULT_MODELS
        elif args.content_models:
            content_models = parse_comma_separated(args.content_models)
        elif args.content_model:
            content_models = [args.content_model]
        else:
            print("❌ 콘텐츠 모델을 지정해야 합니다. (--content-model, --content-models, 또는 --all-content-models)")
            return 1
        
        # 벤치마크 ID 결정
        if args.all_benchmarks:
            benchmark_ids = [1, 2, 3, 4, 5]
        elif args.benchmark_ids:
            benchmark_ids = parse_benchmark_ids(args.benchmark_ids)
        elif args.benchmark_id:
            benchmark_ids = [args.benchmark_id]
        else:
            print("❌ 벤치마크 ID를 지정해야 합니다. (--benchmark-id, --benchmark-ids, 또는 --all-benchmarks)")
            return 1
        
        # GPU 설정
        try:
            gpus = [int(gpu.strip()) for gpu in args.gpus.split(",")]
        except ValueError:
            print(f"❌ GPU ID는 숫자여야 합니다: {args.gpus}")
            return 1
        
        # 평가 설정 생성
        config = EvaluationConfig(
            content_type=content_type,
            eval_model=args.eval_model,
            content_models=content_models,
            benchmark_ids=benchmark_ids,
            rubrics=args.rubrics,
            date=args.date,
            run_id=args.run_id,
            gpus=gpus,
            limit=args.limit,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            data_store_path=args.data_store
        )
        
        # 평가 실행
        evaluator = UnifiedEvaluator(config)
        batch_result = evaluator.evaluate_batch()
        
        # 결과 분석 및 출력
        ResultAnalyzer.print_summary(batch_result)
        
        # 리포트 저장
        if args.output_report:
            ResultAnalyzer.save_report(batch_result, Path(args.output_report))
        
        return 0 if batch_result.total_evaluations > 0 else 1
        
    except Exception as e:
        print(f"❌ 스크립트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

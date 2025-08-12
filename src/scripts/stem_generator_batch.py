#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2025-08-08 EXAONE passage 데이터 기반으로 4개 모델이 stem 생성하는 배치 스크립트
- Clean Architecture: UseCase + Repository 패턴
- 대상: stem 생성 작업
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any


# ---- project path bootstrap ----
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.extend([str(PROJECT_ROOT), str(SRC_ROOT)])
from src.data.repositories.llm_gateway_impl import LLMGatewayImpl

# ---- domain usecases & repos ----
from domain.usecases.benchmark.get_benchmark_set_by_id import (
    GetBenchmarkSetByIdUseCase, GetBenchmarkSetByIdInput
)
from domain.usecases.stem.fill_missing_stems import (
    FillMissingStemsUseCase, FillMissingStemsInput
)
from domain.repositories.stem_repository import StemRepository
from data.repositories.benchmark_repository_impl import BenchmarkRepositoryImpl
from data.repositories.stem_repository_impl import StemRepositoryImpl
from data.repositories.content_repository_impl import ContentRepositoryImpl

# -------------------------------
# settings & helpers  
# -------------------------------
DEFAULT_BENCHMARK_FILE = "iSKA-Gen_Benchmark_v1.1.0_20250808.json"
# -------------------------------
# settings & helpers  
# -------------------------------
DEFAULT_BENCHMARK_FILE = "iSKA-Gen_Benchmark_v1.1.0_20250808.json"

# 벤치마크별 passage 템플릿 매핑
DEFAULT_TEMPLATE_BY_ID = {
    1: "passage_agent.create_passage_rubric_aware",
    2: "passage_agent.create_domestic_passage", 
    3: "passage_agent.create_dialogue_passage",
    4: "passage_agent.create_dialogue_passage",
    5: "passage_agent.create_image_caption_and_situation"
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate stems from EXAONE passages using multiple models.")
    p.add_argument("--passage-model", default="EXAONE-3.5-7.8B-Instruct", help="Passage 생성에 사용된 모델명")
    p.add_argument("--stem-models", default="A.X-4.0-Light,EXAONE-3.5-7.8B-Instruct,llama3.1_korean_v1.1_sft_by_aidx,Midm-2.0-Base-Instruct", 
                   help="Stem 생성에 사용할 모델들 (CSV)")
    p.add_argument("--gpus", default="0", help="GPU 인덱스 CSV (예: 0 또는 0,1)")
    p.add_argument("--bench-ids", default="1,2,3,4,5", help="대상 벤치마크 ID CSV")
    p.add_argument("--benchmark-file", default=DEFAULT_BENCHMARK_FILE, help="벤치마크 JSON 파일명 또는 절대경로")
    p.add_argument("--benchmark-version", default="v1.1.0", help="벤치마크 버전 표기")
    p.add_argument("--date", default="2025-08-08", help="raw_outputs 날짜 디렉토리")
    p.add_argument("--template-key", default="stem_agent.few_shot_new", help="Stem 생성 템플릿")
    p.add_argument("--max-retries", type=int, default=3, help="최대 재시도 횟수")
    p.add_argument("--test", action="store_true", help="단일 벤치마크 테스트 모드")
    return p.parse_args()

def split_benchmark_root_and_name(arg: str) -> tuple[Path, str]:
    """
    절대경로면 (부모, 파일명)으로 쪼개고,
    상대경로면 data_store/benchmarks/v1 아래에 있다고 가정.
    """
    p = Path(arg)
    if p.is_absolute():
        return p.parent, p.name
    return Path("data_store/benchmarks/v1"), arg

def load_passages_for_benchmark(benchmark_id: int, passage_model: str, date_str: str, content_repo) -> List[Dict[str, Any]]:
    """특정 벤치마크의 passage 데이터 로드"""
    template_key = DEFAULT_TEMPLATE_BY_ID.get(benchmark_id)
    if not template_key:
        print(f"❌ 벤치마크 ID {benchmark_id}에 대한 템플릿을 찾을 수 없습니다.")
        return []
    
    passages = content_repo.load_passage_rows(
        model=passage_model,
        benchmark_id=benchmark_id,
        version="v1.1.0",
        template_key=template_key,
        date_str=date_str
    )
    
    if not passages:
        print(f"❌ 벤치마크 ID {benchmark_id} passage 데이터를 찾을 수 없습니다.")
        return []
    
    # 실제로 생성된 passage가 있는 항목만 필터링
    valid_passages = [p for p in passages if p.get('generated_passage')]
    print(f"✅ 벤치마크 ID {benchmark_id}: {len(valid_passages)}개의 유효한 passage 확인")
    return valid_passages

def generate_stems_for_model(
    model_name: str, 
    benchmark_id: int, 
    passages: List[Dict[str, Any]], 
    problem_types: List[str], 
    eval_goals: List[str],
    template_key: str,
    date_str: str,
    passage_model: str,
    gpus: List[int],
    max_retries: int
) -> bool:
    """특정 모델로 stem 생성"""
    print(f"\n🤖 모델 '{model_name}'로 벤치마크 ID {benchmark_id} stem 생성 중...")
    
    try:
        # LLM Gateway 초기화
        llm = LLMGatewayImpl(
            client_type="local",
            model_name=model_name,
            default_params={"temperature": 0.7},
            gpus=gpus,
        )
        
        # Repository 초기화 
        stem_repo = StemRepositoryImpl(llm=llm)
        
        # UseCase 초기화
        fill_stems_uc = FillMissingStemsUseCase(stem_repo)
        
        # Stem 생성 실행
        result = fill_stems_uc.execute(FillMissingStemsInput(
            model_name=model_name,
            template_key=template_key,
            benchmark_id=benchmark_id,
            benchmark_version="v1.1.0",
            problem_types=problem_types,
            eval_goals=eval_goals,
            passages=passages,
            date_str=date_str,
            max_retries=max_retries,
            passage_model_name=passage_model
        ))
        
        print(f"✅ 모델 '{model_name}' 벤치마크 ID {benchmark_id} stem 생성 완료")
        print(f"   📈 생성 성공: {len(result.filled_indices)}개")
        print(f"   ❌ 생성 실패: {len(result.failed_indices)}개")
        print(f"   📊 전체: {result.total_after}개")
        
        return result.success
        
    except Exception as e:
        print(f"❌ 모델 '{model_name}' 벤치마크 ID {benchmark_id} 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

# -------------------------------
# main
# -------------------------------
def main() -> None:
    args = parse_args()
    passage_model: str = args.passage_model
    stem_models: List[str] = [x.strip() for x in args.stem_models.split(",") if x.strip()]
    gpus: List[int] = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    bench_ids: List[int] = [int(x) for x in args.bench_ids.split(",") if x.strip() != ""]
    benchmarks_root, benchmark_filename = split_benchmark_root_and_name(args.benchmark_file)
    
    print("🚀 2025-08-08 EXAONE Passage 기반 Stem 생성 배치 시작")
    print(f"📅 날짜: {args.date}")
    print(f"📄 Passage 모델: {passage_model}")
    print(f"🤖 Stem 생성 모델들: {', '.join(stem_models)}")
    print(f"🔧 템플릿: {args.template_key}")
    
    # --- repositories wiring (manual DI) ---
    benchmark_repo = BenchmarkRepositoryImpl(
        benchmarks_root=benchmarks_root,
        benchmark_filename=benchmark_filename,
    )
    content_repo = ContentRepositoryImpl()
    
    # --- usecases wiring ---
    get_set_uc = GetBenchmarkSetByIdUseCase(benchmark_repo)
    
    # 전체 결과 집계
    total_success = 0
    total_attempts = 0
    
    # 각 벤치마크에 대해 처리
    for benchmark_id in bench_ids:
        print(f"\n{'='*60}")
        print(f"📝 벤치마크 ID {benchmark_id} 처리 시작")
        print(f"{'='*60}")
        
        try:
            # 벤치마크 정보 가져오기
            set_out = get_set_uc.execute(GetBenchmarkSetByIdInput(benchmark_id=benchmark_id))
            problem_types = set_out.benchmark_set.problem_types
            eval_goals = set_out.benchmark_set.eval_goals
            print(f"📋 문제 유형: {problem_types}")
            print(f"🎯 평가 목표: {eval_goals}")
            
            # Passage 데이터 로드
            passages = load_passages_for_benchmark(benchmark_id, passage_model, args.date, content_repo)
            if not passages:
                continue
                
            # 각 모델로 stem 생성
            for model_name in stem_models:
                total_attempts += 1
                success = generate_stems_for_model(
                    model_name=model_name,
                    benchmark_id=benchmark_id, 
                    passages=passages, 
                    problem_types=problem_types, 
                    eval_goals=eval_goals,
                    template_key=args.template_key,
                    date_str=args.date,
                    passage_model=passage_model,
                    gpus=gpus,
                    max_retries=args.max_retries
                )
                if success:
                    total_success += 1
                    
        except Exception as e:
            print(f"❌ 벤치마크 ID {benchmark_id} 전체 처리 중 오류: {e}")
            continue
    
    # 최종 결과 출력
    print(f"\n{'='*60}")
    print("🎉 배치 작업 완료!")
    print(f"✅ 성공: {total_success}/{total_attempts}")
    print(f"❌ 실패: {total_attempts - total_success}/{total_attempts}")
    success_rate = (total_success / total_attempts * 100) if total_attempts > 0 else 0
    print(f"📊 성공률: {success_rate:.1f}%")
    print(f"{'='*60}")

def test_single_benchmark() -> None:
    """단일 벤치마크 테스트 (디버깅용)"""
    print("🧪 단일 벤치마크 테스트 - 벤치마크 ID 2")
    
    # 테스트용 설정
    passage_model = "EXAONE-3.5-7.8B-Instruct"
    test_model = "EXAONE-3.5-7.8B-Instruct"
    template_key = "stem_agent.few_shot_new"
    date_str = "2025-08-08"
    benchmark_id = 2
    gpus = [0]
    
    benchmarks_root, benchmark_filename = split_benchmark_root_and_name(DEFAULT_BENCHMARK_FILE)
    
    # Repository 초기화
    benchmark_repo = BenchmarkRepositoryImpl(
        benchmarks_root=benchmarks_root,
        benchmark_filename=benchmark_filename,
    )
    content_repo = ContentRepositoryImpl()
    get_set_uc = GetBenchmarkSetByIdUseCase(benchmark_repo)
    
    # 벤치마크 정보 가져오기
    set_out = get_set_uc.execute(GetBenchmarkSetByIdInput(benchmark_id=benchmark_id))
    problem_types = set_out.benchmark_set.problem_types
    eval_goals = set_out.benchmark_set.eval_goals
    
    # Passage 데이터 로드
    passages = load_passages_for_benchmark(benchmark_id, passage_model, date_str, content_repo)
    if not passages:
        return
    
    # 테스트를 위해 처음 3개만 사용
    test_passages = passages[:3]
    print(f"🧪 테스트용 passage 수: {len(test_passages)}개")
    
    # Stem 생성
    success = generate_stems_for_model(
        model_name=test_model,
        benchmark_id=benchmark_id,
        passages=test_passages,
        problem_types=problem_types,
        eval_goals=eval_goals,
        template_key=template_key,
        date_str=date_str,
        passage_model=passage_model,
        gpus=gpus,
        max_retries=3
    )
    
    if success:
        print("✅ 단일 벤치마크 테스트 성공!")
    else:
        print("❌ 단일 벤치마크 테스트 실패")

if __name__ == "__main__":
    args = parse_args()
    
    if args.test:
        # 테스트 모드
        test_single_benchmark()
    else:
        # 전체 배치 실행
        main()

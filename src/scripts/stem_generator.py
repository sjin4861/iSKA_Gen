# src/scripts/stem_generator.py
"""
passage_repository처럼 클린 아키텍처로 stem 생성하는 예시 스크립트
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional

from src.domain.repositories.stem_repository import StemRepository
from src.data.repositories.stem_repository_impl import StemRepositoryImpl
from src.domain.usecases.stem.fill_missing_stems import FillMissingStemsUseCase, FillMissingStemsInput
from src.domain.usecases.stem.generate_single_stem import GenerateSingleStemUseCase, GenerateSingleStemInput

from src.data.repositories.benchmark_repository_impl import BenchmarkRepositoryImpl
from src.data.repositories.content_repository_impl import ContentRepositoryImpl

# 설정
MODEL = "EXAONE-3.5-7.8B-Instruct"  
PASSAGE_MODEL = "EXAONE-3.5-7.8B-Instruct"
TEMPLATE_KEY = "stem_agent.few_shot"
DATE = "2025-08-10"
BENCH_ID_LIST = [2]  # 실제 데이터가 있는 벤치마크 ID로 수정

def main():
    print("🚀 Stem 생성기 (Clean Architecture) 시작")
    
    # Repository 초기화
    stem_repo: StemRepository = StemRepositoryImpl(
        client_type="local",
        model_name=MODEL,
        gpus=[0],
        default_llm_params={"temperature": 0.7}
    )
    
    benchmark_repo = BenchmarkRepositoryImpl()
    content_repo = ContentRepositoryImpl()
    
    # UseCase 초기화  
    fill_stems_uc = FillMissingStemsUseCase(stem_repo)
    
    for bench_id in BENCH_ID_LIST:
        print(f"\n📝 벤치마크 ID {bench_id}에 대한 stem 생성 중...")
        
        try:
            # 벤치마크 정보 가져오기
            benchmark = benchmark_repo.get_benchmark_by_id(bench_id, "v1.1.0")
            if not benchmark:
                print(f"❌ 벤치마크 ID {bench_id}를 찾을 수 없습니다.")
                continue
                
            problem_types = benchmark.problem_types
            eval_goals = benchmark.eval_goals
            
            # passage 데이터 로드 (실제 존재하는 템플릿 키 사용)
            passages = content_repo.load_passage_list(
                model_name=PASSAGE_MODEL,
                benchmark_id=bench_id,
                benchmark_version="v1.1.0",
                template_key="passage_agent.create_domestic_passage",  # 실제 존재하는 템플릿
                date_str=DATE
            )
            
            if not passages:
                print(f"❌ 모델 '{PASSAGE_MODEL}'의 벤치마크 ID {bench_id} passage 데이터를 찾을 수 없습니다.")
                continue
                
            print(f"✅ {len(passages)}개의 passage 로드 완료")
            
            # 실제로 생성된 passage가 있는 항목만 필터링
            valid_passages = [p for p in passages if p.get('generated_passage')]
            if not valid_passages:
                print("❌ 생성된 passage가 없습니다.")
                continue
                
            print(f"✅ {len(valid_passages)}개의 유효한 passage 확인")
            
            # Stem 생성 실행
            result = fill_stems_uc.execute(FillMissingStemsInput(
                model_name=MODEL,
                template_key=TEMPLATE_KEY,
                benchmark_id=bench_id,
                benchmark_version="v1.1.0",
                problem_types=problem_types,
                eval_goals=eval_goals,
                passages=valid_passages[:3],  # 테스트를 위해 처음 3개만 사용
                date_str=DATE,
                max_retries=3,
                passage_model_name=PASSAGE_MODEL
            ))
            
            print(f"✅ 벤치마크 ID {bench_id}에 대한 stem 생성 완료")
            print(f"   📈 생성 성공: {len(result.filled_indices)}개")
            print(f"   ❌ 생성 실패: {len(result.failed_indices)}개") 
            print(f"   📊 전체: {result.total_after}개")
            
        except Exception as e:
            print(f"❌ 벤치마크 ID {bench_id} 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            continue

def test_single_stem():
    """단일 stem 생성 테스트"""
    print("\n🧪 단일 stem 생성 테스트")
    
    # Repository 초기화
    stem_repo: StemRepository = StemRepositoryImpl(
        client_type="local", 
        model_name=MODEL,
        gpus=[0]
    )
    
    # UseCase 초기화
    single_stem_uc = GenerateSingleStemUseCase(stem_repo)
    
    # 테스트 데이터
    sample_passage = """한국의 설날은 음력 1월 1일로, 가족들이 모여 차례를 지내고 떡국을 먹는 명절입니다. 
    반면, 서구권의 새해 첫날(New Year's Day)은 양력 1월 1일로, 주로 파티를 열거나 불꽃놀이를 보며 새해를 맞이하는 축제 분위기가 강습니다."""
    
    result = single_stem_uc.execute(GenerateSingleStemInput(
        passage=sample_passage,
        problem_type="자문화와 비교하기",
        eval_goal="문화적 차이 이해 및 표현 능력 평가",
        model_name=MODEL,
        template_key=TEMPLATE_KEY,
        max_retries=3
    ))
    
    if result.success:
        print("✅ 단일 stem 생성 성공!")
        print(f"생성된 stem: {result.stem}")
    else:
        print("❌ 단일 stem 생성 실패")

if __name__ == "__main__":
    # 단일 stem 테스트 먼저 실행
    test_single_stem()
    
    # 전체 stem 생성 실행
    # main()

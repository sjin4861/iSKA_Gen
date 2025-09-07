#!/usr/bin/env python3
"""
통합 콘텐츠 생성 스크립트

이 스크립트는 다음 기능들을 통합합니다:
1. Passage, Audio Script, Image Caption 생성 (generator.py)
2. Stem 생성 (stem_generator.py)
3. Content 생성 (필요시 확장 가능)

사용법:
    python unified_generator.py --mode passage --models "A.X-4.0-Light,EXAONE-3.5-7.8B-Instruct"
    python unified_generator.py --mode stem --models "Midm-2.0-Base-Instruct" --bench-ids "1,2"
    python unified_generator.py --mode all --models "A.X-4.0-Light" --date "2025-08-19"
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Repository implementations
from src.data.repositories.passage_repository_impl import PassageRepositoryImpl
from src.data.repositories.audio_repository_impl import AudioRepositoryImpl
from src.data.repositories.image_repository_impl import ImageRepositoryImpl
from src.data.repositories.benchmark_repository_impl import BenchmarkRepositoryImpl
from src.data.repositories.stem_repository_impl import StemRepositoryImpl

# Data sources
from src.data.datasources.fs.raw_output_fs import RawOutputFSDataSource
from src.data.datasources.fs.data_store_fs import DataStoreFSDataSource
from src.data.datasources.fs.templates_fs import TemplatesFSDataSource
from src.data.datasources.fs.text_generation import TextGenerationDataSource

# Model client
from src.modules.model_client import LocalModelClient, create_model_client

# Use cases and entities
from src.domain.usecases.benchmark.load_collection import LoadCollectionUseCase
from src.domain.entities.enums import ContentType
from src.domain.entities.output_query import OutputQuery
from datetime import datetime

# ================= 설정 및 템플릿 =================

DEFAULT_MODELS = [
    "A.X-4.0-Light", 
    "EXAONE-3.5-7.8B-Instruct", 
    "Midm-2.0-Base-Instruct", 
    "llama3.1_korean_v1.1_sft_by_aidx"
]

# 벤치마크별 기본 템플릿
DEFAULT_TEMPLATES = {
    1: "passage_agent.create_passage_rubric_aware",
    2: "passage_agent.create_domestic_passage", 
    3: "audio_agent.create_dialogue_passage",
    4: "audio_agent.create_dialogue_passage",
    5: "image_agent.create_image_caption_and_situation",
}

# Stem 생성용 템플릿
STEM_TEMPLATE = "stem_agent.few_shot_new"

# ================= 유틸리티 함수 =================

def _strip_braces(s: str | None) -> str:
    """이미지 주제에서 중괄호 제거"""
    s = (s or "").strip()
    return s[1:-1] if len(s) >= 2 and s.startswith("{") and s.endswith("}") else s

def parse_comma_separated(value: str) -> List[str]:
    """쉼표로 구분된 문자열을 리스트로 변환"""
    return [item.strip() for item in value.split(",") if item.strip()]

def parse_bench_ids(value: str) -> List[int]:
    """벤치마크 ID 문자열을 정수 리스트로 변환"""
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"벤치마크 ID는 숫자여야 합니다: {value}")

# ================= 콘텐츠 생성 함수들 =================

class UnifiedGenerator:
    """통합 콘텐츠 생성기"""
    
    def __init__(self, date: str = None, benchmark_version: str = "v1.1.0"):
        self.date = date or "2025-08-19"
        self.benchmark_version = benchmark_version
        
        # 벤치마크 로드
        benchmarks_root = Path("data_store/benchmarks/v1")
        benchmark_filename = "iSKA-Gen_Benchmark_v1.1.0_20250808_test.json"
        benchmark_repo = BenchmarkRepositoryImpl(benchmarks_root, benchmark_filename)
        load_collection_uc = LoadCollectionUseCase(benchmark_repo)
        collection_output = load_collection_uc.execute()
        self.benchmarks = collection_output.collection.benchmarks
        print(f"✅ 벤치마크 로드 완료: {len(self.benchmarks)}개 세트")
        
        # Data sources 초기화
        self.raw_output_ds = RawOutputFSDataSource(Path("data_store/raw_outputs"))
        self.data_store_ds = DataStoreFSDataSource(Path("data_store"))
        self.templates_ds = TemplatesFSDataSource(agent="iska", user_path=Path("src/config/prompts"))

    def generate_passages(self, models: List[str], bench_ids: List[int], 
                         templates: Dict[int, str] = None, gpus: List[int] = [1]):
        """Passage, Audio Script, Image Caption 생성"""
        templates = templates or DEFAULT_TEMPLATES
        
        for model in models:
            print(f"\n🚀 모델 시작: {model}")
            model_client = None
            
            try:
                model_client = LocalModelClient(model_name=model, gpus=gpus)
                textgen_ds = TextGenerationDataSource(model_client)
                
                # Repository 초기화
                passage_repo = PassageRepositoryImpl(
                    self.raw_output_ds, self.data_store_ds, self.templates_ds, textgen_ds
                )
                audio_repo = AudioRepositoryImpl(
                    self.raw_output_ds, self.data_store_ds, self.templates_ds, textgen_ds
                )
                image_repo = ImageRepositoryImpl(
                    self.raw_output_ds, self.data_store_ds, self.templates_ds, textgen_ds
                )
                
                for bench_id in bench_ids:
                    if bench_id > len(self.benchmarks):
                        print(f"⚠️ 벤치마크 ID {bench_id}가 범위를 벗어남 (총 {len(self.benchmarks)}개)")
                        continue
                    
                    template_key = templates.get(bench_id)
                    if not template_key:
                        print(f"⚠️ 벤치마크 ID {bench_id}에 대한 템플릿이 없습니다")
                        continue
                    
                    bench = self.benchmarks[bench_id - 1]
                    print(f"\n📝 벤치마크 {bench_id} 처리 중... (템플릿: {template_key})")
                    
                    try:
                        # 벤치마크 타입별 처리
                        if bench_id in (1, 2):
                            self._generate_passage_content(
                                bench_id, bench, template_key, model, passage_repo
                            )
                        elif bench_id in (3, 4):
                            self._generate_audio_content(
                                bench_id, bench, template_key, model, audio_repo
                            )
                        elif bench_id == 5:
                            self._generate_image_content(
                                bench_id, bench, template_key, model, image_repo
                            )
                        else:
                            print(f"⚠️ 미지원 벤치마크 ID: {bench_id} (스킵)")
                            
                    except Exception as e:
                        print(f"    ❌ 벤치마크 {bench_id} 처리 중 오류: {e}")
                        
            except Exception as e:
                print(f"❌ 모델 {model} 초기화 실패: {e}")
                
            finally:
                if model_client:
                    model_client.close()
                # 메모리 정리
                torch.cuda.empty_cache()
                gc.collect()

    def _generate_passage_content(self, bench_id: int, bench, template_key: str, 
                                 model: str, passage_repo):
        """Passage 콘텐츠 생성 (벤치마크 1, 2)"""
        if bench_id == 1:
            # 비교형: korean_* / foreign_* 사용
            sources = []
            for idx, item in enumerate(bench.items):
                sources.append({
                    "source_id": f"bench_{bench_id}_item_{idx}",
                    "korean_topic": getattr(item, "korean_topic", None) or getattr(item, "topic", "") or "",
                    "korean_context": getattr(item, "korean_context", None) or getattr(item, "context", "") or "",
                    "foreign_topic": getattr(item, "foreign_topic", None),
                    "foreign_context": getattr(item, "foreign_context", None),
                })
        else:
            # 단일형: topic/context 사용
            sources = []
            for idx, item in enumerate(bench.items):
                sources.append({
                    "source_id": f"bench_{bench_id}_item_{idx}",
                    "topic": getattr(item, "topic", None) or getattr(item, "korean_topic", "") or "",
                    "context": getattr(item, "context", None) or getattr(item, "korean_context", "") or "",
                })

        min_length, max_length = 200, 800
        print(f"  📖 Passage 생성 중... (템플릿: {template_key})")
        
        result = passage_repo.generate_and_fill_missing(
            model_name=model,
            template_key=template_key,
            benchmark_id=bench_id,
            benchmark_version=self.benchmark_version,
            problem_types=bench.problem_types,
            eval_goals=bench.eval_goals,
            sources=sources,
            date_str=self.date,
            min_length=min_length,
            max_length=max_length,
            max_retries=10,
        )
        print(f"    ✅ Passage: {len(result['filled'])} 생성, {len(result['failed'])} 실패")

    def _generate_audio_content(self, bench_id: int, bench, template_key: str, 
                               model: str, audio_repo):
        """Audio 콘텐츠 생성 (벤치마크 3, 4)"""
        sources = []
        for idx, item in enumerate(bench.items):
            sources.append({
                "source_id": f"bench_{bench_id}_item_{idx}",
                "topic": getattr(item, "topic", None) or getattr(item, "korean_topic", "") or "",
                "context": getattr(item, "context", None) or getattr(item, "korean_context", "") or "",
            })

        min_length, max_length = 300, 1000
        print(f"  🎵 Audio script 생성 중... (템플릿: {template_key})")
        
        result = audio_repo.generate_and_fill_missing(
            model_name=model,
            template_key=template_key,
            benchmark_id=bench_id,
            benchmark_version=self.benchmark_version,
            problem_types=bench.problem_types,
            eval_goals=bench.eval_goals,
            sources=sources,
            date_str=self.date,
            min_length=min_length,
            max_length=max_length,
            max_retries=10,
        )
        print(f"    ✅ Audio: {len(result['filled'])} 생성, {len(result['failed'])} 실패")

    def _generate_image_content(self, bench_id: int, bench, template_key: str, 
                               model: str, image_repo):
        """Image 콘텐츠 생성 (벤치마크 5)"""
        sources = []
        for idx, item in enumerate(bench.items):
            raw_topic = getattr(item, "topic", None) or getattr(item, "korean_topic", "") or ""
            sources.append({
                "source_id": f"bench_{bench_id}_item_{idx}",
                "topic": _strip_braces(raw_topic),
            })

        min_length, max_length = 200, 800
        print(f"  🖼️ Image caption 생성 중... (템플릿: {template_key})")
        
        result = image_repo.generate_and_fill_missing(
            model_name=model,
            template_key=template_key,
            benchmark_id=bench_id,
            benchmark_version=self.benchmark_version,
            problem_types=bench.problem_types,
            eval_goals=bench.eval_goals,
            sources=sources,
            date_str=self.date,
            min_length=min_length,
            max_length=max_length,
            max_retries=10,
        )
        print(f"    ✅ Image: {len(result['filled'])} 생성, {len(result['failed'])} 실패")

    def generate_stems(self, models: List[str], bench_ids: List[int], 
                      template_key: str = STEM_TEMPLATE, gpus: List[int] = [2]):
        """Stem 생성"""
        for model in models:
            print(f"\n🚀 Stem 생성 - 모델: {model}")
            model_client = None
            
            try:
                model_client = LocalModelClient(model_name=model, gpus=gpus)
                textgen_ds = TextGenerationDataSource(model_client)
                stem_repo = StemRepositoryImpl(
                    self.raw_output_ds, self.data_store_ds, self.templates_ds, textgen_ds
                )
                
                for bench_id in bench_ids:
                    if bench_id > len(self.benchmarks):
                        print(f"⚠️ 벤치마크 ID {bench_id}가 범위를 벗어남")
                        continue
                        
                    print(f"\n📝 벤치마크 {bench_id}에 대한 stem 생성 중...")
                    benchmark = self.benchmarks[bench_id - 1]
                    
                    # 기존 passage 데이터 조회
                    q = OutputQuery(
                        date_from=datetime.strptime(self.date, "%Y-%m-%d"),
                        date_to=datetime.strptime(self.date, "%Y-%m-%d"),
                        model_name=model,
                        benchmark_id=bench_id,
                        limit=None,
                    )
                    
                    candidates = list(self.raw_output_ds.find_candidates(ContentType.passage, q))
                    
                    result = stem_repo.generate_and_fill_missing(
                        model_name=model,
                        template_key=template_key,
                        benchmark_id=bench_id,
                        benchmark_version=self.benchmark_version,
                        problem_types=benchmark.problem_types,
                        eval_goals=benchmark.eval_goals,
                        contents=candidates,
                        date_str=self.date,
                        max_retries=3,
                        content_model_name=model,
                    )
                    
                    print(f"    ✅ Stem: {len(result.get('filled', []))} 생성, "
                          f"{len(result.get('failed', []))} 실패, "
                          f"전체: {result.get('total', 0)}개")
                    
            except Exception as e:
                print(f"❌ 모델 {model} Stem 생성 실패: {e}")
                
            finally:
                if model_client:
                    model_client.close()
                torch.cuda.empty_cache()
                gc.collect()

# ================= 명령줄 인터페이스 =================

def create_parser():
    """명령줄 인자 파서 생성"""
    parser = argparse.ArgumentParser(
        description="통합 콘텐츠 생성 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # Passage/Audio/Image 콘텐츠 생성 (기본 템플릿 사용)
  python unified_generator.py --mode passage --models "A.X-4.0-Light,EXAONE-3.5-7.8B-Instruct"
  
  # 특정 벤치마크만 생성
  python unified_generator.py --mode passage --models "Midm-2.0-Base-Instruct" --bench-ids "1,2"
  
  # Stem 생성
  python unified_generator.py --mode stem --models "A.X-4.0-Light" --bench-ids "1,2"
  
  # 모든 콘텐츠 생성
  python unified_generator.py --mode all --models "EXAONE-3.5-7.8B-Instruct" --date "2025-08-23"
  
  # GPU 설정
  python unified_generator.py --mode passage --models "A.X-4.0-Light" --gpus "0,1"
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["passage", "stem", "all"],
        required=True,
        help="생성 모드: passage(passage/audio/image), stem(stem), all(모든 콘텐츠)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        help="사용할 모델 리스트 (쉼표로 구분). 기본값: 모든 기본 모델"
    )
    
    parser.add_argument(
        "--bench-ids",
        type=str,
        help="처리할 벤치마크 ID (쉼표로 구분). 기본값: 모드에 따라 결정"
    )
    
    parser.add_argument(
        "--date",
        type=str,
        default="2025-08-19",
        help="생성 날짜 (YYYY-MM-DD 형식). 기본값: 2025-08-19"
    )
    
    parser.add_argument(
        "--benchmark-version",
        type=str,
        default="v1.1.0",
        help="벤치마크 버전. 기본값: v1.1.0"
    )
    
    parser.add_argument(
        "--gpus",
        type=str,
        default="1",
        help="사용할 GPU ID (쉼표로 구분). 기본값: 1"
    )
    
    parser.add_argument(
        "--stem-template",
        type=str,
        default=STEM_TEMPLATE,
        help=f"Stem 생성용 템플릿. 기본값: {STEM_TEMPLATE}"
    )
    
    return parser

def main():
    """메인 실행 함수"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 모델 리스트 파싱
    if args.models:
        models = parse_comma_separated(args.models)
    else:
        models = DEFAULT_MODELS
    
    # 벤치마크 ID 파싱
    if args.bench_ids:
        bench_ids = parse_bench_ids(args.bench_ids)
    else:
        # 모드에 따른 기본 벤치마크 ID
        if args.mode == "passage":
            bench_ids = [1, 2, 3, 4, 5]  # 모든 콘텐츠 타입
        elif args.mode == "stem":
            bench_ids = [1, 2]  # Stem은 주로 passage 기반
        else:  # all
            bench_ids = [1, 2, 3, 4, 5]
    
    # GPU 설정 파싱
    try:
        gpus = [int(gpu.strip()) for gpu in args.gpus.split(",")]
    except ValueError:
        print(f"❌ 오류: GPU ID는 숫자여야 합니다: {args.gpus}")
        return 1
    
    print(f"🚀 통합 콘텐츠 생성기 시작")
    print(f"   모드: {args.mode}")
    print(f"   모델: {', '.join(models)}")
    print(f"   벤치마크 ID: {bench_ids}")
    print(f"   날짜: {args.date}")
    print(f"   GPU: {gpus}")
    
    try:
        generator = UnifiedGenerator(
            date=args.date,
            benchmark_version=args.benchmark_version
        )
        
        if args.mode == "passage":
            generator.generate_passages(models, bench_ids, gpus=gpus)
            
        elif args.mode == "stem":
            generator.generate_stems(models, bench_ids, args.stem_template, gpus)
            
        elif args.mode == "all":
            # 먼저 passage 생성, 그 다음 stem 생성
            print("\n🔄 1단계: Passage/Audio/Image 콘텐츠 생성")
            generator.generate_passages(models, bench_ids, gpus=gpus)
            
            print("\n🔄 2단계: Stem 생성")
            # Stem은 passage 기반이므로 passage 관련 벤치마크만
            stem_bench_ids = [bid for bid in bench_ids if bid in [1, 2]]
            if stem_bench_ids:
                generator.generate_stems(models, stem_bench_ids, args.stem_template, gpus)
            else:
                print("⚠️ Stem 생성을 위한 passage 벤치마크가 없습니다")
        
        print(f"\n🎉 모든 생성 작업 완료!")
        return 0
        
    except Exception as e:
        print(f"\n❌ 스크립트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

#!/usr/bin/env p# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # iSKA_Gen 디렉토리
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'modules'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'utils'))
# -*- coding: utf-8 -*-
"""
PassageAgent와 모델 클라이언트를 통한 실제 지문 생성 테스트

벤치마크 데이터를 사용하여 실제 지문 생성 파이프라인을 테스트합니다.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'modules'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'utils'))

from modules.model_client import OpenAIModelClient, LocalModelClient, VLLMOpenAIClient, list_local_models
from modules.iska.passage_agent import PassageAgent
from utils.benchmark_loader import load_benchmarks

class PassageGenerationTester:
    """지문 생성 테스트 클래스"""
    
    def __init__(self):
        self.benchmark_file = "iSKA-Gen_Benchmark_v1.1.0_20250808_small.json"
        self.test_templates = {
            1: "passage_agent.create_passage_rubric_aware",  # 비교형
            2: "passage_agent.create_domestic_passage",      # 단일 주제형
            3: "passage_agent.create_dialogue_passage",      # 대화형
            4: "passage_agent.create_dialogue_passage",      # 대화형
            5: "passage_agent.create_image_caption_and_situation"  # 이미지 캡션
        }
    
    def load_test_data(self) -> Optional[List[Dict[str, Any]]]:
        """테스트용 벤치마크 데이터 로드"""
        try:
            benchmarks = load_benchmarks(self.benchmark_file)
            print(f"✅ 벤치마크 데이터 로드 성공: {len(benchmarks)}개 항목")
            return benchmarks
        except Exception as e:
            print(f"❌ 벤치마크 데이터 로드 실패: {e}")
            return None
    
    def test_passage_generation(
        self, 
        client_type: str, 
        model_name: str, 
        benchmark_id: int = 1,
        item_index: int = 0,
        **client_kwargs
    ) -> Dict[str, Any]:
        """지문 생성 테스트"""
        print(f"\n{'='*60}")
        print(f"📝 지문 생성 테스트")
        print(f"   클라이언트: {client_type}")
        print(f"   모델: {model_name}")
        print(f"   벤치마크 ID: {benchmark_id}")
        print(f"   아이템 인덱스: {item_index}")
        print(f"{'='*60}")
        
        result = {
            "client_type": client_type,
            "model_name": model_name,
            "benchmark_id": benchmark_id,
            "item_index": item_index,
            "success": False,
            "error": None,
            "generated_passage": None,
            "generation_time": None,
            "passage_length": 0
        }
        
        try:
            # 벤치마크 데이터 로드
            benchmarks = self.load_test_data()
            if not benchmarks:
                result["error"] = "벤치마크 데이터 로드 실패"
                return result
            
            if benchmark_id < 1 or benchmark_id > len(benchmarks):
                result["error"] = f"잘못된 벤치마크 ID: {benchmark_id}"
                return result
            
            benchmark = benchmarks[benchmark_id - 1]
            items = benchmark["items"]
            
            if item_index < 0 or item_index >= len(items):
                result["error"] = f"잘못된 아이템 인덱스: {item_index}"
                return result
            
            item = items[item_index]
            problem_types = benchmark["problem_types"]
            eval_goals = benchmark["eval_goals"]
            template_key = self.test_templates[benchmark_id]
            
            print(f"📋 테스트 데이터:")
            if benchmark_id == 1:
                print(f"   한국 주제: {item.get('korean_topic', 'N/A')}")
                print(f"   외국 주제: {item.get('foreign_topic', 'N/A')}")
            elif benchmark_id == 5:
                print(f"   주제: {item.get('topic', 'N/A')}")
            else:
                print(f"   주제: {item.get('topic', 'N/A')}")
            print(f"   템플릿: {template_key}")
            
            # 클라이언트 생성
            print("🔄 모델 클라이언트 생성 중...")
            if client_type.lower() == "openai":
                client = OpenAIModelClient(model_name=model_name, **client_kwargs)
            elif client_type.lower() == "local":
                client = LocalModelClient(model_name=model_name, **client_kwargs)
            elif client_type.lower() == "vllm":
                client = VLLMOpenAIClient(model_name=model_name, **client_kwargs)
            else:
                result["error"] = f"지원하지 않는 클라이언트 타입: {client_type}"
                return result
            
            # PassageAgent 생성
            print("🤖 PassageAgent 생성 중...")
            passage_agent = PassageAgent(llm_client=client)
            
            # 지문 생성
            print("📝 지문 생성 중...")
            start_time = time.time()
            
            if benchmark_id == 5:
                # 이미지 캡션 생성
                topic = item.get('topic', '')
                generated_passage = passage_agent.generate_image_caption_and_situation(topic)
            else:
                # 일반 지문 생성
                if benchmark_id == 1:
                    # 비교형
                    korean_topic = item.get('korean_topic', '')
                    korean_context = item.get('korean_context', '')
                    foreign_topic = item.get('foreign_topic', '')
                    foreign_context = item.get('foreign_context', '')
                else:
                    # 단일 주제형
                    korean_topic = item.get('topic', '')
                    korean_context = item.get('context', '')
                    foreign_topic = ""
                    foreign_context = ""
                
                generated_passage = passage_agent.generate_passage(
                    korean_topic=korean_topic,
                    korean_context=korean_context,
                    foreign_topic=foreign_topic,
                    foreign_context=foreign_context,
                    problem_types=problem_types,
                    eval_goals=eval_goals,
                    template_key=template_key
                )
            
            end_time = time.time()
            
            result["generated_passage"] = generated_passage
            result["generation_time"] = round(end_time - start_time, 2)
            result["passage_length"] = len(generated_passage) if generated_passage else 0
            result["success"] = bool(generated_passage and len(generated_passage.strip()) > 0)
            
            if result["success"]:
                print(f"✅ 지문 생성 성공!")
                print(f"   생성 시간: {result['generation_time']}초")
                print(f"   지문 길이: {result['passage_length']}자")
                print(f"   지문 미리보기: {generated_passage[:150]}{'...' if len(generated_passage) > 150 else ''}")
            else:
                print(f"❌ 지문 생성 실패: 빈 결과")
                result["error"] = "빈 결과 반환"
                
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ 오류 발생: {e}")
            traceback.print_exc()
            
        return result
    
    def test_multiple_benchmarks(
        self, 
        client_type: str, 
        model_name: str,
        benchmark_ids: List[int] = [1, 2, 3, 5],
        **client_kwargs
    ) -> List[Dict[str, Any]]:
        """여러 벤치마크에 대한 지문 생성 테스트"""
        print(f"\n{'='*80}")
        print(f"🔄 다중 벤치마크 지문 생성 테스트")
        print(f"   클라이언트: {client_type}")
        print(f"   모델: {model_name}")
        print(f"   벤치마크 IDs: {benchmark_ids}")
        print(f"{'='*80}")
        
        results = []
        
        for benchmark_id in benchmark_ids:
            print(f"\n📋 벤치마크 ID {benchmark_id} 테스트 중...")
            result = self.test_passage_generation(
                client_type=client_type,
                model_name=model_name,
                benchmark_id=benchmark_id,
                item_index=0,
                **client_kwargs
            )
            results.append(result)
            
            # 메모리 정리를 위한 잠시 대기
            time.sleep(1)
        
        # 결과 요약
        self.print_generation_summary(results)
        
        return results
    
    def print_generation_summary(self, results: List[Dict[str, Any]]):
        """지문 생성 결과 요약 출력"""
        print(f"\n{'='*80}")
        print("📊 지문 생성 테스트 결과 요약")
        print(f"{'='*80}")
        
        success_count = 0
        total_time = 0
        total_length = 0
        
        for result in results:
            benchmark_id = result.get("benchmark_id", "?")
            success = result.get("success", False)
            generation_time = result.get("generation_time", 0)
            passage_length = result.get("passage_length", 0)
            error = result.get("error")
            
            if success:
                success_count += 1
                status = "✅ 성공"
                detail = f"{generation_time}초, {passage_length}자"
                total_time += generation_time
                total_length += passage_length
            else:
                status = "❌ 실패"
                detail = f"오류: {error}" if error else "알 수 없는 오류"
            
            print(f"• 벤치마크 {benchmark_id:2} | {status} | {detail}")
        
        total_count = len(results)
        if total_count > 0:
            success_rate = (success_count / total_count) * 100
            avg_time = total_time / success_count if success_count > 0 else 0
            avg_length = total_length / success_count if success_count > 0 else 0
            
            print(f"\n🎯 전체 성공률: {success_count}/{total_count} ({success_rate:.1f}%)")
            if success_count > 0:
                print(f"⏱️ 평균 생성 시간: {avg_time:.2f}초")
                print(f"📏 평균 지문 길이: {avg_length:.0f}자")
        
        print(f"{'='*80}")
    
    def run_comprehensive_test(self):
        """포괄적인 지문 생성 테스트"""
        print("🚀 포괄적인 지문 생성 테스트 시작")
        print("="*90)
        
        all_results = []
        
        # 1. OpenAI 테스트 (API 키가 있는 경우)
        if os.getenv("OPENAI_API_KEY"):
            print("\n🔧 OpenAI 모델 테스트...")
            openai_results = self.test_multiple_benchmarks(
                client_type="openai",
                model_name="gpt-4o-mini",
                benchmark_ids=[1, 2, 3, 5]
            )
            all_results.extend(openai_results)
        else:
            print("\n⚠️ OPENAI_API_KEY가 설정되지 않아 OpenAI 테스트를 건너뜁니다.")
        
        # 2. 로컬 모델 테스트 (사용 가능한 모델이 있는 경우)
        local_models = list_local_models()
        if local_models:
            print(f"\n🔧 로컬 모델 테스트... (모델: {local_models[0]})")
            local_results = self.test_multiple_benchmarks(
                client_type="local",
                model_name=local_models[0],
                benchmark_ids=[1, 2],  # 로컬 모델은 시간이 오래 걸리므로 일부만
                gpus=[0]
            )
            all_results.extend(local_results)
        else:
            print("\n⚠️ 사용 가능한 로컬 모델이 없어 로컬 모델 테스트를 건너뜁니다.")
        
        print(f"\n{'='*90}")
        print("🎯 전체 테스트 완료!")
        print(f"📊 총 {len(all_results)}개의 지문 생성 테스트가 수행되었습니다.")
        print(f"{'='*90}")
        
        return all_results

def main():
    """메인 실행 함수"""
    tester = PassageGenerationTester()
    
    # 인자 파싱
    import argparse
    parser = argparse.ArgumentParser(description="지문 생성 테스트")
    parser.add_argument("--client", choices=["openai", "local", "vllm", "all"], 
                       default="all", help="테스트할 클라이언트 타입")
    parser.add_argument("--model", type=str, help="테스트할 모델명")
    parser.add_argument("--benchmark", type=int, choices=[1,2,3,4,5], 
                       help="테스트할 벤치마크 ID")
    parser.add_argument("--item", type=int, default=0, help="테스트할 아이템 인덱스")
    parser.add_argument("--gpus", nargs="+", type=int, help="사용할 GPU 인덱스 (로컬 모델용)")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", 
                       help="vLLM 서버 URL")
    
    args = parser.parse_args()
    
    if args.client == "all":
        # 포괄적인 테스트 실행
        tester.run_comprehensive_test()
    else:
        # 개별 테스트 실행
        client_kwargs = {}
        if args.client == "local" and args.gpus:
            client_kwargs["gpus"] = args.gpus
        elif args.client == "vllm":
            client_kwargs["base_url"] = args.url
        
        if args.model:
            model_name = args.model
        elif args.client == "openai":
            model_name = "gpt-4o-mini"
        elif args.client == "local":
            local_models = list_local_models()
            if not local_models:
                print("❌ 사용 가능한 로컬 모델이 없습니다.")
                return
            model_name = local_models[0]
        else:
            model_name = "test-model"
        
        if args.benchmark:
            # 단일 벤치마크 테스트
            tester.test_passage_generation(
                client_type=args.client,
                model_name=model_name,
                benchmark_id=args.benchmark,
                item_index=args.item,
                **client_kwargs
            )
        else:
            # 다중 벤치마크 테스트
            tester.test_multiple_benchmarks(
                client_type=args.client,
                model_name=model_name,
                **client_kwargs
            )

if __name__ == "__main__":
    main()

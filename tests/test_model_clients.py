#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모델 클라이언트들을 테스트하는 스크립트

각 클라이언트 타입(OpenAI, Local, vLLM)별로 기본적인 호출 테스트를 수행합니다.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # iSKA_Gen 디렉토리
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'modules'))

from modules.model_client import OpenAIModelClient, LocalModelClient, VLLMOpenAIClient, list_local_models
from modules.client_factory import ModelClientFactory

class ModelClientTester:
    """모델 클라이언트 테스트 클래스"""
    
    def __init__(self):
        self.test_messages = [
            {"role": "user", "content": "안녕하세요! 간단한 인사말로 답변해주세요."}
        ]
        self.test_messages_long = [
            {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
            {"role": "user", "content": "한국의 전통 음식 중 하나인 김치에 대해 3-4문장으로 설명해주세요."}
        ]
        
    def test_openai_client(self, model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
        """OpenAI 클라이언트 테스트"""
        print(f"\n{'='*60}")
        print(f"🔧 OpenAI 클라이언트 테스트 ({model_name})")
        print(f"{'='*60}")
        
        result = {
            "client_type": "openai",
            "model_name": model_name,
            "success": False,
            "error": None,
            "response": None,
            "response_time": None
        }
        
        try:
            # API 키 확인
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                result["error"] = "OPENAI_API_KEY 환경변수가 설정되지 않았습니다."
                print(f"❌ {result['error']}")
                return result
            
            print(f"✅ API 키 확인됨: {api_key[:10]}...{api_key[-4:]}")
            
            # 클라이언트 생성
            print("🔄 OpenAI 클라이언트 생성 중...")
            client = OpenAIModelClient(model_name=model_name)
            
            # 테스트 호출
            print("🤖 테스트 호출 중...")
            start_time = time.time()
            response = client.call(self.test_messages)
            end_time = time.time()
            
            result["response"] = response
            result["response_time"] = round(end_time - start_time, 2)
            result["success"] = bool(response and not response.startswith("❌"))
            
            if result["success"]:
                print(f"✅ 응답 성공 ({result['response_time']}초)")
                print(f"📝 응답: {response[:100]}{'...' if len(response) > 100 else ''}")
            else:
                print(f"❌ 응답 실패: {response}")
                result["error"] = response
                
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ 오류 발생: {e}")
            traceback.print_exc()
            
        return result
    
    def test_local_client(self, model_name: str, gpus: Optional[List[int]] = None) -> Dict[str, Any]:
        """로컬 모델 클라이언트 테스트"""
        print(f"\n{'='*60}")
        print(f"🔧 로컬 모델 클라이언트 테스트 ({model_name})")
        if gpus:
            print(f"🎮 사용 GPU: {gpus}")
        print(f"{'='*60}")
        
        result = {
            "client_type": "local",
            "model_name": model_name,
            "gpus": gpus,
            "success": False,
            "error": None,
            "response": None,
            "response_time": None
        }
        
        try:
            # 로컬 모델 디렉토리 확인
            models_dir = os.getenv('LOCAL_MODELS_PATH', os.path.expanduser('~/models'))
            model_path = os.path.join(models_dir, model_name)
            
            if not os.path.exists(model_path):
                result["error"] = f"모델 경로가 존재하지 않습니다: {model_path}"
                print(f"❌ {result['error']}")
                print(f"🔍 사용 가능한 모델들: {list_local_models()}")
                return result
                
            print(f"✅ 모델 경로 확인됨: {model_path}")
            
            # 클라이언트 생성
            print("🔄 로컬 모델 클라이언트 생성 중...")
            client_kwargs = {}
            if gpus is not None:
                client_kwargs['gpus'] = gpus
                
            client = LocalModelClient(model_name=model_name, **client_kwargs)
            
            # 테스트 호출
            print("🤖 테스트 호출 중...")
            start_time = time.time()
            response = client.call(self.test_messages)
            end_time = time.time()
            
            result["response"] = response
            result["response_time"] = round(end_time - start_time, 2)
            result["success"] = bool(response and len(response.strip()) > 0)
            
            if result["success"]:
                print(f"✅ 응답 성공 ({result['response_time']}초)")
                print(f"📝 응답: {response[:100]}{'...' if len(response) > 100 else ''}")
            else:
                print(f"❌ 응답 실패: 빈 응답 또는 오류")
                result["error"] = "빈 응답"
                
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ 오류 발생: {e}")
            traceback.print_exc()
            
        return result
    
    def test_vllm_client(self, model_name: str, base_url: str = "http://localhost:8000/v1") -> Dict[str, Any]:
        """vLLM 클라이언트 테스트"""
        print(f"\n{'='*60}")
        print(f"🔧 vLLM 클라이언트 테스트 ({model_name})")
        print(f"🌐 서버 URL: {base_url}")
        print(f"{'='*60}")
        
        result = {
            "client_type": "vllm",
            "model_name": model_name,
            "base_url": base_url,
            "success": False,
            "error": None,
            "response": None,
            "response_time": None
        }
        
        try:
            # 클라이언트 생성
            print("🔄 vLLM 클라이언트 생성 중...")
            client = VLLMOpenAIClient(model_name=model_name, base_url=base_url)
            
            # 테스트 호출
            print("🤖 테스트 호출 중...")
            start_time = time.time()
            response = client.call(self.test_messages)
            end_time = time.time()
            
            result["response"] = response
            result["response_time"] = round(end_time - start_time, 2)
            result["success"] = bool(response and len(response.strip()) > 0)
            
            if result["success"]:
                print(f"✅ 응답 성공 ({result['response_time']}초)")
                print(f"📝 응답: {response[:100]}{'...' if len(response) > 100 else ''}")
            else:
                print(f"❌ 응답 실패: 빈 응답 또는 서버 오류")
                result["error"] = "빈 응답 또는 서버 연결 실패"
                
        except Exception as e:
            result["error"] = str(e)
            print(f"❌ 오류 발생: {e}")
            traceback.print_exc()
            
        return result
    
    def test_factory_creation(self) -> Dict[str, Any]:
        """팩토리를 통한 클라이언트 생성 테스트"""
        print(f"\n{'='*60}")
        print("🏭 ModelClientFactory 테스트")
        print(f"{'='*60}")
        
        results = []
        
        # 사용 가능한 클라이언트 타입 확인
        print("📋 사용 가능한 클라이언트 타입:")
        available_types = ModelClientFactory.get_available_client_types()
        for client_type, description in available_types.items():
            print(f"  - {client_type}: {description}")
        
        # 각 타입별 기본 모델 확인
        print("\n🎯 각 타입별 기본 모델:")
        for client_type in available_types.keys():
            default_model = ModelClientFactory.get_default_model_for_client(client_type)
            print(f"  - {client_type}: {default_model}")
            
            # 설정 검증
            validation = ModelClientFactory.validate_client_config(client_type, default_model)
            if validation["valid"]:
                print(f"    ✅ 설정 유효")
            else:
                print(f"    ❌ 설정 오류: {validation['errors']}")
            if validation["warnings"]:
                print(f"    ⚠️ 경고: {validation['warnings']}")
        
        return {"factory_test": "completed"}
    
    def run_comprehensive_test(self):
        """포괄적인 테스트 실행"""
        print("🚀 모델 클라이언트 종합 테스트 시작")
        print("="*80)
        
        all_results = []
        
        # 1. 팩토리 테스트
        factory_result = self.test_factory_creation()
        all_results.append(factory_result)
        
        # 2. OpenAI 테스트 (API 키가 있는 경우)
        if os.getenv("OPENAI_API_KEY"):
            openai_result = self.test_openai_client("gpt-4o-mini")
            all_results.append(openai_result)
        else:
            print("\n⚠️ OPENAI_API_KEY가 설정되지 않아 OpenAI 테스트를 건너뜁니다.")
        
        # 3. 로컬 모델 테스트 (사용 가능한 모델이 있는 경우)
        local_models = list_local_models()
        if local_models:
            print(f"\n🔍 발견된 로컬 모델: {local_models}")
            # 첫 번째 모델로 테스트
            first_model = local_models[0]
            local_result = self.test_local_client(first_model, gpus=[0])
            all_results.append(local_result)
        else:
            print("\n⚠️ 사용 가능한 로컬 모델이 없어 로컬 모델 테스트를 건너뜁니다.")
            print(f"📁 모델 디렉토리: {os.getenv('LOCAL_MODELS_PATH', '~/models')}")
        
        # 4. vLLM 테스트 (선택적)
        vllm_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        print(f"\n🌐 vLLM 서버 테스트 시도: {vllm_url}")
        vllm_result = self.test_vllm_client("test-model", vllm_url)
        all_results.append(vllm_result)
        
        # 결과 요약
        self.print_test_summary(all_results)
        
        return all_results
    
    def print_test_summary(self, results: List[Dict[str, Any]]):
        """테스트 결과 요약 출력"""
        print(f"\n{'='*80}")
        print("📊 테스트 결과 요약")
        print(f"{'='*80}")
        
        success_count = 0
        total_count = 0
        
        for result in results:
            if "factory_test" in result:
                continue
                
            total_count += 1
            client_type = result.get("client_type", "unknown")
            model_name = result.get("model_name", "unknown")
            success = result.get("success", False)
            response_time = result.get("response_time")
            error = result.get("error")
            
            if success:
                success_count += 1
                status = "✅ 성공"
                detail = f"응답시간: {response_time}초" if response_time else ""
            else:
                status = "❌ 실패"
                detail = f"오류: {error}" if error else ""
            
            print(f"• {client_type.upper():8} | {model_name:20} | {status} | {detail}")
        
        if total_count > 0:
            success_rate = (success_count / total_count) * 100
            print(f"\n🎯 전체 성공률: {success_count}/{total_count} ({success_rate:.1f}%)")
        
        print(f"{'='*80}")

def main():
    """메인 실행 함수"""
    tester = ModelClientTester()
    
    # 인자 파싱 (간단한 버전)
    import argparse
    parser = argparse.ArgumentParser(description="모델 클라이언트 테스트")
    parser.add_argument("--client", choices=["openai", "local", "vllm", "all"], 
                       default="all", help="테스트할 클라이언트 타입")
    parser.add_argument("--model", type=str, help="테스트할 모델명")
    parser.add_argument("--gpus", nargs="+", type=int, help="사용할 GPU 인덱스 (로컬 모델용)")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", 
                       help="vLLM 서버 URL")
    
    args = parser.parse_args()
    
    if args.client == "all":
        # 종합 테스트 실행
        tester.run_comprehensive_test()
    elif args.client == "openai":
        model_name = args.model or "gpt-4o-mini"
        tester.test_openai_client(model_name)
    elif args.client == "local":
        if not args.model:
            local_models = list_local_models()
            if not local_models:
                print("❌ 사용 가능한 로컬 모델이 없습니다.")
                return
            model_name = local_models[0]
            print(f"🎯 첫 번째 사용 가능한 모델 사용: {model_name}")
        else:
            model_name = args.model
        tester.test_local_client(model_name, args.gpus)
    elif args.client == "vllm":
        model_name = args.model or "test-model"
        tester.test_vllm_client(model_name, args.url)

if __name__ == "__main__":
    main()

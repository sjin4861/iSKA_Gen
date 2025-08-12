#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
테스트 실행 스크립트

모든 테스트를 간편하게 실행할 수 있는 통합 스크립트입니다.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # iSKA_Gen 디렉토리
sys.path.append(str(PROJECT_ROOT))

def run_command(command: List[str], description: str) -> Dict[str, Any]:
    """명령어 실행 및 결과 반환"""
    print(f"\n🔄 {description}")
    print(f"   명령어: {' '.join(command)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            cwd=PROJECT_ROOT,
            timeout=300  # 5분 타임아웃
        )
        
        if result.returncode == 0:
            print("✅ 성공!")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ 실패!")
            if result.stderr:
                print("오류:", result.stderr)
            if result.stdout:
                print("출력:", result.stdout)
        
        return {
            "command": " ".join(command),
            "description": description,
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print("⏰ 타임아웃!")
        return {
            "command": " ".join(command),
            "description": description,
            "success": False,
            "error": "타임아웃",
            "returncode": -1
        }
    except Exception as e:
        print(f"❌ 오류: {e}")
        return {
            "command": " ".join(command),
            "description": description,
            "success": False,
            "error": str(e),
            "returncode": -1
        }

def main():
    """메인 실행 함수"""
    print("🧪 iSKA-Gen 모델 클라이언트 테스트 실행기")
    print("=" * 80)
    
    # Python 실행 명령어 결정
    python_cmd = "python3" if subprocess.run(["which", "python3"], capture_output=True).returncode == 0 else "python"
    
    test_results = []
    
    # 1. 기본 모델 클라이언트 테스트
    test_results.append(run_command(
        [python_cmd, "tests/test_model_clients.py", "--client", "all"],
        "기본 모델 클라이언트 연결 테스트"
    ))
    
    # 2. OpenAI 클라이언트 개별 테스트 (API 키가 있는 경우)
    if os.getenv("OPENAI_API_KEY"):
        test_results.append(run_command(
            [python_cmd, "tests/test_model_clients.py", "--client", "openai", "--model", "gpt-4o-mini"],
            "OpenAI 클라이언트 상세 테스트"
        ))
        
        # 3. 지문 생성 테스트 (OpenAI)
        test_results.append(run_command(
            [python_cmd, "tests/test_passage_generation.py", "--client", "openai", "--benchmark", "1"],
            "OpenAI 모델을 사용한 지문 생성 테스트"
        ))
    else:
        print("\n⚠️ OPENAI_API_KEY가 설정되지 않아 OpenAI 관련 테스트를 건너뜁니다.")
    
    # 4. 로컬 모델 테스트 (모델이 있는 경우)
    local_models_check = subprocess.run(
        [python_cmd, "-c", "from src.modules.model_client import list_local_models; print(len(list_local_models()))"],
        capture_output=True, text=True, cwd=PROJECT_ROOT
    )
    
    if local_models_check.returncode == 0 and local_models_check.stdout.strip() != "0":
        test_results.append(run_command(
            [python_cmd, "tests/test_model_clients.py", "--client", "local"],
            "로컬 모델 클라이언트 테스트"
        ))
        
        # 로컬 모델 지문 생성 테스트 (간단한 벤치마크만)
        test_results.append(run_command(
            [python_cmd, "tests/test_passage_generation.py", "--client", "local", "--benchmark", "2"],
            "로컬 모델을 사용한 지문 생성 테스트"
        ))
    else:
        print("\n⚠️ 사용 가능한 로컬 모델이 없어 로컬 모델 테스트를 건너뜁니다.")
    
    # 5. vLLM 서버 테스트 (서버가 실행 중인 경우)
    test_results.append(run_command(
        [python_cmd, "tests/test_model_clients.py", "--client", "vllm"],
        "vLLM 서버 연결 테스트"
    ))
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 테스트 결과 요약")
    print("=" * 80)
    
    success_count = 0
    total_count = len(test_results)
    
    for i, result in enumerate(test_results, 1):
        success = result.get("success", False)
        description = result.get("description", "Unknown")
        
        if success:
            success_count += 1
            status = "✅ 성공"
        else:
            status = "❌ 실패"
            error = result.get("error") or result.get("stderr", "")
            if error:
                status += f" ({error[:50]}...)" if len(error) > 50 else f" ({error})"
        
        print(f"{i:2}. {description:<40} | {status}")
    
    if total_count > 0:
        success_rate = (success_count / total_count) * 100
        print(f"\n🎯 전체 성공률: {success_count}/{total_count} ({success_rate:.1f}%)")
    
    print("=" * 80)
    
    # 환경 정보 출력
    print("\n🔧 환경 정보:")
    print(f"   Python: {python_cmd}")
    print(f"   작업 디렉토리: {PROJECT_ROOT}")
    print(f"   OPENAI_API_KEY: {'설정됨' if os.getenv('OPENAI_API_KEY') else '미설정'}")
    print(f"   LOCAL_MODELS_PATH: {os.getenv('LOCAL_MODELS_PATH', '미설정 (기본값 ~/models 사용)')}")
    
    return test_results

if __name__ == "__main__":
    main()

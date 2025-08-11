#!/usr/bin/env python3
"""
🚀 vLLM Server Manager - 멋진 vLLM 서버 관리 유틸리티

이 모듈은 vLLM 서버를 쉽게 시작하고 관리할 수 있는 기능을 제공합니다.
"""

import subprocess
import time
import requests
import psutil
import signal
import os
from pathlib import Path
from typing import Optional, Dict, List
import json


class VLLMServerManager:
    """
    vLLM 서버를 관리하는 멋진 클래스
    
    🎯 Features:
    - 서버 자동 시작/중지
    - 상태 모니터링
    - 프로세스 관리
    - 설정 관리
    """
    
    def __init__(
        self,
        model_name: str = "gpt-oss-20b",
        host: str = "0.0.0.0",
        port: int = 8000,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 4096,
        tensor_parallel_size: int = 1
    ):
        """
        vLLM 서버 매니저 초기화
        
        Args:
            model_name: 로드할 모델명
            host: 서버 호스트
            port: 서버 포트
            gpu_memory_utilization: GPU 메모리 사용률
            max_model_len: 최대 모델 길이
            tensor_parallel_size: 텐서 병렬 크기
        """
        self.model_name = model_name
        self.host = host
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        
        self.base_url = f"http://{host}:{port}"
        self.health_url = f"{self.base_url}/health"
        self.models_url = f"{self.base_url}/v1/models"
        
        self.process: Optional[subprocess.Popen] = None
        
        print(f"🚀 vLLM Server Manager initialized")
        print(f"   🤖 Model: {model_name}")
        print(f"   🌐 URL: {self.base_url}")
    
    def get_start_command(self) -> List[str]:
        """
        vLLM 서버 시작 명령어 생성
        
        Returns:
            명령어 리스트
        """
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--disable-log-requests"  # 로그 요청 비활성화로 성능 향상
        ]
        return cmd
    
    def is_server_running(self) -> bool:
        """
        서버가 실행 중인지 확인
        
        Returns:
            bool: 서버 실행 상태
        """
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_server(self, timeout: int = 300) -> bool:
        """
        서버가 준비될 때까지 대기
        
        Args:
            timeout: 최대 대기 시간 (초)
            
        Returns:
            bool: 서버 준비 완료 여부
        """
        print(f"⏳ Waiting for server to be ready (timeout: {timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                print(f"✅ Server is ready!")
                return True
            
            elapsed = int(time.time() - start_time)
            print(f"   🔄 Still waiting... ({elapsed}s elapsed)")
            time.sleep(10)
        
        print(f"❌ Server failed to start within {timeout}s")
        return False
    
    def start_server(self, wait: bool = True, timeout: int = 300) -> bool:
        """
        vLLM 서버 시작
        
        Args:
            wait: 서버 준비까지 대기할지 여부
            timeout: 최대 대기 시간
            
        Returns:
            bool: 시작 성공 여부
        """
        if self.is_server_running():
            print(f"ℹ️  Server is already running at {self.base_url}")
            return True
        
        print(f"🚀 Starting vLLM server...")
        print(f"   🤖 Model: {self.model_name}")
        print(f"   🌐 Address: {self.base_url}")
        
        cmd = self.get_start_command()
        print(f"   💻 Command: {' '.join(cmd)}")
        
        try:
            # 환경 변수 설정
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'  # 필요에 따라 조정
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True
            )
            
            print(f"   🔢 Process ID: {self.process.pid}")
            
            if wait:
                if self.wait_for_server(timeout):
                    print(f"🎉 vLLM server started successfully!")
                    return True
                else:
                    self.stop_server()
                    return False
            else:
                print(f"🚀 Server starting in background...")
                return True
                
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """
        vLLM 서버 중지
        
        Returns:
            bool: 중지 성공 여부
        """
        print(f"🛑 Stopping vLLM server...")
        
        stopped = False
        
        # 프로세스가 있으면 종료
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
                print(f"   ✅ Process terminated gracefully")
                stopped = True
            except subprocess.TimeoutExpired:
                print(f"   ⚠️  Process didn't terminate, forcing kill...")
                self.process.kill()
                self.process.wait()
                print(f"   💀 Process killed")
                stopped = True
            except Exception as e:
                print(f"   ❌ Error stopping process: {e}")
            
            self.process = None
        
        # 포트를 사용하는 다른 프로세스 찾아서 종료
        killed_pids = self.kill_processes_on_port(self.port)
        if killed_pids:
            print(f"   🔥 Killed additional processes: {killed_pids}")
            stopped = True
        
        if stopped:
            print(f"✅ Server stopped successfully")
        else:
            print(f"⚠️  No running server found")
        
        return stopped
    
    def kill_processes_on_port(self, port: int) -> List[int]:
        """
        특정 포트를 사용하는 프로세스들을 종료
        
        Args:
            port: 포트 번호
            
        Returns:
            종료된 프로세스 ID 리스트
        """
        killed_pids = []
        
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.info['connections'] or []:
                    if conn.laddr.port == port:
                        print(f"   🎯 Found process {proc.info['pid']} ({proc.info['name']}) using port {port}")
                        proc.terminate()
                        killed_pids.append(proc.info['pid'])
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        return killed_pids
    
    def get_server_info(self) -> Optional[Dict]:
        """
        서버 정보 조회
        
        Returns:
            서버 정보 딕셔너리
        """
        if not self.is_server_running():
            return None
        
        try:
            response = requests.get(self.models_url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"❌ Error getting server info: {e}")
        
        return None
    
    def restart_server(self, timeout: int = 300) -> bool:
        """
        서버 재시작
        
        Args:
            timeout: 최대 대기 시간
            
        Returns:
            bool: 재시작 성공 여부
        """
        print(f"🔄 Restarting vLLM server...")
        self.stop_server()
        time.sleep(5)  # 잠시 대기
        return self.start_server(wait=True, timeout=timeout)
    
    def __enter__(self):
        """Context manager 진입"""
        self.start_server()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.stop_server()


def quick_start_gpt_oss_20b(port: int = 8000) -> VLLMServerManager:
    """
    gpt-oss-20b 모델을 위한 빠른 시작 함수
    
    Args:
        port: 서버 포트
        
    Returns:
        VLLMServerManager 인스턴스
    """
    print(f"🚀 Quick starting gpt-oss-20b on port {port}...")
    
    manager = VLLMServerManager(
        model_name="gpt-oss-20b",
        port=port,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        tensor_parallel_size=1
    )
    
    manager.start_server()
    return manager


if __name__ == "__main__":
    """
    직접 실행 시 데모
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Server Manager")
    parser.add_argument("action", choices=["start", "stop", "restart", "status"], help="Action to perform")
    parser.add_argument("--model", default="gpt-oss-20b", help="Model name")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    manager = VLLMServerManager(model_name=args.model, port=args.port)
    
    if args.action == "start":
        manager.start_server()
    elif args.action == "stop":
        manager.stop_server()
    elif args.action == "restart":
        manager.restart_server()
    elif args.action == "status":
        if manager.is_server_running():
            print(f"✅ Server is running at {manager.base_url}")
            info = manager.get_server_info()
            if info:
                print(f"📊 Server info: {json.dumps(info, indent=2)}")
        else:
            print(f"❌ Server is not running")

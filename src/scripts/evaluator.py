#!/usr/bin/env python
# coding: utf-8
import os
import sys
import json
from pathlib import Path
import gc  # <-- 해결책 2: 가비지 컬렉터 모듈 임포트
import torch
import pandas as pd
from datetime import datetime
# 프로젝트 루트를 Python 경로에 추가

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # iSKA_Gen 디렉토리
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'src'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'modules'))
sys.path.append(str(PROJECT_ROOT / 'src' / 'utils'))


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.eval_passage import evaluate_passages
from utils.create_passage_pairs import save_passage_pairs
from modules.iska.stem_eval import StemEvaluator
from modules.model_client import OpenAIModelClient, LocalModelClient, VLLMOpenAIClient, create_model_client
from utils.output_saver import save_model_output, DEFAULT_EVALUATION_DIR
from utils.vllm_server_manager import VLLMServerManager, quick_start_gpt_oss_20b


# 🎯 평가 설정
EVALUATION_CONFIG = {
    "gpt-oss-20b": {
        "client_type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "max_tokens": 2048,
        "temperature": 0.1,  # 평가에는 낮은 temperature 사용
        "timeout": 180
    },
    "gpt-4o-mini": {
        "client_type": "openai"
    },
}

MODEL_LIST = ["gpt-oss-20b"]  # 🚀 새로운 평가 모델 사용
BENCH_ID_LIST = [1, 2, 3, 4, 5]
BENCH_FILE = "v1/iSKA-Gen_Benchmark_v1.0.0_20250725_Initial.json"
BENCH_FILE_SMALL = "v1/iSKA-Gen_Benchmark_v1.0.0_20250725_Initial_small.json"
RUBRICS = ["completeness_for_guidelines", "clarity_of_core_theme", "reference_groundedness", "logical_flow", "korean_quality", "l2_learner_suitability"]

def create_evaluator_client(model_name: str):
    """
    평가 모델에 맞는 클라이언트를 생성하는 멋진 팩토리 함수
    
    Args:
        model_name: 평가에 사용할 모델명
        
    Returns:
        적절한 클라이언트 인스턴스
    """
    print(f"🔧 Creating evaluator client for {model_name}...")
    
    if model_name not in EVALUATION_CONFIG:
        print(f"⚠️  Model {model_name} not in config, using default OpenAI settings")
        return OpenAIModelClient(model_name=model_name)
    
    config = EVALUATION_CONFIG[model_name]
    client_type = config["client_type"]
    
    print(f"   📋 Client type: {client_type}")
    
    if client_type == "vllm":
        print(f"   🚀 Initializing vLLM client...")
        print(f"   🌐 Server URL: {config['base_url']}")
        
        return VLLMOpenAIClient(
            model_name=model_name,
            base_url=config["base_url"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            timeout=config["timeout"]
        )
    elif client_type == "openai":
        print(f"   🤖 Initializing OpenAI client...")
        return OpenAIModelClient(model_name=model_name)
    elif client_type == "local":
        print(f"   💻 Initializing local client...")
        return LocalModelClient(model_name=model_name)
    else:
        raise ValueError(f"Unknown client type: {client_type}")


# 🚀 Main execution with style and server management
if __name__ == "__main__":
    print("🎯 iSKA Stem Evaluation Pipeline")
    print("="*50)
    
    # Configuration
    TARGET_MODELS = ["A.X-4.0-Light"] 
    EVALUATOR_MODEL = "gpt-oss-20b"  # 🔥 Our shiny new evaluator
    DATE_STR = "2025-08-08"
    
    print(f"🎮 Target models: {TARGET_MODELS}")
    print(f"🤖 Evaluator model: {EVALUATOR_MODEL}")
    print(f"📅 Date: {DATE_STR}")
    print()
    
    # 🚀 Smart server management for vLLM models
    server_manager = None
    
    try:
        if EVALUATOR_MODEL == "gpt-oss-20b":
            print(f"🚀 Setting up vLLM server for {EVALUATOR_MODEL}...")
            
            # Check if server is already running
            temp_manager = VLLMServerManager(model_name=EVALUATOR_MODEL)
            if not temp_manager.is_server_running():
                print("🔧 Starting vLLM server...")
                server_manager = quick_start_gpt_oss_20b()
            else:
                print("✅ vLLM server already running")
        
        # Run evaluations
        for model in TARGET_MODELS:
            try:
                print(f"🚀 Starting evaluation for {model}...")
                evaluate_stems_completeness(
                    model_name=model, 
                    evaluator_model=EVALUATOR_MODEL,
                    date_str=DATE_STR
                )
                print(f"✅ {model} evaluation complete!\n")
                
            except Exception as e:
                print(f"❌ Error evaluating {model}: {e}\n")
        
        print("🎊 All evaluations complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
    finally:
        # Clean up: stop server if we started it
        if server_manager:
            print("\n🛑 Cleaning up vLLM server...")
            server_manager.stop_server()
            print("✅ Cleanup complete")


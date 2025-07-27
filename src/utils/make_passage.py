import os
import sys
import json
from pathlib import Path
import gc  # <-- 해결책 2: 가비지 컬렉터 모듈 임포트
import torch
import pandas as pd
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path.cwd().parent.parent))
sys.path.append(str(Path.cwd().parent.parent))
sys.path.append(str(Path.cwd().parent / 'modules'))

from modules.iska.passage_agent import PassageAgent
from modules.model_client import LocalModelClient
from utils.output_saver import save_model_output
from utils.benchmark_loader import load_benchmarks

def generate_passage(benchmark_file : str, model_name : str,  template_key : str, benchmark_version: str = "v1.0.0", gpus : list = [2,3], BENCH_ID_LIST : list =  [1, 2, 3, 4, 5]):
    benchmarks = load_benchmarks(benchmark_file)

    llm_client = LocalModelClient(model_name=model_name, gpus = gpus)
    passage_agent = PassageAgent(llm_client=llm_client)
    
    for id in BENCH_ID_LIST:
        benchmark = benchmarks[id - 1]  # id는 1부터 시작하므로 -1을 해줌
        problem_types = benchmark['problem_types']
        eval_goals = benchmark['eval_goals']
        passage_datas = []
        for item in benchmark['items']:
            korean_topic = item['korean_topic']
            korean_context = item['korean_context']
            foreign_topic = item['foreign_topic']
            foreign_context = item['foreign_context']
        
            source_item = {
                "korean_topic": korean_topic,
                "korean_context": korean_context,
                "foreign_topic": foreign_topic,
                "foreign_context": foreign_context
            }
            generated_passage = passage_agent.generate_passage(korean_topic=korean_topic, korean_context=korean_context, foreign_topic=foreign_topic, foreign_context=foreign_context, problem_types=problem_types, eval_goals=eval_goals, template_key=template_key)

            # Json 형태로 저장
            passage_data = {
                "source_item": source_item,
                "generated_passage": generated_passage
            }
            passage_datas.append(passage_data)
        
        # output_saver를 사용하여 결과 저장
        saved_file = save_model_output(
            model_name=model_name,
            benchmark_id=id,
            benchmark_version=benchmark_version,
            template_key=template_key,
            data=passage_datas
        )
        print(f"Generated passage for benchmark ID {id} and saved to {saved_file}")

 
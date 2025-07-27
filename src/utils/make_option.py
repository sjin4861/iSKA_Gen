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
sys.path.append(str(Path.cwd().parent / 'modules'))

from modules.iska.options_agent import OptionsAgent
from modules.model_client import LocalModelClient
from utils.output_saver import save_model_output
from utils.benchmark_loader import load_benchmarks

def generate_options(benchmark_file : str, stem_path : Path, model_name : str, template_key : str, benchmark_version: str = "v1.0.0", gpus : list = [0], BENCH_ID_LIST : list = [1, 3, 7, 8, 10]):
    """
    벤치마크와 생성된 stem 데이터를 기반으로 선택지(options)를 생성합니다.
    
    Args:
        benchmark_file (str): 벤치마크 JSON 파일 이름
        stem_path (Path): 생성된 stem 데이터가 있는 디렉토리 경로
        model_name (str): 사용할 모델 이름
        template_key (str): 사용할 템플릿 키
        benchmark_version (str): 벤치마크 버전
        gpus (list): 사용할 GPU 리스트
        BENCH_ID_LIST (list): 처리할 벤치마크 ID 리스트
    """
    benchmarks = load_benchmarks(benchmark_file)

    llm_client = LocalModelClient(model_name=model_name, gpus=gpus)
    options_agent = OptionsAgent(llm_client=llm_client)

    for id in BENCH_ID_LIST:
        # stem 데이터 파일 경로 생성 (stem 생성 시 저장된 경로와 일치해야 함)
        new_stem_path = stem_path / f'{model_name}' / f'benchmark_{id}_{benchmark_version}_stem_*.json'
        
        # 가장 최근 stem 파일 찾기 (glob 패턴 사용)
        stem_files = list(stem_path.glob(f'{model_name}/benchmark_{id}_{benchmark_version}_stem_*.json'))
        if not stem_files:
            print(f"Warning: No stem files found for benchmark ID {id}. Skipping...")
            continue
            
        # 가장 최근 파일 선택 (파일명의 타임스탬프 기준)
        latest_stem_file = sorted(stem_files)[-1]
        
        with open(latest_stem_file, 'r', encoding='utf-8') as f:
            stems = json.load(f)
            
        benchmark = benchmarks[id - 1]  # id는 1부터 시작하므로 -1을 해줌
        problem_types = benchmark['problem_types']
        eval_goals = benchmark['eval_goals']
        
        options_datas = []
        for stem in stems:
            source_passage = stem['source_passage']
            
            # Json 형태로 저장
            options_data = {
                "source_passage": source_passage
            }
            
            for i in range(len(problem_types)):
                problem_type = problem_types[i]
                eval_goal = eval_goals[i]
                stem_text = stem.get(f'stem_{i+1}', '')
                
                if stem_text and stem_text != "문항 생성 실패":
                    # stem과 과제 유형에 맞는 선택지 생성
                    generated_options = options_agent.generate_options(
                        passage=source_passage,
                        stem=stem_text,
                        problem_type=problem_type,
                        eval_goal=eval_goal,
                        template=template_key
                    )
                    
                    options_data[f'problem_type_{i+1}'] = problem_type
                    options_data[f'eval_goal_{i+1}'] = eval_goal
                    options_data[f'stem_{i+1}'] = stem_text
                    options_data[f'options_{i+1}'] = generated_options if generated_options else "선택지 생성 실패"
                else:
                    options_data[f'problem_type_{i+1}'] = problem_type
                    options_data[f'eval_goal_{i+1}'] = eval_goal
                    options_data[f'stem_{i+1}'] = "문항 생성 실패"
                    options_data[f'options_{i+1}'] = "선택지 생성 실패 (stem 없음)"
                    
            options_datas.append(options_data)
        
        # output_saver를 사용하여 결과 저장
        saved_file = save_model_output(
            model_name=model_name,
            benchmark_id=id,
            benchmark_version=benchmark_version,
            template_key=f"{template_key}_options",  # options임을 명시
            data=options_datas
        )
        print(f"Generated options for benchmark ID {id} and saved to {saved_file}")

        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

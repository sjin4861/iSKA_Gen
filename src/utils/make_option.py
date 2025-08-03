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
from utils.output_loader import load_stems


def generate_options(
    benchmark_file: str,
    stem_model_name: str,
    model_name: str,
    template_key: str,
    stem_template_key: str = None,
    benchmark_version: str = "v1.0.0",
    gpus: list = [0],
    BENCH_ID_LIST: list = [1, 3, 7, 8, 10],
    date_str: str = None
):
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
        stems = load_stems(
            model_name=stem_model_name,
            benchmark_id=id,
            benchmark_version=benchmark_version,
            template_key=stem_template_key,
            date_str=date_str
        )
        if not stems:
            print(f"Warning: No stem data found for benchmark ID {id} in date {date_str}. Skipping...")
            continue

        benchmark = benchmarks[id - 1]  # id는 1부터 시작하므로 -1을 해줌
        problem_types = benchmark['problem_types']
        eval_goals = benchmark['eval_goals']

        options_datas = []
        for stem in stems:
            source_passage = stem['source_passage']
            problem_type = problem_types[0]
            eval_goal = eval_goals[0]
            stem_text = stem.get('stem_1', '')

            options_data = {
                "source_passage": source_passage,
                "problem_type_1": problem_type,
                "eval_goal_1": eval_goal,
                "stem_1": stem_text,
            }

            if stem_text and stem_text != "문항 생성 실패":
                generated_options = options_agent.generate_options(
                    passage=source_passage,
                    stem=stem_text,
                    template_key=template_key,
                )
                options_data["options"] = generated_options if generated_options else "선택지 생성 실패"
            else:
                options_data["options"] = "선택지 생성 실패 (stem 없음)"

            options_datas.append(options_data)

        saved_file = save_model_output(
            model_name=model_name,
            benchmark_id=id,
            benchmark_version=benchmark_version,
            template_key=f"{template_key}_options",
            data=options_datas,
            date_str=date_str
        )
        print(f"Generated options for benchmark ID {id} and saved to {saved_file}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

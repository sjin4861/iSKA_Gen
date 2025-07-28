import os
import sys
import json
from pathlib import Path
from typing import Optional
import gc  # <-- 해결책 2: 가비지 컬렉터 모듈 임포트
import torch
import pandas as pd
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path.cwd().parent.parent))
sys.path.append(str(Path.cwd().parent / 'modules'))

from modules.iska.stem_agent import StemAgent
from modules.model_client import LocalModelClient
from utils.output_saver import save_model_output
from utils.benchmark_loader import load_benchmarks, get_benchmark_by_id
from utils.output_loader import load_passages, debug_available_files

def generate_stem(benchmark_file : str, passage_model_name : str, model_name : str,  template_key : str, passage_template_key: Optional[str] = None, benchmark_version: str = "v1.0.0", gpus : list = [0], BENCH_ID_LIST : list =  [1, 3, 7, 8, 10], date_str: Optional[str] = None):
    """
    벤치마크와 생성된 passage 데이터를 기반으로 stem을 생성합니다.
    
    Args:
        benchmark_file (str): 벤치마크 파일 이름
        passage_model_name (str): passage를 생성한 모델 이름 (데이터 로드용)
        model_name (str): stem 생성에 사용할 모델 이름
        template_key (str): stem 생성에 사용할 템플릿 키
        passage_template_key (Optional[str]): passage 생성에 사용된 템플릿 키 (None이면 자동 검색)
        benchmark_version (str): 벤치마크 버전
        gpus (list): 사용할 GPU 리스트
        BENCH_ID_LIST (list): 처리할 벤치마크 ID 리스트
    """
    benchmarks = load_benchmarks(benchmark_file)

    llm_client = LocalModelClient(model_name=model_name, gpus = gpus)
    stem_agent = StemAgent(llm_client=llm_client)

    for id in BENCH_ID_LIST:
        print(f"\n📝 벤치마크 ID {id}에 대한 stem 생성 중...")

        # output_loader를 사용하여 passage 데이터 로드 (date_str 추가)
        passages = load_passages(
            model_name=passage_model_name,
            benchmark_id=id,
            benchmark_version=benchmark_version,
            template_key=passage_template_key,
            date_str=date_str
        )

        if not passages:
            print(f"❌ 모델 '{passage_model_name}'의 벤치마크 ID {id} passage 데이터를 찾을 수 없습니다.")
            print("🔍 디버깅 정보:")
            debug_available_files(passage_model_name)
            continue

        print(f"✅ {len(passages)}개의 passage 로드 완료")

        # 벤치마크 정보 가져오기
        benchmark = get_benchmark_by_id(benchmark_file, id)
        if not benchmark:
            print(f"❌ 벤치마크 ID {id}를 찾을 수 없습니다.")
            continue
        problem_types = benchmark['problem_types']
        eval_goals = benchmark['eval_goals']
        stem_datas = []

        for i, passage_data in enumerate(passages):
            print(f"  📄 Passage {i+1}/{len(passages)} 처리 중...")
            stem_data = {
                "source_passage": passage_data['generated_passage']
            }
            for j in range(len(problem_types)):
                problem_type = problem_types[j]
                eval_goal = eval_goals[j]
                # 첫 번째 stem은 객관식 문제 생성 프롬프트 사용
                if j == 0:
                    prompt_key = "stem_agent.mcq_few_shot"
                else:
                    prompt_key = template_key
                generated_stem = stem_agent.generate_stem(
                    passage=passage_data['generated_passage'],
                    problem_type=problem_type,
                    eval_goal=eval_goal,
                    template=prompt_key
                )
                stem_data[f'problem_type_{j+1}'] = problem_type
                stem_data[f'eval_goal_{j+1}'] = eval_goal
                stem_data[f'stem_{j+1}'] = generated_stem if generated_stem else "문항 생성 실패"
            stem_datas.append(stem_data)

        # output_saver를 사용하여 결과 저장
        saved_file = save_model_output(
            model_name=model_name,
            benchmark_id=id,
            benchmark_version=benchmark_version,
            template_key=f"{template_key}",  # stem임을 명시
            data=stem_datas
        )
        print(f"✅ 벤치마크 ID {id}에 대한 stem 생성 완료 및 저장: {saved_file}")

        # 메모리 정리
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

 
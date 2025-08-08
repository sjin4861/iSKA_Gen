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


import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.eval_passage import evaluate_passages
from utils.create_passage_pairs import save_passage_pairs


MODEL_LIST = ["Gemini-2.5-Pro"]#["Midm-2.0-Base-Instruct", "EXAONE-3.5-7.8B-Instruct"]#, "A.X-4.0-Light", "llama3.1_korean_v1.1_sft_by_aidx", "llama3-bllossom-3b"]
USER_AGENT = "iSKA (sjun24530@gmail.com)"
BENCH_ID_LIST = [1, 2, 3, 4, 5]
BENCH_FILE = "v1/iSKA-Gen_Benchmark_v1.0.0_20250725_Initial.json"
BENCH_FILE_SMALL = "v1/iSKA-Gen_Benchmark_v1.0.0_20250725_Initial_small.json"
RUBRICS = ["completeness_for_guidelines", "clarity_of_core_theme", "reference_groundedness", "logical_flow", "korean_quality", "l2_learner_suitability"]


for model_name in MODEL_LIST:
    evaluate_passages(
        benchmark_file=BENCH_FILE,
        passage_model_name=model_name,
        evaluator_model="gpt-4o",
        passage_template_key="passage_agent.create_passage_rubric_aware",
        template_key="rubric",
        benchmark_version="v1.0.0",
        BENCH_ID_LIST=[1],
        date_str="2025-08-05"  # 특정 날짜 지정
    )
    # evaluate_passages(
    #     benchmark_file=BENCH_FILE,
    #     passage_model_name=model_name,
    #     evaluator_model="gpt-4o",
    #     template_key="passage_eval.binary_rubric_strict",
    #     passage_template_key="passage_agent.create_passage_rubric_aware",
    #     benchmark_version="v1.0.0",
    #     BENCH_ID_LIST=[2],
    #     date_str="2025-08-05"  # 특정 날짜 지정
    # )
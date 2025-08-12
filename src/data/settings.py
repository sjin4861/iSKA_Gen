from pathlib import Path

# 프로젝트 루트 기준으로 data_store를 잡아요.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_STORE = PROJECT_ROOT / "data_store"

# 폴더 레이아웃
BENCHMARKS_DIR = DATA_STORE / "benchmarks" / "v1"
RAW_OUTPUTS_DIR = DATA_STORE / "raw_outputs"
PAIRWISE_DIR = DATA_STORE / "pairwise_data"
EVAL_DIR = DATA_STORE / "evaluations"

# 공통 파일명 포맷
def passage_file_name(benchmark_id: int, version: str, template_key: str) -> str:
    return f"benchmark_{benchmark_id}_{version}_{template_key}.json"

def stem_file_name(benchmark_id: int, version: str, template_key: str) -> str:
    return f"benchmark_{benchmark_id}_{version}_{template_key}.json"

from __future__ import annotations
from pathlib import Path
from typing import Any, List
from .file_system import read_json
from ...settings import BENCHMARKS_DIR

class BenchmarkFSStore:
    """
    data_store/benchmarks/v1/ 아래 JSON을 그대로 읽어오는 역할만.
    """
    def load_file(self, filename: str) -> Any:
        path = Path(BENCHMARKS_DIR) / filename
        data = read_json(path)
        if data is None:
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        return data

    def list_files(self) -> List[Path]:
        return list(Path(BENCHMARKS_DIR).glob("*.json"))

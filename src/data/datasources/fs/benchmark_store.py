from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional
from .file_system import read_json
from .data_store_fs import DataStoreFSDataSource

class BenchmarkFSStore:
    """
    data_store/benchmarks/ 아래 JSON을 그대로 읽어오는 역할만.
    DataStoreFSDataSource를 기반으로 한 레거시 호환성 래퍼
    """
    def __init__(self, data_store: Optional[DataStoreFSDataSource] = None):
        self.data_store = data_store or DataStoreFSDataSource()
    
    def load_file(self, filename: str) -> Any:
        path = self.data_store.benchmarks_path / filename
        data = read_json(path)
        if data is None:
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        return data

    def list_files(self) -> List[Path]:
        return list(self.data_store.benchmarks_path.rglob("*.json"))

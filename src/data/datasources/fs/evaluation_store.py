from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from .file_system import read_json, write_json_atomic
from ...settings import EVAL_DIR

class EvaluationFSStore:
    """
    자동/휴먼 평가 결과 적재용(선택). 필요 시만 사용.
    """
    def path_for(self, name: str) -> Path:
        return Path(EVAL_DIR) / f"{name}.json"

    def load(self, name: str) -> Optional[Any]:
        return read_json(self.path_for(name))

    def save(self, name: str, data: Any) -> Path:
        return write_json_atomic(self.path_for(name), data)

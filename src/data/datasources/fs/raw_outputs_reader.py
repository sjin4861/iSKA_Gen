from __future__ import annotations
import json, re
from pathlib import Path
from typing import List, Dict, Any

class RawOutputsFSReader:
    """
    data_store/raw_outputs/<date>/passage/ 아래 JSON들을 읽어들여
    각 아이템에 model/task/benchmark_id/file_path 메타데이터를 붙여 반환
    """
    def __init__(self, base_root: Path = Path("data_store")):
        self.base_root = base_root

    def list_passage_records_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        base_path = self.base_root / "raw_outputs" / date_str / "passage"
        if not base_path.exists():
            return []

        out: List[Dict[str, Any]] = []
        json_files = list(base_path.glob("**/*.json"))

        for fp in json_files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue

            # path: <date>/passage/<model>/<task>/<file>
            rel = fp.relative_to(base_path)
            parts = rel.parts
            model_name = parts[0] if len(parts) > 0 else "unknown_model"
            task_name  = parts[1] if len(parts) > 1 else "unknown_task"

            m = re.search(r"benchmark_(\d+)", fp.name)
            benchmark_id = int(m.group(1)) if m else -1

            for item in data if isinstance(data, list) else []:
                item = dict(item)
                item["model_name"] = model_name
                item["task_name"] = task_name
                item["benchmark_id"] = benchmark_id
                item["file_path"] = str(fp)
                out.append(item)
        return out

from __future__ import annotations
import json, re
from pathlib import Path
from typing import List, Dict, Any

class EvaluationsFSReader:
    """
    data_store/evaluations/<date>/misc/**/eval_rubric/*.json 읽어서
    모델/벤치마크/점수들을 납작하게 풀어 반환
    """
    def __init__(self, base_root: Path = Path("data_store")):
        self.base_root = base_root

    def list_evaluation_records_by_date(self, date_str: str) -> List[Dict[str, Any]]:
        base_path = self.base_root / "evaluations" / date_str / "misc"
        if not base_path.exists():
            return []

        out: List[Dict[str, Any]] = []
        json_files = list(base_path.glob("**/eval_rubric/*.json"))
        for fp in json_files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue

            # .../misc/<MODEL>_evaluation/eval_rubric/benchmark_1_...json
            parts = fp.parts
            model_dir = next((p for p in parts if p.endswith("_evaluation")), None)
            model_name = model_dir[:-11] if model_dir else "unknown_model"

            m = re.search(r"benchmark_(\d+)", fp.name)
            benchmark_id = int(m.group(1)) if m else -1

            for item in data if isinstance(data, list) else []:
                row: Dict[str, Any] = {
                    "model_name": model_name,
                    "benchmark_id": benchmark_id,
                    "file_path": str(fp),
                }
                eval_dict = item.get("evaluation", {})
                if isinstance(eval_dict, dict):
                    for k, v in eval_dict.items():
                        if k.endswith("_score"):
                            row[k] = v
                out.append(row)
        return out

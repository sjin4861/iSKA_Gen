# src/data/repositories/rubric_evaluation_repository_impl.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.domain.repositories.rubric_evaluation_repository import RubricEvaluationRepository
from src.domain.entities.content_types import ArtifactKind
from src.domain.entities.rubrics import RubricID
from src.modules.client_factory import ModelClientFactory
from src.utils.prompt_loader import get_prompt

_RUBRIC_KEYS_3 = "- clarity_of_core_theme\n- logical_flow\n- korean_quality"

def _first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

def _safe_json_extract(text: str) -> Dict[str, Any]:
    """
    모델이 앞/뒤에 군더더기를 붙여도 JSON 객체만 뽑아 파싱.
    """
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        # 흔한 꼬임 방지(줄바꿈/따옴표 등 매우 가볍게만)
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*]", "]", s)
        try:
            return json.loads(s)
        except Exception:
            return {}

def _iter_passage_files(
    base: Path,
    date_str: str,
    bench_id: int,
    *,
    model_filter: Optional[List[str]],
    template_filter: Optional[List[str]],
    benchmark_version: str,
) -> Iterable[Tuple[str, str, Path]]:
    """
    yields (model_name, template_key, file_path)
    구조: {base}/{date}/passage/{model}/{template}/benchmark_{id}_{ver}_{template}.json
    """
    root = base / date_str / "passage"
    if not root.exists():
        return
    for model_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        model_name = model_dir.name
        if model_filter and model_name not in model_filter:
            continue
        for tpl_dir in sorted([d for d in model_dir.iterdir() if d.is_dir()]):
            template_key = tpl_dir.name
            if template_filter and template_key not in template_filter:
                continue
            # 보수적으로 패턴 매칭
            patt = f"benchmark_{bench_id}_{benchmark_version}_{template_key}.json"
            f = tpl_dir / patt
            if f.exists():
                yield (model_name, template_key, f)

class RubricEvaluationRepositoryImpl(RubricEvaluationRepository):
    """
    - 입력: data_store/raw_outputs (또는 src/data/raw_outputs) 구조를 그대로 순회
    - 평가: vLLM(OpenAI 호환) 클라이언트로 evaluator.rate_passage_full 프롬프트 호출
    - 출력: data_store/evaluations/{date}/misc/{model}_evaluation/eval_rubric/benchmark_{id}_{ver}_eval_rubric.json
            (각 파일은 리스트(JSON array)로 저장)
    """

    def __init__(self) -> None:
        # lazy init for client
        self._client = None

    def _ensure_client(
        self,
        *,
        client_type: str,
        model_name: str,
        client_kwargs: Optional[Dict[str, Any]],
    ):
        if self._client:
            return
        client_kwargs = client_kwargs or {}
        self._client = ModelClientFactory.create_model_client(
            client_type=client_type,
            model_name=model_name,
            **client_kwargs
        )

    def _load_rows(self, file_path: Path) -> List[Dict[str, Any]]:
        try:
            return json.loads(file_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ JSON 로드 실패: {file_path} ({e})")
            return []

    def _save_eval_list(
        self,
        *,
        base_eval_root: Path,
        date_str: str,
        model_name: str,
        bench_id: int,
        benchmark_version: str,
        items: List[Dict[str, Any]],
    ) -> Path:
        out_dir = base_eval_root / date_str / "misc" / f"{model_name}_evaluation" / "eval_rubric"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / f"benchmark_{bench_id}_{benchmark_version}_eval_rubric.json"
        out_fp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_fp

    def _pick_base_roots(self) -> Tuple[Path, Path]:
        raw_roots = [Path("data_store/raw_outputs"), Path("src/data/raw_outputs")]
        eval_roots = [Path("data_store/evaluations"), Path("src/data/evaluations")]
        raw_base = _first_existing(raw_roots)
        if not raw_base:
            raise FileNotFoundError("raw_outputs 루트 디렉토리를 찾을 수 없습니다 (data_store/raw_outputs 또는 src/data/raw_outputs).")
        eval_base = eval_roots[0]  # 우선 data_store에 저장
        eval_base.mkdir(parents=True, exist_ok=True)
        return raw_base, eval_base

    def _build_prompt(self, passage: str) -> str:
        return get_prompt(
            "evaluator.rate_passage_full",
            agent="iska",
            rubric_keys=_RUBRIC_KEYS_3,
            passage=passage,
        )

    def _call_llm_json(self, prompt: str) -> Dict[str, Any]:
        # vLLM(OpenAI 호환) 클라이언트는 ChatCompletions
        messages = [{"role": "user", "content": prompt}]
        resp = self._client.call(messages, temperature=0.0, max_tokens=512)
        return _safe_json_extract(resp)

    def evaluate_and_save(
        self,
        *,
        date_str: str,
        target_mode: str,
        artifact_kind: ArtifactKind,
        bench_ids: List[int],
        benchmark_version: str,
        rubric_ids: List[RubricID],
        source_model_filter: Optional[List[str]] = None,
        template_filter: Optional[List[str]] = None,
        limit_per_benchmark: Optional[int] = None,
        evaluator_client_type: str = "vllm",
        evaluator_model_name: str = "gpt-oss-20b",
        evaluator_client_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        assert artifact_kind == ArtifactKind.passage, "오늘은 passage만 지원"
        assert target_mode == "content", "오늘은 content-only만 지원"

        self._ensure_client(
            client_type=evaluator_client_type,
            model_name=evaluator_model_name,
            client_kwargs=evaluator_client_kwargs or {},
        )

        raw_base, eval_base = self._pick_base_roots()
        summary: Dict[str, Any] = {"total_evaluated": 0, "by_benchmark": {}}

        for bid in bench_ids:
            collected: List[Dict[str, Any]] = []
            for model_name, template_key, fp in _iter_passage_files(
                raw_base, date_str, bid,
                model_filter=source_model_filter,
                template_filter=template_filter,
                benchmark_version=benchmark_version,
            ):
                rows = self._load_rows(fp)
                if not rows:
                    continue

                for idx, row in enumerate(rows):
                    if limit_per_benchmark and len(collected) >= limit_per_benchmark:
                        break
                    passage = (row or {}).get("generated_passage") or ""
                    if not passage:
                        continue

                    prompt = self._build_prompt(passage)
                    j = self._call_llm_json(prompt)

                    # 안전하게 스키마 맞추기
                    item = {
                        "model_name": model_name,
                        "task_name": template_key,
                        "benchmark_id": bid,
                        "index": idx,
                        "file_path": str(fp),
                        "evaluation": {
                            "clarity_of_core_theme_score": int(j.get("clarity_of_core_theme_score", 0) or 0),
                            "clarity_of_core_theme_feedback": str(j.get("clarity_of_core_theme_feedback", ""))[:200],
                            "logical_flow_score": int(j.get("logical_flow_score", 0) or 0),
                            "logical_flow_feedback": str(j.get("logical_flow_feedback", ""))[:200],
                            "korean_quality_score": int(j.get("korean_quality_score", 0) or 0),
                            "korean_quality_feedback": str(j.get("korean_quality_feedback", ""))[:200],
                        }
                    }

                    # 점수 클램프(1..5)
                    for k in ("clarity_of_core_theme_score", "logical_flow_score", "korean_quality_score"):
                        v = item["evaluation"][k]
                        if not isinstance(v, int):
                            v = 0
                        item["evaluation"][k] = max(1, min(5, v)) if v else v

                    collected.append(item)

                if limit_per_benchmark and len(collected) >= limit_per_benchmark:
                    break

            # 저장
            if collected:
                out_fp = self._save_eval_list(
                    base_eval_root=eval_base,
                    date_str=date_str,
                    model_name=source_model_filter[0] if (source_model_filter and len(source_model_filter)==1) else "mixed",
                    bench_id=bid,
                    benchmark_version=benchmark_version,
                    items=collected,
                )
                summary["by_benchmark"][str(bid)] = {"count": len(collected), "path": str(out_fp)}
                summary["total_evaluated"] += len(collected)
            else:
                summary["by_benchmark"][str(bid)] = {"count": 0, "path": None}

        return summary

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPG all-in-one 테스트 데이터 생성기

- data_store/raw_outputs/*/test_stem/<MODEL>/**/benchmark_{1..5}_v*_*.json 자동 스캔
- (벤치, 모델) 조합별 최신 파일만 사용
- 각 row × 루브릭(R1..R6)으로 prompt/ chosen 생성
- 출력: data_store/cpg/all_in_one_test.jsonl  (JSONL, 키는 prompt/ chosen만)
"""

from __future__ import annotations
import json, glob, os
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict
import re
import yaml

# --------------------------- 고정 설정 ---------------------------
BENCH_IDS = [1, 2, 3, 4, 5]
RUBRICS = ["R1", "R2", "R3", "R4", "R5", "R6"]
BENCHMARK_FILE = Path("data_store/benchmarks/v1/iSKA-Gen_Benchmark_v1.1.0_20250808_test.json")
PROMPT_BASE = Path("src/config/prompts/iska")
OUT_PATH  = Path("saves/all_in_one/all_in_one_rm_test2.jsonl")

# --------------------------- 유틸 ---------------------------

_ITEM_SID_RE = re.compile(r"^bench_(\d+)_item_(\d+)$", re.IGNORECASE)

def _resolve_source_id(bench_id: int, row: Dict[str, Any], idx: int) -> str:
    sid = (row.get("source_id") or "").strip()
    if sid and _ITEM_SID_RE.match(sid):
        return sid
    return f"bench_{bench_id}_item_{idx}"

def _infer_model_from_path(p: Path) -> str:
    """
    예상 경로:
      .../test_stem/{MODEL}/{TEMPLATE}/benchmark_{id}_v{ver}_{tkey}.json
    """
    try:
        return p.parent.parent.name  # 템플릿 디렉터리의 상위가 모델 디렉터리
    except Exception:
        return ""

def _resolve_source_id(bench_id: int, row: Dict[str, Any], idx: int) -> str:
    sid = (row.get("source_id") or "").strip()
    if sid and _ITEM_SID_RE.match(sid):
        return sid
    # 위 패턴이 아니면 bench_id + 로컬 인덱스로 표준화
    return f"bench_{bench_id}_item_{idx}"
    
def bench_kind(bench_id: int) -> str:
    if bench_id in (1, 2):
        return "text"
    if bench_id in (3, 4):
        return "listening"
    if bench_id == 5:
        return "visual"
    raise ValueError(f"Unsupported bench id: {bench_id}")

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_json_any(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        return data if isinstance(data, list) else []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows

def _pick_latest(paths: List[str]) -> Optional[Path]:
    if not paths:
        return None
    return Path(max(paths, key=os.path.getmtime))

def prompt_file_for(kind: str) -> Path:
    if kind == "text":
        return PROMPT_BASE / "preference_eval.yaml"
    if kind == "listening":
        return PROMPT_BASE / "preference_eval_listening.yaml"
    if kind == "visual":
        return PROMPT_BASE / "preference_eval_visual.yaml"
    raise ValueError(kind)

RUBRIC_KEYS = {
    "R1": "completeness_for_guidelines",
    "R2": "clarity_of_core_theme",
    "R3": "reference_groundedness",
    "R4": "logical_flow",
    "R5": "korean_quality",
    "R6": "l2_learner_suitability",
}
ROOT_KEYS = {
    "text": "preference_evaluation",
    "listening": "preference_evaluation_listening",
    "visual": "preference_evaluation_visual",
}

class SafeDict(defaultdict):
    def __missing__(self, key):
        return ""

def _extract_primary_text(row: Dict[str, Any], kind: str, *, for_rejected: bool) -> str:
    order_map = {
        "text": (
            ["generated_passage", "content", "passage"],    # rejected
            ["source_passage", "passage", "content"],       # chosen
        ),
        "listening": (
            ["generated_audio_script", "generated_dialogue", "content", "audio_script", "dialogue"],
            ["audio_script", "dialogue", "source_passage", "content"],
        ),
        "visual": (
            ["generated_caption", "generated_image_caption", "content", "source_passage", "caption", "image_caption"],
            ["image_caption", "caption", "source_passage", "content"],
        ),
    }
    keys = order_map[kind][0] if for_rejected else order_map[kind][1]
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Stem 세트의 원문 키 대체 (repo가 source_content로 저장한 경우 대비)
    for alt in ("source_content", "source_passage"):
        v = row.get(alt)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def _render_payload(kind: str, main_text: str, stems: List[str], *, include_stems: bool) -> str:
    if kind == "text":
        head = "[지문]"
    elif kind == "listening":
        head = "[대화]"
    else:
        head = "[이미지 설명/상황 제시]"
    lines = [f"{head}\n{main_text.strip()}"]
    if include_stems:
        usable = [s.strip() for s in stems if isinstance(s, str) and s.strip()]
        if usable:
            lines.append("[문항 세트]")
            for i, s in enumerate(usable, start=1):
                lines.append(f"- {i}) {s}")
    return "\n".join(lines)

def load_benchmark_meta(
    bench_id: int,
    *,
    explicit_file: Optional[Path] = BENCHMARK_FILE,
) -> Dict[str, List[str]]:
    target = explicit_file
    if not target or not target.exists():
        raise FileNotFoundError(f"benchmark file not found: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    benches = data.get("benchmark")
    for b in benches:
        try:
            if int(b.get("id", -1)) == bench_id:
                return {
                    "problem_types": b.get("problem_types", []) or [],
                    "eval_goals": b.get("eval_goals", []) or [],
                }
        except Exception:
            continue
    raise KeyError(f"benchmark id {bench_id} not found in {target}")

def unify_reference_by_bench(source_item: Dict[str, Any], bench_id: int) -> str:
    if not source_item:
        return ""
    # B1(비교형): 한국/외국 컨텍스트 합치기
    if bench_id == 1:
        kc = (source_item.get("korean_context") or "").strip()
        fc = (source_item.get("foreign_context") or "").strip()
        parts = []
        if kc: parts.append(kc)
        if fc: parts.append(fc)
        return "\n\n".join(parts).strip()
    # B2/B3/B4(단일형): context 우선 → fallback topic
    ctx = (source_item.get("context") or source_item.get("korean_context") or "").strip()
    if ctx:
        return ctx
    topic = (source_item.get("topic") or source_item.get("korean_topic") or "").strip()
    return topic

def build_conversation_value(
    bench_id: int,
    rubric_code: str,
    chosen_row: Dict[str, Any],
    bench_meta: Dict[str, List[str]],
) -> str:
    kind = bench_kind(bench_id)
    yml_path = prompt_file_for(kind)
    yml = load_yaml(yml_path)
    root = ROOT_KEYS[kind]
    sub = RUBRIC_KEYS[rubric_code]
    try:
        tmpl = yml[root][sub]
    except Exception:
        raise KeyError(f"프롬프트 키 누락: {root}.{sub} @ {yml_path}")

    sd = SafeDict()
    if rubric_code == "R1":
        pts = bench_meta.get("problem_types", [])
        egs = bench_meta.get("eval_goals", [])
        # 최소 3개까지만 방어적으로 매핑
        for i in range(3):
            sd[f"problem_type{i+1}"] = pts[i] if i < len(pts) else ""
            sd[f"eval_goal{i+1}"] = egs[i] if i < len(egs) else ""

    if rubric_code == "R3":
        ref = unify_reference_by_bench(chosen_row.get("source_item") or {}, bench_id)
        sd["reference"] = ref

    return tmpl.format_map(sd).strip()

def build_chosen_value(bench_id: int, rubric_code: str, chosen_row: Dict[str, Any]) -> str:
    kind = bench_kind(bench_id)
    passage_or_script = _extract_primary_text(chosen_row, kind, for_rejected=False)
    # stems: 리스트가 없으면 stem_1..stem_3에서 구성
    stems = chosen_row.get("stems")
    if not isinstance(stems, list):
        stems = [chosen_row.get("stem_1", ""), chosen_row.get("stem_2", ""), chosen_row.get("stem_3", "")]
    include_stems = rubric_code in ("R1", "R6")
    if include_stems and not any((s or "").strip() for s in stems):
        include_stems = False
    return _render_payload(kind, passage_or_script, stems, include_stems=include_stems)

# --------------------------- 스캐너 ---------------------------
def find_latest_stem_files_for_bench(bench_id: int) -> List[Path]:
    """
    모든 날짜/모델의 test_stem 경로에서 benchmark_{bench_id} 파일을 찾고,
    (모델 기준) 최신 파일만 반환.
    """
    pat = f"data_store/raw_outputs/*/test_stem/*/**/benchmark_{bench_id}_v*_*.json"
    all_paths = glob.glob(pat, recursive=True)
    # 모델 기준 최신 1개 선택
    latest_by_model: Dict[str, Path] = {}
    for p in all_paths:
        path = Path(p)
        # .../test_stem/<MODEL>/<TKEY>/file.json
        try:
            model = path.parents[1].name  # <MODEL>
        except Exception:
            continue
        cur = latest_by_model.get(model)
        if cur is None or os.path.getmtime(p) > os.path.getmtime(str(cur)):
            latest_by_model[model] = path
    return list(latest_by_model.values())

# --------------------------- 메인 파이프라인 ---------------------------
def build_all_in_one() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for bench_id in BENCH_IDS:
        bench_files = find_latest_stem_files_for_bench(bench_id)
        if not bench_files:
            print(f"⚠️ bench {bench_id}: 파일 없음 (test_stem)")
            continue
        bench_meta = load_benchmark_meta(bench_id)
        for file in bench_files:
            file = Path(file)
            model_name = _infer_model_from_path(file)
            rows = load_json_any(file)
            if not rows:
                print(f"⚠️ 빈 파일: {file}")
                continue
            for idx, row in enumerate(rows):
                sid = _resolve_source_id(bench_id, row, idx)
                for rc in RUBRICS:
                    prompt = build_conversation_value(bench_id, rc, row, bench_meta)
                    chosen = build_chosen_value(bench_id, rc, row)
                    out.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "source_id": sid,
                        "model": model_name,  # ✅ 모델명 추가
                    })
    return out

def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False))
            f.write("\n")

def main():
    print("🚀 CPG all-in-one 생성 시작")
    items = build_all_in_one()
    write_jsonl(OUT_PATH, items)
    print(f"✅ 완료: {len(items)}개 라인 → {OUT_PATH}")

if __name__ == "__main__":
    main()

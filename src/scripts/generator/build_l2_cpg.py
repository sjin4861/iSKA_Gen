#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPG 데이터셋 생성기 (L2 적합성 데이터용)

- 2025-08-23 디렉토리에 생성된, stem과 violate_content가 통합된 JSON 파일을 읽어 CPG 데이터셋을 생성.
- conversation.value: 루브릭별 프롬프트(YAML) 로드 후 포맷
- chosen.value: 통합 JSON의 원본 지문/문항
- rejected.value: 통합 JSON의 violate_content 필드 + 원본 문항

사용 예:
  python build_l2_cpg.py \
    --input-dir data_store/raw_outputs/2025-08-23 \
    --output-file data_store/rm_pair/cpg/l2_train.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml


# --------------------------- 유틸 공통 (원본 스크립트에서 재사용) ---------------------------

def bench_kind(bench_id: int) -> str:
    if bench_id in (1, 2):
        return "text"
    if bench_id in (3, 4):
        return "listening"
    if bench_id == 5:
        return "visual"
    raise ValueError(f"Unsupported bench id: {bench_id}")

def _extract_primary_text(row: Dict[str, Any], kind: str, *, for_rejected: bool) -> str:
    order_map = {
        "text": (["violate_content", "generated_passage", "content"], ["source_passage", "passage", "content"]),
        "listening": (["violate_content", "generated_audio_script"], ["audio_script", "dialogue", "source_passage"]),
        "visual": (["violate_content", "generated_caption"], ["image_caption", "caption", "source_passage"]),
    }
    keys = order_map[kind][0] if for_rejected else order_map[kind][1]
    for k in keys:
        v = row.get(k)
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

class SafeDict(defaultdict):
    def __missing__(self, key):
        return ""

def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_json_any(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def unify_reference(source_item: Dict[str, Any]) -> str:
    if not source_item:
        return ""
    c = (source_item.get("topic") or "").strip()
    parts = []
    if c:
        parts.append(c)
    return "\n\n".join(parts).strip()

# --------------------------- 프롬프트 키/파일 (원본 스크립트에서 재사용) ---------------------------

RUBRIC_KEYS = {
    "R1": "completeness_for_guidelines", "R2": "clarity_of_core_theme",
    "R3": "reference_groundedness", "R4": "logical_flow",
    "R5": "korean_quality", "R6": "l2_learner_suitability",
}

VIOL_KEY_BY_RUBRIC = {
    "R1": "violate_completeness_for_guidelines_severely", "R2": "violate_clarity_of_core_theme_severely",
    "R3": "violate_reference_groundedness_severely", "R4": "violate_flow_severely",
    "R5": "violate_korean_quality_severely", "R6": "violate_l2_suitability_severely",
}
# 역방향 매핑 추가
RUBRIC_CODE_BY_VIOL_KEY = {v: k for k, v in VIOL_KEY_BY_RUBRIC.items()}


ROOT_KEYS = {"text": "preference_evaluation", "listening": "preference_evaluation_listening", "visual": "preference_evaluation_visual"}

def prompt_file_for(kind: str) -> Path:
    # 스크립트 위치 기반으로 상대 경로 계산
    script_dir = Path(__file__).parent.parent
    base = script_dir / "config/prompts/iska"
    if kind == "text": return base / "preference_eval.yaml"
    if kind == "listening": return base / "preference_eval_listening.yaml"
    if kind == "visual": return base / "preference_eval_visual.yaml"
    raise ValueError(kind)

# --------------------------- 빌더 (수정됨) ---------------------------

def build_conversation_value(bench_id: int, rubric_code: str, item: Dict[str, Any]) -> str:
    kind = bench_kind(bench_id)
    yml_path = prompt_file_for(kind)
    yml = load_yaml(yml_path)
    root = ROOT_KEYS[kind]
    sub = RUBRIC_KEYS[rubric_code]
    try:
        tmpl = yml[root][sub]
    except KeyError:
        raise KeyError(f"프롬프트 키 누락: {root}.{sub} @ {yml_path}")
    sd = SafeDict()
    if rubric_code == "R1":
        sd["problem_type1"], sd["problem_type2"], sd["problem_type3"] = item.get("problem_type_1", ""), item.get("problem_type_2", ""), item.get("problem_type_3", "")
        sd["eval_goal1"], sd["eval_goal2"], sd["eval_goal3"] = item.get("eval_goal_1", ""), item.get("eval_goal_2", ""), item.get("eval_goal_3", "")
    if rubric_code == "R3":
        sd["reference"] = unify_reference(item.get("source_item") or {})
    return tmpl.format_map(sd).strip()

def build_chosen_value(bench_id: int, rubric_code: str, item: Dict[str, Any]) -> str:
    kind = bench_kind(bench_id)
    passage_or_script = _extract_primary_text(item, kind, for_rejected=False)
    stems = [item.get("stem_1", ""), item.get("stem_2", ""), item.get("stem_3", "")]
    include_stems = rubric_code in ("R1", "R6")
    return _render_payload(kind, passage_or_script, stems, include_stems=include_stems)

def build_rejected_value(bench_id: int, rubric_code: str, item: Dict[str, Any]) -> str:
    kind = bench_kind(bench_id)
    text = _extract_primary_text(item, kind, for_rejected=True) # for_rejected=True 사용
    stems = [item.get("stem_1", ""), item.get("stem_2", ""), item.get("stem_3", "")]
    include_stems = rubric_code in ("R1", "R6")
    return _render_payload(kind, text, stems, include_stems=include_stems)

# --------------------------- 메인 파이프라인 (재작성됨) ---------------------------

def build_l2_cpg_from_combined_data(input_dir: str) -> List[Dict[str, Any]]:
    dataset = []
    search_pattern = os.path.join(input_dir, '**', '*.json')
    
    print(f"Searching for files in: {search_pattern}")
    file_paths = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(file_paths)} files to process.")

    for file_path in file_paths:
        path = Path(file_path)
        
        # 파일 경로에서 벤치마크 ID와 위반 키 추출
        try:
            bench_match = re.search(r'benchmark_(\d+)_', path.name)
            if not bench_match: continue
            bench_id = int(bench_match.group(1))

            # violation 키는 폴더 이름에 있음
            violation_key = path.parent.name
            rubric_code = next((code for key, code in RUBRIC_CODE_BY_VIOL_KEY.items() if key in violation_key), None)
            
            if not rubric_code: continue

        except (ValueError, IndexError):
            print(f"Could not parse metadata from path: {file_path}. Skipping.")
            continue

        items = load_json_any(path)
        if not items:
            print(f"No items found in {file_path}. Skipping.")
            continue

        for i, item in enumerate(items):
            conv_value = build_conversation_value(bench_id, rubric_code, item)
            chosen_val = build_chosen_value(bench_id, rubric_code, item)
            rejected_val = build_rejected_value(bench_id, rubric_code, item)

            source_id = (item.get("source_item") or {}).get("source_id") or f"idx_{i}"

            entry = {
                "conversations": [{"from": "human", "value": conv_value}],
                "chosen":   {"from": "gpt", "value": chosen_val},
                "rejected": {"from": "gpt", "value": rejected_val},
                "meta": {
                    "benchmark_id": bench_id,
                    "rubric": rubric_code,
                    "source_id": source_id,
                    "source_file": str(path),
                }
            }
            dataset.append(entry)
            
    return dataset

def main():
    ap = argparse.ArgumentParser(description="Build CPG dataset from combined stem/violation data.")
    ap.add_argument("--input-dir", default="data_store/raw_outputs/2025-08-23", help="Directory containing combined JSON files.")
    ap.add_argument("--output-file", default="data_store/rm_pair/cpg/l2_train.jsonl", help="Output path for the .jsonl file.")
    args = ap.parse_args()

    # 절대 경로로 변환
    input_dir_abs = Path.cwd() / args.input_dir
    output_file_abs = Path.cwd() / args.output_file

    dataset = build_l2_cpg_from_combined_data(str(input_dir_abs))

    output_file_abs.parent.mkdir(parents=True, exist_ok=True)
    with output_file_abs.open("w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ CPG 데이터 {len(dataset)}개 저장: {output_file_abs}")

if __name__ == "__main__":
    main()

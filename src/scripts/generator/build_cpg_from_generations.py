#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPG 데이터셋 생성기 (Benchmark 1 우선 지원)

- conversation.value: 루브릭별 프롬프트(YAML) 로드 후 포맷
- chosen.value: stem 생성 JSON에서 본문/지시문 파싱
- rejected.value: violation passage JSON에서 본문 파싱 (R1/R6이면 stem은 chosen 재사용)
- R3(reference_groundedness): source_item의 한국/외국 컨텍스트를 합쳐 단일 reference로 conversation에 주입

사용 예:
  python build_cpg.py --bench-id 1 --rubrics R1 R2 R3 R4 R5 R6 --chosen-date 2025-08-08 --rejected-date 2025-08-16 --model A.X-4.0-Light \
    --out data_store/cpg/benchmark_1_A.X-4.0-Light.json

프롬프트 파일:
- B1/B2(지문):        src/config/prompts/iska/preference_eval.yaml
- B3/B4(듣고 말하기): src/config/prompts/iska/preference_eval_listening.yaml
- B5(보고 말하기):    src/config/prompts/iska/preference_eval_visual.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml


# --------------------------- 유틸 공통 ---------------------------

def bench_kind(bench_id: int) -> str:
    """
    벤치마크 ID → 컨텐츠 유형
    """
    if bench_id in (1, 2):
        return "text"
    if bench_id in (3, 4):
        return "listening"
    if bench_id == 5:
        return "visual"
    raise ValueError(f"Unsupported bench id: {bench_id}")


def _extract_primary_text(row: Dict[str, Any], kind: str, *, for_rejected: bool) -> str:
    """
    chosen/rejected JSON에서 대표 텍스트를 우선순위에 따라 추출.
    kind: 'text' | 'listening' | 'visual'
    for_rejected=True면 violation 샘플에서 자주 쓰는 키를 먼저 본다.
    """
    order_map = {
        "text": (
            # rejected 우선순위
            ["generated_passage", "content", "passage"],
            # chosen 우선순위
            ["source_passage", "passage", "content"],
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
    return ""


def _render_payload(kind: str, main_text: str, stems: List[str], *, include_stems: bool) -> str:
    """
    LLM 입력/학습에 바로 쓸 수 있게 라벨을 붙여 합친 텍스트 블록 생성.
    """
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
    """
    파일이 [ ... ] 배열(json) 이든 JSONL이든 리스트[dict]로 반환.
    """
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


def unify_reference(source_item: Dict[str, Any]) -> str:
    """
    B1 특화: korean_context + foreign_context를 합쳐 단일 reference 문자열로.
    """
    if not source_item:
        return ""
    c = (source_item.get("topic") or "").strip()
    parts = []
    if c:
        parts.append(c)
    return "\n\n".join(parts).strip()


# --------------------------- 프롬프트 키/파일 ---------------------------

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

def prompt_file_for(kind: str) -> Path:
    base = Path("src/config/prompts/iska")
    if kind == "text":
        return base / "preference_eval.yaml"
    if kind == "listening":
        return base / "preference_eval_listening.yaml"
    if kind == "visual":
        return base / "preference_eval_visual.yaml"
    raise ValueError(kind)


# --------------------------- 파일 시스템 탐색 ---------------------------

def _pick_latest(paths: list[str]) -> Path:
    if not paths:
        raise FileNotFoundError("대상 파일이 없습니다.")
    # 가장 최근 수정 시간 기준 선택
    return Path(max(paths, key=os.path.getmtime))

def find_chosen_stem_file(chosen_date: str, model: str, bench_id: int) -> Path:
    """
    예시(B1):
    data_store/raw_outputs/2025-08-08/stem/A.X-4.0-Light/**/benchmark_1_v*_*.json
    """
    pat = f"data_store/raw_outputs/{chosen_date}/stem/{model}/**/benchmark_{bench_id}_v*_*.json"
    paths = glob.glob(pat, recursive=True)
    if not paths:
        raise FileNotFoundError(f"chosen stem 파일을 찾을 수 없음: {pat}")
    return _pick_latest(paths)

VIOL_KEY_BY_RUBRIC = {
    "R1": "image_agent.violate_completeness_for_guidelines_severely",
    "R2": "image_agent.violate_clarity_of_core_theme_severely",
    "R3": "image_agent.violate_reference_groundedness_severely",
    "R4": "image_agent.violate_flow_severely",
    "R5": "image_agent.violate_korean_quality_severely",
    "R6": "image_agent.violate_l2_suitability_severely",
}

def find_rejected_violation_file(rejected_date: str, model: str, bench_id: int, rubric_code: str) -> Path:
    """
    예시(B1 - passage 기준):
    data_store/raw_outputs/2025-08-16/passage/A.X-4.0-Light/passage_agent.violate_clarity_of_core_theme_severely/benchmark_1_v*_*.json
    """
    tpl = VIOL_KEY_BY_RUBRIC[rubric_code]
    pat = f"data_store/raw_outputs/{rejected_date}/image_caption/{model}/{tpl}/benchmark_{bench_id}_v*_*.json"
    paths = glob.glob(pat)
    if not paths:
        raise FileNotFoundError(f"rejected violation 파일을 찾을 수 없음: {pat}")
    return _pick_latest(paths)


# --------------------------- 빌더들 ---------------------------

def build_conversation_value(bench_id: int,
                             rubric_code: str,
                             chosen_row: Dict[str, Any]) -> str:
    """
    템플릿(YAML) 로드 → 루브릭별 프롬프트 포맷.
    - R1: problem_type/eval_goal 플레이스홀더 채움
    - R3: reference 플레이스홀더 채움
    """
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
        sd["problem_type1"] = chosen_row.get("problem_type_1", "")
        sd["problem_type2"] = chosen_row.get("problem_type_2", "")
        sd["problem_type3"] = chosen_row.get("problem_type_3", "")
        sd["eval_goal1"] = chosen_row.get("eval_goal_1", "")
        sd["eval_goal2"] = chosen_row.get("eval_goal_2", "")
        sd["eval_goal3"] = chosen_row.get("eval_goal_3", "")

    if rubric_code == "R3":
        ref = unify_reference(chosen_row.get("source_item") or {})
        sd["reference"] = ref

    return tmpl.format_map(sd).strip()


def build_chosen_value(bench_id: int, rubric_code: str, chosen_row: Dict[str, Any]) -> str:
    """
    chosen(JSON) → [지문/대화/이미지] + [문항 세트] 블록으로 변환
    """
    kind = bench_kind(bench_id)
    passage_or_script = _extract_primary_text(chosen_row, kind, for_rejected=False)
    stems = [
        chosen_row.get("stem_1", ""),
        chosen_row.get("stem_2", ""),
        chosen_row.get("stem_3", ""),
    ]
    include_stems = rubric_code in ("R1", "R6")
    if include_stems and not any(s.strip() for s in stems if isinstance(s, str)):
        print("⚠️ R1/R6인데 stem이 비어 있습니다. stems를 포함하지 않고 진행합니다.")
        include_stems = False
    return _render_payload(kind, passage_or_script, stems, include_stems=include_stems)


def build_rejected_value(bench_id: int,
                         rubric_code: str,
                         rejected_row: Dict[str, Any],
                         chosen_row: Dict[str, Any]) -> str:
    """
    rejected(JSON) → [지문/대화/이미지] (+ chosen의 stems) 블록으로 변환
    """
    kind = bench_kind(bench_id)
    text = _extract_primary_text(rejected_row, kind, for_rejected=True)
    stems = [
        chosen_row.get("stem_1", ""),
        chosen_row.get("stem_2", ""),
        chosen_row.get("stem_3", ""),
    ]
    include_stems = rubric_code in ("R1", "R6")
    if include_stems and not any(s.strip() for s in stems if isinstance(s, str)):
        print("⚠️ R1/R6인데 stem이 비어 있습니다. stems를 포함하지 않고 진행합니다.")
        include_stems = False
    return _render_payload(kind, text, stems, include_stems=include_stems)


def index_by_source_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    source_id → 행 매핑 생성 (chosen과 rejected를 매칭하기 위한 기준)
    """
    out = {}
    for r in rows:
        sid = (r.get("source_id")
               or (r.get("source_item") or {}).get("source_id")
               or "")
        if sid:
            out[sid] = r
    return out


# --------------------------- 메인 파이프라인 ---------------------------

def build_cpg(bench_id: int,
              rubric_codes: List[str],
              chosen_date: str,
              rejected_date: str,
              model: str) -> List[Dict[str, Any]]:
    """
    B1 기준:
    - chosen: stem/*/benchmark_{bench}_v*.json 에서 지문/문항/지침 추출
    - rejected: passage/<violate_key>/benchmark_{bench}_v*.json 에서 위반 지문 추출
    - source_id 기준으로 매칭
    """
    chosen_path = find_chosen_stem_file(chosen_date, model, bench_id)
    chosen_rows = load_json_any(chosen_path)
    if not chosen_rows:
        print(f"⚠️ chosen({chosen_path}) 비어있음")

    # ⚠️ 변경: 루브릭별로 rejected를 '리스트'와 '맵' 둘 다 보관
    rejected_rows_by_rubric: Dict[str, List[Dict[str, Any]]] = {}
    rejected_index_by_rubric: Dict[str, Dict[str, Dict[str, Any]]] = {}
    rejected_path_by_rubric: Dict[str, Path] = {}

    for rc in rubric_codes:
        rej_path = find_rejected_violation_file(rejected_date, model, bench_id, rc)
        rej_rows = load_json_any(rej_path)
        rejected_rows_by_rubric[rc] = rej_rows
        rejected_index_by_rubric[rc] = index_by_source_id(rej_rows)
        rejected_path_by_rubric[rc] = rej_path
        if not rej_rows:
            print(f"⚠️ rejected({rej_path}) 비어있음")

    dataset: List[Dict[str, Any]] = []

    # ⚠️ 변경: enumerate로 chosen 인덱스 보존
    for i, row in enumerate(chosen_rows):
        source_id = (row.get("source_item") or {}).get("source_id") \
                    or row.get("source_id") \
                    or None  # ⚠️ bench_{id}_item_0로 고정하지 않음 (오탐 유발 방지)

        for rc in rubric_codes:
            rej_map = rejected_index_by_rubric.get(rc, {})
            rej_list = rejected_rows_by_rubric.get(rc, [])  # 인덱스 fallback용

            # 1) source_id 매칭 시도
            rejected_row = rej_map.get(source_id) if source_id else None

            # 2) 실패 시 인덱스 정렬 fallback
            if rejected_row is None:
                if i < len(rej_list):
                    rejected_row = rej_list[i]
                else:
                    # 그래도 없으면 스킵
                    print(f"⚠️ 매칭 실패: rc={rc}, i={i}, source_id={source_id} (스킵)")
                    continue

            conv_value = build_conversation_value(bench_id, rc, row)
            chosen_val = build_chosen_value(bench_id, rc, row)
            rejected_val = build_rejected_value(bench_id, rc, rejected_row, row)

            entry = {
                "conversations": [{"from": "human", "value": conv_value}],
                "chosen":   {"from": "gpt", "value": chosen_val},
                "rejected": {"from": "gpt", "value": rejected_val},
                "meta": {
                    "benchmark_id": bench_id,
                    "rubric": rc,
                    "source_id": source_id or f"idx_{i}",  # ⚠️ 매칭 근거 기록
                    "chosen_file": str(chosen_path),
                    "rejected_file": str(rejected_path_by_rubric[rc]),
                    "match_strategy": "source_id" if source_id in rejected_index_by_rubric.get(rc, {}) else "index_fallback",
                    "rejected_index": i if source_id not in rejected_index_by_rubric.get(rc, {}) else None,
                }
            }
            dataset.append(entry)

    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-id", type=int, required=True)
    ap.add_argument("--rubrics", nargs="+", required=True, help="예: R1 R2 R3 R4 R5 R6")
    ap.add_argument("--chosen-date", required=True, help="예: 2025-08-08")
    ap.add_argument("--rejected-date", required=True, help="예: 2025-08-16")
    ap.add_argument("--model", required=True, help="예: A.X-4.0-Light")
    ap.add_argument("--out", required=True, help="출력 경로(.json)")
    args = ap.parse_args()

    dataset = build_cpg(
        bench_id=args.bench_id,
        rubric_codes=args.rubrics,
        chosen_date=args.chosen_date,
        rejected_date=args.rejected_date,
        model=args.model,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"✅ CPG 데이터 {len(dataset)}개 저장: {out_path}")


if __name__ == "__main__":
    main()

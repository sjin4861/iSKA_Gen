#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python src/scripts/build_llm_pairs_from_evals.py --base /home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/evaluations/l2_learner_suitability --out /home/sjin4861/25-1/HCLT/iSKA_Gen/llm_pairs_l2_learner_suitability.csv

l2_learner_suitability 평가 JSONL들에서 모델 간 점수 차가 나는 아이템을 추출해
CSV(벤치마크 id, 아이템 id, 루브릭 이름, chosen content, chosen stems, chosen 모델명,
    rejected content, rejected stems, rejected 모델명)로 저장.

- 디렉터리 구조(예시):
  /home/.../data_store/evaluations/l2_learner_suitability/{content_type}/llm/
    eval_20250818_110146_b3_A.X-4.0-Light/
      20250818_benchmarkunknown_benchmark_EXAONE-4.0-32B.jsonl
    eval_20250818_110146_b3_llama3.1_korean_v1.1_sft_by_aidx/
      ...

- 벤치마크 id는 상위 eval 폴더명에서 `_b{num}_` 패턴으로 추출 (예: b3 → 3)
- 아이템 id는 JSONL 파일의 라인 인덱스(0 시작)
- 모델명은 상위 eval 폴더명의 `_b{num}_` 뒤에 오는 전체 문자열 (예: A.X-4.0-Light)
- 채점 LLM(EXAONE-4.0-32B 등)은 파일명에 있으나 **무시**

- llm_score = notes 문자열 맨 앞 정수 (예: "3  \n..." -> 3)
  * 없거나 파싱 실패 시: score.value가 bool이면 1/0으로 사용
"""

from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple

BASE_DIR_DEFAULT = "/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/evaluations/l2_learner_suitability"
CONTENT_TYPES = ["passage", "audio_script", "image_caption"]

# 경로 파싱용 정규식: .../eval_YYYYMMDD_..._b{num}_{model}/
EVAL_DIR_RE = re.compile(r"eval_[^/]*_b(\d+)_([^/]+)$")
LEADING_INT_RE = re.compile(r"^\s*(-?\d+)")
RUBRIC_NAME = "l2_learner_suitability"

def parse_llm_score(obj: Dict[str, Any]) -> int:
    notes = obj.get("notes", "")
    if isinstance(notes, (int, float)):
        return int(notes)
    if isinstance(notes, str):
        m = LEADING_INT_RE.match(notes)
        if m:
            return int(m.group(1))
    score = obj.get("score", {})
    val = score.get("value")
    if isinstance(val, bool):
        return 1 if val else 0
    try:
        return int(val)
    except Exception:
        return 0

def collect_records(base_dir: Path) -> Dict[str, Dict[int, Dict[int, Dict[str, Dict[str, Any]]]]]:
    """
    data[content_type][benchmark_id][item_id][model_name] = {
        "content": str,
        "stems": list,
        "llm_score": int
    }
    """
    data: Dict[str, Dict[int, Dict[int, Dict[str, Dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )

    for ctype in CONTENT_TYPES:
        llm_root = base_dir / ctype / "llm"
        if not llm_root.exists():
            continue
        # eval_* directories
        for eval_dir in llm_root.glob("eval_*"):
            if not eval_dir.is_dir():
                continue
            m = EVAL_DIR_RE.search(str(eval_dir))
            if not m:
                # 스킵(벤치마크/모델명 패턴 불일치)
                continue
            benchmark_id = int(m.group(1))
            model_name = m.group(2)

            # 내부 JSONL들
            for jf in eval_dir.glob("*.jsonl"):
                with jf.open("r", encoding="utf-8") as f:
                    for idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            # 손상 라인 스킵
                            continue
                        # item_id: 파일 내 라인 인덱스
                        item_id = idx
                        tgt = obj.get("target", {}) or {}
                        content = tgt.get("content", "")
                        stems = tgt.get("stems", [])
                        score = parse_llm_score(obj)

                        # 이미 같은 (ctype,bench,item,model) 키가 있더라도, 먼저 온 걸 유지(안정성)
                        if model_name not in data[ctype][benchmark_id][item_id]:
                            data[ctype][benchmark_id][item_id][model_name] = {
                                "content": content,
                                "stems": stems,
                                "llm_score": score,
                            }
    return data

# --- 생략: 기존 import/상수/parse_llm_score/collect_records 동일 ---

def choose_pairs_all_levels(
    models_dict: Dict[str, Dict[str, Any]]
) -> List[Tuple[str, Dict[str, Any], str, Dict[str, Any]]]:
    """
    같은 아이템에 대해 점수 차가 있을 때
    (모든 상이한 점수쌍) = (각 상위 점수 그룹 × 각 하위 점수 그룹)의 모든 조합을 반환.
    반환: [(chosen_name, chosen_rec, rejected_name, rejected_rec), ...]
    """
    pairs: List[Tuple[str, Dict[str, Any], str, Dict[str, Any]]] = []
    if not models_dict or len(models_dict) < 2:
        return pairs

    # 점수별 버킷
    by_score: Dict[int, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for mname, rec in models_dict.items():
        by_score[int(rec["llm_score"])].append((mname, rec))

    if len(by_score) <= 1:
        # 전부 동일 점수면 비교쌍 없음
        return pairs

    # 점수 오름차순 정렬
    scores = sorted(by_score.keys())  # e.g., [3, 4, 5]

    # 각 점수 버킷 내 모델은 모델명 사전순으로 안정 정렬
    for s in scores:
        by_score[s] = sorted(by_score[s], key=lambda x: x[0])

    # s_hi > s_lo 인 모든 점수쌍에 대해 카르테시안 곱 생성
    # (예: 5,4,3 → (4>3), (5>3), (5>4))
    for hi_idx in range(1, len(scores)):
        s_hi = scores[hi_idx]
        for lo_idx in range(0, hi_idx):
            s_lo = scores[lo_idx]
            for (hi_name, hi_rec) in by_score[s_hi]:
                for (lo_name, lo_rec) in by_score[s_lo]:
                    pairs.append((hi_name, hi_rec, lo_name, lo_rec))
    return pairs


def build_rows(data: Dict[str, Dict[int, Dict[int, Dict[str, Dict[str, Any]]]]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for ctype in CONTENT_TYPES:
        if ctype not in data:
            continue
        for bench_id, items in sorted(data[ctype].items()):
            for item_id, models_dict in sorted(items.items()):
                pairs = choose_pairs_all_levels(models_dict)  # ← 변경
                if not pairs:
                    continue

                for chosen_name, chosen_rec, rejected_name, rejected_rec in pairs:
                    chosen_stems   = json.dumps(chosen_rec.get("stems", []), ensure_ascii=False)
                    rejected_stems = json.dumps(rejected_rec.get("stems", []), ensure_ascii=False)

                    rows.append([
                        str(bench_id),                             # 벤치마크 id
                        str(item_id),                              # 아이템 id
                        RUBRIC_NAME,                               # 루브릭 이름
                        chosen_rec.get("content", ""),             # chosen content
                        chosen_stems,                              # chosen stems
                        chosen_name,                               # chosen 모델명
                        rejected_rec.get("content", ""),           # rejected content
                        rejected_stems,                            # rejected stems
                        rejected_name,                             # rejected 모델명
                    ])
    return rows

def write_csv(rows: List[List[str]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "벤치마크 id","아이템 id","루브릭 이름",
            "chosen content","chosen stems","chosen 모델명",
            "rejected content","rejected stems","rejected 모델명"
        ])
        writer.writerows(rows)
    print(f"[완료] {len(rows)}개 쌍 저장 -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DIR_DEFAULT,
                    help="평가 데이터 루트 디렉터리 (default: %(default)s)")
    ap.add_argument("--out", default="/home/sjin4861/25-1/HCLT/iSKA_Gen/llm_pairs_l2_learner_suitability.csv",
                    help="출력 CSV 경로 (default: %(default)s)")
    args = ap.parse_args()

    base_dir = Path(args.base)
    if not base_dir.exists():
        raise FileNotFoundError(f"기준 디렉터리 없음: {base_dir}")

    data = collect_records(base_dir)
    rows = build_rows(data)
    write_csv(rows, Path(args.out))

if __name__ == "__main__":
    main()

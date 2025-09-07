#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV(벤치마크 id, 아이템 id, 루브릭 이름, chosen/rejected content & stems & 모델명) -> JSONL (prompt / chosen / rejected)

- 벤치마크 ID로 과제 유형 자동 판별:
    1,2 -> 지문(text) / 3,4 -> 대화(dialogue) / 5 -> 이미지(image)
- 유형별 프롬프트/본문 정규화:
    * 지문: [지문] 머리말 통일
    * 대화: [대화] 머리말 통일
    * 이미지: [이미지 설명], [상황 제시] 블록 정리
- [문항 세트]는 CSV의 stems(JSON)로 구성(번호 매김)

사용 예:
python src/scripts/csv_to_empg_jsonl.py \
  --csv /home/sjin4861/25-1/HCLT/iSKA_Gen/llm_pairs_l2_learner_suitability.csv \
  --out /home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/rm_pair/empg/empg_train_new.jsonl
"""

from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from typing import List

# ----------------------------- 벤치마크 → 유형 -----------------------------
BENCH_KIND = {
    1: "text",      # 지문
    2: "text",
    3: "dialogue",  # 대화
    4: "dialogue",
    5: "image",     # 이미지
}
DEFAULT_KIND = "text"

# ----------------------------- 유형별 프롬프트 -----------------------------
PROMPT_TEXT = (
    "아래 [지문]과 [문항 세트]가 주어집니다.\n\n"
    "[평가 기준]\n"
    "- (난이도) 어휘 수준·문장 길이·구문 복잡도가 과도하지 않은가?\n"
    "- (명료성) 전문 용어·암묵적 배경지식 의존을 피하고, 필요한 경우 간단한 정의/예시로 해소되는가?\n"
    "- (문항 적합성) 각 stem이 **명확하고 과도한 추론을 요구하지 않으며**, 지문 근거로 답할 수 있는가?\n\n"
    "위 기준에 따라 [지문]과 [문항 세트]를 L2 한국어 학습자를 가정하여 적절한 난이도·표현·구조인지 평가하세요."
)

PROMPT_DIALOGUE = (
    "아래 [대화]와 [문항 세트]가 주어집니다.\n\n"
    "[평가 기준]\n"
    "- (난이도) 어휘 수준·문장 길이·구문 복잡도가 과도하지 않은가? 구어체 축약·속담·은어 남용은 감점.\n"
    "- (명료성) 암묵적 배경지식 의존 없이 의미가 전달되는가? 필요한 경우 간단한 정의·예시로 해소되는가?\n"
    "- (문항 적합성) 각 stem이 **명확하고 과도한 추론을 요구하지 않으며**, 발화 근거로 답할 수 있는가?\n\n"
    "위 기준에 따라 [대화]와 [문항 세트]를 L2 한국어 학습자를 가정하여 적절한 난이도·표현·구조인지 평가하세요."
)

PROMPT_IMAGE = (
    "아래 [이미지 설명/상황 제시]와 [문항 세트]가 주어집니다.\n"
    "   [평가 기준]\n"
    "- (난이도) 어휘 수준·문장 길이·구문 복잡도가 과도하지 않은가? 관용구·은어·한자어 남용은 감점.\n"
    "- (명료성) 암묵적 배경지식 없이도 시각 단서가 이해되도록 설명되는가? 필요한 경우 간단한 정의/예시로 보완되는가?\n"
    "- (문항 적합성) 각 stem이 **명확하고 과도한 추론을 요구하지 않으며**, 텍스트 근거로 답할 수 있는가?\n\n"
    "위 기준에 따라 [이미지 설명/상황 제시]와 [문항 세트]를 L2 한국어 학습자를 가정하여 적절한 난이도·표현·구조인지 평가하세요."
)

PROMPTS = {
    "text": PROMPT_TEXT,
    "dialogue": PROMPT_DIALOGUE,
    "image": PROMPT_IMAGE,
}

# ----------------------------- 공통 유틸 -----------------------------
QUESTION_SPLIT = re.compile(r'\n\s*\[문항\s*세트\]\s*\n', re.UNICODE)

def split_before_questions(text: str) -> str:
    """본문만 반환([문항 세트] 포함 시 앞부분만)."""
    parts = QUESTION_SPLIT.split(text or "", maxsplit=1)
    return parts[0].strip()

def format_stems_block(stems_json: str) -> str:
    """CSV stems(JSON 문자열) → 번호 매김 블록."""
    if not stems_json or not stems_json.strip():
        return ""
    try:
        stems = json.loads(stems_json)
        if not isinstance(stems, list):
            return ""
    except Exception:
        return ""
    stems = [str(s).strip() for s in stems if str(s).strip()]
    if not stems:
        return ""
    return "\n".join(f"{i+1}) {s}" for i, s in enumerate(stems))

# ----------------------------- 지문/대화 정규화 -----------------------------
LEADING_TEXT_BLOCK = re.compile(
    r'^\s*\[\s*지문\s*\]\s*\n', flags=re.IGNORECASE
)
TEXT_TITLE = re.compile(
    r'^\s*(?:\*\*)?\s*지문\s*[:\-]?\s*(?:\*\*)?\s*', flags=re.IGNORECASE
)

LEADING_DIALOGUE_BLOCK = re.compile(
    r'^\s*\[\s*대화\s*\]\s*\n', flags=re.IGNORECASE
)
DIALOGUE_TITLE = re.compile(
    r'^\s*(?:\*\*)?\s*대화\s*[:\-]?\s*(?:\*\*)?\s*', flags=re.IGNORECASE
)

def normalize_passage_body(text: str) -> str:
    """[지문] 머리말 통일 + 앞쪽 군더더기 제거."""
    body = split_before_questions(text)
    # 선행 [지문] 블록 반복 제거
    while True:
        new = LEADING_TEXT_BLOCK.sub("", body).strip()
        if new == body:
            break
        body = new
    # 줄별 '지문:' 같은 타이틀 제거
    lines = [TEXT_TITLE.sub("", ln) for ln in body.splitlines()]
    core = "\n".join(lines).strip()
    return f"[지문]\n{core}" if core else "[지문]"

def normalize_dialogue_body(text: str) -> str:
    """[대화] 머리말 통일 + 앞쪽 군더더기 제거."""
    body = split_before_questions(text)
    # 선행 [대화] 블록 반복 제거
    while True:
        new = LEADING_DIALOGUE_BLOCK.sub("", body).strip()
        if new == body:
            break
        body = new
    # 줄별 '대화:' 같은 타이틀 제거
    lines = [DIALOGUE_TITLE.sub("", ln) for ln in body.splitlines()]
    core = "\n".join(lines).strip()
    return f"[대화]\n{core}" if core else "[대화]"

# ----------------------------- 이미지 정규화 -----------------------------
LEADING_IMG_BLOCK = re.compile(
    r'^\s*\[(?:이미지\s*설명(?:\s*/\s*상황\s*제시)?|상황\s*제시|문제\s*상황)\]\s*\n',
    flags=re.IGNORECASE
)

IMG_TITLE = re.compile(
    r'^\s*(?:\*\*)?\s*이미지\s*설명\s*[:\-]?\s*(?:\*\*)?\s*', flags=re.IGNORECASE
)

SITU_LINE = re.compile(
    r'^\s*(?:\[\s*문제\s*상황\s*\]|\[\s*상황\s*제시\s*\]|(?:\*\*)?\s*문제\s*상황\s*[:\-]?\s*(?:\*\*)?)\s*$',
    flags=re.IGNORECASE
)
SITU_INLINE = re.compile(
    r'^\s*(?:\[\s*문제\s*상황\s*\]|\[\s*상황\s*제시\s*\]|(?:\*\*)?\s*문제\s*상황\s*[:\-]?\s*(?:\*\*)?)\s*',
    flags=re.IGNORECASE
)

def normalize_image_body(text: str) -> str:
    """
    - 선행 [이미지 설명], [이미지 설명/상황 제시], [상황 제시], [문제 상황] 등 제거(여러 줄 반복 허용)
    - 이미지 설명 / 상황 제시 분리 및 머리말 통일
    """
    body = split_before_questions(text or "")

    # 선행 브래킷 블록 반복 제거(두 줄 연속으로 나오는 케이스 대응)
    while True:
        new = LEADING_IMG_BLOCK.sub("", body).strip()
        if new == body:
            break
        body = new

    lines = body.splitlines()

    # [상황 제시] 시작 위치 탐색
    situ_start = None
    for i, ln in enumerate(lines):
        if SITU_LINE.match(ln) or SITU_INLINE.match(ln):
            situ_start = i
            break

    # 상황 타이틀 없으면 전부 이미지 설명
    if situ_start is None:
        img_lines = [IMG_TITLE.sub("", ln) for ln in lines]
        img_text = "\n".join(img_lines).strip()
        return f"[이미지 설명]\n{img_text}" if img_text else "[이미지 설명]"

    img_part = lines[:situ_start]
    situ_part = lines[situ_start:]

    # 이미지 설명 정리
    img_part = [IMG_TITLE.sub("", ln) for ln in img_part]
    img_text = "\n".join(img_part).strip()

    # 상황 제목 제거(+ 같은 줄 본문 보존)
    if situ_part:
        first = situ_part[0]
        if SITU_LINE.match(first):
            situ_part = situ_part[1:]
        else:
            situ_part[0] = SITU_INLINE.sub("", first)
    situ_text = "\n".join(l.rstrip() for l in situ_part).strip()

    out = "[이미지 설명]"
    if img_text:
        out += "\n" + img_text
    if situ_text:
        out += "\n\n[상황 제시]\n" + situ_text
    return out.strip()

# ----------------------------- 빌더 -----------------------------
def build_example_for_kind(kind: str, content: str, stems_json: str) -> str:
    if kind == "text":
        main = normalize_passage_body(content or "")
    elif kind == "dialogue":
        main = normalize_dialogue_body(content or "")
    elif kind == "image":
        main = normalize_image_body(content or "")
    else:
        # 폴백: 지문 처리
        main = normalize_passage_body(content or "")

    stems_block = format_stems_block(stems_json)
    return f"{main}\n\n[문항 세트]\n{stems_block}" if stems_block else main

# ----------------------------- 변환 -----------------------------
REQUIRED_COLS = [
    "벤치마크 id","아이템 id","루브릭 이름",
    "chosen content","chosen stems","chosen 모델명",
    "rejected content","rejected stems","rejected 모델명"
]

def convert(csv_path: Path, out_jsonl: Path):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with csv_path.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        missing = [c for c in REQUIRED_COLS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

        for row in reader:
            try:
                bench_id = int(str(row.get("벤치마크 id","")).strip())
            except Exception:
                bench_id = None
            kind = BENCH_KIND.get(bench_id, DEFAULT_KIND)
            prompt = PROMPTS.get(kind, PROMPT_TEXT)

            chosen_text = build_example_for_kind(kind, row.get("chosen content",""), row.get("chosen stems",""))
            rejected_text = build_example_for_kind(kind, row.get("rejected content",""), row.get("rejected stems",""))

            obj = {
                "prompt": prompt,
                "chosen": chosen_text,
                "rejected": rejected_text,
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"[완료] {n}개 항목 저장 -> {out_jsonl}")

# ----------------------------- 엔트리 포인트 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="입력 CSV 경로")
    ap.add_argument("--out", required=True, help="출력 JSONL 경로")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    if not csv_path.exists():
        raise FileNotFoundError(f"입력 CSV가 없습니다: {csv_path}")

    convert(csv_path, out_path)

if __name__ == "__main__":
    main()

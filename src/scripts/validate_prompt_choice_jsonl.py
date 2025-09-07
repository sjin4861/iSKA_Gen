#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prompt/Chosen/Rejected JSONL 검증기
python src/scripts/validate_prompt_choice_jsonl.py \
  /home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/rm_pair/empg/empg_train_new.jsonl \
  --check-rejected \
  --report /home/sjin4861/25-1/HCLT/iSKA_Gen/qa_reports/empg_train_new_validation.tsv


검증 항목
1) 각 레코드가 prompt / chosen / rejected 3필드를 모두 보유하며 문자열인지
2) 프롬프트로 유형 감지:
   - passage  : prompt에 [지문] 존재 → chosen은 [지문] ... [문항 세트] 구성
   - dialogue : prompt에 [대화] 존재 → chosen은 [대화] ... [문항 세트] 구성
   - image    : prompt에 [이미지 설명/상황 제시] 존재 → chosen은
                [이미지 설명] → (옵션) [상황 제시] → [문항 세트] 순서
3) [문항 세트] 아래 항목이 최소 1개 이상이며 '^\d+\)' 번호 매김인지

옵션
- --allow-missing-situation : 이미지 유형에서 [상황 제시]가 없어도 통과
- --check-rejected          : rejected도 chosen과 동일 규칙으로 검사
- --report <path>           : 오류 리포트를 TSV로 저장
- --limit N                 : 앞 N개 레코드만 점검(샘플링)

종료코드: 오류가 있으면 1, 없으면 0
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# 유형 감지용
PROMPT_IS_PASSAGE = re.compile(r'\[지문\]')
PROMPT_IS_DIALOG  = re.compile(r'\[대화\]')
PROMPT_IS_IMAGE   = re.compile(r'\[이미지\s*설명/상황\s*제시\]')

# 섹션 헤더 패턴
HEADER_PASSAGE = re.compile(r'^\s*\[지문\]\s*$', re.MULTILINE)
HEADER_DIALOG  = re.compile(r'^\s*\[대화(?:문)?\]\s*$', re.MULTILINE)
HEADER_IMAGE   = re.compile(r'^\s*\[이미지\s*설명\]\s*$', re.MULTILINE)
HEADER_SITU    = re.compile(r'^\s*\[상황\s*제시\]\s*$', re.MULTILINE)
HEADER_QSET    = re.compile(r'^\s*\[문항\s*세트\]\s*$', re.MULTILINE)

# 문항 번호 매김 (1) 또는 1) 형태 모두 지원 → 기본은 1) 권장
QITEM_LINE = re.compile(r'^\s*\d+\)\s+')

def detect_type(prompt: str) -> str:
    """prompt 내용으로 유형 추론: passage|dialogue|image|unknown"""
    if PROMPT_IS_IMAGE.search(prompt):
        return "image"
    if PROMPT_IS_DIALOG.search(prompt):
        return "dialogue"
    if PROMPT_IS_PASSAGE.search(prompt):
        return "passage"
    return "unknown"

def split_sections(text: str) -> Dict[str, Tuple[int, int]]:
    """
    본문에서 주요 섹션 헤더들의 '시작 위치'를 반환.
    섹션 존재 여부와 순서 검사에 사용.
    """
    found = {}
    for label, pat in [
        ("passage", HEADER_PASSAGE),
        ("dialogue", HEADER_DIALOG),
        ("image", HEADER_IMAGE),
        ("situation", HEADER_SITU),
        ("qset", HEADER_QSET),
    ]:
        m = pat.search(text)
        if m:
            found[label] = (m.start(), m.end())
    return found

def extract_qset_block(text: str) -> str:
    """[문항 세트] 이후 텍스트 블록 추출 (없으면 빈 문자열)"""
    m = HEADER_QSET.search(text)
    if not m:
        return ""
    return text[m.end():].strip()

def validate_qset_items(qset_text: str) -> List[str]:
    errs: List[str] = []
    if not qset_text:
        errs.append("문항 세트 본문이 비어 있음")
        return errs
    lines = [ln for ln in qset_text.splitlines() if ln.strip()]
    # 최소 1개
    if not lines:
        errs.append("문항 세트 항목이 없음")
        return errs
    # 번호 매김 확인
    bad = [i for i, ln in enumerate(lines) if not QITEM_LINE.match(ln)]
    if bad:
        errs.append(f"문항 세트 번호 매김 형식 아님: 라인 {', '.join(str(i+1) for i in bad[:5])} ...")
    return errs

def validate_by_type(chosen_text: str, content_type: str, allow_missing_situ: bool) -> List[str]:
    errs: List[str] = []
    sec = split_sections(chosen_text)

    if content_type == "passage":
        if "passage" not in sec:
            errs.append("[지문] 헤더 없음")
        if "qset" not in sec:
            errs.append("[문항 세트] 헤더 없음")
        # 순서: [지문] < [문항 세트]
        if "passage" in sec and "qset" in sec and sec["passage"][0] > sec["qset"][0]:
            errs.append("[지문]이 [문항 세트] 뒤에 옴(순서 오류)")
        qerrs = validate_qset_items(extract_qset_block(chosen_text)) if "qset" in sec else ["[문항 세트] 없음으로 문항 검증 불가"]
        errs.extend(qerrs)

    elif content_type == "dialogue":
        if "dialogue" not in sec:
            errs.append("[대화] 헤더 없음")
        if "qset" not in sec:
            errs.append("[문항 세트] 헤더 없음")
        if "dialogue" in sec and "qset" in sec and sec["dialogue"][0] > sec["qset"][0]:
            errs.append("[대화]가 [문항 세트] 뒤에 옴(순서 오류)")
        qerrs = validate_qset_items(extract_qset_block(chosen_text)) if "qset" in sec else ["[문항 세트] 없음으로 문항 검증 불가"]
        errs.extend(qerrs)

    elif content_type == "image":
        if "image" not in sec:
            errs.append("[이미지 설명] 헤더 없음")
        if not allow_missing_situ and "situation" not in sec:
            errs.append("[상황 제시] 헤더 없음")
        if "qset" not in sec:
            errs.append("[문항 세트] 헤더 없음")
        # 순서: [이미지 설명] < (상황 제시) < [문항 세트]
        if "image" in sec and "qset" in sec and sec["image"][0] > sec["qset"][0]:
            errs.append("[이미지 설명]이 [문항 세트] 뒤에 옴(순서 오류)")
        if "image" in sec and "situation" in sec and sec["image"][0] > sec["situation"][0]:
            errs.append("[이미지 설명]이 [상황 제시] 뒤에 옴(순서 오류)")
        if "situation" in sec and "qset" in sec and sec["situation"][0] > sec["qset"][0]:
            # OK
            pass
        elif "situation" in sec and "qset" in sec and sec["situation"][0] > sec["qset"][0]:
            errs.append("[상황 제시]가 [문항 세트] 뒤에 옴(순서 오류)")
        # 문항 세트 항목
        qerrs = validate_qset_items(extract_qset_block(chosen_text)) if "qset" in sec else ["[문항 세트] 없음으로 문항 검증 불가"]
        errs.extend(qerrs)

    else:
        errs.append("프롬프트로 유형 감지 실패(unknown)")

    return errs

def validate_record(obj: Dict, idx: int, allow_missing_situ: bool, check_rejected: bool) -> List[str]:
    errs: List[str] = []
    # 1) 기본 키 검사
    for k in ("prompt", "chosen", "rejected"):
        if k not in obj:
            errs.append(f"{idx}: 필드 누락: {k}")
            return errs  # 핵심 필드가 빠지면 나머지는 스킵
        if not isinstance(obj[k], str):
            errs.append(f"{idx}: {k} 타입이 문자열이 아님")
            return errs

    # 2) 유형 감지
    ctype = detect_type(obj["prompt"])
    if ctype == "unknown":
        errs.append(f"{idx}: 프롬프트 유형 감지 실패")
        return errs

    # 3) chosen 검사
    chosen_errs = validate_by_type(obj["chosen"], ctype, allow_missing_situ)
    errs.extend([f"{idx}: {e}" for e in chosen_errs])

    # 4) rejected 옵션 검사
    if check_rejected:
        rej_errs = validate_by_type(obj["rejected"], ctype, allow_missing_situ)
        errs.extend([f"{idx}: (rejected) {e}" for e in rej_errs])

    return errs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="검증할 JSONL 경로")
    ap.add_argument("--allow-missing-situation", action="store_true",
                    help="이미지 유형에서 [상황 제시]가 없어도 통과")
    ap.add_argument("--check-rejected", action="store_true",
                    help="rejected도 chosen과 동일 규칙으로 검사")
    ap.add_argument("--report", help="오류 리포트 TSV 저장 경로")
    ap.add_argument("--limit", type=int, default=0, help="앞 N개 레코드만 점검")
    args = ap.parse_args()

    src = Path(args.jsonl)
    if not src.exists():
        print(f"[에러] 파일이 존재하지 않습니다: {src}", file=sys.stderr)
        sys.exit(2)

    all_errs: List[str] = []
    checked = 0
    with src.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if args.limit and checked >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                all_errs.append(f"{idx}: JSON 파싱 실패: {e}")
                continue
            errs = validate_record(obj, idx, args.allow_missing_situation, args.check_rejected)
            all_errs.extend(errs)
            checked += 1

    # 결과 출력
    if all_errs:
        print("[검증 실패] 총 오류:", len(all_errs))
        for e in all_errs[:200]:
            print(" -", e)
        if len(all_errs) > 200:
            print(f" ... (총 {len(all_errs)}건, 상위 200건만 표시)")
        if args.report:
            rpt = Path(args.report)
            rpt.parent.mkdir(parents=True, exist_ok=True)
            with rpt.open("w", encoding="utf-8") as out:
                out.write("index\treason\n")
                for e in all_errs:
                    # "idx: message" 포맷을 분리
                    if ": " in e:
                        idx_str, msg = e.split(": ", 1)
                        out.write(f"{idx_str}\t{msg}\n")
                    else:
                        out.write(f"\t{e}\n")
            print(f"[리포트 저장] {rpt}")
        sys.exit(1)
    else:
        print(f"[검증 통과] 총 {checked}개 레코드 이상 없음 ✅")
        sys.exit(0)

if __name__ == "__main__":
    main()

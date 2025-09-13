#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.modules.model_client import get_vllm_chat
from src.modules.evaluation_chain import build_evaluation_chain
from src.modules.rubric_prompts import RUBRIC_DESCRIPTIONS, resolve_rubrics

# ----------------------- Utils -----------------------

from typing import Optional

SPECS_DIR = Path("specifications")

def _spec_filename(spec_type: str) -> str:
    # small/train/test 대응
    return f"iSKA-Gen_Spec_v1.1.0_{spec_type}.json"

def load_reference_from_spec(sample: Dict[str, Any]) -> str:
    """spec 파일에서 bench_id에 해당하는 items를 합쳐 reference 텍스트 생성."""
    spec_type = (sample.get("spec_type") or "small").strip()
    bench_id = int(sample.get("bench_id", 0))
    spec_path = SPECS_DIR / _spec_filename(spec_type)
    if not spec_path.exists():
        return ""

    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    # data = list[benchmarks]
    target = next((b for b in data if int(b.get("id", -1)) == bench_id), None)
    if not target:
        return ""

    # items 배열에서 context류를 모아 참고자료 문자열 구성
    parts: list[str] = []
    for it in target.get("items", []):
        # 한국어/외국어 topic/context 등을 가능한 한 붙여줌
        kt, kc = it.get("korean_topic"), it.get("korean_context")
        ft, fc = it.get("foreign_topic"), it.get("foreign_context")
        if kt: parts.append(f"[주제] {kt}")
        if kc: parts.append(f"[지문] {kc}")
        if ft: parts.append(f"[Foreign Topic] {ft}")
        if fc: parts.append(f"[Reference] {fc}")
        # 필요하면 다른 필드도 추가 가능
    return "\n\n".join(p for p in parts if p)

def load_single_sample(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        line = f.readline().strip()
        if not line:
            raise ValueError("빈 파일 혹은 첫 줄 없음")
        return json.loads(line)

def parse_strict_score(output: str) -> Tuple[Any, str]:
    text = (output or "").strip()
    up = text.upper()
    if up.startswith("PASS"):
        just = text[len("PASS"):].strip(" :\t")
        return 1, just
    if up.startswith("FAIL"):
        just = text[len("FAIL"):].strip(" :\t")
        return 0, just
    if text and text[0].isdigit() and text[0] in "12345":
        score = int(text[0])
        just = text[1:].strip(" :\t")
        return score, just
    return None, text

# ----------------------- Core -----------------------

def evaluate(args):
    input_path = Path(args.input)
    sample = load_single_sample(input_path)

    stems: List[Dict[str, Any]] = sample.get('stems') or []
    if not stems:
        print('⚠️ stems 비어있음')
        return 1

    # top-k
    if args.top_k_stems and args.top_k_stems < len(stems):
        stems = stems[:args.top_k_stems]

    # rubrics 해석 ("all" 지원)
    rubrics = resolve_rubrics(args.rubrics)
    print(f"✅ 평가 루브릭: {rubrics}")

    # 평가용 LLM (EXAONE @ 8001)
    eval_llm = get_vllm_chat(
        model_name=args.eval_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.eval_base_url,
        api_key=args.eval_api_key,
    )

    chain_cache: Dict[str, Any] = {}
    def get_chain_for_rubric(rubric_id: str):
        if rubric_id not in chain_cache:
            chain_cache[rubric_id] = build_evaluation_chain(rubric_id)
        return chain_cache[rubric_id]

    full_passage = sample.get('passage', '') or ''
    clip = full_passage[:300] if (args.include_passage and full_passage) else None
    reference_text = load_reference_from_spec(sample)  # ← spec에서 참고자료 로드

    out_records: List[Dict[str, Any]] = []

    # --------- R1: 샘플 단위 (지문 + 3 stem 동시) ---------
    if "completeness_for_guidelines" in rubrics:
        if len(stems) >= 3:
            r = "completeness_for_guidelines"
            chain = get_chain_for_rubric(r)

            s1, s2, s3 = stems[0], stems[1], stems[2]
            vars = {
                "llm": eval_llm,
                "passage": full_passage,  # R1은 전체 지문
                "stem1": s1.get("stem", ""),
                "problem_type1": s1.get("problem_type", ""),
                "eval_goal1": s1.get("eval_goal", ""),
                "stem2": s2.get("stem", ""),
                "problem_type2": s2.get("problem_type", ""),
                "eval_goal2": s2.get("eval_goal", ""),
                "stem3": s3.get("stem", ""),
                "problem_type3": s3.get("problem_type", ""),
                "eval_goal3": s3.get("eval_goal", ""),
            }
            raw_out = chain.invoke(vars)
            score, just = parse_strict_score(raw_out)
            out_records.append({
                'sample_path': str(input_path),
                'bench_id': sample.get('bench_id'),
                'spec_type': sample.get('spec_type'),
                'stem_index': -1,
                'rubric_id': r,
                'score': score,
                'justification': just,
                'answer_raw': raw_out,
                'stem_text': None,
                'model': args.eval_model,
                'rubric_desc': RUBRIC_DESCRIPTIONS.get(r, ''),
                'timestamp': datetime.utcnow().isoformat(),
                'passage_length': sample.get('passage_length'),
                'included_passage': False,
                'sample_level': True,
                'used_stem_indices': [0, 1, 2],
            })
            print(f"[sample] {r} => {score} | {str(just)[:60]}")
        else:
            print("⚠️ R1(completeness_for_guidelines) 건너뜀: stems < 3")

    # --------- R2~R6: 샘플 단위 (지문만) 각 한 번씩 ---------
    for r in [x for x in rubrics if x != "completeness_for_guidelines"]:
        chain = get_chain_for_rubric(r)
        vars = {"llm": eval_llm, "passage": full_passage}
        if r == "reference_groundedness":
            # 참고 자료 전달 (없으면 빈 문자열)
            vars["reference"] = reference_text

        raw_out = chain.invoke(vars)
        score, just = parse_strict_score(raw_out)
        out_records.append({
            'sample_path': str(input_path),
            'bench_id': sample.get('bench_id'),
            'spec_type': sample.get('spec_type'),
            'stem_index': -1,                 # 샘플 단위
            'rubric_id': r,
            'score': score,
            'justification': just,
            'answer_raw': raw_out,
            'stem_text': None,
            'model': args.eval_model,
            'rubric_desc': RUBRIC_DESCRIPTIONS.get(r, ''),
            'timestamp': datetime.utcnow().isoformat(),
            'passage_length': sample.get('passage_length'),
            'included_passage': False,
            'sample_level': True,
        })
        print(f"[sample] {r} => {score} | {str(just)[:60]}")

    if args.output:
        out_path = Path(args.output)
    else:
        # input.jsonl 옆에 *_eval.jsonl 저장
        in_path = Path(args.input)
        out_path = in_path.with_name(in_path.stem + "_eval.jsonl")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', encoding='utf-8') as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"💾 저장 완료: {out_path} ({len(out_records)} lines)")
    return 0

# ----------------------- CLI -----------------------

def parse_args():
    p = argparse.ArgumentParser(description='단일 sample_* stems 평가')
    p.add_argument('--input', required=True, help='sample_*.jsonl 파일 경로')
    p.add_argument('--eval-model', required=True, help='평가 모델명 (예: EXAONE-4.0-32B)')
    p.add_argument('--rubrics', nargs='+', default=['l2_learner_suitability'],
                   help='루브릭 ID 목록 또는 "all"')
    p.add_argument('--top-k-stems', type=int)
    p.add_argument('--temperature', type=float, default=0.1)
    p.add_argument('--output', default=None,
                help='평가 결과 저장 경로 (기본: 입력 파일 옆 *_eval.jsonl)')
    p.add_argument('--include-passage', action='store_true')
    p.add_argument('--max-tokens', dest='max_tokens', type=int, help='vLLM max_tokens (선택)')
    p.add_argument('--eval-base-url', default='http://localhost:8001/v1',
                   help='평가용 vLLM base URL (기본: http://localhost:8001/v1)')
    p.add_argument('--eval-api-key', default=None,
                   help='평가용 vLLM API Key (미지정 시 VLLM_API_KEY 사용)')
    return p.parse_args()

def main():
    args = parse_args()
    return evaluate(args)

if __name__ == '__main__':
    raise SystemExit(main())

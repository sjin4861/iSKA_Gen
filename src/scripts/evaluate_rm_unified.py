#!/usr/bin/env python3
"""
통합된 Reward Model 평가 스크립트

기능:
1. 단일 JSONL 파일 스코어링 (--mode score)
2. 데이터셋 정확도 평가 (--mode evaluate)
3. 디렉토리 내 모든 JSONL 파일 스코어링 (--mode score-dir)

사용법:
    python evaluate_rm_unified.py --mode score --input path/to/file.jsonl
    python evaluate_rm_unified.py --mode evaluate --input path/to/dataset.jsonl
    python evaluate_rm_unified.py --mode score-dir --input path/to/directory/
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datetime import datetime
from pathlib import Path
import re
import random
import numpy as np
import sys

# --- 랜덤 시드 고정 ---
def set_seed(seed=42):
    """재현 가능한 결과를 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- 프로젝트 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- 기본 설정 ---
DEFAULT_BASE_MODEL = "K-intelligence/Midm-2.0-Mini-Instruct"
DEFAULT_ADAPTER_PATH = "./saves/l2_v3_5ep/checkpoint-445"  # 기본값
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 모델 로딩 및 스코어링 =================

def load_and_merge_model(base_path: str, adapter_path: str):
    """베이스 모델에 어댑터를 로드하고 병합하여 최종 모델을 반환합니다."""
    print(f"\n🔄 '{adapter_path}'에서 모델 로딩 및 병합 시작...")
    
    # 1. 베이스 모델 로드
    model = AutoModelForSequenceClassification.from_pretrained(
        base_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 2. LoRA 어댑터 적용
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # 3. 어댑터를 베이스 모델에 완전히 병합
    model = model.merge_and_unload()
    print("  - ✅ 모델 병합 완료!")
    
    # 4. 토크나이저 로드 및 pad_token 설정
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        
    model.to(device).eval()
    return model, tokenizer

def get_score(prompt: str, response: str, model, tokenizer, max_length: int = 2048):
    """주어진 프롬프트와 응답으로 점수를 계산합니다."""
    # 훈련 시 사용한 f-string 템플릿과 동일하게 구성
    full_text = (
        f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    )
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    inputs.pop("token_type_ids", None)
    
    with torch.no_grad():
        return model(**inputs).logits[0].item()

# ================= 루브릭 자동 감지 =================

_RUBRIC_HEADER_RE = re.compile(r"#\s*R\s*([1-6])\b", re.IGNORECASE)
# 키워드 힌트(헤더 없을 때 대비)
_RUBRIC_HINTS = [
    ("R1", ["지침 완전성", "completeness", "평가 지침"]),
    ("R2", ["핵심 주제", "core theme", "주제가 명확", "일관"]),
    ("R3", ["참고 자료", "grounded", "자료 기반", "reference"]),
    ("R4", ["논리적", "flow", "전개", "연결"]),
    ("R5", ["한국어 품질", "문법", "어휘", "표현", "korean quality"]),
    ("R6", ["L2", "학습자", "초급", "난이도", "suitability"]),
]

def detect_rubric(prompt: str) -> str:
    """
    프롬프트에 '# R{n}' 표기가 있으면 그대로 사용.
    없으면 힌트 키워드로 추정(여러 개 일치 시 우선순위 상위 반환).
    실패 시 'R?' 반환.
    """
    m = _RUBRIC_HEADER_RE.search(prompt or "")
    if m:
        return f"R{m.group(1)}"
    p = (prompt or "").lower()
    for code, hints in _RUBRIC_HINTS:
        for h in hints:
            if h.lower() in p:
                return code
    return "R?"

# ================= 데이터 처리 함수 =================

def _iter_jsonl(path: Path):
    """JSONL 파일을 라인별로 읽어 파싱된 객체를 반환"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _write_jsonl(path: Path, rows):
    """객체 리스트를 JSONL 파일로 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

def parse_chosen_response(chosen_full: str):
    """chosen 응답을 passage와 questions로 분리"""
    parts = chosen_full.split('[문항 세트]', 1)
    if len(parts) == 2:
        passage = parts[0].strip()
        questions = '[문항 세트]' + parts[1]
        return passage, questions.strip()
    else:
        # 분리 실패 시 전체를 응답으로 간주
        return None, chosen_full

# ================= 메인 기능 함수들 =================

def score_single_file(input_file: Path, model, tokenizer, base_model: str, adapter_path: str, 
                     default_rubric: str = None) -> Path:
    """
    단일 JSONL 파일을 스코어링하여 새로운 파일로 저장
    
    Args:
        input_file: 입력 JSONL 파일 경로
        model: 로드된 모델
        tokenizer: 토크나이저
        base_model: 베이스 모델 이름
        adapter_path: 어댑터 경로
        default_rubric: 기본 루브릭 (None이면 자동 감지)
    
    Returns:
        저장된 파일 경로
    """
    print(f"\n📄 스코어링 시작: {input_file.name}")
    scored_rows = []
    total_items = sum(1 for _ in _iter_jsonl(input_file))

    for idx, row in enumerate(_iter_jsonl(input_file)):
        print(f"  - 처리 중... {idx + 1}/{total_items}", end='\r')
        
        instruction = row.get("prompt", "")
        chosen_full = row.get("chosen", "")

        # chosen을 passage와 questions로 분리
        passage, questions = parse_chosen_response(chosen_full)
        
        if passage is not None:
            prompt_for_rm = f"{instruction}\n\n{passage}"
            response_for_rm = questions
        else:
            prompt_for_rm = instruction
            response_for_rm = chosen_full

        # 점수 계산
        score = get_score(prompt_for_rm, response_for_rm, model, tokenizer)

        # 루브릭 결정
        if default_rubric:
            row_rubric = default_rubric
        else:
            row_rubric = (
                row.get("rubric")
                or (row.get("meta") or {}).get("rubric")
                or detect_rubric(instruction)
            )

        # 결과 키 추가
        row["rm_score"] = score
        row["rubric"] = row_rubric
        row["scored_by"] = {
            "base_model": base_model,
            "adapter": adapter_path,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        scored_rows.append(row)

    # 출력 파일 생성
    out_file = input_file.with_suffix("").with_suffix(".scored.jsonl")
    _write_jsonl(out_file, scored_rows)
    print(f"\n✅ 저장 완료: {out_file} (총 {len(scored_rows)}개)")
    return out_file

def evaluate_dataset_accuracy(data_path: Path, model, tokenizer) -> tuple:
    """
    chosen/rejected 쌍이 있는 데이터셋으로 모델 정확도 평가
    
    Returns:
        (accuracy, total_count, detailed_results)
    """
    print(f"\n📊 정확도 평가 시작: {data_path.name}")
    
    correct_predictions = 0
    total_predictions = 0
    detailed_results = []

    for i, data in enumerate(_iter_jsonl(data_path)):
        instruction = data["prompt"]
        chosen_full = data["chosen"]
        rejected_full = data["rejected"]

        # Chosen 처리
        chosen_passage, chosen_questions = parse_chosen_response(chosen_full)
        if chosen_passage is not None:
            prompt_for_chosen = f"{instruction}\n\n{chosen_passage}"
            response_for_chosen = chosen_questions
        else:
            prompt_for_chosen = instruction
            response_for_chosen = chosen_full

        # Rejected 처리
        rejected_passage, rejected_questions = parse_chosen_response(rejected_full)
        if rejected_passage is not None:
            prompt_for_rejected = f"{instruction}\n\n{rejected_passage}"
            response_for_rejected = rejected_questions
        else:
            prompt_for_rejected = instruction
            response_for_rejected = rejected_full
        
        score_chosen = get_score(prompt_for_chosen, response_for_chosen, model, tokenizer)
        score_rejected = get_score(prompt_for_rejected, response_for_rejected, model, tokenizer)
        
        score_diff = score_chosen - score_rejected
        prediction = "Correct" if score_diff > 0 else "Incorrect"
        
        detailed_results.append({
            "pair_id": i + 1,
            "chosen_score": score_chosen,
            "rejected_score": score_rejected,
            "score_difference": score_diff,
            "prediction": prediction,
            "rubric": detect_rubric(instruction)
        })
        
        if score_chosen > score_rejected:
            correct_predictions += 1
        total_predictions += 1
        
        if (i + 1) % 10 == 0:
            print(f"  - 처리 중... {i + 1}", end='\r')
    
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"\n✅ 평가 완료: 정확도 {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    
    return accuracy, total_predictions, detailed_results

def score_directory(input_dir: Path, model, tokenizer, base_model: str, adapter_path: str):
    """디렉토리 내 모든 JSONL 파일을 스코어링"""
    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"⚠️ 디렉터리 내 *.jsonl 파일이 없습니다: {input_dir}")
        return
    
    print(f"\n📁 디렉터리 스코어링 시작: {input_dir}")
    print(f"   찾은 파일: {len(jsonl_files)}개")
    
    for file_path in jsonl_files:
        score_single_file(file_path, model, tokenizer, base_model, adapter_path)

# ================= 메인 실행 부분 =================

def create_parser():
    """명령줄 인자 파서 생성"""
    parser = argparse.ArgumentParser(
        description="통합된 Reward Model 평가 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 파일 스코어링
  python evaluate_rm_unified.py --mode score --input data/test.jsonl
  
  # 정확도 평가 (chosen/rejected 쌍 필요)
  python evaluate_rm_unified.py --mode evaluate --input data/eval.jsonl
  
  # 디렉터리 내 모든 JSONL 파일 스코어링
  python evaluate_rm_unified.py --mode score-dir --input data/
  
  # 사용자 정의 모델/어댑터 경로
  python evaluate_rm_unified.py --mode score --input data/test.jsonl \\
    --base-model "custom/model" --adapter-path "./custom/adapter"
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["score", "evaluate", "score-dir"],
        required=True,
        help="실행 모드: score(단일파일 스코어링), evaluate(정확도 평가), score-dir(디렉터리 스코어링)"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="입력 파일 또는 디렉터리 경로"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"베이스 모델 경로 (기본값: {DEFAULT_BASE_MODEL})"
    )
    
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help=f"어댑터 경로 (기본값: {DEFAULT_ADAPTER_PATH})"
    )
    
    parser.add_argument(
        "--rubric",
        type=str,
        help="기본 루브릭 설정 (예: R6). 지정하지 않으면 자동 감지"
    )
    
    return parser

def main():
    """메인 실행 함수"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 입력 경로 검증
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 오류: 입력 경로를 찾을 수 없습니다: {input_path}")
        return 1
    
    # 모드별 경로 타입 검증
    if args.mode == "score-dir" and not input_path.is_dir():
        print(f"❌ 오류: score-dir 모드는 디렉터리 경로가 필요합니다: {input_path}")
        return 1
    elif args.mode in ["score", "evaluate"] and not input_path.is_file():
        print(f"❌ 오류: {args.mode} 모드는 파일 경로가 필요합니다: {input_path}")
        return 1
    
    print(f"🚀 시작: {args.mode} 모드")
    print(f"   입력: {input_path}")
    print(f"   베이스 모델: {args.base_model}")
    print(f"   어댑터: {args.adapter_path}")
    print(f"   장치: {device}")
    
    try:
        # 모델 로드
        model, tokenizer = load_and_merge_model(args.base_model, args.adapter_path)
        
        # 모드별 실행
        if args.mode == "score":
            score_single_file(input_path, model, tokenizer, args.base_model, args.adapter_path, args.rubric)
            
        elif args.mode == "evaluate":
            accuracy, total, results = evaluate_dataset_accuracy(input_path, model, tokenizer)
            
            # 결과 요약 출력
            print("\n" + "="*50)
            print("🏆 평가 결과 요약")
            print("="*50)
            print(f"총 샘플 수: {total}")
            print(f"정확도: {accuracy:.2f}%")
            
            if results:
                avg_diff = sum(r['score_difference'] for r in results) / len(results)
                print(f"평균 점수 차이: {avg_diff:.4f}")
                
                # 가장 구별하기 어려운 샘플 출력
                sorted_results = sorted(results, key=lambda x: x['score_difference'])
                print(f"\n📉 구별하기 어려운 샘플 (Top 3):")
                for result in sorted_results[:3]:
                    print(f"  Pair #{result['pair_id']}: Diff={result['score_difference']:.4f} -> {result['prediction']}")
            
        elif args.mode == "score-dir":
            score_directory(input_path, model, tokenizer, args.base_model, args.adapter_path)
        
        print("\n" + "=" * 50)
        print("🏁 모든 작업이 성공적으로 완료되었습니다.")
        print("=" * 50)
        return 0
        
    except Exception as e:
        print(f"\n❌ 스크립트 실행 중 오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

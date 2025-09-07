from datetime import datetime
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import yaml
import sys
import random
import numpy as np
from pathlib import Path
import re

# --- 랜덤 시드 고정 ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
# --- 1. 프로젝트 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- 2. 모듈 임포트 ---
from src.model_loader import load_model_for_reward_training

BASE_MODEL = "K-intelligence/Midm-2.0-Mini-Instruct"
ADAPTER_PATH = "./saves/l2/checkpoint TODO" # 훈련 결과 경로
# TRAIN_DATA_PATH = "saves/l2/l2_rm_train.jsonl"
# EVAL_DATA_PATH = "saves/l2/l2_rm_eval.jsonl"
TEST_DATA_PATH = Path("saves/l2/l2_rm_test.jsonl")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
def load_and_merge_model(base_path, adapter_path):
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

def get_score(prompt: str, response: str, model, tokenizer):
    """주어진 프롬프트와 응답으로 점수를 계산합니다."""
    # 훈련 시 사용한 f-string 템플릿과 동일하게 구성
    full_text = (
        f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    )
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    inputs.pop("token_type_ids", None)
    
    with torch.no_grad():
        return model(**inputs).logits[0].item()

# -------- 루브릭 자동 감지(프롬프트에서) --------
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


# ================= JSONL 스코어링 & 저장 =================
def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

def score_jsonl_file(in_file: Path, model, tokenizer) -> Path:
    """
    입력 JSONL (각 라인: {prompt, chosen, ...})을 읽어
    rm_score/rubric/scored_by를 추가한 *.scored.jsonl 로 저장.
    루브릭은 프롬프트 내용 기반으로 자동 감지.
    """
    print(f"📄 스코어링: {in_file}")
    scored_rows = []

    for idx, row in enumerate(_iter_jsonl(in_file)):
        instruction = row.get("prompt", "")
        chosen_full = row.get("chosen", "")

        # [수정] chosen_full을 passage와 questions로 분리
        parts = chosen_full.split('[문항 세트]', 1)
        if len(parts) == 2:
            passage = parts[0].strip()
            questions = '[문항 세트]' + parts[1]
            # RM을 위한 프롬프트와 응답 재구성
            prompt_for_rm = f"{instruction}\n\n{passage}"
            response_for_rm = questions.strip()
        else:
            # 분리 실패 시 기존 방식대로 처리
            prompt_for_rm = instruction
            response_for_rm = chosen_full

        # 점수 계산
        score = get_score(prompt_for_rm, response_for_rm, model, tokenizer)

        # ❶ 루브릭: 이미 있으면 존중, 없으면 프롬프트에서 자동 감지
        row_rubric = (
            row.get("rubric")
            or (row.get("meta") or {}).get("rubric")
            or detect_rubric(instruction)
        )

        # ❷ 결과 키 추가
        row["rm_score"] = score
        row["rubric"] = row_rubric
        row["scored_by"] = {
            "base_model": BASE_MODEL,
            "adapter": ADAPTER_PATH,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        scored_rows.append(row)

    out_file = in_file.with_suffix("").with_suffix(".scored.jsonl")
    _write_jsonl(out_file, scored_rows)
    print(f"✅ 저장 완료: {out_file} (총 {len(scored_rows)}개)")
    return out_file


def score_path(test_path: Path, model, tokenizer):
    """
    test_path가 파일이면 그 파일 하나를, 디렉터리면 하위의 *.jsonl 전부 스코어링.
    """
    if test_path.is_dir():
        files = sorted(test_path.glob("*.jsonl"))
        if not files:
            print(f"⚠️ 디렉터리 내 *.jsonl 없음: {test_path}")
            return
        for f in files:
            score_jsonl_file(f, model, tokenizer)
    else:
        if not test_path.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없음: {test_path}")
        score_jsonl_file(test_path, model, tokenizer)


def evaluate_dataset(data_path: str, model, tokenizer):
    """주어진 데이터셋 파일로 모델의 정확도를 평가합니다."""
    correct_predictions = 0
    total_predictions = 0
    detailed_results = [] # 결과를 저장할 리스트

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            instruction = data["prompt"]
            chosen_full = data["chosen"]
            rejected_full = data["rejected"]

            # [수정] chosen/rejected를 passage와 questions로 분리
            # Chosen 처리
            chosen_parts = chosen_full.split('[문항 세트]', 1)
            if len(chosen_parts) == 2:
                chosen_passage = chosen_parts[0].strip()
                chosen_questions = '[문항 세트]' + chosen_parts[1]
                prompt_for_chosen = f"{instruction}\n\n{chosen_passage}"
                response_for_chosen = chosen_questions.strip()
            else:
                prompt_for_chosen = instruction
                response_for_chosen = chosen_full

            # Rejected 처리
            rejected_parts = rejected_full.split('[문항 세트]', 1)
            if len(rejected_parts) == 2:
                rejected_passage = rejected_parts[0].strip()
                rejected_questions = '[문항 세트]' + rejected_parts[1]
                prompt_for_rejected = f"{instruction}\n\n{rejected_passage}"
                response_for_rejected = rejected_questions.strip()
            else:
                prompt_for_rejected = instruction
                response_for_rejected = rejected_full
            
            score_chosen = get_score(prompt_for_chosen, response_for_chosen, model, tokenizer)
            score_rejected = get_score(prompt_for_rejected, response_for_rejected, model, tokenizer)
            
            # ✨ **핵심 변경: 점수 차이 계산 및 상세 결과 저장**
            score_diff = score_chosen - score_rejected
            
            prediction = "Correct" if score_diff > 0 else "Incorrect"
            
            detailed_results.append({
                "pair_id": i + 1,
                "chosen_score": score_chosen,
                "rejected_score": score_rejected,
                "score_difference": score_diff,
                "prediction": prediction,
                "rubric": detect_rubric(instruction) # 루브릭 추가
            })
            # ---------------------------------------------
            
            if score_chosen > score_rejected:
                correct_predictions += 1
            total_predictions += 1
            
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    return accuracy, total_predictions, detailed_results

# ==============================================================================
# 3. 메인 실행 블록
# ==============================================================================

# if __name__ == "__main__":
#     try:
#         model, tokenizer = load_and_merge_model(BASE_MODEL, ADAPTER_PATH)
#         # --- 훈련 데이터셋 평가 ---
#         # print("\n--- 훈련 데이터셋 평가 시작 ---")
#         # train_accuracy, train_total, train_results = evaluate_dataset(TRAIN_DATA_PATH, model, tokenizer)
        
#         # --- 검증 데이터셋 평가 ---
#         print("\n--- 검증 데이터셋 평가 시작 ---")
#         eval_accuracy, eval_total, eval_results = evaluate_dataset(EVAL_DATA_PATH, model, tokenizer)

#         # --- 테스트 데이터셋 평가 ---
#         # print("\n--- 테스트 데이터셋 평가 시작 ---")
#         # test_accuracy, test_total, test_results = evaluate_dataset(TEST_DATA_PATH, model, tokenizer)

#         # --- 최종 결과 출력 ---
#         print("\n" + "="*50)
#         print("🏆 최종 평가 결과 요약")
#         print("="*50)
        
#         # 훈련셋 결과 요약
#         # if train_total > 0:
#         #     train_avg_diff = sum(r['score_difference'] for r in train_results) / len(train_results)
#         #     print(f"훈련 데이터셋 ({train_total}개 샘플):")
#         #     print(f"  - 정확도: {train_accuracy:.2f}%")
#         #     print(f"  - 평균 점수 차이 (Chosen - Rejected): {train_avg_diff:.4f}")

#         # 검증셋 결과 요약
#         if eval_total > 0:
#             eval_avg_diff = sum(r['score_difference'] for r in eval_results) / len(eval_results)
#             print(f"\n검증 데이터셋 ({eval_total}개 샘플):")
#             print(f"  - 정확도: {eval_accuracy:.2f}%")
#             print(f"  - 평균 점수 차이 (Chosen - Rejected): {eval_avg_diff:.4f}")

#         # 테스트셋 결과 요약
#         # if test_total > 0:
#         #     test_avg_diff = sum(r['score_difference'] for r in test_results) / len(test_results)
#         #     print(f"\n테스트 데이터셋 ({test_total}개 샘플):")
#         #     print(f"  - 정확도: {test_accuracy:.2f}%")
#         #     print(f"  - 평균 점수 차이 (Chosen - Rejected): {test_avg_diff:.4f}")

#         # 검증셋 상세 결과 샘플 출력
#         # print("\n--- 검증셋 상세 결과 샘플 ---")
#         # for result in test_results[:3]: # 처음 3개 샘플만 출력
#         #     print(
#         #         f"Pair #{result['pair_id']}: "
#         #         f"Chosen Score={result['chosen_score']:.2f}, "
#         #         f"Rejected Score={result['rejected_score']:.2f}, "
#         #         f"Diff={result['score_difference']:.2f} "
#         #         f"-> {result['prediction']}"
#         #     )
#         # print("="*50)
#         # # --- ✨ 핵심 변경 사항: 점수 차이가 가장 작은 샘플 5개 출력 ---
#         # print("\n" + "="*60)
#         # print("📉 모델이 가장 구별하기 어려워한 샘플 (Top 5)")
#         # print("="*60)

#         # 점수 차이(score_difference)를 기준으로 오름차순 정렬
#         # sorted_results = sorted(test_results, key=lambda x: x['score_difference'])
        
#         # # 점수 차이가 가장 작은 (가장 많이 틀린) 5개 샘플을 출력
#         # for result in sorted_results[:5]:
#         #     print(
#         #         f"\nPair #{result['pair_id']} (Diff: {result['score_difference']:.4f}) -> {result['prediction']}"
#         #     )
            
#         #     # 해당 샘플의 원본 텍스트를 불러와서 함께 출력
#         #     with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
#         #         for i, line in enumerate(f):
#         #             if i + 1 == result['pair_id']:
#         #                 data = json.loads(line)
#         #                 print(f"  [Prompt]: {data['prompt'][:100]}...")
#         #                 print(f"  [Chosen] (Score: {result['chosen_score']:.4f}): {data['chosen'][:100]}...")
#         #                 print(f"  [Rejected] (Score: {result['rejected_score']:.4f}): {data['rejected'][:100]}...")
#         #                 break
#         # print("\n" + "="*60)
#     except Exception as e:
#         print(f"\n❌ 스크립트 실행 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    try:
        model, tokenizer = load_and_merge_model(BASE_MODEL, ADAPTER_PATH)
        print("\n--- 스코어링 시작 ---")
        score_path(TEST_DATA_PATH, model, tokenizer)

        print("\n" + "=" * 50)
        print("🏁 스코어링 완료")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ 스크립트 실행 중 오류가 발생했습니다: {e}")
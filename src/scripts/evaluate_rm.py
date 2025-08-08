import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import yaml
import sys
import random
import numpy as np
from pathlib import Path

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
ADAPTER_PATH = "./saves/all_in_one/checkpoint-2196" # 훈련 결과 경로
TRAIN_DATA_PATH = "saves/all_in_one/all_in_one_rm_train.jsonl"
EVAL_DATA_PATH = "saves/all_in_one/all_in_one_rm_eval.jsonl"
TEST_DATA_PATH = "saves/all_in_one/all_in_one_rm_test.jsonl"
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

def evaluate_dataset(data_path: str, model, tokenizer):
    """주어진 데이터셋 파일로 모델의 정확도를 평가합니다."""
    correct_predictions = 0
    total_predictions = 0
    detailed_results = [] # 결과를 저장할 리스트

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            prompt = data["prompt"]
            chosen = data["chosen"]
            rejected = data["rejected"]
            
            score_chosen = get_score(prompt, chosen, model, tokenizer)
            score_rejected = get_score(prompt, rejected, model, tokenizer)
            
            # ✨ **핵심 변경: 점수 차이 계산 및 상세 결과 저장**
            score_diff = score_chosen - score_rejected
            
            prediction = "Correct" if score_diff > 0 else "Incorrect"
            
            detailed_results.append({
                "pair_id": i + 1,
                "chosen_score": score_chosen,
                "rejected_score": score_rejected,
                "score_difference": score_diff,
                "prediction": prediction,
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

if __name__ == "__main__":
    try:
        model, tokenizer = load_and_merge_model(BASE_MODEL, ADAPTER_PATH)
        # --- 훈련 데이터셋 평가 ---
        # print("\n--- 훈련 데이터셋 평가 시작 ---")
        # train_accuracy, train_total, train_results = evaluate_dataset(TRAIN_DATA_PATH, model, tokenizer)
        
        # --- 검증 데이터셋 평가 ---
        # print("\n--- 검증 데이터셋 평가 시작 ---")
        # eval_accuracy, eval_total, eval_results = evaluate_dataset(EVAL_DATA_PATH, model, tokenizer)

        # --- 테스트 데이터셋 평가 ---
        print("\n--- 테스트 데이터셋 평가 시작 ---")
        test_accuracy, test_total, test_results = evaluate_dataset(TEST_DATA_PATH, model, tokenizer)

        # --- 최종 결과 출력 ---
        print("\n" + "="*50)
        print("🏆 최종 평가 결과 요약")
        print("="*50)
        
        # 훈련셋 결과 요약
        # if train_total > 0:
        #     train_avg_diff = sum(r['score_difference'] for r in train_results) / len(train_results)
        #     print(f"훈련 데이터셋 ({train_total}개 샘플):")
        #     print(f"  - 정확도: {train_accuracy:.2f}%")
        #     print(f"  - 평균 점수 차이 (Chosen - Rejected): {train_avg_diff:.4f}")

        # 검증셋 결과 요약
        # if eval_total > 0:
        #     eval_avg_diff = sum(r['score_difference'] for r in eval_results) / len(eval_results)
        #     print(f"\n검증 데이터셋 ({eval_total}개 샘플):")
        #     print(f"  - 정확도: {eval_accuracy:.2f}%")
        #     print(f"  - 평균 점수 차이 (Chosen - Rejected): {eval_avg_diff:.4f}")

        # 테스트셋 결과 요약
        if test_total > 0:
            test_avg_diff = sum(r['score_difference'] for r in test_results) / len(test_results)
            print(f"\n테스트 데이터셋 ({test_total}개 샘플):")
            print(f"  - 정확도: {test_accuracy:.2f}%")
            print(f"  - 평균 점수 차이 (Chosen - Rejected): {test_avg_diff:.4f}")

        # 검증셋 상세 결과 샘플 출력
        print("\n--- 검증셋 상세 결과 샘플 ---")
        for result in test_results[:3]: # 처음 3개 샘플만 출력
            print(
                f"Pair #{result['pair_id']}: "
                f"Chosen Score={result['chosen_score']:.2f}, "
                f"Rejected Score={result['rejected_score']:.2f}, "
                f"Diff={result['score_difference']:.2f} "
                f"-> {result['prediction']}"
            )
        print("="*50)
        # --- ✨ 핵심 변경 사항: 점수 차이가 가장 작은 샘플 5개 출력 ---
        print("\n" + "="*60)
        print("📉 모델이 가장 구별하기 어려워한 샘플 (Top 5)")
        print("="*60)

        # 점수 차이(score_difference)를 기준으로 오름차순 정렬
        sorted_results = sorted(test_results, key=lambda x: x['score_difference'])
        
        # 점수 차이가 가장 작은 (가장 많이 틀린) 5개 샘플을 출력
        for result in sorted_results[:5]:
            print(
                f"\nPair #{result['pair_id']} (Diff: {result['score_difference']:.4f}) -> {result['prediction']}"
            )
            
            # 해당 샘플의 원본 텍스트를 불러와서 함께 출력
            with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i + 1 == result['pair_id']:
                        data = json.loads(line)
                        print(f"  [Prompt]: {data['prompt'][:100]}...")
                        print(f"  [Chosen] (Score: {result['chosen_score']:.4f}): {data['chosen'][:100]}...")
                        print(f"  [Rejected] (Score: {result['rejected_score']:.4f}): {data['rejected'][:100]}...")
                        break
        print("\n" + "="*60)
    except Exception as e:
        print(f"\n❌ 스크립트 실행 중 오류가 발생했습니다: {e}")
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datetime import datetime
from pathlib import Path
import re
import random
import numpy as np

# --- 0. 설정 ---
# TODO: 이 경로를 실제 훈련된 어댑터(체크포인트) 경로로 수정하세요.
ADAPTER_PATH = "./saves/l2_v3_5ep/checkpoint-445" 
BASE_MODEL = "K-intelligence/Midm-2.0-Mini-Instruct"
INPUT_FILE = "/home/sjin4861/25-1/HCLT/iSKA_Gen/data_store/final_prompted_dataset/v4/final_prompted_dataset_chosen_2.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")

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

# --- 1. 모델 로딩 및 스코어링 함수 (제공된 코드 기반) ---

def load_and_merge_model(base_path, adapter_path):
    """베이스 모델에 어댑터를 로드하고 병합하여 최종 모델을 반환합니다."""
    print(f"\n🔄 '{adapter_path}'에서 모델 로딩 및 병합 시작...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        base_path,
        num_labels=1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    print("  - ✅ 모델 병합 완료!")
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
        
    model.to(device).eval()
    return model, tokenizer

def get_score(prompt: str, response: str, model, tokenizer):
    """주어진 프롬프트와 응답으로 점수를 계산합니다."""
    full_text = (
        f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"
    )
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    with torch.no_grad():
        score = model(**inputs).logits[0].item()
    return score

# --- 2. JSONL 파일 처리 함수 ---

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            yield json.loads(line)

def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

def score_jsonl_file(in_file: Path, model, tokenizer) -> Path:
    """
    입력 JSONL (각 라인: {prompt, chosen})을 읽어
    rm_score, rubric, scored_by를 추가한 *.scored.jsonl 로 저장합니다.
    """
    print(f"\n📄 스코어링 시작: {in_file.name}")
    scored_rows = []
    total_items = sum(1 for _ in _iter_jsonl(in_file)) # 전체 항목 수 계산

    for idx, row in enumerate(_iter_jsonl(in_file)):
        print(f"  - 처리 중... {idx + 1}/{total_items}", end='\r')
        
        instruction = row.get("prompt", "")
        chosen_full = row.get("chosen", "")

        # 'chosen'을 본문과 문항 세트로 분리하여 RM의 입력 형식에 맞게 재구성
        parts = chosen_full.split('[문항 세트]', 1)
        if len(parts) == 2:
            passage = parts[0].strip()
            questions = '[문항 세트]' + parts[1]
            prompt_for_rm = f"{instruction}\n\n{passage}"
            response_for_rm = questions.strip()
        else:
            # 분리 실패 시, 전체를 응답으로 간주
            prompt_for_rm = instruction
            response_for_rm = chosen_full

        # 점수 계산
        score = get_score(prompt_for_rm, response_for_rm, model, tokenizer)

        # 결과 필드 추가
        row["rm_score"] = score
        row["rubric"] = "R6"  # 루브릭을 'R6'로 고정
        row["scored_by"] = {
            "base_model": BASE_MODEL,
            "adapter": ADAPTER_PATH,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        scored_rows.append(row)

    # 출력 파일 경로 생성 (*.scored.jsonl)
    out_file = in_file.with_suffix("").with_suffix(".scored_chosen_1.jsonl")
    _write_jsonl(out_file, scored_rows)
    print(f"\n✅ 저장 완료: {out_file} (총 {len(scored_rows)}개)")
    return out_file

# --- 3. 메인 실행 블록 ---

if __name__ == "__main__":
    try:
        input_path = Path(INPUT_FILE)
        if not input_path.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

        # 1. 모델과 토크나이저 로드
        model, tokenizer = load_and_merge_model(BASE_MODEL, ADAPTER_PATH)
        
        # 2. 지정된 파일 스코어링
        score_jsonl_file(input_path, model, tokenizer)

        print("\n" + "=" * 50)
        print("🏁 모든 작업이 성공적으로 완료되었습니다.")
        print("=" * 50)

    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
    except Exception as e:
        print(f"\n❌ 스크립트 실행 중 오류가 발생했습니다: {e}")


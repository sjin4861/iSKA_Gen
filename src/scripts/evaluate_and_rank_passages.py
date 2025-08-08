import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from pathlib import Path
import json
import pandas as pd
import yaml

# ==============================================================================
# 1. 설정
# ==============================================================================
RM_BASE_MODEL_PATH = "K-intelligence/Midm-2.0-Mini-Instruct"
INPUT_FILE = Path("src/data/rm_testing/ranking_test_data.jsonl") # 평가할 지문이 담긴 파일
OUTPUT_FILE = Path("src/outputs/ranked_passages_result.jsonl") # 최종 결과 저장 파일
PROMPT_YAML_PATH = Path("src/config/prompts/iska/preference_eval.yaml")
RM_SAVES_DIR= Path("saves")  # Reward Model 체크포인트가 저장된 디렉토리
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- ✨ 중요: 각 루브릭과 실제 훈련된 모델 폴더/프롬프트 이름을 연결 ---
RM_INFO = {
    "r1_score": {"folder_name": "completeness_for_guidelines", "prompt_key": "completeness_for_guidelines"},
    "r2_score": {"folder_name": "clarity_of_core_theme", "prompt_key": "clarity_of_core_theme"},
    "r3_score": {"folder_name": "reference_groundedness", "prompt_key": "reference_groundedness"},
    "r4_score": {"folder_name": "logical_flow", "prompt_key": "logical_flow"},
    "r5_score": {"folder_name": "korean_quality", "prompt_key": "korean_quality"},
    "r6_score": {"folder_name": "l2_learner_suitability", "prompt_key": "l2_learner_suitability"}
}

# --- ✨ 프롬프트 Placeholder를 채울 정적 데이터 ---
# 평가 대상 지문들이 '회식 문화'에 대한 것이므로, 관련 정보를 미리 정의합니다.
STATIC_SOURCE_ITEM = {
    "korean_topic": "회식 문화",
    "korean_context": "회식은 한국 직장 문화의 중요한 부분으로, 업무가 끝난 후 동료들과 함께 식사하며 친목을 다지는 활동입니다...",
    "foreign_topic": "Happy Hour Culture",
    "foreign_context": "Happy hour is a social tradition in many Western countries where colleagues gather at a bar or pub after work...",
    "problem_type1": "제목을 붙인 근거 설명하기",
    "problem_type2": "자문화와 비교하기", 
    "problem_type3": "원인과 전망 예측하기",
    "eval_goal1": "글의 전체적인 주제 파악 능력 평가",
    "eval_goal2": "문화 비교 설명 능력 평가",
    "eval_goal3": "원인 추론 및 전망 예측 능력 평가"
}
# --------------------------------------------------

# ==============================================================================
# 2. 헬퍼 함수
# ==============================================================================
def load_prompts(yaml_path):
    """YAML 파일에서 프롬프트 템플릿을 로드합니다."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def format_prompt(template, source_item):
    """Jinja2와 유사한 형식의 프롬프트 템플릿을 채웁니다."""
    for key, value in source_item.items():
        template = template.replace(f"{{{{ {key} }}}}", str(value))
    return template

def load_and_merge_model(base_path, adapter_path):
    # ... (이전과 동일한 모델 로딩 함수)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_path, num_labels=1, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    model.to(device).eval()
    return model, tokenizer

def cleanup_model(model, tokenizer):
    """모델과 토크나이저를 메모리에서 해제합니다."""
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    torch.cuda.empty_cache()  # GPU 메모리 정리

def get_rm_score(prompt, passage, rm_model, rm_tokenizer):
    """주어진 프롬프트와 지문으로 점수를 계산합니다."""
    full_text = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{passage}<|eot_id|>"
    inputs = rm_tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        return rm_model(**inputs).logits[0].item()

# ==============================================================================
# 3. 메인 실행 블록
# ==============================================================================
if __name__ == "__main__":
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            passages_data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ 평가할 지문 파일을 찾을 수 없습니다: {INPUT_FILE}")
        exit()

    prompts = load_prompts(PROMPT_YAML_PATH)
    
    # 각 RM에 사용할 프롬프트를 미리 생성
    formatted_prompts = {
        score_key: format_prompt(prompts["preference_evaluation"][info['prompt_key']], STATIC_SOURCE_ITEM)
        for score_key, info in RM_INFO.items()
    }

    print("\n" + "="*60)
    print("순차적으로 Reward Model을 로딩하여 채점합니다...")
    
    # 각 모델을 순차적으로 로드하고 채점
    for score_key, info in RM_INFO.items():
        print(f"\n🔄 '{info['folder_name']}' 모델 로딩 중...")
        checkpoint_path = max(list((RM_SAVES_DIR / info['folder_name']).glob("checkpoint-*")), key=lambda p: int(p.name.split('-')[-1]))
        
        if checkpoint_path:
            # 모델 로드
            model, tokenizer = load_and_merge_model(RM_BASE_MODEL_PATH, checkpoint_path)
            prompt_to_use = formatted_prompts[score_key]
            
            print(f"  ✅ '{info['folder_name']}' 모델 로딩 완료!")
            print(f"  🔄 모든 지문 채점 중...")
            
            # 모든 지문에 대해 현재 모델로 채점
            for i, data in enumerate(passages_data):
                passage = data['passage']
                score = get_rm_score(prompt_to_use, passage, model, tokenizer)
                data[score_key] = score
                if (i + 1) % 10 == 0 or i == len(passages_data) - 1:
                    print(f"    - 진행상황: {i+1}/{len(passages_data)} 지문 완료")
            
            print(f"  ✅ '{info['folder_name']}' 모델 채점 완료!")
            
            # 메모리 정리
            cleanup_model(model, tokenizer)
            print(f"  🧹 '{info['folder_name']}' 모델 메모리 해제 완료!")
        else:
            print(f"  ❌ '{info['folder_name']}' 체크포인트를 찾을 수 없습니다!")
            
    print("\n✅ 모든 모델 채점 완료!")
    
    # 총합 점수 계산 및 순위 부여
    df = pd.DataFrame(passages_data)
    score_columns = list(RM_INFO.keys())
    df['total_score'] = df[score_columns].sum(axis=1)
    df = df.sort_values(by="total_score", ascending=False)
    df['rm_ranking'] = range(1, len(df) + 1)
    df = df.drop(columns=['total_score'])
    
    # 최종 결과 저장
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(OUTPUT_FILE, orient='records', lines=True, force_ascii=False)

    print("\n\n" + "="*60)
    print("🏆 최종 채점 및 순위 부여 완료!")
    print("="*60)
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', 150)
    print(df[['rm_ranking'] + score_columns + ['passage']].round(4))
    print("\n✅ 모든 결과가 아래 파일에 저장되었습니다:")
    print(f"   {OUTPUT_FILE.resolve()}")
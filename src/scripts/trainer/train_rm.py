import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType
from trl import RewardConfig, RewardTrainer
import yaml
import sys
from pathlib import Path

# --- 1. 프로젝트 경로 설정 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- 2. 모듈 임포트 ---
from src.model_loader import load_model_for_reward_training
from src.data_loader import load_and_preprocess_data_chat # 대화형 데이터 로더 임포트


def main():
    # --- 3. 설정 파일 로드 ---
    # PROJECT_ROOT를 사용하여 정확한 경로를 지정합니다.
    with open(PROJECT_ROOT / "src/config/model_config.yaml", 'r') as f:
        model_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "src/config/training_args.yaml", 'r') as f:
        training_config = yaml.safe_load(f)

    # --- 4. 모델, 토크나이저, peft 설정 로드 ---
    model, tokenizer, peft_config = load_model_for_reward_training(model_config)

    # --- 5. 데이터셋 로드 및 전처리 ---
    train_dataset = load_and_preprocess_data_chat(
        str(PROJECT_ROOT / "src/data/all_in_one_rm_train.jsonl"), # 데이터 파일 경로
        tokenizer, 
        training_config["max_length"]
    )
    eval_dataset = load_and_preprocess_data_chat(
        str(PROJECT_ROOT / "src/data/all_in_one_rm_eval.jsonl"), # 데이터 파일 경로
        tokenizer, 
        training_config["max_length"]
    )
    
    # --- 6. 훈련 인자 설정 ---
    training_args = RewardConfig(**training_config)

    # --- 7. RewardTrainer 생성 및 훈련 시작 ---
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    
    # --- ✨ 핵심 변경 사항: 체크포인트 경로를 trainer.train()에 전달 ---
    trainer.train()
    print(f"✅ 훈련 완료! 모델이 '{training_args.output_dir}'에 저장되었습니다.")

# --- 8. 메인 실행 블록 ---
if __name__ == "__main__":
    main()
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, TaskType

def load_model_for_reward_training(config: dict):
    """설정 파일(dict)을 바탕으로 RM 훈련을 위한 모델과 토크나이저, peft 설정을 로드합니다."""
    base_model_path = config["base_model_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Also update the model's configuration to recognize the new pad token ID

    model_config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.num_labels = 1

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        config=model_config, # 수정된 config 전달
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
        
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        **config["peft_config"]
    )
    
    return model, tokenizer, peft_config
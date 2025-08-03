from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess_data(data_path: str, tokenizer: AutoTokenizer, max_length: int):
    """JSONL 데이터셋을 로드하고 토크나이징 전처리 함수를 적용합니다."""
    
    def formatting_func(examples):
        kwargs = {"padding": "max_length", "truncation": True, "max_length": max_length}
        tokenized_chosen = tokenizer(examples["chosen"], **kwargs)
        tokenized_rejected = tokenizer(examples["rejected"], **kwargs)
        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }

    dataset = load_dataset("json", data_files=data_path, split="train")
    dataset = dataset.map(formatting_func, batched=True)
    return dataset

from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess_data_chat(data_path: str, tokenizer: AutoTokenizer, max_length: int):
    """
    대화형 템플릿을 적용하여 RM 훈련용 데이터셋을 전처리합니다.
    데이터셋은 'prompt', 'chosen', 'rejected' 키를 포함해야 합니다.
    """
    
    def formatting_func(examples):
        kwargs = {"padding": "max_length", "truncation": True, "max_length": max_length}
        
        # --- ✨ 핵심 변경 사항: apply_chat_template을 사용하여 프롬프트와 응답을 결합 ---
        chosen_conversations = []
        for prompt, chosen in zip(examples["prompt"], examples["chosen"]):
        # 시스템 프롬프트를 제외하고 필요한 부분만 조합합니다.
        # 모델의 특수 토큰을 사용하여 대화 턴을 명확히 구분해 줍니다.
            full_text = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{chosen}<|eot_id|>"
            )
            chosen_conversations.append(full_text)

        rejected_conversations = []
        for prompt, rejected in zip(examples["prompt"], examples["rejected"]):
            full_text = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n{rejected}<|eot_id|>"
            )
            rejected_conversations.append(full_text)
        # -------------------------------------------------------------

        tokenized_chosen = tokenizer(chosen_conversations, **kwargs)
        tokenized_rejected = tokenizer(rejected_conversations, **kwargs)

        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
        }

    # 데이터셋 로드
    dataset = load_dataset("json", data_files=data_path, split="train")
    # `batched=True`로 설정하여 여러 샘플을 한 번에 효율적으로 처리합니다.
    dataset = dataset.map(formatting_func, batched=True)
    return dataset
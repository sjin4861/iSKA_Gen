"""간단한 로컬 HuggingFace AutoModelForCausalLM Chat wrapper.

목표:
 - vLLM 서버 없이 로컬 경로(EXAONE 등) 직접 로드
 - LangChain ChatOpenAI 유사 interface (invoke(messages=[...])) 최소 구현
 - greedy / temperature sampling 단순 지원

제한:
 - 스트리밍 미구현
 - 시스템 / tool 메세지 단순 무시
 - 긴 컨텍스트 truncation (max_new_tokens + input_len > model_max_len 시 앞부분 절단)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ChatMessage:
    role: str
    content: str


class LocalHFChat:
    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        dtype: str = "bfloat16",
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" and torch.cuda.is_available() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        self.model.eval()

    # 최소 호환: LangChain Chat 모델에서 invoke(inputs) 형태 흉내
    def invoke(self, inputs: Dict[str, Any]):
        messages: List[Dict[str, str]] = inputs.get("messages") or []
        text_parts = []
        for m in messages:
            if isinstance(m, dict):
                role, content = m.get("role", "user"), m.get("content", "")
            else:  # LangChain Message 객체 호환
                role, content = getattr(m, "type", "user"), getattr(m, "content", str(m))

            if role == "system":
                text_parts.append(f"[시스템]\n{content}\n")
            elif role == "user":
                text_parts.append(f"[사용자]\n{content}\n")
            elif role == "assistant":
                text_parts.append(f"[assistant]\n{content}\n")

        prompt = "\n".join(text_parts).strip() + "\n[assistant]\n"

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **input_ids,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen = out[0][input_ids["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        return {"role": "assistant", "content": text}
    # LangChain 호환성용 단순 별칭
    def __call__(self, messages: List[Dict[str, str]]):
        return self.invoke({"messages": messages})

def load_local_chat(model_path: str, **kwargs) -> LocalHFChat:
    return LocalHFChat(model_path, **kwargs)

__all__ = ["LocalHFChat", "load_local_chat"]

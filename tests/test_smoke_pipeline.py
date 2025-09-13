"""Lightweight smoke tests for core pipeline components.

These tests avoid external network / model downloads by:
 - Monkeypatching ChatOpenAI with a dummy implementation.
 - Using a DummyTokenizer instead of HF tokenizer.
 - Conditionally testing data loader only if `datasets` is installed.

Run with: `uv run pytest` (pytest is declared in pyproject.toml dependencies).
"""
from __future__ import annotations

import os
import types
import importlib
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Helpers / Dummies
# ---------------------------------------------------------------------------

class DummyChatResponse:
    def __init__(self, content: str):
        self.content = content


class DummyChatOpenAI:
    """Minimal stand‑in for langchain_openai.ChatOpenAI used in tests.

    Stores init kwargs for later assertions; implements `invoke` with a
    deterministic echo style to verify chaining logic.
    """

    def __init__(self, **kwargs):  # noqa: D401
        self.kwargs = kwargs

    def invoke(self, messages: List[Dict[str, Any]]):  # LangChain style
        # Use last user message content if present.
        last = messages[-1]["content"] if messages else ""
        return DummyChatResponse(content=f"DUMMY_COMPLETION::{last[:25]}")


class DummyTokenizer:
    """Simple tokenizer mimicking HF tokenizer output shape.

    Produces token ids = index of word truncated/padded to max_length.
    """

    def __call__(self, texts, padding: str, truncation: bool, max_length: int):  # noqa: D401
        if isinstance(texts, str):
            texts = [texts]
        input_ids = []
        attn = []
        for t in texts:
            # naive whitespace split
            tokens = t.split()
            ids = list(range(1, len(tokens) + 1))  # 1..N
            if truncation and len(ids) > max_length:
                ids = ids[: max_length]
            # pad
            if padding == "max_length" and len(ids) < max_length:
                ids = ids + [0] * (max_length - len(ids))
            input_ids.append(ids)
            attn.append([1 if i != 0 else 0 for i in ids])
        return {"input_ids": input_ids, "attention_mask": attn}


# ---------------------------------------------------------------------------
# Prompt Loader Tests
# ---------------------------------------------------------------------------


def test_prompt_loader_list_agents_and_prompts():
    pl = importlib.import_module("src.utils.prompt_loader")
    agents = pl.list_available_agents()
    assert "iska" in agents, "Expected 'iska' agent to be available"
    prompt_files = pl.list_agent_prompts("iska")
    assert "stem_agent" in prompt_files, "stem_agent.yaml should exist"


def test_get_prompt_nested_key_and_format():
    pl = importlib.import_module("src.utils.prompt_loader")
    prompt = pl.get_prompt(
        "stem_agent.few_shot_new",
        agent="iska",
        content="<PASSAGE>",
        problem_type="유형",
        eval_goal="목표",
    )
    # Should have had placeholders replaced:
    assert "<PASSAGE>" in prompt
    assert "유형" in prompt
    assert "목표" in prompt
    # Template contains an instruction token like '[출력]:' near end
    assert "[출력]" in prompt


# ---------------------------------------------------------------------------
# Stem Chain Test
# ---------------------------------------------------------------------------


def test_stem_chain_runs_with_dummy_llm(monkeypatch):
    stem_chain_mod = importlib.import_module("src.modules.stem_chain")
    # Build chain with existing template key from prompts
    chain = stem_chain_mod.build_stem_chain("stem_agent.few_shot_new")

    dummy_llm = DummyChatOpenAI(model="dummy")
    out = chain.invoke(
        {
            "passage": "이것은 테스트 지문입니다.",
            "problem_type": "비교하기",
            "eval_goal": "목표",
            "llm": dummy_llm,
            "k": 1,
        }
    )
    assert isinstance(out, str) and out.startswith("DUMMY_COMPLETION::"), out


# ---------------------------------------------------------------------------
# Model Client Tests (routing logic only; ChatOpenAI monkeypatched)
# ---------------------------------------------------------------------------


def test_model_client_routing_and_params(monkeypatch):
    mc = importlib.import_module("src.modules.model_client")

    # Monkeypatch ChatOpenAI symbol used inside module.
    monkeypatch.setattr(mc, "ChatOpenAI", DummyChatOpenAI)

    # Ensure OPENAI key so get_openai_chat doesn't raise.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    # vLLM path selection by model name pattern
    vllm_a = mc.get_vllm_chat("A.X-4.0-Light")
    assert vllm_a.kwargs["base_url"].endswith(":8000/v1")

    vllm_exa = mc.get_vllm_chat("EXAONE-4.0-32B")
    assert vllm_exa.kwargs["base_url"].endswith(":8001/v1")

    # OpenAI path (auto)
    openai = mc.auto_get_chat("gpt-4o-mini")
    assert openai.kwargs.get("model") == "gpt-4o-mini"
    assert "base_url" not in openai.kwargs, "OpenAI path should not inject base_url"

    # Explicit env based fetch falls back to vLLM when not an OpenAI model
    monkeypatch.setenv("VLLM_MODEL", "A.X-4.0-Light")
    model_env = mc.get_chat_from_env()
    assert model_env.kwargs.get("model") == "A.X-4.0-Light"


# ---------------------------------------------------------------------------
# Data Loader (optional) - skipped gracefully if datasets missing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(importlib.util.find_spec("datasets") is None, reason="datasets package not installed")
def test_data_loader_local_jsonl(tmp_path):
    dl = importlib.import_module("src.utils.data_loader")

    # Create a minimal JSONL file with chosen/rejected fields
    p = tmp_path / "mini.jsonl"
    p.write_text(
        """{"chosen": "좋은 선택", "rejected": "나쁜 선택"}\n{"chosen": "두번째 좋은", "rejected": "두번째 나쁜"}\n""",
        encoding="utf-8",
    )

    tokenizer = DummyTokenizer()
    dataset = dl.load_and_preprocess_data(str(p), tokenizer, max_length=8)

    # Basic shape assertions
    assert len(dataset) == 2
    first = dataset[0]
    assert "input_ids_chosen" in first and "input_ids_rejected" in first
    assert len(first["input_ids_chosen"]) == 8  # padded


@pytest.mark.skipif(importlib.util.find_spec("datasets") is None, reason="datasets package not installed")
def test_data_loader_chat_format(tmp_path):
    dl = importlib.import_module("src.utils.data_loader")

    p = tmp_path / "mini_chat.jsonl"
    p.write_text(
        """{"prompt": "안녕?", "chosen": "좋습니다", "rejected": "별로"}\n""",
        encoding="utf-8",
    )
    tokenizer = DummyTokenizer()
    dataset = dl.load_and_preprocess_data_chat(str(p), tokenizer, max_length=10)
    assert len(dataset) == 1
    sample = dataset[0]
    assert len(sample["input_ids_chosen"]) == 10


# ---------------------------------------------------------------------------
# Sanity: ensure tests themselves do not rely on real network / API keys
# ---------------------------------------------------------------------------


def test_no_real_api_keys_exposed():
    # Guard against accidentally using production keys in CI
    for key in ["OPENAI_API_KEY", "VLLM_API_KEY"]:
        val = os.environ.get(key, "")
        if val:
            assert "test" in val.lower() or val.startswith("sk-test"), f"Unsafe key value present for {key}"

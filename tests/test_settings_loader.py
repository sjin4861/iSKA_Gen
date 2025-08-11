import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가하여 src 모듈을 불러올 수 있게 합니다.
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.settings_loader import get_merged_setting


def test_get_merged_setting_prefers_config_over_env():
    # set environment variable to a different value
    original = os.environ.get("LLM_MAX_TOKENS")
    os.environ["LLM_MAX_TOKENS"] = "2048"
    try:
        value = get_merged_setting("llm.max_tokens", "LLM_MAX_TOKENS")
    finally:
        if original is None:
            del os.environ["LLM_MAX_TOKENS"]
        else:
            os.environ["LLM_MAX_TOKENS"] = original
    assert value == 1024

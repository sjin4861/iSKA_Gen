# utils/settings_loader.py
from pathlib import Path
import yaml
import os
import sys

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))

_CFG_PATH = Path(__file__).resolve().parents[2] / "src/config/settings/settings.yaml"

class Settings:
    """설정 클래스 - 편리한 속성 접근을 제공"""
    
    def __init__(self, config_dict: dict):
        self._config = config_dict
    
    @property
    def ska_db_path(self) -> str:
        """SKA 데이터베이스 경로"""
        return self._config.get("db", {}).get("ska_path", "src/db/ska.sqlite3")
    
    @property
    def iska_db_path(self) -> str:
        """iSKA 데이터베이스 경로"""
        return self._config.get("db", {}).get("iska_path", "src/db/iska.sqlite3")
    
    def get(self, key: str, default=None):
        """딕셔너리 스타일 접근"""
        return self._config.get(key, default)

def get_settings() -> Settings:
    """설정 파일을 로드하여 Settings 객체로 반환합니다."""
    with _CFG_PATH.open(encoding="utf-8") as f:
        settings = yaml.safe_load(f) or {}
    
    # 설정 파일 값이 없을 때만 환경 변수로 대체 (설정 파일 우선순위)
    llm_env_tokens = os.getenv('LLM_MAX_TOKENS') or os.getenv('VLLM_MAX_TOKENS')  # 하위 호환성
    llm_env_temp = os.getenv('LLM_TEMPERATURE') or os.getenv('VLLM_TEMPERATURE')  # 하위 호환성
    
    if 'llm' in settings:
        # max_tokens가 설정 파일에 없을 때만 환경 변수 사용
        if 'max_tokens' not in settings['llm'] and llm_env_tokens is not None:
            try:
                settings['llm']['max_tokens'] = int(llm_env_tokens)
            except ValueError:
                pass
        # temperature가 설정 파일에 없을 때만 환경 변수 사용
        if 'temperature' not in settings['llm'] and llm_env_temp is not None:
            try:
                settings['llm']['temperature'] = float(llm_env_temp)
            except ValueError:
                pass
    elif llm_env_tokens is not None or llm_env_temp is not None:
        # llm 섹션이 없으면 새로 생성하고 환경 변수 적용
        settings['llm'] = {}
        if llm_env_tokens is not None:
            try:
                settings['llm']['max_tokens'] = int(llm_env_tokens)
            except ValueError:
                pass
        if llm_env_temp is not None:
            try:
                settings['llm']['temperature'] = float(llm_env_temp)
            except ValueError:
                pass
    
    return Settings(settings)

def get_evaluation_settings() -> dict:
    """평가 관련 설정을 반환합니다."""
    settings = get_settings()
    return settings.get("evaluation", {})

def get_score_threshold(tier: str) -> float:
    """지정된 tier의 점수 임계값을 반환합니다."""
    eval_settings = get_evaluation_settings()
    thresholds = eval_settings.get("score_thresholds", {})
    return thresholds.get(tier, 6.0)  # 기본값 6.0

def get_evaluation_criteria(tier: str) -> list:
    """지정된 tier의 평가 기준을 반환합니다."""
    eval_settings = get_evaluation_settings()
    criteria = eval_settings.get("criteria", {})
    return criteria.get(tier, [])

def get_api_settings(service: str) -> dict:
    """지정된 외부 서비스의 API 설정을 반환합니다."""
    settings = get_settings()
    external_services = settings.get("external_services", {})
    return external_services.get(service, {})

def get_setting(setting_path: str, default=None):
    """설정 경로로 값을 가져옵니다. (예: "db.ska_path")"""
    settings = get_settings()
    
    # 설정 경로를 점으로 분리하여 중첩된 딕셔너리 접근
    keys = setting_path.split('.')
    value = settings._config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

def get_merged_setting(setting_path: str, env_var: str) -> str:
    """설정 파일에서 값을 가져오고, 환경 변수로 대체 가능한 값을 처리합니다."""
    settings = get_settings()
    
    # 설정 경로를 점으로 분리하여 중첩된 딕셔너리 접근
    keys = setting_path.split('.')
    value = settings
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            value = None
            break
    
    # 환경 변수 치환 처리 (${VAR_NAME} 형태)
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_var_name = value[2:-1]  # ${ 와 } 제거
        value = os.getenv(env_var_name)
    
    # 환경 변수로 대체
    if value is None:
        value = os.getenv(env_var)
    
    return value

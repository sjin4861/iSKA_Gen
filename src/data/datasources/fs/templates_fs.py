# src/data/datasources/fs/template_fs.py
from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional
from pathlib import Path

# YAML 파싱은 prompt_loader에 위임(경로/에이전트 해석 포함)
from src.utils.prompt_loader import (
    get_prompt,
    _resolve_yaml_path,   # 내부 함수이지만, 경로 해석 일관성을 위해 사용
    _load_yaml,           # 내부 함수이지만, 키 나열/원본 조회에 사용
)

class TemplatesFSDataSource:
    """
    YAML 기반 프롬프트 템플릿 로더(DataSource).

    - 에이전트(예: "iska") 단위로 디렉토리(예: src/config/prompts/iska/*.yaml)를 로드
    - dot notation 키("passage_eval.system", "preference_eval.judge")로 템플릿 조회
    - get(): 바로 format까지 수행 (kwargs 없으면 raw 반환)
    - get_raw(): 포매팅 없이 원문 반환
    - list_keys(): 사용 가능한 모든 키 나열
    - as_dict(): 플랫한 dict(key -> str)로 전체 템플릿 반환

    참고:
      * src/utils/prompt_loader.py 의 경로/캐시 전략을 그대로 재사용.
      * 환경변수 ISKA_PROMPT_FILE 및 src/config/prompts/{agent}/ 구조를 지원.
    """

    def __init__(self, *, agent: str = "iska", user_path: Optional[str | Path] = None) -> None:
        self.agent = agent
        self._root: Path = _resolve_yaml_path(user_path, agent=self.agent)

    # ----------------------------
    # Public API
    # ----------------------------
    def get(self, name: str, /, **format_kwargs: Any) -> str:
        """
        지정한 키의 템플릿을 가져와 str.format(**kwargs) 적용 후 반환.
        (format 인자가 없으면 원문 그대로 반환)
        """
        return get_prompt(name, agent=self.agent, **format_kwargs)

    def get_raw(self, name: str, /) -> str:
        """
        지정한 키의 템플릿 원문을 반환(포매팅 없음).
        """
        # get_prompt는 kwargs가 없으면 원문을 반환하므로 그대로 사용
        return get_prompt(name, agent=self.agent)

    def list_keys(self) -> List[str]:
        """
        현재 에이전트 디렉토리(또는 지정 파일)에 존재하는 모든 문자열 템플릿 키를
        dot notation으로 나열한다.
        """
        data = _load_yaml(self._root, agent=self.agent)  # Mapping[str, Any]
        flat: List[str] = []
        self._flatten_keys(data, prefix="", out=flat)
        return sorted(flat)

    def as_dict(self) -> Dict[str, str]:
        """
        모든 템플릿을 평탄화하여 { 'a.b.c': '...'} 형태로 반환.
        문자열이 아닌 값은 건너뛰며, 키 충돌 시 후승(마지막 파일) 우선.
        """
        data = _load_yaml(self._root, agent=self.agent)
        out: Dict[str, str] = {}
        self._collect_flat(data, prefix="", out=out)
        return out

    def root_path(self) -> Path:
        """실제 로딩에 사용된 YAML 파일/디렉토리 경로를 반환."""
        return self._root

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _flatten_keys(self, node: Any, *, prefix: str, out: List[str]) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                self._flatten_keys(v, prefix=key, out=out)
        else:
            # 템플릿은 문자열만 대상으로 삼음
            if isinstance(node, str):
                out.append(prefix)

    def _collect_flat(self, node: Any, *, prefix: str, out: Dict[str, str]) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                self._collect_flat(v, prefix=key, out=out)
        else:
            if isinstance(node, str):
                out[prefix] = node

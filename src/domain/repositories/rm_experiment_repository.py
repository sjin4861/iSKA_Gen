from __future__ import annotations
from typing import Protocol
from src.domain.entities.rm_config import RMExperimentConfig

class RmExperimentRepository(Protocol):
    """RM 실험 설정 로드/저장(필요 시)"""
    def load(self) -> RMExperimentConfig: ...

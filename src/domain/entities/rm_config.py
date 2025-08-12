from __future__ import annotations
from typing import Optional, List
from enum import Enum
from pydantic import Field
from .base import DomainModel
from .rubrics import RubricID

class DatasetType(str, Enum):
    SPF = "SPF"   # Supervised Preference by Filtering
    IMP = "IMP"   # Inter-Model Performance Preference
    ICP = "ICP"   # Intra-Model Contrastive Preference

class RMDatasetSpec(DomainModel):
    """루브릭별 선호도 쌍 데이터셋 스펙"""
    dataset_type: DatasetType
    rubric_ids: List[RubricID] = Field(..., description="이 데이터셋에서 학습할 루브릭 ID 목록")
    pairs_target: int = Field(1000, ge=1, description="루브릭당 목표 쌍 수")

class RMHyperparams(DomainModel):
    base_model: str = "K-intelligence/Midm-2.0-Mini-Instruct"
    learning_rate: float = 5e-5
    epochs: int = 3
    batch_size_per_device: int = 2
    grad_accum_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    optimizer: str = "AdamW"
    center_rewards_coefficient: float = 0.01

class EvaluationProtocolConfig(DomainModel):
    """실험 프로토콜(문서의 3번 항목)을 구조화"""
    test_samples: int = 100
    ensembles: int = 6
    top_k: int = 25
    score_aggregation: str = "mean_vs_min"  # "mean" | "min" | "mean_vs_min"
    gold_refs: List[str] = Field(default_factory=lambda: ["gpt4o_top25", "expert_top25"])

class RMExperimentConfig(DomainModel):
    """전체 실험 설정"""
    datasets: List[RMDatasetSpec]
    hyperparams: RMHyperparams = RMHyperparams()
    eval_protocol: EvaluationProtocolConfig = EvaluationProtocolConfig()

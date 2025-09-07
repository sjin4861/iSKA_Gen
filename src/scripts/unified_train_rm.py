#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 Reward Model 학습 스크립트

이 스크립트는 다양한 설정으로 Reward Model을 학습할 수 있는 통합 도구입니다.

주요 기능:
1. 유연한 설정 관리 (YAML 파일 + 명령줄 인자)
2. 다양한 데이터셋 지원
3. 실험 추적 및 로깅
4. 체크포인트 관리
5. 모델 평가 및 검증

사용법 예시:
    # 기본 설정으로 학습
    python unified_train_rm.py --experiment-name "rm_l2_v1"
    
    # 사용자 정의 데이터셋으로 학습
    python unified_train_rm.py --train-data "path/to/train.jsonl" --eval-data "path/to/eval.jsonl"
    
    # 특정 모델과 설정으로 학습
    python unified_train_rm.py --base-model "custom/model" --config "custom_config.yaml"
    
    # 체크포인트에서 재개
    python unified_train_rm.py --resume-from "saves/checkpoint-100"
    
    # GPU 설정
    python unified_train_rm.py --gpus "0,1" --batch-size 16
"""

import argparse
import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType
from trl import RewardConfig, RewardTrainer

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# 모듈 임포트
from src.model_loader import load_model_for_reward_training
from src.data_loader import load_and_preprocess_data_chat

# ================= 설정 및 상수 =================

DEFAULT_BASE_MODELS = [
    "K-intelligence/Midm-2.0-Mini-Instruct",
    "microsoft/DialoGPT-medium",
    "facebook/opt-350m"
]

DEFAULT_DATASETS = {
    "l2_v3": {
        "train": "data_store/rm_pair/v3/l2_train.jsonl",
        "eval": "data_store/rm_pair/v3/l2_eval.jsonl"
    },
    "korean_quality": {
        "train": "data_store/rm_pair/korean_quality/train.jsonl",
        "eval": "data_store/rm_pair/korean_quality/eval.jsonl"
    },
    "completeness": {
        "train": "data_store/rm_pair/completeness/train.jsonl",
        "eval": "data_store/rm_pair/completeness/eval.jsonl"
    }
}

# ================= 설정 데이터 클래스 =================

@dataclass
class ModelConfig:
    """모델 설정"""
    base_model: str = "K-intelligence/Midm-2.0-Mini-Instruct"
    num_labels: int = 1
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    
    # LoRA 설정
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

@dataclass
class TrainingConfig:
    """학습 설정"""
    # 기본 학습 설정
    output_dir: str = "saves/rm_training"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 1024
    
    # 평가 및 저장
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # 로깅
    logging_dir: str = None
    logging_steps: int = 10
    report_to: List[str] = None
    
    # 기타
    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    
    def __post_init__(self):
        if self.logging_dir is None:
            self.logging_dir = f"{self.output_dir}/logs"
        if self.report_to is None:
            self.report_to = ["tensorboard"]

@dataclass
class DataConfig:
    """데이터 설정"""
    dataset_name: str = "l2_v3"
    train_file: str = None
    eval_file: str = None
    max_samples: Optional[int] = None
    test_size: float = 0.1
    shuffle: bool = True
    
    def get_data_files(self) -> Dict[str, str]:
        """데이터 파일 경로 반환"""
        if self.train_file and self.eval_file:
            return {"train": self.train_file, "eval": self.eval_file}
        elif self.dataset_name in DEFAULT_DATASETS:
            return DEFAULT_DATASETS[self.dataset_name]
        else:
            raise ValueError(f"알 수 없는 데이터셋: {self.dataset_name}")

@dataclass
class ExperimentConfig:
    """실험 설정"""
    experiment_name: str = f"rm_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    description: str = ""
    tags: List[str] = None
    resume_from_checkpoint: Optional[str] = None
    seed: int = 42
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

# ================= 유틸리티 함수 =================

def setup_logging(log_level: str = "INFO"):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

def load_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """YAML 파일에서 설정 로드"""
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def save_config(config: Dict[str, Any], output_path: Path):
    """설정을 파일로 저장"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # YAML 저장
    yaml_path = output_path.with_suffix('.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # JSON 저장 (더 쉬운 파싱을 위해)
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logging.info(f"설정 저장: {yaml_path}, {json_path}")

def parse_gpu_list(gpu_str: str) -> List[int]:
    """GPU 문자열을 리스트로 변환"""
    try:
        return [int(gpu.strip()) for gpu in gpu_str.split(",") if gpu.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"GPU ID는 숫자여야 합니다: {gpu_str}")

def set_seed(seed: int):
    """시드 설정"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CUDA 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================= 메인 학습 클래스 =================

class RewardModelTrainer:
    """Reward Model 학습기"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig, 
                 data_config: DataConfig, experiment_config: ExperimentConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.experiment_config = experiment_config
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 시드 설정
        set_seed(self.experiment_config.seed)
        
        # 출력 디렉토리 설정
        self.output_dir = Path(self.training_config.output_dir) / self.experiment_config.experiment_name
        self.training_config.output_dir = str(self.output_dir)
        self.training_config.logging_dir = str(self.output_dir / "logs")
        
    def setup_model_and_tokenizer(self):
        """모델과 토크나이저 설정"""
        self.logger.info(f"모델 로드 시작: {self.model_config.base_model}")
        
        # 기존 모델 로더 사용하되, 설정을 딕셔너리로 변환
        model_config_dict = {
            "model_name": self.model_config.base_model,
            "num_labels": self.model_config.num_labels,
            "torch_dtype": self.model_config.torch_dtype,
            "trust_remote_code": self.model_config.trust_remote_code,
            "lora": {
                "r": self.model_config.lora_r,
                "alpha": self.model_config.lora_alpha,
                "dropout": self.model_config.lora_dropout,
                "target_modules": self.model_config.target_modules
            }
        }
        
        self.model, self.tokenizer, self.peft_config = load_model_for_reward_training(model_config_dict)
        self.logger.info("모델 로드 완료")
    
    def setup_datasets(self):
        """데이터셋 설정"""
        self.logger.info("데이터셋 로드 시작")
        
        data_files = self.data_config.get_data_files()
        
        # 절대 경로로 변환
        train_path = PROJECT_ROOT / data_files["train"]
        eval_path = PROJECT_ROOT / data_files["eval"]
        
        if not train_path.exists():
            raise FileNotFoundError(f"학습 데이터를 찾을 수 없습니다: {train_path}")
        if not eval_path.exists():
            raise FileNotFoundError(f"평가 데이터를 찾을 수 없습니다: {eval_path}")
        
        # 데이터 로드
        self.train_dataset = load_and_preprocess_data_chat(
            str(train_path), 
            self.tokenizer, 
            self.training_config.max_length
        )
        
        self.eval_dataset = load_and_preprocess_data_chat(
            str(eval_path), 
            self.tokenizer, 
            self.training_config.max_length
        )
        
        # 샘플 수 제한
        if self.data_config.max_samples:
            if len(self.train_dataset) > self.data_config.max_samples:
                self.train_dataset = self.train_dataset.select(range(self.data_config.max_samples))
        
        self.logger.info(f"데이터셋 로드 완료 - 학습: {len(self.train_dataset)}, 평가: {len(self.eval_dataset)}")
    
    def setup_trainer(self):
        """트레이너 설정"""
        self.logger.info("트레이너 설정 시작")
        
        # TrainingConfig를 RewardConfig로 변환
        training_args_dict = asdict(self.training_config)
        training_args_dict.pop('max_length', None)  # RewardConfig에 없는 필드 제거
        
        training_args = RewardConfig(**training_args_dict)
        
        self.trainer = RewardTrainer(
            model=self.model,
            args=training_args,
            processing_class=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            peft_config=self.peft_config,
        )
        
        self.logger.info("트레이너 설정 완료")
    
    def save_experiment_config(self):
        """실험 설정 저장"""
        config_dict = {
            "experiment": asdict(self.experiment_config),
            "model": asdict(self.model_config),
            "training": asdict(self.training_config),
            "data": asdict(self.data_config),
            "timestamp": datetime.now().isoformat()
        }
        
        config_path = self.output_dir / "experiment_config"
        save_config(config_dict, config_path)
    
    def train(self):
        """학습 실행"""
        self.logger.info(f"=== 실험 시작: {self.experiment_config.experiment_name} ===")
        self.logger.info(f"설명: {self.experiment_config.description}")
        self.logger.info(f"태그: {', '.join(self.experiment_config.tags)}")
        
        try:
            # 설정
            self.setup_model_and_tokenizer()
            self.setup_datasets()
            self.setup_trainer()
            
            # 실험 설정 저장
            self.save_experiment_config()
            
            # 학습 시작
            self.logger.info("학습 시작")
            
            if self.experiment_config.resume_from_checkpoint:
                self.logger.info(f"체크포인트에서 재개: {self.experiment_config.resume_from_checkpoint}")
                train_result = self.trainer.train(resume_from_checkpoint=self.experiment_config.resume_from_checkpoint)
            else:
                train_result = self.trainer.train()
            
            # 최종 모델 저장
            self.trainer.save_model()
            
            self.logger.info(f"✅ 학습 완료! 모델이 '{self.output_dir}'에 저장되었습니다.")
            self.logger.info(f"학습 통계: {train_result}")
            
            return train_result
            
        except Exception as e:
            self.logger.error(f"❌ 학습 중 오류 발생: {e}")
            raise
    
    def evaluate(self):
        """모델 평가"""
        self.logger.info("모델 평가 시작")
        eval_result = self.trainer.evaluate()
        self.logger.info(f"평가 결과: {eval_result}")
        return eval_result

# ================= 명령줄 인터페이스 =================

def create_parser():
    """명령줄 인자 파서 생성"""
    parser = argparse.ArgumentParser(
        description="통합 Reward Model 학습 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 설정으로 학습
  python unified_train_rm.py --experiment-name "rm_l2_v1"
  
  # 사용자 정의 데이터와 모델
  python unified_train_rm.py --base-model "custom/model" --train-data "data/train.jsonl" --eval-data "data/eval.jsonl"
  
  # 배치 크기와 에폭 수 조정
  python unified_train_rm.py --batch-size 8 --epochs 5 --learning-rate 1e-5
  
  # 체크포인트에서 재개
  python unified_train_rm.py --resume-from "saves/checkpoint-100"
  
  # GPU 설정
  python unified_train_rm.py --gpus "0,1" --experiment-name "multi_gpu_exp"
        """
    )
    
    # 실험 설정
    parser.add_argument("--experiment-name", type=str,
                        default=f"rm_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="실험 이름")
    parser.add_argument("--description", type=str, default="",
                        help="실험 설명")
    parser.add_argument("--tags", type=str, nargs="+", default=[],
                        help="실험 태그")
    parser.add_argument("--resume-from", type=str,
                        help="재개할 체크포인트 경로")
    
    # 모델 설정
    parser.add_argument("--base-model", type=str,
                        default="K-intelligence/Midm-2.0-Mini-Instruct",
                        help="베이스 모델")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    
    # 데이터 설정
    parser.add_argument("--dataset", type=str, default="l2_v3",
                        choices=list(DEFAULT_DATASETS.keys()),
                        help="사용할 데이터셋")
    parser.add_argument("--train-data", type=str,
                        help="학습 데이터 파일 경로")
    parser.add_argument("--eval-data", type=str,
                        help="평가 데이터 파일 경로")
    parser.add_argument("--max-samples", type=int,
                        help="최대 샘플 수")
    
    # 학습 설정
    parser.add_argument("--output-dir", type=str, default="saves/rm_training",
                        help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=3,
                        help="에폭 수")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="배치 크기")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="학습률")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="최대 시퀀스 길이")
    
    # 하드웨어 설정
    parser.add_argument("--gpus", type=str, default="0",
                        help="사용할 GPU (쉼표로 구분)")
    parser.add_argument("--fp16", action="store_true",
                        help="FP16 사용")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="BF16 사용")
    
    # 기타
    parser.add_argument("--config", type=str,
                        help="사용자 정의 설정 파일 (YAML)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="로그 레벨")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드")
    parser.add_argument("--dry-run", action="store_true",
                        help="설정만 확인하고 실제 학습은 하지 않음")
    
    return parser

def main():
    """메인 실행 함수"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # GPU 설정
        gpus = parse_gpu_list(args.gpus)
        if gpus:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
            logger.info(f"GPU 설정: {gpus}")
        
        # 설정 로드 및 생성
        base_config = {}
        if args.config:
            base_config = load_config_from_yaml(Path(args.config))
            logger.info(f"사용자 정의 설정 로드: {args.config}")
        
        # 설정 객체 생성 (명령줄 인자가 우선)
        experiment_config = ExperimentConfig(
            experiment_name=args.experiment_name,
            description=args.description,
            tags=args.tags,
            resume_from_checkpoint=args.resume_from,
            seed=args.seed
        )
        
        model_config = ModelConfig(
            base_model=args.base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            **base_config.get("model", {})
        )
        
        training_config = TrainingConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            fp16=args.fp16,
            bf16=args.bf16,
            **base_config.get("training", {})
        )
        
        data_config = DataConfig(
            dataset_name=args.dataset,
            train_file=args.train_data,
            eval_file=args.eval_data,
            max_samples=args.max_samples,
            **base_config.get("data", {})
        )
        
        # 설정 요약 출력
        logger.info("=== 실험 설정 요약 ===")
        logger.info(f"실험명: {experiment_config.experiment_name}")
        logger.info(f"모델: {model_config.base_model}")
        logger.info(f"데이터셋: {data_config.dataset_name}")
        logger.info(f"출력 경로: {training_config.output_dir}")
        logger.info(f"에폭: {training_config.num_train_epochs}")
        logger.info(f"배치 크기: {training_config.per_device_train_batch_size}")
        logger.info(f"학습률: {training_config.learning_rate}")
        
        if args.dry_run:
            logger.info("🏁 Dry run 완료 - 실제 학습은 수행하지 않습니다.")
            return 0
        
        # 학습 실행
        trainer = RewardModelTrainer(model_config, training_config, data_config, experiment_config)
        train_result = trainer.train()
        
        logger.info("🎉 모든 작업이 성공적으로 완료되었습니다!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 스크립트 실행 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

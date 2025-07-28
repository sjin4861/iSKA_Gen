"""
RM 훈련 데이터셋 생성을 위한 유틸리티 모듈

이 모듈은 세 가지 유형의 선호도 쌍 데이터셋을 생성합니다:
1. SPF (Supervised Preference Dataset by Filtering)
2. IMP (Inter-Model Performance Preference Dataset)  
3. ICP (Intra-Model Contrastive Preference Dataset)
"""

import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'modules'))
sys.path.append(str(project_root / 'utils'))

from modules.model_client import OpenAIModelClient
from utils.prompt_loader import get_prompt


class RMDatasetGenerator:
    """RM 훈련용 데이터셋 생성기"""
    
    def __init__(self, openai_model: str = "gpt-4o"):
        """
        초기화
        
        Args:
            openai_model (str): 사용할 OpenAI 모델명
        """
        self.openai_model = openai_model
        self.client = OpenAIModelClient(model_name=openai_model)
        self.data_dir = Path(__file__).parent.parent / "data"
        self.rm_training_dir = self.data_dir / "rm_training"
        
        # 최신 6개 루브릭 기준 정의 (RM_Experiment_v1.0.0.md)
        self.rubrics = [
            "completeness_for_guidelines",      # 평가 지침 완전성
            "core_theme_clarity",               # 핵심 주제 명확성
            "reference_groundedness",           # 참고 자료 기반성
            "logical_flow_and_structure",       # 논리적 흐름 및 구조
            "korean_quality",                   # 한국어 품질
            "l2_learner_suitability"            # L2 학습자 적합성
        ]
        
        print(f"🎯 RM 데이터셋 생성기 초기화 완료")
        print(f"   OpenAI 모델: {openai_model}")
        print(f"   데이터 디렉토리: {self.rm_training_dir}")
        print(f"   루브릭 수: {len(self.rubrics)}")

    def load_base_passages(self, file_path: str) -> List[Dict[str, Any]]:
        """
        기본 지문 데이터 로드
        
        Args:
            file_path (str): 지문 파일 경로
            
        Returns:
            List[Dict[str, Any]]: 로드된 지문 데이터
        """
        full_path = self.data_dir / "base_passages" / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"기본 지문 파일을 찾을 수 없습니다: {full_path}")
        
        with open(full_path, 'r', encoding='utf-8') as f:
            passages = json.load(f)
        
        print(f"✅ 기본 지문 로드 완료: {len(passages)}개")
        return passages

    def evaluate_passage_with_gpt4o(self, passage: str, rubric: str) -> Dict[str, Any]:
        """
        GPT-4o를 사용하여 특정 루브릭 기준으로 지문 평가
        
        Args:
            passage (str): 평가할 지문
            rubric (str): 평가 기준 (루브릭)
            
        Returns:
            Dict[str, Any]: 평가 결과 (pass/fail, score, reason)
        """
        try:
            # 루브릭별 평가 프롬프트 생성
            prompt = get_prompt(
                f"rubric_evaluation.{rubric}", 
                agent="iska",
                passage=passage
            )
            
            response = self.client.call([{"role": "user", "content": prompt}])
            
            # 응답 파싱 (예: "PASS" 또는 "FAIL"로 시작하는 응답)
            if response.strip().upper().startswith("PASS"):
                result = {
                    "rubric": rubric,
                    "result": "pass",
                    "score": 1.0,
                    "reason": response.strip()
                }
            elif response.strip().upper().startswith("FAIL"):
                result = {
                    "rubric": rubric,
                    "result": "fail", 
                    "score": 0.0,
                    "reason": response.strip()
                }
            else:
                # 애매한 경우 기본값
                result = {
                    "rubric": rubric,
                    "result": "uncertain",
                    "score": 0.5,
                    "reason": response.strip()
                }
            
            return result
            
        except Exception as e:
            print(f"⚠️ GPT-4o 평가 실패 ({rubric}): {e}")
            return {
                "rubric": rubric,
                "result": "error",
                "score": 0.0,
                "reason": f"평가 실패: {str(e)}"
            }

    def generate_spf_dataset(
        self, 
        passages: List[Dict[str, Any]], 
        target_pairs_per_rubric: int = 1000
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        SPF (Supervised Preference Dataset by Filtering) 생성
        
        Args:
            passages (List[Dict[str, Any]]): 기본 지문 데이터
            target_pairs_per_rubric (int): 루브릭당 목표 쌍 개수
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 루브릭별 선호도 쌍 데이터
        """
        print(f"🔄 SPF 데이터셋 생성 시작...")
        print(f"   입력 지문 수: {len(passages)}")
        print(f"   루브릭당 목표 쌍 수: {target_pairs_per_rubric}")
        
        spf_dataset = {}
        
        for rubric in self.rubrics:
            print(f"\n📊 {rubric} 루브릭 처리 중...")
            
            positive_passages = []
            negative_passages = []
            
            # 각 지문을 해당 루브릭으로 평가
            for i, passage_data in enumerate(passages):
                if i % 100 == 0:
                    print(f"   진행률: {i}/{len(passages)}")
                
                passage_text = passage_data.get("text", passage_data.get("generated_passage", ""))
                
                evaluation = self.evaluate_passage_with_gpt4o(passage_text, rubric)
                
                passage_with_eval = passage_data.copy()
                passage_with_eval["evaluation"] = evaluation
                
                if evaluation["result"] == "pass":
                    positive_passages.append(passage_with_eval)
                elif evaluation["result"] == "fail":
                    negative_passages.append(passage_with_eval)
            
            print(f"   ✅ {rubric}: Positive {len(positive_passages)}개, Negative {len(negative_passages)}개")
            
            # 선호도 쌍 생성
            pairs = []
            min_count = min(len(positive_passages), len(negative_passages), target_pairs_per_rubric)
            
            # 랜덤 샘플링으로 쌍 생성
            positive_sample = random.sample(positive_passages, min_count)
            negative_sample = random.sample(negative_passages, min_count)
            
            for pos, neg in zip(positive_sample, negative_sample):
                pair = {
                    "pair_id": f"spf_{rubric}_{len(pairs)+1:04d}",
                    "rubric": rubric,
                    "chosen": pos.get("text", pos.get("generated_passage", "")),
                    "rejected": neg.get("text", neg.get("generated_passage", "")),
                    "chosen_metadata": pos,
                    "rejected_metadata": neg,
                    "dataset_type": "SPF",
                    "created_at": datetime.now().isoformat()
                }
                pairs.append(pair)
            
            spf_dataset[rubric] = pairs
            print(f"   📝 {rubric}: {len(pairs)}개 쌍 생성 완료")
        
        return spf_dataset

    def generate_imp_dataset(
        self,
        high_performance_passages: List[Dict[str, Any]],
        low_performance_passages: List[Dict[str, Any]],
        target_pairs_per_rubric: int = 1000
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        IMP (Inter-Model Performance Preference Dataset) 생성
        
        Args:
            high_performance_passages: 고성능 모델 생성 지문 (98% 달성률)
            low_performance_passages: 저성능 모델 생성 지문 (40% 달성률)
            target_pairs_per_rubric: 루브릭당 목표 쌍 개수
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 루브릭별 선호도 쌍 데이터
        """
        print(f"🔄 IMP 데이터셋 생성 시작...")
        print(f"   고성능 지문 수: {len(high_performance_passages)}")
        print(f"   저성능 지문 수: {len(low_performance_passages)}")
        
        imp_dataset = {}
        
        for rubric in self.rubrics:
            print(f"\n📊 {rubric} 루브릭 처리 중...")
            
            pairs = []
            min_count = min(
                len(high_performance_passages), 
                len(low_performance_passages), 
                target_pairs_per_rubric
            )
            
            # 랜덤 샘플링으로 쌍 생성
            high_sample = random.sample(high_performance_passages, min_count)
            low_sample = random.sample(low_performance_passages, min_count)
            
            for high, low in zip(high_sample, low_sample):
                pair = {
                    "pair_id": f"imp_{rubric}_{len(pairs)+1:04d}",
                    "rubric": rubric,
                    "chosen": high.get("text", high.get("generated_passage", "")),
                    "rejected": low.get("text", low.get("generated_passage", "")),
                    "chosen_metadata": high,
                    "rejected_metadata": low,
                    "dataset_type": "IMP",
                    "created_at": datetime.now().isoformat()
                }
                pairs.append(pair)
            
            imp_dataset[rubric] = pairs
            print(f"   📝 {rubric}: {len(pairs)}개 쌍 생성 완료")
        
        return imp_dataset

    def generate_contrasted_passage(
        self, 
        base_passage: str, 
        rubric: str, 
        violation_type: str = "negative"
    ) -> str:
        """
        기본 지문을 바탕으로 특정 루브릭을 위반하는 대조 지문 생성
        
        Args:
            base_passage (str): 기본 지문
            rubric (str): 위반할 루브릭
            violation_type (str): 위반 유형
            
        Returns:
            str: 대조 지문
        """
        try:
            prompt = get_prompt(
                f"contrastive_generation.{rubric}_{violation_type}",
                agent="iska", 
                base_passage=base_passage
            )
            
            contrasted_passage = self.client.call([{"role": "user", "content": prompt}])
            return contrasted_passage.strip()
            
        except Exception as e:
            print(f"⚠️ 대조 지문 생성 실패 ({rubric}): {e}")
            return base_passage  # 실패시 원본 반환

    def generate_icp_dataset(
        self,
        base_passages: List[Dict[str, Any]],
        target_pairs_per_rubric: int = 1000
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        ICP (Intra-Model Contrastive Preference Dataset) 생성
        
        Args:
            base_passages: 기본 지문 데이터
            target_pairs_per_rubric: 루브릭당 목표 쌍 개수
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 루브릭별 선호도 쌍 데이터
        """
        print(f"🔄 ICP 데이터셋 생성 시작...")
        print(f"   기본 지문 수: {len(base_passages)}")
        
        icp_dataset = {}
        
        for rubric in self.rubrics:
            print(f"\n📊 {rubric} 루브릭 처리 중...")
            
            pairs = []
            sample_passages = random.sample(
                base_passages, 
                min(len(base_passages), target_pairs_per_rubric)
            )
            
            for i, passage_data in enumerate(sample_passages):
                if i % 100 == 0:
                    print(f"   진행률: {i}/{len(sample_passages)}")
                
                base_text = passage_data.get("text", passage_data.get("generated_passage", ""))
                
                # 루브릭 위반 지문 생성
                violated_text = self.generate_contrasted_passage(base_text, rubric)
                
                pair = {
                    "pair_id": f"icp_{rubric}_{len(pairs)+1:04d}",
                    "rubric": rubric,
                    "chosen": base_text,  # 원본 (품질 기준 충족)
                    "rejected": violated_text,  # 위반 지문
                    "chosen_metadata": passage_data,
                    "rejected_metadata": {
                        "violation_type": rubric,
                        "base_passage_id": passage_data.get("id", f"base_{i}")
                    },
                    "dataset_type": "ICP",
                    "created_at": datetime.now().isoformat()
                }
                pairs.append(pair)
            
            icp_dataset[rubric] = pairs
            print(f"   📝 {rubric}: {len(pairs)}개 쌍 생성 완료")
        
        return icp_dataset

    def save_dataset(
        self, 
        dataset: Dict[str, List[Dict[str, Any]]], 
        dataset_type: str
    ) -> Dict[str, str]:
        """
        생성된 데이터셋을 파일로 저장
        
        Args:
            dataset: 저장할 데이터셋
            dataset_type: 데이터셋 유형 (spf, imp, icp)
            
        Returns:
            Dict[str, str]: 저장된 파일 경로들
        """
        dataset_dir = self.rm_training_dir / dataset_type.lower()
        dataset_dir.mkdir(exist_ok=True)
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for rubric, pairs in dataset.items():
            filename = f"{dataset_type.upper()}_{rubric}_{timestamp}.json"
            file_path = dataset_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(pairs, f, ensure_ascii=False, indent=2)
            
            saved_files[rubric] = str(file_path)
            print(f"💾 {rubric}: {filename} 저장됨 ({len(pairs)}개 쌍)")
        
        # 전체 데이터셋도 저장
        full_filename = f"{dataset_type.upper()}_complete_{timestamp}.json"
        full_path = dataset_dir / full_filename
        
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        saved_files["complete"] = str(full_path)
        print(f"💾 전체 데이터셋: {full_filename} 저장됨")
        
        return saved_files

    def load_dataset(self, dataset_type: str, rubric: Optional[str] = None) -> Dict[str, Any]:
        """
        저장된 데이터셋 로드
        
        Args:
            dataset_type: 데이터셋 유형 (spf, imp, icp)
            rubric: 특정 루브릭 (None이면 전체 로드)
            
        Returns:
            Dict[str, Any]: 로드된 데이터셋
        """
        dataset_dir = self.rm_training_dir / dataset_type.lower()
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"데이터셋 디렉토리가 없습니다: {dataset_dir}")
        
        if rubric:
            # 특정 루브릭 파일 찾기
            pattern = f"{dataset_type.upper()}_{rubric}_*.json"
            files = list(dataset_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(f"루브릭 데이터를 찾을 수 없습니다: {pattern}")
            
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 전체 데이터셋 파일 찾기
            pattern = f"{dataset_type.upper()}_complete_*.json"
            files = list(dataset_dir.glob(pattern))
            if not files:
                raise FileNotFoundError(f"전체 데이터셋을 찾을 수 없습니다: {pattern}")
            
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)

    def get_dataset_stats(self, dataset: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        데이터셋 통계 정보 반환
        
        Args:
            dataset: 분석할 데이터셋
            
        Returns:
            Dict[str, Any]: 통계 정보
        """
        stats = {
            "rubrics": list(dataset.keys()),
            "total_rubrics": len(dataset),
            "pairs_per_rubric": {},
            "total_pairs": 0
        }
        
        for rubric, pairs in dataset.items():
            stats["pairs_per_rubric"][rubric] = len(pairs)
            stats["total_pairs"] += len(pairs)
        
        stats["average_pairs_per_rubric"] = stats["total_pairs"] / stats["total_rubrics"] if stats["total_rubrics"] > 0 else 0
        
        return stats


# 편의 함수들
def create_spf_dataset(
    passages_file: str,
    target_pairs_per_rubric: int = 1000,
    openai_model: str = "gpt-4o"
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """
    SPF 데이터셋 생성 편의 함수
    
    Returns:
        Tuple[dataset, saved_files]: 생성된 데이터셋과 저장된 파일 경로들
    """
    generator = RMDatasetGenerator(openai_model)
    passages = generator.load_base_passages(passages_file)
    dataset = generator.generate_spf_dataset(passages, target_pairs_per_rubric)
    saved_files = generator.save_dataset(dataset, "SPF")
    
    stats = generator.get_dataset_stats(dataset)
    print(f"\n📊 SPF 데이터셋 통계:")
    print(f"   총 루브릭 수: {stats['total_rubrics']}")
    print(f"   총 쌍 수: {stats['total_pairs']}")
    print(f"   루브릭당 평균 쌍 수: {stats['average_pairs_per_rubric']:.1f}")
    
    return dataset, saved_files


def create_imp_dataset(
    high_perf_file: str,
    low_perf_file: str,
    target_pairs_per_rubric: int = 1000,
    openai_model: str = "gpt-4o"
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """
    IMP 데이터셋 생성 편의 함수
    
    Returns:
        Tuple[dataset, saved_files]: 생성된 데이터셋과 저장된 파일 경로들
    """
    generator = RMDatasetGenerator(openai_model)
    high_passages = generator.load_base_passages(high_perf_file)
    low_passages = generator.load_base_passages(low_perf_file)
    dataset = generator.generate_imp_dataset(high_passages, low_passages, target_pairs_per_rubric)
    saved_files = generator.save_dataset(dataset, "IMP")
    
    stats = generator.get_dataset_stats(dataset)
    print(f"\n📊 IMP 데이터셋 통계:")
    print(f"   총 루브릭 수: {stats['total_rubrics']}")
    print(f"   총 쌍 수: {stats['total_pairs']}")
    print(f"   루브릭당 평균 쌍 수: {stats['average_pairs_per_rubric']:.1f}")
    
    return dataset, saved_files


def create_icp_dataset(
    base_passages_file: str,
    target_pairs_per_rubric: int = 1000,
    openai_model: str = "gpt-4o"
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    """
    ICP 데이터셋 생성 편의 함수
    
    Returns:
        Tuple[dataset, saved_files]: 생성된 데이터셋과 저장된 파일 경로들
    """
    generator = RMDatasetGenerator(openai_model)
    passages = generator.load_base_passages(base_passages_file)
    dataset = generator.generate_icp_dataset(passages, target_pairs_per_rubric)
    saved_files = generator.save_dataset(dataset, "ICP")
    
    stats = generator.get_dataset_stats(dataset)
    print(f"\n📊 ICP 데이터셋 통계:")
    print(f"   총 루브릭 수: {stats['total_rubrics']}")
    print(f"   총 쌍 수: {stats['total_pairs']}")
    print(f"   루브릭당 평균 쌍 수: {stats['average_pairs_per_rubric']:.1f}")
    
    return dataset, saved_files


if __name__ == "__main__":
    print("🚀 RM 데이터셋 생성기 모듈 로드됨")
    print("📚 사용 가능한 함수:")
    print("   - create_spf_dataset(): SPF 데이터셋 생성")
    print("   - create_imp_dataset(): IMP 데이터셋 생성") 
    print("   - create_icp_dataset(): ICP 데이터셋 생성")
    print("   - RMDatasetGenerator: 메인 생성기 클래스")

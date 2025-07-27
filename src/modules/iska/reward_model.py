import sys
import torch
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from trl import AutoModelForCausalLMWithValueHead
    TRL_AVAILABLE = True
except ImportError:
    TRL_AVAILABLE = False
    print("⚠️ TRL 라이브러리가 설치되지 않았습니다. 일반 모델로 대체합니다.")

try:
    from peft import AutoPeftModelForCausalLM, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️ PEFT 라이브러리가 설치되지 않았습니다. PEFT 모델 지원이 제한됩니다.")


class RewardModel:
    """
    Reward Model을 사용하여 텍스트 품질을 평가하는 클래스
    """
    
    def __init__(self, model_path: str, device: str = "auto", seed: int = 42, gpus: Optional[List[int]] = None):
        """
        Reward Model 초기화
        
        Args:
            model_path (str): 모델 경로
            device (str): 사용할 디바이스 ("auto", "cuda", "cpu")
            seed (int): 랜덤 시드
            gpus (Optional[List[int]]): 사용할 GPU 리스트 (예: [0, 1])
        """
        # 모델 경로 틸드 확장
        self.model_path = str(Path(model_path).expanduser())
        self.seed = seed
        self.gpus = gpus
        self.set_seed(seed)
        
        # 디바이스 설정
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🎯 Reward Model 초기화 중...")
        print(f"   모델 경로: {self.model_path}")
        print(f"   디바이스: {self.device}")
        print(f"   GPU 리스트: {self.gpus}")
        print(f"   시드: {seed}")
        
        # GPU 설정
        if self.gpus is not None and torch.cuda.is_available():
            if isinstance(self.gpus, list) and all(isinstance(i, int) for i in self.gpus):
                # GPU 환경 변수 설정
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpus))
                print(f"🎯 CUDA_VISIBLE_DEVICES 설정: {','.join(map(str, self.gpus))}")
                torch.cuda.empty_cache()
        
        # 모델과 토크나이저 로드
        self._load_model()
    
    def set_seed(self, seed: int):
        """Deterministic 설정을 위한 seed 고정"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _load_model(self):
        """모델과 토크나이저 로드"""
        try:
            print(f"📥 토크나이저 로딩 중...")
            
            # PEFT 모델인지 확인
            adapter_config_path = Path(self.model_path) / "adapter_config.json"
            is_peft_model = adapter_config_path.exists()
            
            if is_peft_model and PEFT_AVAILABLE:
                print(f"🔧 PEFT 어댑터 모델 감지됨")
                try:
                    # PEFT 모델로 로드
                    self.model = AutoPeftModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        device_map=self._get_device_map()
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self.model_type = "peft"
                    print("✅ PEFT 모델 로딩 완료")
                except Exception as e:
                    print(f"⚠️ PEFT 모델 로딩 실패: {e}")
                    print("� 일반 모델로 fallback 시도...")
                    self._load_standard_model_fallback()
            else:
                # 일반 모델 로딩
                print(f"🔧 일반 모델로 로딩 중...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self._load_standard_model_with_device_map()
            
            self.model.eval()
            
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            raise
    
    def _get_device_map(self):
        """디바이스 매핑 설정 반환"""
        if self.gpus is not None and torch.cuda.is_available():
            if len(self.gpus) == 1:
                return {"": "cuda:0"}  # 단일 GPU
            else:
                return "auto"  # 다중 GPU
        elif self.device.type == "cuda":
            return "auto"
        else:
            return None  # CPU
    
    def _load_standard_model_with_device_map(self):
        """일반 모델을 디바이스 매핑과 함께 로드"""
        device_map = self._get_device_map()
        
        # TRL이 사용 가능하면 Value Head가 있는 모델 시도
        if TRL_AVAILABLE:
            try:
                self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map=device_map
                )
                self.model_type = "value_head"
                print("✅ Value Head가 있는 Reward Model 로딩 완료")
            except Exception as e:
                print(f"⚠️ Value Head 모델 로딩 실패, 일반 모델로 시도: {e}")
                self._load_standard_model(device_map)
        else:
            self._load_standard_model(device_map)
    
    def _load_standard_model_fallback(self):
        """PEFT 실패 시 fallback 모델 로딩"""
        try:
            import json
            adapter_config_path = Path(self.model_path) / "adapter_config.json"
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            
            base_model_path = adapter_config.get("base_model_name_or_path", "Qwen/Qwen3-4B")
            print(f"🔄 베이스 모델로 fallback: {base_model_path}")
            
            # 베이스 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map=self._get_device_map()
            )
            self.model_type = "fallback_base"
            print("✅ 베이스 모델로 fallback 완료")
            
        except Exception as e:
            print(f"❌ Fallback도 실패: {e}")
            raise
    
    def _load_standard_model(self, device_map=None):
        """일반 CausalLM 모델 로딩"""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=device_map
        )
        self.model_type = "standard"
        print("✅ 일반 CausalLM 모델 로딩 완료 (Logit을 점수로 사용)")
    
    def evaluate_text(self, text: str, max_length: int = 512) -> float:
        """
        단일 텍스트의 품질을 평가
        
        Args:
            text (str): 평가할 텍스트
            max_length (int): 최대 토큰 길이
            
        Returns:
            float: 품질 점수
        """
        # 입력 토큰화
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 시드 재설정 (일관된 결과를 위해)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            
            outputs = self.model(**inputs)
            
            # 모델 타입에 따른 점수 추출
            if self.model_type == "value_head":
                return self._extract_value_score(outputs)
            else:
                return self._extract_logit_score(outputs)
    
    def _extract_value_score(self, outputs) -> float:
        """Value Head에서 점수 추출"""
        values = None
        
        # 다양한 출력 형태 처리
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            values = outputs[2]  # (logits, past_key_values, values)
        elif hasattr(outputs, 'value'):
            values = outputs.value
        elif hasattr(outputs, 'values'):
            values = outputs.values
        
        if values is not None:
            # 마지막 토큰의 reward 점수 추출
            reward_score = values[0, -1].item()
            return reward_score
        else:
            print("⚠️ Value를 찾을 수 없어 Logit을 사용합니다.")
            return self._extract_logit_score(outputs)
    
    def _extract_logit_score(self, outputs) -> float:
        """일반 모델에서 Logit 기반 점수 추출"""
        if hasattr(outputs, 'logits'):
            # 마지막 토큰의 모든 logit 중 평균값을 점수로 사용
            last_token_logits = outputs.logits[0, -1, :]
            reward_score = last_token_logits.mean().item()
            return reward_score
        else:
            print(f"❌ 예상과 다른 출력 구조: {type(outputs)}")
            return 0.0
    
    def evaluate_multiple(self, texts: List[str], max_length: int = 512) -> List[Dict[str, Any]]:
        """
        여러 텍스트를 일괄 평가
        
        Args:
            texts (List[str]): 평가할 텍스트 리스트
            max_length (int): 최대 토큰 길이
            
        Returns:
            List[Dict[str, Any]]: 평가 결과 리스트
        """
        results = []
        
        for i, text in enumerate(texts):
            print(f"📊 텍스트 {i+1}/{len(texts)} 평가 중...")
            
            score = self.evaluate_text(text, max_length)
            
            result = {
                "text": text,
                "score": score,
                "index": i
            }
            results.append(result)
        
        return results
    
    def compare_texts(self, chosen_text: str, rejected_text: str, max_length: int = 512) -> Dict[str, Any]:
        """
        두 텍스트를 비교하여 어느 것이 더 좋은지 판단
        
        Args:
            chosen_text (str): 선택된(좋은) 텍스트
            rejected_text (str): 거부된(나쁜) 텍스트
            max_length (int): 최대 토큰 길이
            
        Returns:
            Dict[str, Any]: 비교 결과
        """
        chosen_score = self.evaluate_text(chosen_text, max_length)
        rejected_score = self.evaluate_text(rejected_text, max_length)
        
        return {
            "chosen_text": chosen_text,
            "rejected_text": rejected_text,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "score_difference": chosen_score - rejected_score,
            "correct_preference": chosen_score > rejected_score
        }
    
    def batch_compare(self, pairs: List[Dict[str, str]], max_length: int = 512) -> Dict[str, Any]:
        """
        여러 텍스트 쌍을 일괄 비교
        
        Args:
            pairs (List[Dict[str, str]]): [{"chosen": str, "rejected": str}, ...] 형태의 쌍 리스트
            max_length (int): 최대 토큰 길이
            
        Returns:
            Dict[str, Any]: 일괄 비교 결과
        """
        results = []
        correct_count = 0
        
        for i, pair in enumerate(pairs):
            print(f"🔄 쌍 {i+1}/{len(pairs)} 비교 중...")
            
            comparison = self.compare_texts(
                pair["chosen"], 
                pair["rejected"], 
                max_length
            )
            
            results.append(comparison)
            if comparison["correct_preference"]:
                correct_count += 1
        
        accuracy = correct_count / len(pairs) if pairs else 0
        
        return {
            "comparisons": results,
            "total_pairs": len(pairs),
            "correct_preferences": correct_count,
            "accuracy": accuracy,
            "model_info": {
                "model_path": self.model_path,
                "model_type": self.model_type,
                "device": str(self.device)
            }
        }


def evaluate_passages_with_reward_model(
    model_path: str,
    passages: List[Dict[str, Any]],
    device: str = "auto",
    max_length: int = 512,
    seed: int = 42,
    gpus: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Reward Model을 사용하여 passage들을 평가하는 함수
    
    Args:
        model_path (str): Reward Model 경로
        passages (List[Dict[str, Any]]): 평가할 passage 리스트
        device (str): 사용할 디바이스
        max_length (int): 최대 토큰 길이
        seed (int): 랜덤 시드
        gpus (Optional[List[int]]): 사용할 GPU 리스트
        
    Returns:
        List[Dict[str, Any]]: 평가 결과가 추가된 passage 리스트
    """
    print(f"🎯 Reward Model을 사용한 Passage 평가 시작...")
    print(f"   평가할 passage 개수: {len(passages)}")
    
    # Reward Model 초기화
    reward_model = RewardModel(model_path, device, seed, gpus)
    
    # 평가 실행
    evaluated_passages = []
    
    for i, passage_data in enumerate(passages):
        print(f"📊 Passage {i+1}/{len(passages)} 평가 중...")
        
        passage_text = passage_data.get("generated_passage", "")
        if not passage_text:
            print(f"⚠️ Passage {i+1}: 빈 텍스트, 건너뜀")
            continue
        
        # 점수 계산
        score = reward_model.evaluate_text(passage_text, max_length)
        
        # 결과 추가
        evaluated_passage = passage_data.copy()
        evaluated_passage["reward_score"] = score
        evaluated_passage["evaluation_model"] = model_path
        
        evaluated_passages.append(evaluated_passage)
        
        print(f"   점수: {score:.4f}")
    
    print(f"✅ Reward Model 평가 완료!")
    return evaluated_passages


def compare_passage_pairs_with_reward_model(
    model_path: str,
    passage_pairs: List[Dict[str, Any]],
    device: str = "auto",
    max_length: int = 512,
    seed: int = 42,
    gpus: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Reward Model을 사용하여 passage pair들을 비교 평가하는 함수
    
    Args:
        model_path (str): Reward Model 경로
        passage_pairs (List[Dict[str, Any]]): 비교할 passage pair 리스트
        device (str): 사용할 디바이스
        max_length (int): 최대 토큰 길이
        seed (int): 랜덤 시드
        gpus (Optional[List[int]]): 사용할 GPU 리스트
        
    Returns:
        Dict[str, Any]: 비교 평가 결과
    """
    print(f"🎯 Reward Model을 사용한 Passage Pair 비교 시작...")
    print(f"   비교할 pair 개수: {len(passage_pairs)}")
    
    # Reward Model 초기화
    reward_model = RewardModel(model_path, device, seed, gpus)
    
    # 비교할 pair 데이터 준비
    comparison_pairs = []
    for pair_data in passage_pairs:
        chosen = pair_data.get("chosen", "")
        rejected = pair_data.get("rejected", "")
        
        if chosen and rejected:
            comparison_pairs.append({
                "chosen": chosen,
                "rejected": rejected
            })
    
    # 일괄 비교 실행
    comparison_results = reward_model.batch_compare(comparison_pairs, max_length)
    
    print(f"✅ Reward Model 비교 완료!")
    print(f"   정확도: {comparison_results['accuracy']:.2%}")
    
    return comparison_results


# --- 실행 예시 ---
if __name__ == "__main__":
    # 테스트용 예제
    MODEL_PATH = str(Path("~/models/train_2025-07-26-12-04-23").expanduser())  # 실제 경로로 변경
    
    # 테스트 케이스
    test_passages = [
        {
            "generated_passage": "인공지능 기술이 사회 전반에 확산됨에 따라, 알고리즘의 편향성과 불투명성으로 인해 발생할 수 있는 윤리적 문제에 대한 심도 있는 논의가 요구됩니다.",
            "source_item": {"korean_topic": "AI 윤리"}
        },
        {
            "generated_passage": "AI는 되게 좋지만 가끔 이상해요. 그래서 조심해야 해요.",
            "source_item": {"korean_topic": "AI 윤리"}
        }
    ]
    
    test_pairs = [
        {
            "chosen": "환경 보호는 우리 모두의 책임입니다. 지구 온난화를 막기 위해 재생 가능한 에너지 사용을 늘리고, 플라스틱 사용을 줄여야 합니다.",
            "rejected": "환경은 중요해요. 그래서 좋은 일을 해야 돼요. 나무도 심고 그런 거요."
        }
    ]
    
    try:
        print("=" * 50)
        print("🧪 Reward Model 테스트 시작")
        print("=" * 50)
        
        # 1. 개별 passage 평가 테스트
        print("\n1️⃣ 개별 Passage 평가 테스트:")
        evaluated = evaluate_passages_with_reward_model(
            MODEL_PATH, 
            test_passages,
            max_length=256
        )
        
        for result in evaluated:
            score = result.get("reward_score", 0)
            text = result.get("generated_passage", "")[:50] + "..."
            print(f"   {text} → 점수: {score:.4f}")
        
        # 2. Passage pair 비교 테스트
        print("\n2️⃣ Passage Pair 비교 테스트:")
        comparison_result = compare_passage_pairs_with_reward_model(
            MODEL_PATH,
            test_pairs,
            max_length=256
        )
        
        print(f"   총 {comparison_result['total_pairs']}개 pair 중 {comparison_result['correct_preferences']}개 정확")
        print(f"   정확도: {comparison_result['accuracy']:.2%}")
        
        print("\n🎉 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

import os
import sys
import time
from typing import List, Dict, Optional
from openai import OpenAI, RateLimitError, APITimeoutError
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from dotenv import load_dotenv
from pathlib import Path

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))

# 런타임에 import
try:
    from src.utils.settings_loader import get_settings
except ImportError:
    from utils.settings_loader import get_settings

load_dotenv()
_CFG = get_settings()
_LLM_CFG = _CFG.get('llm', {})
_CHATGPT_CFG = _CFG.get('chatgpt', {})

# === Configuration Constants ===
# LLM (로컬 모델) 설정
_LLM_MAX_TOKENS = int(_LLM_CFG.get('max_tokens', 512))
_LLM_TEMPERATURE = float(_LLM_CFG.get('temperature', 0.7))
_LLM_REPETITION_PENALTY = float(_LLM_CFG.get('repetition_penalty', 1.1))
_LLM_TOP_P = float(_LLM_CFG.get('top_p', 0.9))
_LLM_TOP_K = int(_LLM_CFG.get('top_k', 50))
_LLM_NO_REPEAT_NGRAM_SIZE = int(_LLM_CFG.get('no_repeat_ngram_size', 0))

# ChatGPT (OpenAI) 설정
_CHATGPT_MAX_TOKENS = int(_CHATGPT_CFG.get('max_tokens', 1024))
_CHATGPT_TEMPERATURE = float(_CHATGPT_CFG.get('temperature', 0))
_CHATGPT_TOP_P = float(_CHATGPT_CFG.get('top_p', 1.0))
_CHATGPT_FREQUENCY_PENALTY = float(_CHATGPT_CFG.get('frequency_penalty', 0.0))
_CHATGPT_PRESENCE_PENALTY = float(_CHATGPT_CFG.get('presence_penalty', 0.0))

# === Local Models Configuration ===
_LOCAL_MODELS_DIR = os.getenv('LOCAL_MODELS_PATH') or os.path.expanduser(_LLM_CFG.get('local_models_dir', '~/models'))
_DEFAULT_TORCH_DTYPE = torch.bfloat16
_FALLBACK_TORCH_DTYPE = torch.float16

# === Default Values ===
_DEFAULT_OPENAI_MODEL = 'gpt-4o-mini'
_DEFAULT_LOCAL_MODEL = 'default'
_DEFAULT_DEVICE = 'auto'

# === Batch API Configuration (필요한 경우만) ===
_BATCH_COMPLETION_WINDOW = "24h"
_BATCH_POLL_INTERVAL = 10

# === File Names ===
_TEMP_BATCH_FILE = "temp_batch_input.jsonl"

# === Status Constants ===
_BATCH_COMPLETED_STATUSES = ["completed", "failed", "cancelled"]
_API_ERROR_KEYWORDS = {
    "invalid_key": ["invalid_api_key", "401"],
    "rate_limit": ["rate_limit", "429"]
}

# === Error Messages ===
_ERROR_NO_RESPONSE = "죄송합니다. AI로부터 응답을 받지 못했습니다."
_ERROR_INVALID_API_KEY = "❌ OpenAI API 키가 올바르지 않습니다. 관리자에게 문의하세요."
_ERROR_RATE_LIMIT = "⏳ API 사용량 한도에 도달했습니다. 잠시 후 다시 시도해주세요."

class BaseModelClient:
    """모든 모델 클라이언트의 기본 인터페이스"""
    def call(self, messages: List[Dict], **kwargs) -> str:
        raise NotImplementedError

class OpenAIModelClient(BaseModelClient):
    """최신 openai 라이브러리와 호환되는 클라이언트"""
    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.default_params = {
            "temperature": _CHATGPT_TEMPERATURE,
            "max_tokens": _CHATGPT_MAX_TOKENS,
            "top_p": _CHATGPT_TOP_P,
            "frequency_penalty": _CHATGPT_FREQUENCY_PENALTY,
            "presence_penalty": _CHATGPT_PRESENCE_PENALTY,
            **kwargs
        }
        print(f"✅ OpenAI 클라이언트가 '{self.model_name}' 모델로 초기화되었습니다.")

    def call(self, messages: List[Dict], **kwargs) -> str:
        params = self.default_params.copy()
        params.update(kwargs)
        
        request_payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": params.get('temperature'),
            "max_tokens": params.get('max_tokens'),
            "top_p": params.get('top_p'),
            "frequency_penalty": params.get('frequency_penalty'),
            "presence_penalty": params.get('presence_penalty')
        }

        # JSON 모드 강제 여부 확인
        if params.get('force_json', False):
            request_payload["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**request_payload)
            content = response.choices[0].message.content
            return content or _ERROR_NO_RESPONSE
        except Exception as e:
            print(f"❌ OpenAI API 호출 중 오류 발생: {e}")
            # API 키 에러인지 확인
            error_str = str(e)
            if any(keyword in error_str for keyword in _API_ERROR_KEYWORDS["invalid_key"]):
                return _ERROR_INVALID_API_KEY
            elif any(keyword in error_str for keyword in _API_ERROR_KEYWORDS["rate_limit"]):
                return _ERROR_RATE_LIMIT
            else:
                return f"❌ AI 응답 생성 중 오류가 발생했습니다: {str(e)[:100]}"

     # --- ✨ 주요 변경 사항 시작 ✨ ---

    def _prepare_batch_file(self, batch_of_messages: List[List[Dict]], **kwargs) -> str:
        """배치 API에 업로드할 .jsonl 파일을 생성합니다."""
        batch_requests = []
        for i, messages in enumerate(batch_of_messages):
            request_body = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.default_params.get('temperature'),
                "max_tokens": self.default_params.get('max_tokens')
            }
            # 추가적인 kwargs 파라미터를 body에 업데이트
            request_body.update(kwargs)

            batch_requests.append({
                "custom_id": f"request_{i+1}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request_body
            })

        # 임시 파일로 저장
        file_path = _TEMP_BATCH_FILE
        with open(file_path, "w", encoding="utf-8") as f:
            for req in batch_requests:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        
        return file_path

    def call_batch(self, batch_of_messages: List[List[Dict]], **kwargs) -> List[str]:
        """
        OpenAI 배치 API를 사용하여 여러 요청을 비동기적으로 처리하고 비용을 절감합니다.

        Args:
            batch_of_messages: 각 요소가 message 리스트인 배치 요청.
            **kwargs: 생성 파라미터 (temperature, max_tokens 등).

        Returns:
            List[str]: 각 요청에 대한 응답 콘텐츠 리스트.
        """
        print(f"🚀 {len(batch_of_messages)}개의 요청에 대한 배치 처리를 시작합니다...")

        # 1. 배치 입력 파일 생성
        batch_file_path = self._prepare_batch_file(batch_of_messages, **kwargs)

        try:
            # 2. 파일 업로드
            print(f"  - 1/4: 배치 파일 '{batch_file_path}'을(를) 업로드합니다...")
            batch_input_file = self.client.files.create(
                file=open(batch_file_path, "rb"),
                purpose="batch"
            )

            # 3. 배치 작업 생성
            print(f"  - 2/4: 배치 작업을 생성하고 API에 제출합니다...")
            batch_job = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window=_BATCH_COMPLETION_WINDOW # 24시간 내에 완료되도록 설정
            )

            # 4. 배치 작업 완료 대기 (폴링)
            print(f"  - 3/4: 배치 작업(ID: {batch_job.id})이 완료될 때까지 대기합니다...")
            while batch_job.status not in _BATCH_COMPLETED_STATUSES:
                time.sleep(_BATCH_POLL_INTERVAL) # 10초마다 상태 확인
                batch_job = self.client.batches.retrieve(batch_job.id)
                print(f"    - 현재 상태: {batch_job.status}...")

            if batch_job.status != "completed":
                raise RuntimeError(f"배치 작업이 실패 또는 취소되었습니다. 최종 상태: {batch_job.status}")

            # 5. 결과 파일 다운로드 및 파싱
            print(f"  - 4/4: 작업 완료! 결과 파일을 다운로드하고 파싱합니다...")
            result_file_id = batch_job.output_file_id
            result_content = self.client.files.content(result_file_id).read()
            
            responses = []
            for line in result_content.decode("utf-8").strip().split("\n"):
                data = json.loads(line)
                # 응답 본문에서 content를 추출하여 리스트에 추가
                content = data["response"]["body"]["choices"][0]["message"]["content"]
                responses.append(content)

            print(f"✅ 배치 처리 완료! {len(responses)}개의 응답을 성공적으로 받았습니다.")
            return responses

        except Exception as e:
            print(f"❌ 배치 처리 중 오류 발생: {e}")
            return [f"오류: {e}" for _ in batch_of_messages]
        finally:
            # 임시 파일 삭제
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)

class LocalModelClient(BaseModelClient):
    """
    로컬 HuggingFace 모델 추론용 클라이언트 (강화된 GPU 제어 기능)
    """
    def __init__(self, model_name: str, **kwargs):
        """
        클래스를 초기화하고, 지정된 GPU에 모델을 로드합니다.

        Args:
            model_name (str): '~/models/' 디렉토리 내의 모델 폴더 이름.
            **kwargs: 'gpus' (List[int])와 같은 추가 인자를 받을 수 있습니다.
        """
        self.model_name = model_name
        self.model_path = os.path.join(_LOCAL_MODELS_DIR, model_name)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        # --- 강화된 GPU 설정 로직 ---
        gpus = kwargs.pop('gpus', None)
        device_setting = kwargs.pop('device', _LLM_CFG.get('device', _DEFAULT_DEVICE))
        self.target_gpus = gpus
        
        # 설정 파일의 device 값에 따른 처리
        if device_setting == "cpu":
            # CPU 강제 사용
            self.device_map = "cpu"
            self.target_device = "cpu"
            print("🎯 CPU 모드: 설정에 의한 CPU 사용")
        elif gpus is not None and torch.cuda.is_available():
            if isinstance(gpus, list) and all(isinstance(i, int) for i in gpus):
                # 1. CUDA_VISIBLE_DEVICES를 먼저 설정 (매우 중요!)
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))
                print(f"🎯 CUDA_VISIBLE_DEVICES 설정: {','.join(map(str, gpus))}")
                
                # 2. PyTorch CUDA 캐시 정리
                torch.cuda.empty_cache()
                print("🧹 CUDA 캐시 정리 완료")
                
                # 3. GPU 설정 - CUDA_VISIBLE_DEVICES 설정 후에는 0부터 시작
                if len(gpus) == 1:
                    # 단일 GPU 사용 - CUDA_VISIBLE_DEVICES 설정 후에는 0번 인덱스
                    self.device_map = {"": "cuda:0"}
                    self.target_device = "cuda:0"
                    print(f"🎯 단일 GPU 모드: 실제 GPU {gpus[0]} → 논리적 GPU 0")
                else:
                    # 다중 GPU 사용 - 논리적 GPU 인덱스 매핑
                    gpu_mapping = {f"cuda:{i}": f"cuda:{i}" for i in range(len(gpus))}
                    self.device_map = gpu_mapping
                    self.target_device = "cuda:0"  # 첫 번째 논리적 GPU
                    print(f"🎯 다중 GPU 모드: 실제 GPU {gpus} → 논리적 GPU 0~{len(gpus)-1}")
                    print(f"🔗 디바이스 매핑: {gpu_mapping}")
            else:
                print("⚠️ 'gpus' 인자는 정수 리스트여야 합니다 (예: [2, 3]). auto로 설정합니다.")
                self.device_map = "auto"
                self.target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            # GPU 지정 없음 - 설정 파일 기반 처리
            if device_setting == "auto" and torch.cuda.is_available():
                self.device_map = "auto"
                self.target_device = "cuda:0"
                print("🎯 AUTO 모드: 설정에 의한 자동 디바이스 매핑")
            elif device_setting == "cuda" and torch.cuda.is_available():
                self.device_map = "auto"
                self.target_device = "cuda:0"
                print("🎯 CUDA 모드: 설정에 의한 GPU 사용")
            else:
                self.device_map = "cpu"
                self.target_device = "cpu"
                print("🎯 CPU 모드: CUDA 사용 불가 또는 설정에 의한 CPU 사용")
        # ------------------------------------
            
        print(f"🔄 로컬 모델 로딩 중: {self.model_path}...")
        
        # GPU 메모리 사용량 체크 (CUDA_VISIBLE_DEVICES 설정 후)
        if torch.cuda.is_available() and gpus:
            print("📊 GPU 메모리 상태:")
            for logical_idx in range(len(gpus)):
                try:
                    memory_allocated = torch.cuda.memory_allocated(logical_idx) / 1024**3
                    memory_cached = torch.cuda.memory_reserved(logical_idx) / 1024**3
                    memory_total = torch.cuda.get_device_properties(logical_idx).total_memory / 1024**3
                    actual_gpu = gpus[logical_idx]
                    print(f"   논리적 GPU {logical_idx} (실제 GPU {actual_gpu}): {memory_allocated:.1f}GB 할당, {memory_cached:.1f}GB 캐시, {memory_total:.1f}GB 총용량")
                except Exception as e:
                    print(f"   GPU {logical_idx} 메모리 정보 조회 실패: {e}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            # 메모리 효율적인 로딩 옵션
            load_options = {
                "torch_dtype": _DEFAULT_TORCH_DTYPE,
                "device_map": self.device_map,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,  # CPU 메모리 사용량 최소화
            }
            
            print(f"🔧 모델 로딩 옵션: device_map={self.device_map}")
            
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_options)
            
            # 모델이 로드된 후 디바이스 확인
            if hasattr(self, 'target_device'):
                print(f"🔧 모델 타겟 디바이스: {self.target_device}")
                # 단일 GPU 사용 시 명시적 이동 (안전성 확보)
                if gpus and len(gpus) == 1:
                    try:
                        self.model = self.model.to(self.target_device)
                        print(f"✅ 모델을 {self.target_device}로 명시적 이동 완료")
                    except Exception as e:
                        print(f"⚠️ 모델 디바이스 이동 실패: {e}")
                    
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            # 폴백: 더 보수적인 설정으로 재시도
            print("🔄 보수적인 설정으로 재시도...")
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=_FALLBACK_TORCH_DTYPE,  # bfloat16 대신 float16
                    device_map="cpu",  # 일단 CPU로 로드
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                # 나중에 첫 번째 GPU로 이동
                if torch.cuda.is_available() and gpus:
                    self.target_device = f"cuda:{gpus[0]}"
                    self.model = self.model.to(self.target_device)
                    print(f"🔧 모델을 {self.target_device}로 이동 완료")
                else:
                    self.target_device = "cpu"
            except Exception as e2:
                raise RuntimeError(f"모델 로딩 완전 실패: {e2}")
        
        # pad_token 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # default_params 설정
        self.default_params = {
            "temperature": _LLM_TEMPERATURE,
            "max_new_tokens": _LLM_MAX_TOKENS,
            "repetition_penalty": _LLM_REPETITION_PENALTY,
            "top_p": _LLM_TOP_P,
            "top_k": _LLM_TOP_K,
            "no_repeat_ngram_size": _LLM_NO_REPEAT_NGRAM_SIZE,
        }
            
        print(f"✅ 로컬 모델 로딩 완료 ({self.target_device})")

    def call(self, messages: List[Dict], **kwargs) -> str:
        params = self.default_params.copy()
        params.update(kwargs)
        
        # --- ✨ 추론 기능 토글 추가 ✨ ---
        # 1. 'enable_thinking' 인자를 kwargs에서 추출합니다. (기본값: False)
        enable_thinking = params.pop('enable_thinking', False)
        
        # 2. apply_chat_template에 전달할 인자를 구성합니다.
        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True
        }
        
        # 3. 토글이 True일 경우에만 'enable_thinking' 인자를 추가합니다.
        if enable_thinking:
            chat_template_kwargs['enable_thinking'] = True
            print("💡 추론(Thinking) 모드가 활성화되었습니다.")
        
        prompt = self.tokenizer.apply_chat_template(messages, **chat_template_kwargs)
        # --- ✨ 추론 기능 토글 끝 ✨ ---
        
        # 입력을 올바른 디바이스로 보내기
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.target_device)
        
        try:
            with torch.no_grad():
                # 🔧 model.generate에 전달할 파라미터 필터링
                generate_kwargs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "temperature": params['temperature'],
                    "max_new_tokens": params['max_new_tokens'],
                    "repetition_penalty": params['repetition_penalty'],
                    "top_p": params['top_p'],
                    "top_k": params['top_k'],
                    "no_repeat_ngram_size": params['no_repeat_ngram_size'],
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "do_sample": True,  # temperature > 0일 때 필요
                }
                
                # token_type_ids가 있으면서 모델이 지원하는 경우에만 추가
                if hasattr(inputs, 'token_type_ids') and inputs.token_type_ids is not None:
                    # 모델이 token_type_ids를 지원하는지 확인
                    if 'token_type_ids' in self.model.forward.__code__.co_varnames:
                        generate_kwargs['token_type_ids'] = inputs.token_type_ids
                    # 지원하지 않는 모델의 경우 token_type_ids는 제외
                
                outputs = self.model.generate(**generate_kwargs)
            response_text = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            return response_text.strip()
        except Exception as e:
            print(f"❌ 로컬 모델 추론 중 오류 발생: {e}")
            return ""

# Factory function to create appropriate client
def create_model_client(
    client_type: str,
    model_name: str,
    **kwargs
) -> BaseModelClient:
    """
    Factory function to create appropriate model client.
    
    Args:
        client_type: "openai" or "local"
        model_name: Name of the model
        **kwargs: Additional arguments for specific clients
    
    Returns:
        BaseModelClient instance
    """
    if client_type.lower() == "openai":
        return OpenAIModelClient(model_name=model_name, **kwargs)
    elif client_type.lower() == "local":
        return LocalModelClient(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown client type: {client_type}. Supported: 'openai', 'local'")

# Utility function to list available local models
def list_local_models() -> List[str]:
    """
    List available models in ~/models/ directory.
    
    Returns:
        List of model names
    """
    models_dir = _LOCAL_MODELS_DIR
    
    if not os.path.exists(models_dir):
        return []
    
    try:
        models = [d for d in os.listdir(models_dir) 
                 if os.path.isdir(os.path.join(models_dir, d))]
        return sorted(models)
    except Exception as e:
        print(f"Error listing local models: {e}")
        return []

# Utility function to create client from settings
def create_client_from_settings() -> BaseModelClient:
    """
    Create model client based on settings.yaml configuration.
    
    Returns:
        BaseModelClient instance configured from settings
    """
    client_type = _LLM_CFG.get('client_type', 'openai').lower()
    
    if client_type == "openai":
        external_services = _CFG.get('external_services', {})
        openai_config = external_services.get('openai', {})
        model_name = openai_config.get('model', _DEFAULT_OPENAI_MODEL)
        return OpenAIModelClient(model_name=model_name)
    elif client_type == "local":
        # 로컬 모델 설정 - 기본 llm 설정을 사용하되 local_model에서 오버라이드
        local_config = _LLM_CFG.get('local_model', {})
        return LocalModelClient(
            model_name=local_config.get('name', _DEFAULT_LOCAL_MODEL),
            device=_LLM_CFG.get('device', _DEFAULT_DEVICE)  # llm.device 설정 사용
        )
    else:
        raise ValueError(f"Unknown client type in settings: {client_type}. Supported: 'openai', 'local'")

# Quick test function
def test_client_from_settings():
    """Test client creation from settings"""
    try:
        print("🔄 Creating client from settings...")
        client = create_client_from_settings()
        
        test_messages = [
            {"role": "user", "content": "안녕하세요! 간단히 인사해주세요."}
        ]
        
        print("🤖 Testing client...")
        response = client.call(test_messages)
        print(f"✅ Response: {response[:100]}...")
        
    except Exception as e:
        print(f"❌ Error testing client: {e}")

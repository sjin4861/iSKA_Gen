import os
import sys
import time
from typing import Any, List, Dict, Optional
from openai import OpenAI, RateLimitError, APITimeoutError
import torch
import torch._dynamo  # torch.compile 관련 오류 해결을 위해 추가
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from dotenv import load_dotenv
from pathlib import Path
import logging

# 경로 설정
sys.path.append(str(Path.cwd().parent.parent))

# 런타임에 import
try:
    from src.utils.settings_loader import get_settings
except ImportError:
    from utils.settings_loader import get_settings
    
logger = logging.getLogger(__name__)
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
_DEFAULT_TORCH_DTYPE = torch.float16  # bfloat16 대신 float16으로 통일
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
            "max_new_tokens": _CHATGPT_MAX_TOKENS,
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
            "max_new_tokens": params.get('max_new_tokens'),
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
                "max_new_tokens": self.default_params.get('max_new_tokens')
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
    로컬 HuggingFace 모델 추론용 클라이언트.
    - gpus=[0,1,2] 등으로 사용할 물리 GPU 지정 가능
    - device_map='auto' + (선택) max_memory 기반 멀티 GPU 샤딩
    - flash_attention_2 / 4bit/8bit 양자화 선택적 지원
    """

    def __init__(
        self,
        model_name: str,
        *,
        gpus: Optional[List[int]] = None,
        device: str = None,                    # 'cpu' | 'cuda' | 'auto' (기본: _LLM_CFG.device or 'auto')
        use_max_memory: bool = True,
        attn_impl: Optional[str] = None,       # 'flash_attention_2' 등
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        bnb_4bit_compute_dtype: Optional[str] = None,  # 'bfloat16' | 'float16' 등
        trust_remote_code: bool = True,
        # 생성용 파라미터들 (모델 로딩에는 사용되지 않음)
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        **kwargs: Any
    ):
        """
        Args:
            model_name: ~/models 아래 모델 디렉토리명
            gpus: 사용할 물리 GPU 인덱스 목록 (예: [0,1,2]). 지정 시 해당 GPU만 노출되도록 설정
            device: 'cpu' | 'cuda' | 'auto' (미지정 시 설정파일 기본값/_DEFAULT_DEVICE)
            use_max_memory: True면 각 GPU 총용량-1GiB 상한으로 max_memory 전달
            attn_impl: 'flash_attention_2' 사용 등
            load_in_4bit / load_in_8bit: bitsandbytes 양자화 옵션
            bnb_4bit_compute_dtype: 4bit 시 연산 dtype
            trust_remote_code: 리포의 커스텀 코드 신뢰
            temperature, max_new_tokens, top_p, top_k, repetition_penalty: 생성용 파라미터
            kwargs: from_pretrained에 전달할 추가 인자 (생성용 파라미터는 자동 필터링됨)
        """
        self.model_name = model_name
        self.model_path = os.path.join(_LOCAL_MODELS_DIR, model_name)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
            
        # 생성용 파라미터들을 미리 저장 (default_params에서 사용)
        self._custom_generation_params = {}
        if temperature is not None:
            self._custom_generation_params['temperature'] = temperature
        if max_new_tokens is not None:
            self._custom_generation_params['max_new_tokens'] = max_new_tokens
        if top_p is not None:
            self._custom_generation_params['top_p'] = top_p
        if top_k is not None:
            self._custom_generation_params['top_k'] = top_k
        if repetition_penalty is not None:
            self._custom_generation_params['repetition_penalty'] = repetition_penalty

        # --- 디바이스/환경 결정 ---
        device = device or _LLM_CFG.get('device', _DEFAULT_DEVICE)

        # dtype 해석
        def _to_torch_dtype(name_or_none: Optional[str], default_dtype: torch.dtype) -> torch.dtype:
            if not name_or_none:
                return default_dtype
            name = name_or_none.lower()
            if name in ("bf16", "bfloat16"):
                return torch.bfloat16
            if name in ("fp16", "float16", "half"):
                return torch.float16
            if name in ("fp32", "float32"):
                return torch.float32
            return default_dtype

        torch_dtype = _DEFAULT_TORCH_DTYPE

        # CUDA 사용 가능성
        cuda_available = torch.cuda.is_available()

        # gpus 지정 처리
        if gpus is not None:
            if not isinstance(gpus, list) or not all(isinstance(i, int) for i in gpus):
                raise ValueError("gpus 인자는 정수 리스트여야 합니다. 예: [0,1,2]")
            if not cuda_available:
                raise RuntimeError("CUDA가 사용 불가한 환경에서 gpus가 지정되었습니다.")
            
            # 물리 인덱스 유효성 확인 (CUDA_VISIBLE_DEVICES가 설정되지 않은 경우)
            original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if not original_visible:
                physical_count = torch.cuda.device_count()
                invalid = [i for i in gpus if i < 0 or i >= physical_count]
                if invalid:
                    raise ValueError(f"유효하지 않은 GPU 인덱스: {invalid}. 사용 가능 범위: 0..{physical_count-1}")
            
            # 프로세스 시작시 환경변수로 GPU 제한 (가장 확실한 방법)
            if not torch.cuda.is_initialized() and not original_visible:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
                print(f"🎯 CUDA_VISIBLE_DEVICES를 {gpus}로 설정했습니다.")
                # CUDA 컨텍스트 재초기화
                if 'torch' in sys.modules:
                    torch.cuda.empty_cache()
            else:
                print(f"⚠️ CUDA 이미 초기화됨 또는 CUDA_VISIBLE_DEVICES 기설정됨. device_map으로 GPU {gpus} 지정을 시도합니다.")

        # 최종 가시 GPU 개수(논리 인덱스 기준)  
        logical_cuda = cuda_available and (torch.cuda.device_count() > 0)

        # device_map 결정 - 더 강력한 GPU 지정
        if device == "cpu" or not logical_cuda:
            device_map = "cpu"
            input_device = torch.device("cpu")
        else:
            if gpus is not None and len(gpus) == 1:
                # 단일 GPU: 강제로 해당 GPU만 사용
                target_gpu = gpus[0]
                
                # CUDA_VISIBLE_DEVICES가 설정된 경우 논리적 인덱스 0 사용
                current_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                if current_visible and str(target_gpu) in current_visible.split(","):
                    # 지정된 GPU가 visible devices에 포함된 경우
                    visible_gpus = [int(x) for x in current_visible.split(",")]
                    logical_index = visible_gpus.index(target_gpu)
                    device_map = {"": f"cuda:{logical_index}"}
                    input_device = torch.device(f"cuda:{logical_index}")
                    print(f"🎯 단일 GPU 모드: 물리 GPU {target_gpu}번 → 논리적 cuda:{logical_index} 사용")
                else:
                    # 직접 물리 GPU 인덱스 사용 (위험하지만 필요시)
                    device_map = {"": f"cuda:{target_gpu}"}
                    input_device = torch.device(f"cuda:{target_gpu}")
                    print(f"🎯 단일 GPU 모드: 물리 GPU {target_gpu}번 직접 사용")
                    
            elif gpus is not None and len(gpus) > 1:
                # 멀티 GPU: 지정된 GPU들만 사용
                device_map = "auto"
                input_device = torch.device("cuda:0")
                print(f"🔗 멀티 GPU 모드: 물리 GPU {gpus} 사용")
            else:
                # gpus 미지정시 auto 사용
                device_map = "auto" 
                input_device = torch.device("cuda:0")

        # max_memory 구성(선택) - 보수적 메모리 설정
        max_memory = None
        if logical_cuda and use_max_memory:
            if device_map == "auto":
                # auto 모드: 모든 가시 GPU에 메모리 제한 설정 (보수적으로 3GB 여유)
                max_memory = {}
                for i in range(torch.cuda.device_count()):
                    total_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    # 여유 3GiB 남겨두기 (보수적 설정)
                    available_gb = max(1, int(total_gb - 3))
                    max_memory[i] = f"{available_gb}GiB"
                    print(f"📊 GPU {i} 메모리 제한: {available_gb}GiB / {int(total_gb)}GiB")
            elif isinstance(device_map, dict) and gpus is not None and len(gpus) == 1:
                # 단일 GPU 모드: 지정된 GPU의 메모리 제한 설정 (보수적으로 3GB 여유)
                target_gpu = gpus[0]
                try:
                    # device_map에서 실제 사용될 논리적 GPU 인덱스 찾기
                    device_str = list(device_map.values())[0]  # 예: "cuda:1"
                    if "cuda:" in device_str:
                        logical_gpu = int(device_str.split(":")[1])
                    else:
                        logical_gpu = 0
                        
                    total_gb = torch.cuda.get_device_properties(logical_gpu).total_memory / 1024**3
                    # 여유 3GiB 남겨두기 (보수적 설정)
                    available_gb = max(1, int(total_gb - 3))
                    max_memory = {logical_gpu: f"{available_gb}GiB"}
                    print(f"📊 GPU {target_gpu} (논리적 {logical_gpu}) 메모리 제한: {available_gb}GiB / {int(total_gb)}GiB")
                except Exception as e:
                    print(f"⚠️ GPU {target_gpu} 메모리 정보 가져오기 실패: {e}")
                    max_memory = None

        # Tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        except Exception as e:
            logger.warning(f"토크나이저 로드 실패: {e}")
            # 폴백: 기본 GPT2 토크나이저
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- 모델 로딩 옵션 구성 ---
        load_opts: Dict[str, Any] = dict(
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )

        # attn impl
        if attn_impl:
            load_opts["attn_implementation"] = attn_impl

        # 양자화 옵션 우선 (4bit/8bit) → dtype은 프레임워크가 관리
        if load_in_4bit or load_in_8bit:
            load_opts["device_map"] = device_map
            if max_memory:
                load_opts["max_memory"] = max_memory
            if load_in_4bit:
                load_opts["load_in_4bit"] = True
                if bnb_4bit_compute_dtype:
                    load_opts["bnb_4bit_compute_dtype"] = _to_torch_dtype(bnb_4bit_compute_dtype, torch.float16)
            if load_in_8bit:
                load_opts["load_in_8bit"] = True
        else:
            load_opts["torch_dtype"] = torch_dtype
            load_opts["device_map"] = device_map
            if max_memory:
                load_opts["max_memory"] = max_memory

        # 모델 로딩용 파라미터만 필터링 (생성용 파라미터 제외)
        generation_params = {
            'max_new_tokens', 'temperature', 'top_p', 'top_k', 'repetition_penalty',
            'no_repeat_ngram_size', 'do_sample', 'pad_token_id', 'num_beams',
            'early_stopping', 'length_penalty', 'eos_token_id'
        }
        
        # kwargs에서 모델 로딩용 파라미터만 추출
        model_loading_kwargs = {k: v for k, v in kwargs.items() if k not in generation_params}
        load_opts.update(model_loading_kwargs)

        # --- 모델 로드 ---
        try:
            # torch.compile 관련 설정 비활성화
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 1  # 캐시 제한
            
            # 완전한 torch.compile 비활성화 (환경변수)
            os.environ["TORCH_COMPILE_DEBUG"] = "0"
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **load_opts)
            
            # 모델이 컴파일되어 있다면 비활성화
            if hasattr(self.model, '_orig_mod'):
                print("🔧 torch.compile 감지됨, 원본 모델 사용")
                self.model = self.model._orig_mod

                
        except Exception as e:
            # 보수적 폴백: CPU + fp16, 생성용 파라미터 제거
            logger.warning("모델 로딩 실패. 보수적 설정으로 재시도합니다: %s", e)
            
            # torch.compile 비활성화 후 재시도
            torch._dynamo.config.suppress_errors = True
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
                device_map="cpu",
                torch_dtype=torch.float16,  # 강제로 float16
                **model_loading_kwargs  # 필터링된 kwargs만 사용
            )

        # 기본 생성 파라미터 (커스텀 파라미터가 있으면 덮어씀)
        self._input_device = input_device
        default_temperature = _LLM_TEMPERATURE
        
        self.default_params = {
            "temperature": default_temperature,
            "max_new_tokens": _LLM_MAX_TOKENS,
            "repetition_penalty": _LLM_REPETITION_PENALTY,
            "top_p": _LLM_TOP_P,
            "top_k": _LLM_TOP_K,
            "no_repeat_ngram_size": _LLM_NO_REPEAT_NGRAM_SIZE,
            "do_sample": True if default_temperature > 0.0 else False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        
        # 커스텀 생성 파라미터로 기본값 덮어쓰기
        self.default_params.update(self._custom_generation_params)
        
        # temperature가 0.0이면 do_sample을 False로 강제 설정
        if self.default_params.get("temperature", 0.0) == 0.0:
            self.default_params["do_sample"] = False

    # 간단한 백업 포맷(토크나이저에 chat_template가 없을 때)
    @staticmethod
    def _fallback_chat_format(messages: List[Dict[str, str]]) -> str:
        # ChatML 유사 포맷
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n")
            elif role == "user":
                parts.append(f"[INST] {content} [/INST]")
            else:
                parts.append(content)
        return "\n".join(parts)

    def call(self, messages: List[Dict[str, str]], **kwargs) -> str:
        params = {**self.default_params, **kwargs}
        
        # temperature가 0.0일 때는 do_sample=False로 설정 (greedy decoding)
        # 이는 HuggingFace transformers의 요구사항입니다
        if params.get("temperature", 0.0) == 0.0:
            params["do_sample"] = False
            # greedy decoding일 때는 sampling 관련 파라미터들을 제거
            params.pop("top_p", None)
            params.pop("top_k", None)
        else:
            # temperature > 0일 때만 sampling 사용
            params["do_sample"] = True

        # prompt 생성 (chat_template 우선, 실패 시 fallback)
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = self._fallback_chat_format(messages)

        # 입력 텐서를 입력 디바이스로
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device)
        
        # 모든 텐서를 float16으로 강제 변환 (dtype 불일치 방지)
        target_dtype = torch.float16
        
        # input_ids는 정수형이므로 변환하지 않음
        if hasattr(inputs, 'attention_mask'):
            inputs['attention_mask'] = inputs['attention_mask'].to(dtype=target_dtype)
        
        # 추가적인 입력 텐서들도 모두 float16으로 변환
        for key, tensor in inputs.items():
            if key != 'input_ids' and torch.is_floating_point(tensor):
                inputs[key] = tensor.to(dtype=target_dtype)
                
        # 모델도 강제로 float16으로 설정 (혹시 모를 dtype 불일치 방지)
        if hasattr(self.model, 'dtype') and self.model.dtype != target_dtype:
            print(f"🔄 추론 시 모델 dtype을 {self.model.dtype}에서 {target_dtype}으로 변경")
            self.model = self.model.to(target_dtype)

        try:
            with torch.no_grad():
                # torch.compile 비활성화 (generator 추적 오류 방지)
                torch._dynamo.config.suppress_errors = True
                
                # 첫 번째 시도: 일반적인 generate 호출
                try:
                    # 모델이 컴파일되어 있다면 비컴파일 버전 사용
                    if hasattr(self.model, '_orig_mod'):
                        # torch.compile로 감싸진 경우 원본 모델 사용
                        model_to_use = self.model._orig_mod
                    else:
                        model_to_use = self.model
                    
                    # generate 파라미터 구성 (do_sample 여부에 따라)
                    generate_kwargs = {
                        "input_ids": inputs.input_ids,
                        "attention_mask": inputs.attention_mask,
                        "max_new_tokens": params["max_new_tokens"],
                        "repetition_penalty": params["repetition_penalty"],
                        "no_repeat_ngram_size": params["no_repeat_ngram_size"],
                        "do_sample": params["do_sample"],
                        "pad_token_id": params["pad_token_id"],
                    }
                    
                    # sampling 사용할 때만 temperature, top_p, top_k 추가
                    if params["do_sample"]:
                        generate_kwargs.update({
                            "temperature": params["temperature"],
                            "top_p": params["top_p"],
                            "top_k": params["top_k"],
                        })
                        
                    outputs = model_to_use.generate(**generate_kwargs)
                except RuntimeError as dtype_error:
                    error_msg = str(dtype_error)
                    if "expected scalar type" in error_msg and "but found" in error_msg:
                        print(f"🔧 dtype 오류 감지, 전체 모델을 float16으로 강제 변환: {dtype_error}")
                        # 더 강력한 변환: 모든 파라미터를 float16으로 변환
                        self.model = self.model.float()  # 먼저 float32로
                        self.model = self.model.half()   # 그다음 float16으로
                        
                        # 재시도 - 동일한 generate_kwargs 사용
                        outputs = model_to_use.generate(**generate_kwargs)
                    elif "generator" in error_msg.lower() or "compile" in error_msg.lower():
                        print(f"🔧 torch.compile/generator 오류 감지, 비컴파일 모드로 재시도: {dtype_error}")
                        # torch.compile 완전 비활성화
                        torch._dynamo.reset()
                        os.environ["TORCHDYNAMO_DISABLE"] = "1"
                        
                        # 원본 모델 사용
                        original_model = getattr(self.model, '_orig_mod', self.model)
                        
                        outputs = original_model.generate(**generate_kwargs)
                    elif "temperature" in error_msg.lower() and "strictly positive" in error_msg.lower():
                        print(f"🔧 temperature 오류 감지, greedy decoding으로 전환: {dtype_error}")
                        # temperature 관련 오류 - greedy decoding 강제
                        generate_kwargs_fixed = generate_kwargs.copy()
                        generate_kwargs_fixed["do_sample"] = False
                        generate_kwargs_fixed.pop("temperature", None)
                        generate_kwargs_fixed.pop("top_p", None)
                        generate_kwargs_fixed.pop("top_k", None)
                        
                        outputs = model_to_use.generate(**generate_kwargs_fixed)
                    else:
                        raise dtype_error
                        
            gen = outputs[0][inputs.input_ids.shape[1]:]
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
            return text.strip()
        except Exception as e:
            logger.error("로컬 모델 추론 오류: %s", e)
            return ""

    def close(self):
        """
        모델 리소스를 정리하고 GPU 메모리를 해제합니다.
        사용이 끝난 후 호출하여 메모리를 확보할 수 있습니다.
        """
        try:
            # 1. 모델과 토크나이저를 CPU로 이동
            if hasattr(self, 'model') and self.model is not None:
                print("🔄 모델을 CPU로 이동 중...")
                self.model = self.model.cpu()
                
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                # 토크나이저에도 GPU 텐서가 있을 수 있음
                if hasattr(self.tokenizer, 'model') and hasattr(self.tokenizer.model, 'cpu'):
                    self.tokenizer.model = self.tokenizer.model.cpu()
            
            # 2. 모델 객체 삭제
            if hasattr(self, 'model'):
                print("🗑️ 모델 객체 삭제 중...")
                del self.model
                self.model = None
                
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
                self.tokenizer = None
            
            # 3. GPU 캐시 정리
            if torch.cuda.is_available():
                print("🧹 GPU 메모리 캐시 정리 중...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            # 4. Python 가비지 컬렉션 실행
            import gc
            gc.collect()
            
            print("✅ LocalModelClient 리소스 정리 완료")
            
        except Exception as e:
            print(f"⚠️ 리소스 정리 중 오류 발생: {e}")
            # 오류가 발생해도 최소한 GPU 캐시는 정리 시도
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except:
                    pass

    def __del__(self):
        """소멸자에서도 자동으로 리소스 정리"""
        try:
            self.close()
        except:
            pass  # 소멸자에서는 예외를 무시

    def __enter__(self):
        """Context manager 지원 - with 문 시작"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 지원 - with 문 종료시 자동 정리"""
        self.close()

class VLLMOpenAIClient(BaseModelClient):
    """vLLM(OpenAI 호환 서버) 클라이언트"""
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        timeout: int = 120,
        max_retries: int = 3,
        **default_params,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_params = {
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            **default_params,
        }
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        self._check_server_health()

    def _check_server_health(self) -> bool:
        try:
            models = self.client.models.list()
            ids = {m.id for m in models.data}
            ok = self.model_name in ids or bool(ids)  # 모델명 매치가 안 돼도 서버 응답만 확인해도 충분
            print("✅ vLLM server healthy:", ok, "models:", list(ids)[:3], "...")
            return ok
        except Exception as e:
            print(f"⚠️ vLLM server health check failed: {e} (URL: {self.base_url})")
            return False

    def call(self, messages: List[Dict], **kwargs) -> str:
        params = {**self.default_params, **kwargs}
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    top_p=params.get("top_p", 0.9),
                    frequency_penalty=params.get("frequency_penalty", 0.0),
                    presence_penalty=params.get("presence_penalty", 0.0),
                    response_format=params.get("response_format"),
                    seed=params.get("seed"),
                )
                content = resp.choices[0].message.content or ""
                return content.strip()
            except (RateLimitError, APITimeoutError) as e:
                backoff = 2 ** attempt
                print(f"⏳ vLLM rate/timeout, retry in {backoff}s: {e}")
                time.sleep(backoff)
            except Exception as e:
                print(f"❌ vLLM error (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return ""
                time.sleep(1)
        return ""

    def stream(self, messages: List[Dict], **kwargs):
        """선택: 토큰 스트리밍 제너레이터"""
        params = {**self.default_params, **kwargs}
        with self.client.chat.completions.with_streaming_response.create(
            model=self.model_name,
            messages=messages,
            max_tokens=params["max_tokens"],
            temperature=params["temperature"],
        ) as stream:
            for event in stream:
                if event.type == "token":
                    yield event.data  # 토큰 단위

def create_vllm_client(
    model_name: str,
    base_url: str = "http://localhost:8000/v1",
    **kwargs
) -> VLLMOpenAIClient:
    """
    vLLM OpenAI 호환 클라이언트 생성 함수
    
    Args:
        model_name: 모델명
        base_url: vLLM 서버 URL
        **kwargs: 추가 설정
        
    Returns:
        VLLMOpenAIClient 인스턴스
    """
    print(f"🔧 Creating vLLM client for {model_name}...")
    return VLLMOpenAIClient(model_name=model_name, base_url=base_url, **kwargs)


# Factory function to create appropriate client
def create_model_client(
    client_type: str,
    model_name: str,
    **kwargs
) -> BaseModelClient:
    """
    Factory function to create appropriate model client.
    
    Args:
        client_type: "openai", "local", or "vllm"
        model_name: Name of the model
        **kwargs: Additional arguments for specific clients
    
    Returns:
        BaseModelClient instance
    """
    client_type_lower = client_type.lower()
    if client_type_lower == "openai":
        return OpenAIModelClient(model_name=model_name, **kwargs)
    elif client_type_lower == "local":
        return LocalModelClient(model_name=model_name, **kwargs)
    elif client_type_lower == "vllm":
        return VLLMOpenAIClient(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown client type: {client_type}. Supported: 'openai', 'local', 'vllm'")

def get_model_client(model_name: str, **kwargs) -> BaseModelClient:
    """
    모델명을 기반으로 적절한 클라이언트를 자동 선택하여 생성합니다.
    
    Args:
        model_name: 모델명 (예: "gpt-4", "EXAONE-4.0-32B", "llama-3.1-8b")
        **kwargs: 클라이언트별 추가 인자 (gpus, device 등)
    
    Returns:
        BaseModelClient 인스턴스
    """
    model_name_lower = model_name.lower()
    
    # OpenAI 모델 패턴 확인
    openai_patterns = ["gpt-", "chatgpt", "o1-", "claude-"]
    if any(pattern in model_name_lower for pattern in openai_patterns):
        print(f"🤖 OpenAI 클라이언트로 {model_name} 모델 초기화 중...")
        return OpenAIModelClient(model_name=model_name, **kwargs)
    
    # 로컬 모델로 처리
    print(f"🏠 로컬 클라이언트로 {model_name} 모델 초기화 중...")
    return LocalModelClient(model_name=model_name, **kwargs)

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

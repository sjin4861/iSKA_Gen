# 모델 클라이언트 테스트

이 디렉토리에는 iSKA-Gen 프로젝트의 모델 클라이언트들을 테스트하는 스크립트들이 포함되어 있습니다.

## 테스트 스크립트

### 1. `test_model_clients.py`
기본적인 모델 클라이언트 연결 및 응답 테스트

**사용법:**
```bash
# 모든 클라이언트 타입 테스트
python tests/test_model_clients.py --client all

# OpenAI 클라이언트만 테스트
python tests/test_model_clients.py --client openai --model gpt-4o-mini

# 로컬 모델 클라이언트 테스트
python tests/test_model_clients.py --client local --model EXAONE-3.5-7.8B-Instruct --gpus 0

# vLLM 클라이언트 테스트
python tests/test_model_clients.py --client vllm --model test-model --url http://localhost:8000/v1
```

### 2. `test_passage_generation.py`
실제 벤치마크 데이터를 사용한 지문 생성 테스트

**사용법:**
```bash
# 포괄적인 지문 생성 테스트
python tests/test_passage_generation.py --client all

# 특정 클라이언트와 벤치마크로 테스트
python tests/test_passage_generation.py --client openai --benchmark 1 --item 0

# 로컬 모델로 여러 벤치마크 테스트
python tests/test_passage_generation.py --client local --model EXAONE-3.5-7.8B-Instruct --gpus 0
```

### 3. `run_all_tests.py`
모든 테스트를 자동으로 실행하는 통합 스크립트

**사용법:**
```bash
# 모든 테스트 실행
python tests/run_all_tests.py
```

## 테스트 옵션

### 공통 옵션
- `--client`: 테스트할 클라이언트 타입 (`openai`, `local`, `vllm`, `all`)
- `--model`: 사용할 모델명

### 로컬 모델 전용 옵션
- `--gpus`: 사용할 GPU 인덱스 목록 (예: `--gpus 0 1`)

### vLLM 전용 옵션
- `--url`: vLLM 서버 URL (기본값: `http://localhost:8000/v1`)

### 지문 생성 테스트 전용 옵션
- `--benchmark`: 테스트할 벤치마크 ID (1-5)
- `--item`: 테스트할 아이템 인덱스 (기본값: 0)

## 환경 설정

### 필수 환경변수
- `OPENAI_API_KEY`: OpenAI API 키 (OpenAI 클라이언트 사용 시)

### 선택적 환경변수
- `LOCAL_MODELS_PATH`: 로컬 모델 저장 경로 (기본값: `~/models`)
- `VLLM_BASE_URL`: vLLM 서버 URL (기본값: `http://localhost:8000/v1`)
- `VLLM_API_KEY`: vLLM API 키 (기본값: `EMPTY`)

## 벤치마크 ID 설명

1. **ID 1**: 비교형 지문 (한국/외국 문화 비교)
2. **ID 2**: 단일 주제형 지문 (국내 사회 문제)
3. **ID 3**: 대화형 지문 (찬반 논의)
4. **ID 4**: 대화형 지문 (상황별 대화)
5. **ID 5**: 이미지 캡션 및 상황 설명

## 예상 출력

### 성공적인 테스트
```
✅ 응답 성공 (2.34초)
📝 응답: 안녕하세요! 반갑습니다. 오늘 하루도 좋은 하루 되세요!

✅ 지문 생성 성공!
   생성 시간: 5.67초
   지문 길이: 456자
   지문 미리보기: 회식 문화는 한국 직장 생활에서 중요한 역할을 하고 있습니다...
```

### 실패한 테스트
```
❌ 응답 실패: API 키가 올바르지 않습니다
❌ 지문 생성 실패: 빈 결과 반환
```

## 문제해결

### 자주 발생하는 문제들

1. **OpenAI API 키 오류**
   ```
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **로컬 모델을 찾을 수 없음**
   ```bash
   # 모델 디렉토리 확인
   ls ~/models/
   
   # 또는 환경변수 설정
   export LOCAL_MODELS_PATH="/path/to/your/models"
   ```

3. **vLLM 서버 연결 실패**
   ```bash
   # 서버 상태 확인
   curl http://localhost:8000/v1/models
   ```

4. **GPU 메모리 부족**
   ```bash
   # 더 적은 GPU 사용
   python tests/test_model_clients.py --client local --gpus 0
   ```

## 테스트 결과 해석

- **성공률**: 전체 테스트 중 성공한 비율
- **응답 시간**: 모델이 응답을 생성하는 데 걸린 시간
- **지문 길이**: 생성된 지문의 문자 수
- **오류 메시지**: 실패한 경우의 구체적인 오류 내용

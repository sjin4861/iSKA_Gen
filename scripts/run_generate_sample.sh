#!/usr/bin/env bash
set -euo pipefail

# ---- vLLM 서버 인증키 (dummy로 통일) ----
export VLLM_API_KEY="${VLLM_API_KEY:-dummy}"

# ---- 사용자 환경에 맞게 조정 ----
BENCH_ID="${1:-1}"                 # 첫 번째 인자: 벤치 ID (기본 1)
SPEC_TYPE="${2:-small}"            # 두 번째 인자: 스펙 타입 (기본 small)
MODEL_NAME="${3:-A.X-4.0-Light}"   # 세 번째 인자: 생성 모델명
BASE_URL="${4:-http://localhost:8000/v1}" # 네 번째 인자: 생성 서버 URL

# (선택) 생성 파라미터
PASSAGE_TEMPERATURE="${PASSAGE_TEMPERATURE:-0.7}"
STEM_TEMPERATURE="${STEM_TEMPERATURE:-0.3}"
MAX_TOKENS="${MAX_TOKENS:-}"

# uv/venv 환경에서 실행 (uv를 쓰지 않으면 python 으로 교체)
cmd=(uv run python -m src.scripts.generate_single_sample_langchain
  --model "$MODEL_NAME"
  --bench-id "$BENCH_ID"
  --spec-type "$SPEC_TYPE"
  --gen-base-url "$BASE_URL"
  --passage-temperature "$PASSAGE_TEMPERATURE"
  --stem-temperature "$STEM_TEMPERATURE"
)

if [[ -n "$MAX_TOKENS" ]]; then
  cmd+=(--max-tokens "$MAX_TOKENS")
fi

echo ">>> Running: ${cmd[*]}"
"${cmd[@]}"

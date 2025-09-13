#!/usr/bin/env bash
set -euo pipefail

# ---- 사용자 환경에 맞게 조정 ----
INPUT_PATH="${1:-outputs/samples/2025-09-13/sample_1_0.jsonl}"  # 첫 번째 인자: 입력 sample jsonl
BENCH_ID="${2:-1}"                                              # 두 번째 인자: (호환용, 미사용)
RUBRICS_STR="${3:-all}"                                         # 세 번째 인자: 평가 루브릭 ("all" 또는 공백구분 다중)
MODEL_NAME="${4:-EXAONE-4.0-32B}"                               # 네 번째 인자: 평가 모델명
BASE_URL="${5:-http://localhost:8001/v1}"                       # 다섯 번째 인자: 평가 서버 URL

# rubrics 문자열을 배열로 분리 (예: "korean_quality l2_learner_suitability")
# shellcheck disable=SC2206
RUBRIC_ARGS=($RUBRICS_STR)

# 출력 경로 자동 생성 (입력 파일명 기반)
OUT_DIR="outputs/eval"
mkdir -p "$OUT_DIR"
BASENAME="$(basename "$INPUT_PATH" .jsonl)"

# (선택) 추가 파라미터 환경 변수
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-}"
EVAL_API_KEY="${EVAL_API_KEY:-}"     # vLLM 서버가 키를 요구하면 설정
INCLUDE_PASSAGE="${INCLUDE_PASSAGE:-0}"  # 1이면 --include-passage 추가

cmd=(uv run python -m src.scripts.evaluate_single_sample_langchain
  --input "$INPUT_PATH"
  --rubrics "${RUBRIC_ARGS[@]}"
  --eval-model "$MODEL_NAME"
  --eval-base-url "$BASE_URL"
  --temperature "$TEMPERATURE"
)

if [[ -n "$MAX_TOKENS" ]]; then
  cmd+=(--max-tokens "$MAX_TOKENS")
fi

if [[ -n "$EVAL_API_KEY" ]]; then
  cmd+=(--eval-api-key "$EVAL_API_KEY")
fi

if [[ "$INCLUDE_PASSAGE" == "1" ]]; then
  cmd+=(--include-passage)
fi

echo ">>> Running: ${cmd[*]}"
"${cmd[@]}"

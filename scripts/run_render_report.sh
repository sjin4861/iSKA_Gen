#!/usr/bin/env bash
set -euo pipefail

SAMPLE="${1:-outputs/samples/2025-09-13/sample_1_0.jsonl}"
EVAL="${2:-}"   # ← 빈 값이면 파이썬이 자동 추론

cmd=(uv run python -m src.scripts.render_sample_report --sample "$SAMPLE")
if [[ -n "$EVAL" ]]; then
  cmd+=(--eval "$EVAL")
fi

echo ">>> Running: ${cmd[*]}"
"${cmd[@]}"

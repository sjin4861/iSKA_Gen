#!/bin/bash

# 2025-08-08 EXAONE passage 데이터 기반으로 4개 모델이 stem 생성하는 배치 스크립트
# CUDA 초기화 문제 해결을 위해 각 모델을 별도 프로세스로 실행

echo "🚀 2025-08-08 EXAONE Passage 기반 Stem 생성 배치 시작"
echo "📅 날짜: 2025-08-08"
echo "📄 Passage 모델: EXAONE-3.5-7.8B-Instruct"

# 모델 목록
MODELS=(
    "A.X-4.0-Light"
    "EXAONE-3.5-7.8B-Instruct"
    "llama3.1_korean_v1.1_sft_by_aidx"
    "Midm-2.0-Base-Instruct"
)

# 결과 추적
total_success=0
total_attempts=0

# 각 모델별로 별도 프로세스 실행
for model in "${MODELS[@]}"; do
    echo ""
    echo "========================================================"
    echo "🤖 모델 '$model' 처리 시작"
    echo "========================================================"
    
    # 각 모델을 별도 Python 프로세스로 실행
    python src/scripts/stem_generator_single.py \
        --stem-model "$model" \
        --passage-model "EXAONE-3.5-7.8B-Instruct" \
        --date "2025-08-08" \
        --template-key "stem_agent.few_shot_new" \
        --gpus "0" \
        --bench-ids "1,2,3,4,5"
    
    exit_code=$?
    total_attempts=$((total_attempts + 5))  # 5개 벤치마크
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ 모델 '$model' 처리 성공"
        total_success=$((total_success + 5))  # 모든 벤치마크 성공으로 가정
    else
        echo "❌ 모델 '$model' 처리 실패 (exit code: $exit_code)"
    fi
    
    # 메모리 정리를 위한 잠시 대기
    echo "🧹 메모리 정리 중..."
    sleep 5
done

# 최종 결과
echo ""
echo "========================================================"
echo "🎉 전체 배치 작업 완료!"
echo "✅ 추정 성공: $total_success"
echo "📊 총 시도: $total_attempts"
echo "========================================================"

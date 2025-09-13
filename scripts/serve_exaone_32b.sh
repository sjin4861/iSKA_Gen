# 셸 2
export CUDA_VISIBLE_DEVICES=1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve ~/models/EXAONE-4.0-32B \
  --dtype bfloat16 \
  --pipeline-parallel-size 3 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16384 \
  --enforce-eager \
  --served-model-name EXAONE-4.0-32B \
  --api-key dummy \
  --port 8001

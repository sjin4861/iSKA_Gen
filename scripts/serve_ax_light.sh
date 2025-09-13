# 셸 1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

vllm serve ~/models/A.X-4.0-Light \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 16384 \
  --enforce-eager \
  --served-model-name A.X-4.0-Light \
  --api-key dummy \
  --port 8000

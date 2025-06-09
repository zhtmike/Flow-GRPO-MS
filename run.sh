#!/bin/sh
SD3_PATH="/home/hyx/models/stable-diffusion-3.5-medium"

export PYTHONPATH="/home/mikecheung/gitlocal/mindone:$PYTHONPATH"

# Model path
export QWEN_VL_PATH="/home/mikecheung/model/Qwen2.5-VL-7B-Instruct"
export QWEN_VL_OCR_PATH="/home/mikecheung/model/Qwen2.5-VL-7B-Instruct"
export CLIP_PATH="/mnt/disk4/mikecheung/model/clip-vit-large-patch14"
export PICKSCORE_PATH="/mnt/disk4/mikecheung/model/PickScore_v1"
export UNIFIED_REWARD_PATH="/mnt/disk4/mikecheung/model/UnifiedReward-qwen-7b"

# VLLM URl
export QWEN_VL_VLLM_URL="http://0.0.0.0:9529/v1"
export QWEN_VL_OCR_VLLM_URL="http://0.0.0.0:9529/v1"
export UNIFIED_REWARD_VLLM_URL="http://0.0.0.0:9529/v1"

export TOKENIZERS_PARALLELISM=False

# test with single scorer
# python -m flow_grpo.scorer.multi
# python -m flow_grpo.scorer.qwenvl
# python -m flow_grpo.scorer.compression
# python -m flow_grpo.scorer.aesthetic
# python -m flow_grpo.scorer.pickscore
# python -m flow_grpo.scorer.vllm

# training with one card
msrun --worker_num 1 --local_worker_num 1 --master_port 9527 --join True scripts/train_sd3.py \
    --reward jpeg-compressibility \
    --model $SD3_PATH \

#!/bin/sh
SD3_PATH="/home/hyx/models/stable-diffusion-3.5-medium"
WAN21_PATH="/home/mikecheung/model/Wan2.1-T2V-1.3B-Diffusers"

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
# python -m flow_grpo.scorer.aesthetic
# python -m flow_grpo.scorer.pickscore
# python -m flow_grpo.scorer.vllm
# python -m flow_grpo.scorer.misc

# training sd3 with one card
msrun --worker_num 1 --local_worker_num 1 --master_port 9527 --join True scripts/train_sd3.py \
    --reward jpeg-imcompressibility \
    --model $SD3_PATH

# training wan2.1 with two cards
# msrun --worker_num 2 --local_worker_num 2 --master_port 9527 --join True scripts/train_wan21.py \
#     --reward frame-smoothness \
#     --model $WAN21_PATH

# training wan2.1 with two cards (for quick experiment)
msrun --worker_num 2 --local_worker_num 2 --master_port 9527 --join True scripts/train_wan21.py \
    --reward constrast-change \
    --model $WAN21_PATH \
    --width 400 \
    --height 400 \
    --num-frames 25 \
    --num-steps 40 \
    --no-vae-save-memory

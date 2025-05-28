#!/bin/sh
export PYTHONPATH="/home/mikecheung/gitlocal/mindone:$PYTHONPATH"
export QWEN_VL_PATH="/home/mikecheung/model/Qwen2.5-VL-7B-Instruct"
export SD3_PATH="/home/hyx/models/stable-diffusion-3.5-medium"
export TOKENIZERS_PARALLELISM=False

# test with single scorer
# python -m flow_grpo.scorer.multi
# python -m flow_grpo.scorer.qwenvl
# python -m flow_grpo.scorer.compression

# training with one card
python scripts/train_sd3.py

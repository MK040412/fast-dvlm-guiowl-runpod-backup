#!/usr/bin/env bash
set -euo pipefail
cd /workspace/fast-dvlm-guiowl
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
OUT=/workspace/dvlm_ckpts/ckpt_bard_bd32_steer_sft
LOG=/workspace/paper_logs/steering_sft_300m_bf16_resume.log
/workspace/venv/bin/python train_steering_sft.py \
  --model /workspace/dvlm_ckpts/ckpt_bard_bd32 \
  --out "$OUT" \
  --resume-steering "$OUT" \
  --data /workspace/data/standard \
  --split train \
  --bd 32 \
  --mode teacher_forced_trace_sft \
  --epochs 1 \
  --lr 2e-4 \
  --weight-decay 0.01 \
  --optim adamw_fused \
  --grad-accum 8 \
  --dtype bf16 \
  --steering-dtype bf16 \
  --attn sdpa \
  --max-pixels 100352 \
  --ctx-cap 8192 \
  --tap-layers last4 \
  --inject-layers last8 \
  --steer-dim 2048 \
  --mla-dim 512 \
  --ssd-layers 3 \
  --mla-layers 1 \
  --n-steer-blocks 3 \
  --gate-center 0.35 \
  --gate-width 0.15 \
  --save-every 50 \
  --log-every 10 \
  --freeze-base true \
  --train-steering-only true \
  --struct-weight 1.5 \
  --coord-token-weight 2.0 \
  --lambda-action 0.1 \
  --lambda-coord 0.05 \
  --lambda-res 1e-4 \
  --producers 8 \
  --prefetch 16 2>&1 | tee "$LOG"

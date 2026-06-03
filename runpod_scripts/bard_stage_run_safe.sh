#!/bin/bash
set -euo pipefail

cd /workspace/fast-dvlm-guiowl

export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HOME=${HF_HOME:-/workspace/hf_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/workspace/triton_cache}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-28}

BASE_MODEL=${BASE_MODEL:-/workspace/models/GUI-Owl-1.5-2B-Instruct}
DATA_DIR=${DATA_DIR:-/workspace/data}
CKPT_ROOT=${CKPT_ROOT:-/workspace/dvlm_ckpts}
SHARD_START=${SHARD_START:-0}
SHARD_END=${SHARD_END:-256}
EPOCHS_PER_STAGE=${EPOCHS_PER_STAGE:-1}
MAX_BD=${MAX_BD:-8}
STAGES=${STAGES:-"2 4 8"}
LR=${LR:-3e-5}
WARMUP=${WARMUP:-50}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_PIXELS=${MAX_PIXELS:-100352}
DTYPE=${DTYPE:-bf16}
MAX_EP_STEPS=${MAX_EP_STEPS:-24}
CTX_CAP=${CTX_CAP:-8192}
PREFETCH=${PREFETCH:-32}
PRODUCERS=${PRODUCERS:-8}
SAVE_EVERY=${SAVE_EVERY:-100}
GRAD_CKPT=${GRAD_CKPT:-0}
LOG_EVERY=${LOG_EVERY:-20}
MAX_STEPS_PER_STAGE=${MAX_STEPS_PER_STAGE:-0}
EARLY_STOP_LOSS=${EARLY_STOP_LOSS:-0}
EARLY_STOP_WINDOW=${EARLY_STOP_WINDOW:-80}
EVAL_N=${EVAL_N:-7}
HF_UPLOAD=${HF_UPLOAD:-1}
HF_UPLOAD_SCRIPT=${HF_UPLOAD_SCRIPT:-/workspace/hf_upload_bard_stage.py}

if [ "$GRAD_CKPT" = "1" ]; then
  TRAIN_CKPT_ARG=(--grad-ckpt)
else
  TRAIN_CKPT_ARG=(--no-grad-ckpt)
fi

mkdir -p "$CKPT_ROOT"
mkdir -p /opt
ln -sfn "$CKPT_ROOT" /opt/dvlm_ckpts

echo "[safe] ckpt_root=$CKPT_ROOT opt_link=$(readlink -f /opt/dvlm_ckpts)"
echo "[safe] stages=$STAGES max_bd=$MAX_BD save_every=$SAVE_EVERY dtype=$DTYPE prefetch=$PREFETCH/$PRODUCERS grad_ckpt=$GRAD_CKPT"
echo "[safe] free=$(df -h /workspace | awk 'NR==2 {print $4}')"

src="$BASE_MODEL"
for bd in $STAGES; do
  if [ "$bd" -gt "$MAX_BD" ]; then
    continue
  fi
  out="$CKPT_ROOT/ckpt_bard_bd${bd}"
  log="/workspace/bard_bd${bd}.log"
  eval_log="/workspace/bard_bd${bd}_eval.log"
  echo "[stage] bd=${bd} src=${src} out=${out}" | tee "$log"

  /workspace/venv/bin/python train_full.py \
    --model "$src" --data-dir "$DATA_DIR" --out "$out" \
    --shard-start "$SHARD_START" --shard-end "$SHARD_END" --epochs "$EPOCHS_PER_STAGE" \
    --attn flex "${TRAIN_CKPT_ARG[@]}" --optim adamw_fused --grad-accum "$GRAD_ACCUM" \
    --max-pixels "$MAX_PIXELS" --dtype "$DTYPE" --max-ep-steps "$MAX_EP_STEPS" --ctx-cap "$CTX_CAP" \
    --lr "$LR" --warmup "$WARMUP" --anneal-steps 1 --bd-set "$bd" \
    --save-every "$SAVE_EVERY" --log-every "$LOG_EVERY" \
    --prefetch "$PREFETCH" --producers "$PRODUCERS" \
    --max-steps "$MAX_STEPS_PER_STAGE" \
    --early-stop-loss "$EARLY_STOP_LOSS" --early-stop-window "$EARLY_STOP_WINDOW" \
    2>&1 | tee -a "$log"

  if [ ! -f "$out/model.safetensors" ]; then
    echo "[verify] missing checkpoint file: $out/model.safetensors" | tee -a "$log"
    exit 3
  fi
  du -sh "$out" | awk '{print "[verify] saved_size="$1" path="$2}' | tee -a "$log"

  if [ "$HF_UPLOAD" = "1" ] && [ -f "$HF_UPLOAD_SCRIPT" ]; then
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 /workspace/androidworld_eval/venv/bin/python "$HF_UPLOAD_SCRIPT" \
      --folder "$out" --stage "bd${bd}" 2>&1 | tee -a "$log" || true
  fi

  if [ -f /workspace/data/standard/test-00000-of-00032.parquet ]; then
    echo "[eval] bd=${bd} held-out decode" | tee "$eval_log"
    /workspace/venv/bin/python scripts/decode_demo.py \
      --model "$out" --data /workspace/data/standard/test-00000-of-00032.parquet \
      --n "$EVAL_N" --gen-len 64 --max-pixels "$MAX_PIXELS" 2>&1 | tee -a "$eval_log" || true
  fi
  src="$out"
done

echo "[bard] DONE max_bd=${MAX_BD} final_src=${src}"

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
WAIT_PID_FILE=${WAIT_PID_FILE:-/workspace/bard_stage_retrain.pid}
WAIT_LOG_FILE=${WAIT_LOG_FILE:-/workspace/bard_stage_retrain.latest}
SAVE_EVERY=${SAVE_EVERY:-100}
LOG_EVERY=${LOG_EVERY:-20}
LR=${LR:-3e-5}
WARMUP=${WARMUP:-50}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_PIXELS=${MAX_PIXELS:-100352}
DTYPE=${DTYPE:-bf16}
MAX_EP_STEPS=${MAX_EP_STEPS:-24}
CTX_CAP=${CTX_CAP:-8192}
PREFETCH=${PREFETCH:-32}
PRODUCERS=${PRODUCERS:-8}
PAPER_N=${PAPER_N:-32}
DECODE_N=${DECODE_N:-32}
HF_UPLOAD=${HF_UPLOAD:-1}
HF_UPLOAD_SCRIPT=${HF_UPLOAD_SCRIPT:-/workspace/hf_upload_bard_stage.py}
PAPER_SUMMARY_SCRIPT=${PAPER_SUMMARY_SCRIPT:-/workspace/write_paper_stage_summary.py}
ANDROIDWORLD_BD8=${ANDROIDWORLD_BD8:-1}
ANDROIDWORLD_BD8_SCRIPT=${ANDROIDWORLD_BD8_SCRIPT:-/workspace/run_androidworld_bd8_compare.sh}

mkdir -p "$CKPT_ROOT" /workspace/paper_logs /opt
ln -sfn "$CKPT_ROOT" /opt/dvlm_ckpts

echo "[extend] wait_pid_file=$WAIT_PID_FILE ckpt_root=$CKPT_ROOT paper_n=$PAPER_N"
if [ -f "$WAIT_PID_FILE" ]; then
  wait_pid=$(cat "$WAIT_PID_FILE")
  echo "[extend] waiting for existing run pid=$wait_pid"
  while kill -0 "$wait_pid" 2>/dev/null; do
    sleep 60
  done
  echo "[extend] existing run finished pid=$wait_pid"
fi

if [ -f "$WAIT_LOG_FILE" ]; then
  echo "[extend] previous run log=$(cat "$WAIT_LOG_FILE")"
fi

run_eval_bundle() {
  local stage=$1
  local bd=$2
  local epoch=$3
  local src=$4
  local out=$5
  local train_log=$6
  local stage_dir="/workspace/paper_logs/${stage}"
  local decode_log="${stage_dir}/decode_demo_n${DECODE_N}.log"
  local compare_log="${stage_dir}/compare_grounding_n${PAPER_N}.log"

  mkdir -p "$stage_dir"
  cp "$train_log" "${stage_dir}/train.log" 2>/dev/null || true

  echo "[paper] stage=$stage decode_demo n=$DECODE_N"
  /workspace/venv/bin/python scripts/decode_demo.py \
    --model "$out" \
    --data /workspace/data/standard/test-00000-of-00032.parquet \
    --n "$DECODE_N" --gen-len 64 --max-pixels "$MAX_PIXELS" \
    > "$decode_log" 2>&1 || true

  echo "[paper] stage=$stage compare_grounding n=$PAPER_N"
  /workspace/venv/bin/python scripts/compare_grounding.py \
    --data /workspace/data/standard/test-00000-of-00032.parquet \
    --n "$PAPER_N" --gen-len 64 --max-pixels "$MAX_PIXELS" \
    --spec "${stage}_ar=${out}:ar" \
    --spec "${stage}_dvlm_strict=${out}:dvlm" \
    --spec "${stage}_dvlm_repair=${out}:dvlm:repair" \
    > "$compare_log" 2>&1 || true

  /workspace/androidworld_eval/venv/bin/python "$PAPER_SUMMARY_SCRIPT" \
    --stage "$stage" --bd "$bd" --epoch "$epoch" --source "$src" \
    --checkpoint "$out" --train-log "$train_log" \
    --decode-log "$decode_log" --compare-log "$compare_log" --out-dir "$stage_dir" || true

  if [ "$HF_UPLOAD" = "1" ] && [ -f "$HF_UPLOAD_SCRIPT" ]; then
    /workspace/androidworld_eval/venv/bin/python "$HF_UPLOAD_SCRIPT" \
      --folder "$out" --stage "$stage" --logs-dir "$stage_dir" 2>&1 | tee -a "$train_log" || true
  fi
}

run_train_stage() {
  local stage=$1
  local bd=$2
  local epoch=$3
  local src=$4
  local out=$5
  local log="/workspace/${stage}.log"

  echo "[stage] $stage bd=$bd epoch=$epoch src=$src out=$out" | tee "$log"
  /workspace/venv/bin/python train_full.py \
    --model "$src" --data-dir "$DATA_DIR" --out "$out" \
    --shard-start 0 --shard-end 256 --epochs 1 \
    --attn flex --no-grad-ckpt --optim adamw_fused --grad-accum "$GRAD_ACCUM" \
    --max-pixels "$MAX_PIXELS" --dtype "$DTYPE" --max-ep-steps "$MAX_EP_STEPS" --ctx-cap "$CTX_CAP" \
    --lr "$LR" --warmup "$WARMUP" --anneal-steps 1 --bd-set "$bd" \
    --save-every "$SAVE_EVERY" --log-every "$LOG_EVERY" \
    --prefetch "$PREFETCH" --producers "$PRODUCERS" \
    --max-steps 0 --early-stop-loss 0 --early-stop-window 80 \
    2>&1 | tee -a "$log"

  test -f "$out/model.safetensors"
  du -sh "$out" | awk '{print "[verify] saved_size="$1" path="$2}' | tee -a "$log"
  run_eval_bundle "$stage" "$bd" "$epoch" "$src" "$out" "$log"
}

for stage in bd2_e1 bd4_e1 bd8_e1; do
  case "$stage" in
    bd2_e1) bd=2; out="$CKPT_ROOT/ckpt_bard_bd2"; src="$BASE_MODEL"; train_log="/workspace/bard_bd2.log" ;;
    bd4_e1) bd=4; out="$CKPT_ROOT/ckpt_bard_bd4"; src="$CKPT_ROOT/ckpt_bard_bd2"; train_log="/workspace/bard_bd4.log" ;;
    bd8_e1) bd=8; out="$CKPT_ROOT/ckpt_bard_bd8"; src="$CKPT_ROOT/ckpt_bard_bd4"; train_log="/workspace/bard_bd8.log" ;;
  esac
  if [ -f "$out/model.safetensors" ]; then
    run_eval_bundle "$stage" "$bd" 1 "$src" "$out" "$train_log"
  else
    echo "[extend] missing previous checkpoint for $stage: $out"
    exit 4
  fi
done

if [ "$ANDROIDWORLD_BD8" = "1" ] && [ -x "$ANDROIDWORLD_BD8_SCRIPT" ]; then
  aw_log="/workspace/androidworld_bd8_compare_$(date +%Y%m%d_%H%M%S).log"
  echo "$aw_log" > /workspace/androidworld_bd8_compare.latest
  echo "[extend] AndroidWorld bd8 compare -> $aw_log"
  TASKS=${ANDROIDWORLD_TASKS:-ContactsAddContact,ClockStopWatchRunning} \
    N_TASK_COMBINATIONS=${ANDROIDWORLD_N_TASK_COMBINATIONS:-1} \
    bash "$ANDROIDWORLD_BD8_SCRIPT" > "$aw_log" 2>&1 || true
fi

run_train_stage bd16_e1 16 1 "$CKPT_ROOT/ckpt_bard_bd8" "$CKPT_ROOT/ckpt_bard_bd16"
run_train_stage bd32_e1 32 1 "$CKPT_ROOT/ckpt_bard_bd16" "$CKPT_ROOT/ckpt_bard_bd32_e1"
run_train_stage bd32_e2 32 2 "$CKPT_ROOT/ckpt_bard_bd32_e1" "$CKPT_ROOT/ckpt_bard_bd32_e2"
run_train_stage bd32_e3 32 3 "$CKPT_ROOT/ckpt_bard_bd32_e2" "$CKPT_ROOT/ckpt_bard_bd32"

ln -sfn "$CKPT_ROOT/ckpt_bard_bd32" "$CKPT_ROOT/final"
echo "[extend] DONE final=$CKPT_ROOT/ckpt_bard_bd32 paper_logs=/workspace/paper_logs"

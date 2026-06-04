#!/usr/bin/env bash
set -euo pipefail
source /data2/androidworld_eval/env.sh
TASK_SET=${TASK_SET:-general_core}
MODELS=${MODELS:-base,current}
PARALLEL=${PARALLEL:-1}
N_TASK_COMBINATIONS=${N_TASK_COMBINATIONS:-1}
TASK_RANDOM_SEED=${TASK_RANDOM_SEED:-30}
OUT_ROOT=${OUT_ROOT:-/data2/androidworld_eval/runs/$(date +%Y%m%d_%H%M%S)_${TASK_SET}}
BASE_URL=${BASE_URL:-http://127.0.0.1:8123}
CURRENT_URL=${CURRENT_URL:-http://127.0.0.1:8124}
BASE_CONSOLE=${BASE_CONSOLE:-5554}
BASE_GRPC=${BASE_GRPC:-8554}
if [ "$TASK_SET" = "standard_full" ]; then
  TASKS_FLAG=""
else
  TASKS_FLAG=$(cat "/data2/androidworld_eval/task_sets/${TASK_SET}.txt")
fi
mkdir -p "$OUT_ROOT"
echo "task_set=$TASK_SET" | tee "$OUT_ROOT/plan.txt"
echo "models=$MODELS" | tee -a "$OUT_ROOT/plan.txt"
echo "tasks=$TASKS_FLAG" | tee -a "$OUT_ROOT/plan.txt"
run_model() {
  local model_name=$1 idx=$2 url=$3
  local console=$((BASE_CONSOLE + 2*idx)); local grpc=$((BASE_GRPC + idx))
  NAME="${model_name}_${TASK_SET}" SERVER_URL="$url" CONSOLE_PORT="$console" GRPC_PORT="$grpc" \
  TASKS="$TASKS_FLAG" N_TASK_COMBINATIONS="$N_TASK_COMBINATIONS" TASK_RANDOM_SEED="$TASK_RANDOM_SEED" \
  OUT_ROOT="$OUT_ROOT" /data2/androidworld_eval/scripts/run_one_eval.sh
}
idx=0
pids=()
IFS=',' read -ra arr <<< "$MODELS"
for m in "${arr[@]}"; do
  if [ "$m" = "base" ]; then url="$BASE_URL"; else url="$CURRENT_URL"; fi
  if [ "$PARALLEL" = "1" ]; then
    run_model "$m" "$idx" "$url" >"$OUT_ROOT/${m}.runner.log" 2>&1 & pids+=("$!")
  else
    run_model "$m" "$idx" "$url" | tee "$OUT_ROOT/${m}.runner.log"
  fi
  idx=$((idx+1))
done
if [ "$PARALLEL" = "1" ]; then
  for p in "${pids[@]}"; do wait "$p"; done
fi
echo "DONE $OUT_ROOT"

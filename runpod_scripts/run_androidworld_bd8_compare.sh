#!/bin/bash
set -euo pipefail

source /workspace/androidworld_eval/env.sh

BASE_MODEL=${BASE_MODEL:-/workspace/models/GUI-Owl-1.5-2B-Instruct}
BD8_MODEL=${BD8_MODEL:-/workspace/dvlm_ckpts/ckpt_bard_bd8}
TASKS=${TASKS:-ContactsAddContact,ClockStopWatchRunning}
N_TASK_COMBINATIONS=${N_TASK_COMBINATIONS:-1}
OUT=${OUT:-/workspace/androidworld_eval/runs_bd8_compare}
LOG_ROOT=${LOG_ROOT:-/workspace/paper_logs/androidworld_bd8_compare}
ADB_PATH=${ADB_PATH:-/workspace/android-sdk/platform-tools/adb}
CONSOLE_PORT=${CONSOLE_PORT:-5554}

mkdir -p "$OUT" "$LOG_ROOT"
echo "[androidworld-bd8] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[androidworld-bd8] tasks=$TASKS n=$N_TASK_COMBINATIONS base=$BASE_MODEL bd8=$BD8_MODEL"
df -h /workspace /opt / || true
command -v adb || true
command -v emulator || true
command -v sdkmanager || true
ls -l /dev/kvm || true
adb devices || true

if [ ! -f "$BD8_MODEL/model.safetensors" ]; then
  echo "[androidworld-bd8] missing bd8 checkpoint: $BD8_MODEL/model.safetensors"
  exit 4
fi

if ! adb devices | awk 'NR>1 && $2=="device" {found=1} END {exit found?0:1}'; then
  echo "[androidworld-bd8] no adb device; trying non-KVM emulator for smoke availability"
  if ! pgrep -af "emulator.*Pixel_6_API_33" >/dev/null 2>&1; then
    nohup emulator -avd Pixel_6_API_33 -no-window -no-audio -no-boot-anim \
      -gpu swiftshader_indirect -accel off -no-snapshot -ports 5554,5555 \
      > "$LOG_ROOT/emulator.log" 2>&1 &
    echo $! > "$LOG_ROOT/emulator.pid"
  fi
  for _ in $(seq 1 60); do
    if adb devices | awk 'NR>1 && $2=="device" {found=1} END {exit found?0:1}'; then
      break
    fi
    sleep 10
  done
fi

if ! adb devices | awk 'NR>1 && $2=="device" {found=1} END {exit found?0:1}'; then
  echo "[androidworld-bd8] BLOCKED: no adb device. /dev/kvm is absent, so local emulator may not boot on this pod."
  cp "$LOG_ROOT/emulator.log" "$LOG_ROOT/emulator.tail.log" 2>/dev/null || true
  exit 0
fi

wait_health() {
  local port=$1
  for _ in $(seq 1 90); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

run_one() {
  local name=$1
  local model=$2
  local decode=$3
  local repair=$4
  local port=$5
  local server_log="$LOG_ROOT/${name}_server.log"
  local eval_log="$LOG_ROOT/${name}_eval.log"
  local run_out="$OUT/${name}"

  echo "[androidworld-bd8] server name=$name decode=$decode repair=$repair model=$model"
  GUIOWL_MODEL="$model" GUIOWL_DECODE="$decode" GUIOWL_SERVER_PORT="$port" \
    bash /workspace/androidworld_eval/start_guiowl_server.sh > "$server_log" 2>&1 &
  local pid=$!
  trap 'kill "$pid" 2>/dev/null || true' RETURN
  wait_health "$port"
  GUIOWL_REPAIR="$repair" TASKS="$TASKS" N_TASK_COMBINATIONS="$N_TASK_COMBINATIONS" \
    CONSOLE_PORT="$CONSOLE_PORT" ADB_PATH="$ADB_PATH" SERVER_URL="http://127.0.0.1:${port}" \
    OUT="$run_out" bash /workspace/androidworld_eval/run_guiowl_androidworld.sh > "$eval_log" 2>&1 || true
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

run_one base_ar "$BASE_MODEL" ar 0 8123
run_one bd8_dvlm_repair "$BD8_MODEL" dvlm 1 8124

echo "[androidworld-bd8] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ -f /workspace/hf_upload_bard_stage.py ]; then
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 /workspace/androidworld_eval/venv/bin/python /workspace/hf_upload_bard_stage.py \
    --folder "$BD8_MODEL" --stage androidworld_bd8_compare --logs-dir "$LOG_ROOT" || true
fi

#!/bin/bash
set -euo pipefail

# Requires one emulator for sequential comparison or two emulators for true
# parallel comparison. This pod currently has no adb/KVM, so run this only where
# AndroidWorld can connect to emulator console ports.

BASE_MODEL=${BASE_MODEL:-/workspace/models/GUI-Owl-1.5-2B-Instruct}
CURRENT_MODEL=${CURRENT_MODEL:-/opt/dvlm_ckpts/ckpt_bard_bd8}
TASKS=${TASKS:-ContactsAddContact,ClockStopWatchRunning}
N_TASK_COMBINATIONS=${N_TASK_COMBINATIONS:-1}
OUT=${OUT:-/opt/androidworld_eval/runs_compare}
ADB_PATH=${ADB_PATH:-/workspace/android-sdk/platform-tools/adb}
BASE_CONSOLE_PORT=${BASE_CONSOLE_PORT:-5554}
CURRENT_CONSOLE_PORT=${CURRENT_CONSOLE_PORT:-5556}
BASE_SERVER_PORT=${BASE_SERVER_PORT:-8123}
CURRENT_SERVER_PORT=${CURRENT_SERVER_PORT:-8124}
PARALLEL=${PARALLEL:-0}
GUIOWL_REPAIR=${GUIOWL_REPAIR:-1}

mkdir -p "$OUT"

start_server() {
  local model=$1
  local mode=$2
  local port=$3
  local log=$4
  (
    export GUIOWL_MODEL="$model"
    export GUIOWL_DECODE="$mode"
    export GUIOWL_SERVER_PORT="$port"
    export GUIOWL_REPAIR="$GUIOWL_REPAIR"
    exec /opt/androidworld_eval/start_guiowl_server.sh
  ) >"$log" 2>&1 &
  echo $!
}

run_eval() {
  local name=$1
  local console_port=$2
  local server_port=$3
  local log=$4
  (
    export TASKS="$TASKS"
    export N_TASK_COMBINATIONS="$N_TASK_COMBINATIONS"
    export CONSOLE_PORT="$console_port"
    export ADB_PATH="$ADB_PATH"
    export SERVER_URL="http://127.0.0.1:${server_port}"
    export OUT="${OUT}/${name}"
    exec /opt/androidworld_eval/run_guiowl_androidworld.sh
  ) >"$log" 2>&1
}

base_pid=$(start_server "$BASE_MODEL" ar "$BASE_SERVER_PORT" "$OUT/base_server.log")
current_pid=$(start_server "$CURRENT_MODEL" dvlm "$CURRENT_SERVER_PORT" "$OUT/current_server.log")
trap 'kill "$base_pid" "$current_pid" 2>/dev/null || true' EXIT
sleep 20

if [ "$PARALLEL" = "1" ]; then
  run_eval base "$BASE_CONSOLE_PORT" "$BASE_SERVER_PORT" "$OUT/base_eval.log" &
  p1=$!
  run_eval current "$CURRENT_CONSOLE_PORT" "$CURRENT_SERVER_PORT" "$OUT/current_eval.log" &
  p2=$!
  wait "$p1" "$p2"
else
  run_eval base "$BASE_CONSOLE_PORT" "$BASE_SERVER_PORT" "$OUT/base_eval.log"
  run_eval current "$BASE_CONSOLE_PORT" "$CURRENT_SERVER_PORT" "$OUT/current_eval.log"
fi

echo "Wrote compare logs to $OUT"

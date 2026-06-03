#!/bin/bash
set -euo pipefail

cd /opt/androidworld_eval
TASKS=${TASKS:-ContactsAddContact,ClockStopWatchRunning}
N_TASK_COMBINATIONS=${N_TASK_COMBINATIONS:-1}
CONSOLE_PORT=${CONSOLE_PORT:-5554}
GRPC_PORT=${GRPC_PORT:-8554}
ADB_PATH=${ADB_PATH:-/workspace/android-sdk/platform-tools/adb}
SERVER_URL=${SERVER_URL:-http://127.0.0.1:8123}
OUT=${OUT:-/opt/androidworld_eval/runs}

exec /opt/androidworld_eval/venv/bin/python /opt/androidworld_eval/run_guiowl_androidworld.py \
  --suite_family=android_world \
  --tasks="$TASKS" \
  --n_task_combinations="$N_TASK_COMBINATIONS" \
  --console_port="$CONSOLE_PORT" \
  --grpc_port="$GRPC_PORT" \
  --adb_path="$ADB_PATH" \
  --guiowl_server_url="$SERVER_URL" \
  --output_path="$OUT"

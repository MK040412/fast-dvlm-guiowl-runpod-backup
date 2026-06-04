#!/usr/bin/env bash
set -euo pipefail
source /data2/androidworld_eval/env.sh
source /data2/androidworld_eval/venv/bin/activate
N=${1:-1}
BASE_CONSOLE=${BASE_CONSOLE:-5554}
BASE_GRPC=${BASE_GRPC:-8554}
FORCE_SETUP=${FORCE_SETUP:-0}
for i in $(seq 0 $((N-1))); do
  console=$((BASE_CONSOLE + 2*i)); grpc=$((BASE_GRPC + i))
  log="/data2/androidworld_eval/install_logs/androidworld_setup_${console}.log"
  if [ "$FORCE_SETUP" != "1" ] && [ -f "$log" ] && grep -q "ENV_READY ${console} ${grpc}" "$log"; then
    echo "[androidworld setup] skip ready console=$console grpc=$grpc"
    continue
  fi
  echo "[androidworld setup] console=$console grpc=$grpc"
  python /data2/androidworld_eval/setup_env_once.py \
    --adb_path "$ANDROID_HOME/platform-tools/adb" \
    --console_port "$console" \
    --grpc_port "$grpc" \
    --perform_emulator_setup \
    >"$log" 2>&1
  tail -n 20 "$log"
done

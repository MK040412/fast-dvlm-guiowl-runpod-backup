#!/usr/bin/env bash
set -euo pipefail
source /data2/androidworld_eval/env.sh
N=${1:-1}
BASE_CONSOLE=${BASE_CONSOLE:-5554}
BASE_GRPC=${BASE_GRPC:-8554}
adb devices
for i in $(seq 0 $((N-1))); do
  console=$((BASE_CONSOLE + 2*i)); grpc=$((BASE_GRPC + i)); dev="emulator-${console}"
  echo "--- $dev grpc=$grpc ---"
  timeout 5 adb -s "$dev" shell getprop sys.boot_completed || echo boot_timeout
  timeout 5 adb -s "$dev" shell wm size || echo wm_timeout
  timeout 10 adb -s "$dev" exec-out screencap -p >"/data2/android-emulator-logs/${dev}_check.png" || echo screencap_timeout
  file "/data2/android-emulator-logs/${dev}_check.png" 2>/dev/null || true
  ss -ltn | grep -E ":(${grpc}|${console}|$((console+1))) " || true
done

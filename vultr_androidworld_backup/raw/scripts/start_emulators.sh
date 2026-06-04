#!/usr/bin/env bash
set -euo pipefail
source /data2/androidworld_eval/env.sh
N=${1:-1}
API_LEVEL=${API_LEVEL:-33}
AVD_PREFIX=${AVD_PREFIX:-aw_pixel${API_LEVEL}}
BASE_CONSOLE=${BASE_CONSOLE:-5554}
BASE_GRPC=${BASE_GRPC:-8554}
GPU_MODE=${GPU_MODE:-swiftshader_indirect}
mkdir -p /data2/android-emulator-logs
adb start-server > /data2/android-emulator-logs/adb_start.log 2>&1 || true
for i in $(seq 0 $((N-1))); do
  name="${AVD_PREFIX}_${i}"
  console=$((BASE_CONSOLE + 2*i))
  adbport=$((console + 1))
  grpc=$((BASE_GRPC + i))
  if adb devices | grep -q "emulator-${console}[[:space:]]"; then
    echo "[emu] already running $name console=$console grpc=$grpc"
    continue
  fi
  echo "[emu] start $name console=$console adb=$adbport grpc=$grpc"
  nohup emulator -avd "$name" \
    -ports "${console},${adbport}" \
    -grpc "$grpc" \
    -no-window -no-audio -gpu "$GPU_MODE" \
    -no-snapshot -no-boot-anim -accel on \
    -netdelay none -netspeed full \
    >"/data2/android-emulator-logs/${name}.log" 2>&1 &
  echo $! >"/data2/android-emulator-logs/${name}.pid"
done
for i in $(seq 0 $((N-1))); do
  console=$((BASE_CONSOLE + 2*i))
  dev="emulator-${console}"
  echo "[wait] $dev"
  for t in $(seq 1 150); do
    state=$(adb devices | awk -v d="$dev" '$1==d {print $2}')
    boot=$(adb -s "$dev" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r' || true)
    if [ "$state" = "device" ] && [ "$boot" = "1" ]; then
      echo "[ready] $dev"
      adb -s "$dev" shell settings put global window_animation_scale 0 >/dev/null 2>&1 || true
      adb -s "$dev" shell settings put global transition_animation_scale 0 >/dev/null 2>&1 || true
      adb -s "$dev" shell settings put global animator_duration_scale 0 >/dev/null 2>&1 || true
      break
    fi
    sleep 2
  done
done
adb devices

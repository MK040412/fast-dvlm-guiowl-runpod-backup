#!/usr/bin/env bash
set -euo pipefail
source /data2/androidworld_eval/env.sh
adb devices | awk '/emulator-/{print $1}' | while read -r d; do
  echo "[stop] adb emu kill $d"
  adb -s "$d" emu kill >/dev/null 2>&1 || true
done
pkill -f 'qemu-system-x86_64.*aw_pixel35_' 2>/dev/null || true
sleep 2
adb kill-server >/dev/null 2>&1 || true

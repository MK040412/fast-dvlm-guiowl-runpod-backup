#!/usr/bin/env bash
set -euo pipefail
source /data2/androidworld_eval/env.sh
N=${1:-8}
API_LEVEL=${API_LEVEL:-33}
IMAGE=${IMAGE:-system-images;android-${API_LEVEL};google_apis;x86_64}
DEVICE=${DEVICE:-pixel_2}
AVD_PREFIX=${AVD_PREFIX:-aw_pixel${API_LEVEL}}
mkdir -p "$ANDROID_AVD_HOME" /data2/android-emulator-logs
for i in $(seq 0 $((N-1))); do
  name="${AVD_PREFIX}_${i}"
  echo "[avd] create/update $name image=$IMAGE"
  echo no | avdmanager create avd -n "$name" -k "$IMAGE" -d "$DEVICE" --force >"/data2/android-emulator-logs/${name}_create.log" 2>&1 || {
    cat "/data2/android-emulator-logs/${name}_create.log"
    exit 1
  }
  cfg="$ANDROID_AVD_HOME/${name}.avd/config.ini"
  if [ -f "$cfg" ]; then
    sed -i "s/^hw.ramSize=.*/hw.ramSize=4096/" "$cfg" || true
    grep -q '^hw.keyboard=' "$cfg" && sed -i 's/^hw.keyboard=.*/hw.keyboard=yes/' "$cfg" || echo 'hw.keyboard=yes' >> "$cfg"
    grep -q '^disk.dataPartition.size=' "$cfg" && sed -i 's/^disk.dataPartition.size=.*/disk.dataPartition.size=8G/' "$cfg" || echo 'disk.dataPartition.size=8G' >> "$cfg"
    grep -q '^hw.gpu.enabled=' "$cfg" && sed -i 's/^hw.gpu.enabled=.*/hw.gpu.enabled=yes/' "$cfg" || echo 'hw.gpu.enabled=yes' >> "$cfg"
  fi
done
avdmanager list avd

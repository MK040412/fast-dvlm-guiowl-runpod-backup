#!/usr/bin/env bash
set -euo pipefail

# Skeleton bootstrap for a fresh Vultr/Ubuntu 22.04 AndroidWorld eval host.
# Review RUNBOOK.md before running this. Android SDK commandline-tools must be
# downloaded into /data2/android-sdk/cmdline-tools/latest before sdkmanager works.

apt-get update
apt-get install -y \
  git curl wget unzip rsync openjdk-17-jdk \
  python3.11 python3.11-venv python3-pip qemu-kvm

mkdir -p /data2/androidworld_eval /data2/android-sdk /data2/android-avd /data2/android-emulator-logs

cp vultr_androidworld_backup/env.template.sh /data2/androidworld_eval/env.sh
rsync -a vultr_androidworld_backup/raw/ /data2/androidworld_eval/
chmod +x /data2/androidworld_eval/scripts/*.sh

source /data2/androidworld_eval/env.sh

cat <<'MSG'
Next manual steps:
1. Install Android commandline-tools into /data2/android-sdk/cmdline-tools/latest.
2. Run:
   yes | sdkmanager --licenses
   sdkmanager "platform-tools" "emulator" "platforms;android-33" "system-images;android-33;google_apis;x86_64"
3. Clone AndroidWorld at the commit recorded in androidworld_git_state.txt.
4. Create /data2/androidworld_eval/venv and install packages.
5. Run API_LEVEL=33 bash /data2/androidworld_eval/scripts/prepare_avds.sh 8.
6. Run API_LEVEL=33 bash /data2/androidworld_eval/scripts/start_emulators.sh 8.
7. Start RunPod model servers and create SSH remote forwards for ports 8123-8130.
MSG


# Runbook

This is the minimal procedure for recreating the Vultr AndroidWorld evaluation
host and running the same benchmark style again.

## 1. Prepare Host

Use a machine with KVM access. The previous host used 24 cores / 48 threads and
256 GB RAM. Eight Android emulators were stable on that machine.

Install base packages:

```bash
apt-get update
apt-get install -y git curl wget unzip rsync openjdk-17-jdk python3.11 python3.11-venv python3-pip qemu-kvm
```

Create directories:

```bash
mkdir -p /data2/androidworld_eval /data2/android-sdk /data2/android-avd /data2/android-emulator-logs
```

Restore this backup:

```bash
rsync -a vultr_androidworld_backup/raw/ /data2/androidworld_eval/
chmod +x /data2/androidworld_eval/scripts/*.sh
```

Environment:

```bash
source /data2/androidworld_eval/env.sh
```

The expected env file is:

```bash
export ANDROID_HOME=/data2/android-sdk
export ANDROID_SDK_ROOT=/data2/android-sdk
export ANDROID_AVD_HOME=/data2/android-avd
export PATH="$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$ANDROID_HOME/cmdline-tools/latest/bin:$PATH"
export PYTHONPATH=/data2/androidworld_eval:/data2/androidworld_eval/android_world:${PYTHONPATH:-}
```

## 2. Install Android SDK

Install command line tools under:

```text
/data2/android-sdk/cmdline-tools/latest
```

Then install the SDK packages:

```bash
source /data2/androidworld_eval/env.sh
yes | sdkmanager --licenses
sdkmanager \
  "platform-tools" \
  "emulator" \
  "platforms;android-33" \
  "system-images;android-33;google_apis;x86_64"
```

The original setup also had Android 30/35 AVDs, but the active 8-lane eval used
Android 33 AVDs named `aw_pixel33_0` through `aw_pixel33_7`.

## 3. Install AndroidWorld

Clone AndroidWorld:

```bash
cd /data2/androidworld_eval
git clone https://github.com/google-research/android_world.git
cd android_world
git checkout d9c569f764b3a5629321858de03ff653d0f24056
```

Create venv and install:

```bash
cd /data2/androidworld_eval
python3.11 -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r android_world/requirements.txt
pip install -e android_world
pip install fastapi uvicorn requests opencv-python huggingface_hub
```

`python_freeze.txt` records the exact package versions from the original host.

## 4. Create And Start 8 AVDs

Create AVDs:

```bash
cd /data2/androidworld_eval
source env.sh
source venv/bin/activate
API_LEVEL=33 bash scripts/prepare_avds.sh 8
```

Start emulators:

```bash
API_LEVEL=33 bash scripts/start_emulators.sh 8
```

Expected ports:

```text
console: 5554,5556,5558,5560,5562,5564,5566,5568
grpc:    8554,8555,8556,8557,8558,8559,8560,8561
```

Check:

```bash
adb devices
bash scripts/check_emulators.sh
```

## 5. Start RunPod Model Servers

On the RunPod GPU host, start eight GUI-Owl/Fast-dVLM HTTP servers on ports
`8123-8130`.

For bd32 dVLM eval:

```bash
for p in 8123 8124 8125 8126 8127 8128 8129 8130; do
  GUIOWL_MODEL=/workspace/dvlm_ckpts/ckpt_bard_bd32 \
  GUIOWL_DECODE=dvlm \
  GUIOWL_SERVER_PORT=$p \
  GUIOWL_REPAIR=1 \
  GUIOWL_COORD_MODE=normalized \
  GUIOWL_PROMPT_MODE=mobileagent \
  GUIOWL_INCLUDE_UI_ELEMENTS=0 \
  GUIOWL_GEN_LEN=128 \
  bash /opt/androidworld_eval/start_guiowl_server.sh \
  >"/workspace/paper_logs/androidworld_servers/current_bd32_${p}.log" 2>&1 &
done
```

For base AR comparison, use the base model and AR decode:

```bash
GUIOWL_MODEL=/workspace/models/GUI-Owl-1.5-2B-Instruct \
GUIOWL_DECODE=ar \
GUIOWL_SERVER_PORT=8123 \
GUIOWL_REPAIR=0 \
GUIOWL_COORD_MODE=normalized \
GUIOWL_PROMPT_MODE=mobileagent \
bash /opt/androidworld_eval/start_guiowl_server.sh
```

## 6. Forward RunPod Server Ports To Vultr

The previous setup exposed RunPod model servers to the Vultr runner as local
ports `127.0.0.1:8123-8130` on Vultr.

Run this from the RunPod host, replacing `<VULTR_USER_HOST>` with the new eval
host SSH target:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -R 127.0.0.1:8123:127.0.0.1:8123 \
  -R 127.0.0.1:8124:127.0.0.1:8124 \
  -R 127.0.0.1:8125:127.0.0.1:8125 \
  -R 127.0.0.1:8126:127.0.0.1:8126 \
  -R 127.0.0.1:8127:127.0.0.1:8127 \
  -R 127.0.0.1:8128:127.0.0.1:8128 \
  -R 127.0.0.1:8129:127.0.0.1:8129 \
  -R 127.0.0.1:8130:127.0.0.1:8130 \
  <VULTR_USER_HOST>
```

On Vultr, verify:

```bash
for p in 8123 8124 8125 8126 8127 8128 8129 8130; do
  curl -fsS "http://127.0.0.1:${p}/health"
  echo
done
```

## 7. Run Benchmarks

Current bd32 dVLM, general-core:

```bash
cd /data2/androidworld_eval
source env.sh
source venv/bin/activate

OUT=/data2/androidworld_eval/runs/current_general_core_sweep_$(date +%Y%m%d_%H%M%S)
TASK_SET=general_core \
LANES=8 \
CURRENT_COORD_MODE=normalized \
N_TASK_COMBINATIONS=1 \
TASK_RANDOM_SEED=30 \
OUT_ROOT="$OUT" \
bash scripts/run_eval_current_8lane.sh
```

Current bd32 dVLM, general-focus expanded, three combinations:

```bash
OUT=/data2/androidworld_eval/runs/current_general_focus_multicombo_$(date +%Y%m%d_%H%M%S)
TASK_SET=general_focus_expanded \
LANES=8 \
CURRENT_COORD_MODE=normalized \
N_TASK_COMBINATIONS=3 \
TASK_RANDOM_SEED=41 \
OUT_ROOT="$OUT" \
bash scripts/run_eval_current_8lane.sh
```

Current bd32 dVLM, standard full one combination:

```bash
OUT=/data2/androidworld_eval/runs/current_standard_full_sweep_$(date +%Y%m%d_%H%M%S)
TASK_SET=standard_full \
LANES=8 \
CURRENT_COORD_MODE=normalized \
N_TASK_COMBINATIONS=1 \
TASK_RANDOM_SEED=42 \
OUT_ROOT="$OUT" \
bash scripts/run_eval_current_8lane.sh
```

Base AR vs current dVLM smoke comparison:

```bash
TASK_SET=smoke_norm_core \
MODELS=base,current \
PARALLEL=1 \
BASE_URL=http://127.0.0.1:8123 \
CURRENT_URL=http://127.0.0.1:8124 \
N_TASK_COMBINATIONS=1 \
TASK_RANDOM_SEED=30 \
bash scripts/run_eval_matrix.sh
```

## 8. Summarize Existing Runs

```bash
python scripts/summarize_androidworld_run.py /data2/androidworld_eval/runs/<run_dir>
cat /data2/androidworld_eval/runs/<run_dir>/summary.json
```

The summarizer was patched to handle AndroidWorld `episode_data` stored either
as a list of per-step dicts or as a dict of per-step lists.

## 9. Stop

Stop AndroidWorld runner processes:

```bash
pkill -f /data2/androidworld_eval/run_guiowl_androidworld.py || true
```

Stop emulators:

```bash
bash /data2/androidworld_eval/scripts/stop_emulators.sh
```


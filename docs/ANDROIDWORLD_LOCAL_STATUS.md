# AndroidWorld Local Status

This document records the current local AndroidWorld setup status and what is
safe to claim from it.

## Environment Split

There are two separate local environments:

```text
runner env:
  /home/perelman/androidworld_eval_local/.venv
  purpose: official AndroidWorld runner and task registry
  android_world: installed from local clone

model server env:
  /home/perelman/miniconda3/envs/internvla_a1
  purpose: PyTorch / transformers / GUI-Owl model serving
  torch: 2.7.1+cu128
  transformers: 4.57.1
```

`/home/perelman/local_gmail_eval/venv` exists but is not the useful environment
for current model/AndroidWorld execution.

## Local Model Server Check

Step-3000 checkpoint:

```text
HF repo: KMK040412/fast-dvlm-guiowl-kd-tpu
path: fast-dvlm-kd-tpu/aw-overfit-norm1000-full-pad480-cap96/checkpoint-step003000
```

Local path:

```text
/home/perelman/local_gmail_eval/models/ckpt_kd_aw_step3000/fast-dvlm-kd-tpu/aw-overfit-norm1000-full-pad480-cap96/checkpoint-step003000
```

The local server was started as bd32 dVLM:

```text
mode: dvlm
bd: 32
tau: 0.9
grounded: 1
port: 8123
```

Health check result:

```json
{
  "ok": true,
  "mode": "dvlm",
  "bd": 32,
  "tau": 0.9
}
```

This proves the checkpoint can be loaded and served locally. It does not prove
task success.

## AndroidWorld Runner State

Official AndroidWorld package:

```text
clone root: /home/perelman/androidworld_eval_local/android_world
installed env: /home/perelman/androidworld_eval_local/.venv
task registry: android_world
known task count: 116
```

The runner wrapper and GUI-Owl agent are from:

```text
/home/perelman/fast-dvlm-guiowl-runpod-backup/vultr_androidworld_backup/raw/run_guiowl_androidworld.py
/home/perelman/fast-dvlm-guiowl-runpod-backup/vultr_androidworld_backup/raw/guiowl_androidworld_agent.py
```

The agent supports:

```text
GUIOWL_REPAIR=1
GUIOWL_COORD_MODE=normalized
```

## Emulator State

Local machine has KVM and ADB. The old AVD:

```text
gmail_pixel34
Android 14 / API34 / Google Play image
```

was unstable for official AndroidWorld gRPC runs and exited before completing a
smoke benchmark.

A standard AndroidWorld-friendly image was installed:

```text
platforms;android-33
system-images;android-33;google_apis;x86_64
```

Preferred next AVD:

```text
aw_pixel33_local
API33 Google APIs x86_64
gRPC port: 8554
console/ADB ports: 5554/5555
```

## What Is Not Yet Proven

Do not claim:

- step-3000 passed AndroidWorld
- bd32 dVLM has task-level success from official AndroidWorld
- local benchmark is paper-grade
- repaired metrics are strict metrics

Current safe claim:

```text
The local model server can load and serve the step-3000 checkpoint. The official
AndroidWorld runner environment is installed. Emulator stability remains the
blocking local infrastructure issue for completing official smoke tasks.
```

## Required Reporting Format

Every AndroidWorld run should save:

- output directory
- task list
- random seed
- model checkpoint path
- decode mode, block size, tau
- strict JSON rate
- strict mobile_use rate
- repaired mobile_use rate
- repaired count
- task success/failure per task
- raw model outputs
- parsed/repaired actions
- screenshots/videos when available
- latency per step

Strict and repaired metrics must be separated.
